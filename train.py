from __future__ import print_function

import logging
import numpy
import os
import pickle
import cPickle;

from collections import Counter
from theano import tensor
from theano import  function
from toolz import merge

from blocks.algorithms import (GradientDescent, StepClipping, AdaDelta,
                               CompositeRule)
from blocks.extensions import FinishAfter, Printing
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph, apply_noise, apply_dropout
from blocks.initialization import IsotropicGaussian, Orthogonal, Constant
from blocks.main_loop import MainLoop
from blocks.model import Model
#from blocks.search import BeamSearch
from search import BeamSearch;
from blocks.select import Selector
from blocks.serialization import load_parameter_values

from checkpoint import CheckpointNMT, LoadNMT
from model import BidirectionalEncoder, Decoder, topicalq_transformer
from sampling import BleuValidator, Sampler, SamplingBase
from stream import (get_tr_stream_with_topicalq,get_tr_stream_unsorted, get_dev_stream_with_topicalq,
                                        _ensure_special_tokens)

try:
    from blocks.extras.extensions.plot import Plot
    BOKEH_AVAILABLE = True
except ImportError:
    BOKEH_AVAILABLE = False

logger = logging.getLogger(__name__)


def main(mode, config, use_bokeh=False):

    # Construct model
    logger.info('Building RNN encoder-decoder')
    encoder = BidirectionalEncoder(
        config['src_vocab_size'], config['enc_embed'], config['enc_nhids'])
    decoder = Decoder(
        config['trg_vocab_size'], config['dec_embed'], config['dec_nhids'],
        config['enc_nhids'] * 2,config['topical_embedding_dim'])
    topical_transformer=topicalq_transformer(config['topical_vocab_size'],config['topical_embedding_dim'], config['enc_nhids'],config['topical_word_num'],config['batch_size']);

    if mode == "train":

        # Create Theano variables
        logger.info('Creating theano variables')
        source_sentence = tensor.lmatrix('source')
        source_sentence_mask = tensor.matrix('source_mask')
        target_sentence = tensor.lmatrix('target')
        target_sentence_mask = tensor.matrix('target_mask')
        sampling_input = tensor.lmatrix('input')
        source_topical_word=tensor.lmatrix('source_topical')
        source_topical_mask=tensor.matrix('source_topical_mask')

        # Get training and development set streams
        tr_stream = get_tr_stream_with_topicalq(**config)
        dev_stream = get_dev_stream_with_topicalq(**config)
        topic_embedding=topical_transformer.apply(source_topical_word);
        # Get cost of the model
        representation=encoder.apply(source_sentence, source_sentence_mask);
        tw_representation=topical_transformer.look_up.apply(source_topical_word.T);
        content_embedding=representation[0,:,(representation.shape[2]/2):];

        cost = decoder.cost(
            representation,source_sentence_mask,tw_representation,
            source_topical_mask, target_sentence, target_sentence_mask,topic_embedding,content_embedding);

        logger.info('Creating computational graph')
        cg = ComputationGraph(cost)

        # Initialize model
        logger.info('Initializing model')
        encoder.weights_init = decoder.weights_init = IsotropicGaussian(
            config['weight_scale'])
        encoder.biases_init = decoder.biases_init = Constant(0)
        encoder.push_initialization_config()
        decoder.push_initialization_config()
        encoder.bidir.prototype.weights_init = Orthogonal()
        decoder.transition.weights_init = Orthogonal()
        encoder.initialize()
        decoder.initialize()
        topical_transformer.weights_init=IsotropicGaussian(
            config['weight_scale']);
        topical_transformer.biases_init=Constant(0);
        topical_transformer.push_allocation_config();#don't know whether the initialize is for
        topical_transformer.look_up.weights_init=Orthogonal();
        topical_transformer.transformer.weights_init=Orthogonal();
        topical_transformer.initialize();
        word_topical_embedding=cPickle.load(open(config['topical_embeddings'], 'rb'));
        np_word_topical_embedding=numpy.array(word_topical_embedding,dtype='float32');
        topical_transformer.look_up.W.set_value(np_word_topical_embedding);
        topical_transformer.look_up.W.tag.role=[];


        # apply dropout for regularization
        if config['dropout'] < 1.0:
            # dropout is applied to the output of maxout in ghog
            logger.info('Applying dropout')
            dropout_inputs = [x for x in cg.intermediary_variables
                              if x.name == 'maxout_apply_output']
            cg = apply_dropout(cg, dropout_inputs, config['dropout'])

        # Apply weight noise for regularization
        if config['weight_noise_ff'] > 0.0:
            logger.info('Applying weight noise to ff layers')
            enc_params = Selector(encoder.lookup).get_params().values()
            enc_params += Selector(encoder.fwd_fork).get_params().values()
            enc_params += Selector(encoder.back_fork).get_params().values()
            dec_params = Selector(
                decoder.sequence_generator.readout).get_params().values()
            dec_params += Selector(
                decoder.sequence_generator.fork).get_params().values()
            dec_params += Selector(decoder.state_init).get_params().values()
            cg = apply_noise(
                cg, enc_params+dec_params, config['weight_noise_ff'])

        # Print shapes
        shapes = [param.get_value().shape for param in cg.parameters]
        logger.info("Parameter shapes: ")
        for shape, count in Counter(shapes).most_common():
            logger.info('    {:15}: {}'.format(shape, count))
        logger.info("Total number of parameters: {}".format(len(shapes)))

        # Print parameter names
        enc_dec_param_dict = merge(Selector(encoder).get_parameters(),
                                   Selector(decoder).get_parameters())
        logger.info("Parameter names: ")
        for name, value in enc_dec_param_dict.items():
            logger.info('    {:15}: {}'.format(value.get_value().shape, name))
        logger.info("Total number of parameters: {}"
                    .format(len(enc_dec_param_dict)))

        # Set up training model
        logger.info("Building model")
        training_model = Model(cost)

        # Set extensions
        logger.info("Initializing extensions")
        extensions = [
            FinishAfter(after_n_batches=config['finish_after']),
            TrainingDataMonitoring([cost], after_batch=True),
            Printing(after_batch=True),
            CheckpointNMT(config['saveto'],
                          every_n_batches=config['save_freq'])
        ]
        '''
        # Set up beam search and sampling computation graphs if necessary
        if config['hook_samples'] >= 1 or config['bleu_script'] is not None:
            logger.info("Building sampling model")
            sampling_representation = encoder.apply(
                sampling_input, tensor.ones(sampling_input.shape))
            generated = decoder.generate(
                sampling_input, sampling_representation)
            search_model = Model(generated)
            _, samples = VariableFilter(
                bricks=[decoder.sequence_generator], name="outputs")(
                    ComputationGraph(generated[1]))

        # Add sampling
        if config['hook_samples'] >= 1:
            logger.info("Building sampler")
            extensions.append(
                Sampler(model=search_model, data_stream=tr_stream,
                        hook_samples=config['hook_samples'],
                        every_n_batches=config['sampling_freq'],
                        src_vocab_size=config['src_vocab_size']))

        # Add early stopping based on bleu
        if config['bleu_script'] is not None:
            logger.info("Building bleu validator")
            extensions.append(
                BleuValidator(sampling_input, samples=samples, config=config,
                              model=search_model, data_stream=dev_stream,
                              normalize=config['normalized_bleu'],
                              every_n_batches=config['bleu_val_freq']))
        '''

        # Reload model if necessary
        if config['reload']:
            extensions.append(LoadNMT(config['saveto']))

        # Plot cost in bokeh if necessary
        if use_bokeh and BOKEH_AVAILABLE:
            extensions.append(
                Plot('Cs-En', channels=[['decoder_cost_cost']],
                     after_batch=True))

        # Set up training algorithm
        logger.info("Initializing training algorithm")
        algorithm = GradientDescent(
            cost=cost, parameters=cg.parameters,on_unused_sources='warn',
            step_rule=CompositeRule([StepClipping(config['step_clipping']),
                                     eval(config['step_rule'])()])
        )

        # Initialize main loop
        logger.info("Initializing main loop")
        main_loop = MainLoop(
            model=training_model,
            algorithm=algorithm,
            data_stream=tr_stream,
            extensions=extensions
        )

        # Train!
        main_loop.run()

    elif mode == 'translate':

        # Create Theano variables
        logger.info('Creating theano variables')
        source_sentence = tensor.lmatrix('source')
        source_topical_word=tensor.lmatrix('source_topical')

        # Get test set stream
        test_stream = get_dev_stream_with_topicalq(
            config['test_set'], config['src_vocab'],
            config['src_vocab_size'],config['topical_test_set'],config['topical_vocab'],config['topical_vocab_size'],config['unk_id'])
        ftrans = open(config['test_set'] + '.trans.out', 'w')

        # Helper utilities
        sutils = SamplingBase()
        unk_idx = config['unk_id']
        src_eos_idx = config['src_vocab_size'] - 1
        trg_eos_idx = config['trg_vocab_size'] - 1

        # Get beam search
        logger.info("Building sampling model")
        topic_embedding=topical_transformer.apply(source_topical_word);
        representation=encoder.apply(source_sentence, tensor.ones(source_sentence.shape));
        tw_representation=topical_transformer.look_up.apply(source_topical_word.T);
        content_embedding=representation[0,:,(representation.shape[2]/2):];
        generated = decoder.generate(source_sentence,representation, tw_representation,topical_embedding=topic_embedding,content_embedding=content_embedding);


        _, samples = VariableFilter(
            bricks=[decoder.sequence_generator], name="outputs")(
                ComputationGraph(generated[1]))  # generated[1] is next_outputs
        beam_search = BeamSearch(samples=samples)

        logger.info("Loading the model..")
        model = Model(generated)
        loader = LoadNMT(config['saveto'])
        loader.set_model_parameters(model, loader.load_parameters())

        # Get target vocabulary
        trg_vocab = _ensure_special_tokens(
            pickle.load(open(config['trg_vocab'], 'rb')), bos_idx=0,
            eos_idx=trg_eos_idx, unk_idx=unk_idx)
        trg_ivocab = {v: k for k, v in trg_vocab.items()}

        logger.info("Started translation: ")
        total_cost = 0.0

        for i, line in enumerate(test_stream.get_epoch_iterator()):

            seq = sutils._oov_to_unk(
                line[0], config['src_vocab_size'], unk_idx)
            seq2 = line[1];
            input_ = numpy.tile(seq, (config['beam_size'], 1))
            input_topical=numpy.tile(seq2,(config['beam_size'],1))


            # draw sample, checking to ensure we don't get an empty string back
            trans, costs = \
                beam_search.search(
                    input_values={source_sentence: input_,source_topical_word:input_topical},
                    max_length=10*len(seq), eol_symbol=src_eos_idx,
                    ignore_first_eol=True)
            '''
            # normalize costs according to the sequence lengths
            if config['normalized_bleu']:
                lengths = numpy.array([len(s) for s in trans])
                costs = costs / lengths
            '''
            #best = numpy.argsort(costs)[0]
            best=numpy.argsort(costs)[0:config['beam_size']];
            for b in best:
                try:
                    total_cost += costs[b]
                    trans_out = trans[b]

                    # convert idx to words
                    trans_out = sutils._idx_to_word(trans_out, trg_ivocab)

                except ValueError:
                    logger.info(
                        "Can NOT find a translation for line: {}".format(i+1))
                    trans_out = '<UNK>'

                print(trans_out, file=ftrans)

            if i != 0 and i % 100 == 0:
                logger.info(
                    "Translated {} lines of test set...".format(i))

        logger.info("Total cost of the test: {}".format(total_cost))
        ftrans.close()
    elif mode == 'rerank':
        # Create Theano variables
        ftrans = open(config['val_set'] + '.scores.out', 'w')
        logger.info('Creating theano variables')
        source_sentence = tensor.lmatrix('source')
        source_sentence_mask = tensor.matrix('source_mask')
        target_sentence = tensor.lmatrix('target')
        target_sentence_mask = tensor.matrix('target_mask')

        config['src_data']=config['val_set']
        config['trg_data']=config['val_set_grndtruth']
        config['batch_size']=1;
        config['sort_k_batches']=1;
        test_stream=get_tr_stream_unsorted(**config);
        logger.info("Building sampling model")
        representations= encoder.apply(
            source_sentence,  source_sentence_mask)
        costs = decoder.cost(representations, source_sentence_mask,
            target_sentence, target_sentence_mask)
        logger.info("Loading the model..")
        model = Model(costs)
        loader = LoadNMT(config['saveto'])
        loader.set_model_parameters(model, loader.load_parameters())

        costs_computer = function([source_sentence,source_sentence_mask,
                                  target_sentence,
                                  target_sentence_mask],costs)
        iterator = test_stream.get_epoch_iterator()

        scores = []
        for i, (src, src_mask, trg, trg_mask) in enumerate(iterator):
            costs = costs_computer(*[src, src_mask, trg, trg_mask])
            cost = costs.sum()
            print(i, cost)
            scores.append(cost)
            ftrans.write(str(cost)+"\n");
        ftrans.close();










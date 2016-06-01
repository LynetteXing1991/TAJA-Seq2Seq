def get_config_fr2en():
    config = {}

    # Model related -----------------------------------------------------------

    # Sequences longer than this will be discarded
    config['seq_len'] = 50

    # Number of hidden units in encoder/decoder GRU
    config['enc_nhids'] = 1000
    config['dec_nhids'] = 1000

    # Dimension of the word embedding matrix in encoder/decoder
    config['enc_embed'] = 620
    config['dec_embed'] = 620

    # Where to save model, this corresponds to 'prefix' in groundhog
    config['saveto'] = 'search_model_fr2en'

    # Optimization related ----------------------------------------------------

    # Batch size
    config['batch_size'] = 80

    # This many batches will be read ahead and sorted
    config['sort_k_batches'] = 12

    # Optimization step rule
    config['step_rule'] = 'AdaDelta'

    # Gradient clipping threshold
    config['step_clipping'] = 1.

    # Std of weight initialization
    config['weight_scale'] = 0.01

    # Regularization related --------------------------------------------------

    # Weight noise flag for feed forward layers
    config['weight_noise_ff'] = False

    # Weight noise flag for recurrent layers
    config['weight_noise_rec'] = False

    # Dropout ratio, applied only after readout maxout
    config['dropout'] = 1.0

    # Vocabulary/dataset related ----------------------------------------------

    # Root directory for dataset
    datadir = open('datadir_fr2en', 'rb').readline().rstrip()

    # Module name of the stream that will be used
    config['stream'] = 'stream'

    # Original data
    config['src_original'] = datadir + '200k/' + 'fr'
    config['trg_original'] = datadir + '200k/' + 'en'

    # Source and target vocabularies
    config['src_vocab'] = datadir + '200k/' + 'vocab.fr-en.fr.pkl'
    config['trg_vocab'] = datadir + '200k/' + 'vocab.fr-en.en.pkl'

    # Source and target datasets
    config['src_data'] = datadir + '200k/' + 'fr.tok.shuf'
    config['trg_data'] = datadir + '200k/' + 'en.tok.shuf'

    # Source and target vocabulary sizes, should include bos, eos, unk tokens
    config['src_vocab_size'] = 30000
    config['trg_vocab_size'] = 30000

    # Special tokens and indexes
    config['unk_id'] = 1
    config['bos_token'] = '<S>'
    config['eos_token'] = '</S>'
    config['unk_token'] = '<UNK>'

    # Early stopping based on bleu related ------------------------------------

    # Normalize cost according to sequence length after beam-search
    config['normalized_bleu'] = False

    # Bleu script that will be used (moses multi-perl in this case)
    config['bleu_script'] = datadir + 'multi-bleu.perl'

    # Validation set source file
    config['val_set'] = datadir + 'valid/' + 'fr.tok'

    # Validation set gold file
    config['val_set_grndtruth'] = datadir + 'valid/' + 'en.tok'

    # Print validation output to file
    config['output_val_set'] = False

    # Validation output file
    config['val_set_out'] = config['saveto'] + '/validation_out.txt'

    # Test set
    config['test_set'] = datadir + 'test/' + 'fr.tok'

    # Beam-size
    config['beam_size'] = 12

    # Timing/monitoring related -----------------------------------------------

    # Maximum number of updates
    config['finish_after'] = 1000000

    # Reload model from files if exist
    config['reload'] = True

    # Save model after this many updates
    config['save_freq'] = 50

    # Show samples from model after this many updates
    config['sampling_freq'] = 10

    # Show this many samples at each sampling
    config['hook_samples'] = 3

    # Validate bleu after this many updates
    config['bleu_val_freq'] = 5000

    # Start bleu validation after this many updates
    config['val_burn_in'] = 80000

    return config

def get_config_en2ch():
    config = get_config_fr2en()

    # Where to save model, this corresponds to 'prefix' in groundhog
    config['saveto'] = 'search_model_en2ch'

    # Root directory for dataset
    datadir = open('datadir_en2ch', 'rb').readline().rstrip()

    # Module name of the stream that will be used
    config['stream'] = 'stream'

    # Original data
    config['src_original'] = datadir + '200k/' + 'en'
    config['trg_original'] = datadir + '200k/' + 'ch'

    # Source and target vocabularies
    config['src_vocab'] = datadir + '200k/' + 'vocab.en-ch.en.pkl'
    config['trg_vocab'] = datadir + '200k/' + 'vocab.en-ch.ch.pkl'

    # Source and target datasets
    config['src_data'] = datadir + '200k/' + 'en.tok.shuf'
    config['trg_data'] = datadir + '200k/' + 'ch.tok.shuf'

    # Bleu script that will be used (moses multi-perl in this case)
    config['bleu_script'] = datadir + 'multi-bleu.perl'

    # Validation set source file
    config['val_set'] = datadir + 'valid/' + 'en.tok'

    # Validation set gold file
    config['val_set_grndtruth'] = datadir + 'valid/' + 'ch.tok'

    # Test set
    config['test_set'] = datadir + 'test/' + 'en.tok'

    return config

def get_config_ch2en():
    config = get_config_fr2en()

    # Where to save model, this corresponds to 'prefix' in groundhog
    config['saveto'] = 'search_model_ch2en'

    # Root directory for dataset
    datadir = open('datadir_en2ch', 'rb').readline().rstrip()

    # Module name of the stream that will be used
    config['stream'] = 'stream'

    # Original data
    config['src_original'] = datadir + '200k/' + 'ch'
    config['trg_original'] = datadir + '200k/' + 'en'

    # Source and target vocabularies
    config['src_vocab'] = datadir + '200k/' + 'vocab.en-ch.ch.pkl'
    config['trg_vocab'] = datadir + '200k/' + 'vocab.en-ch.en.pkl'

    # Source and target datasets
    config['src_data'] = datadir + '200k/' + 'ch.tok.shuf'
    config['trg_data'] = datadir + '200k/' + 'en.tok.shuf'

    # Bleu script that will be used (moses multi-perl in this case)
    config['bleu_script'] = datadir + 'multi-bleu.perl'

    # Validation set source file
    config['val_set'] = datadir + 'valid/' + 'ch.tok'

    # Validation set gold file
    config['val_set_grndtruth'] = datadir + 'valid/' + 'en.tok'

    # Test set
    config['test_set'] = datadir + 'test/' + 'ch.tok'

    return config

def get_config_qa():
    config = get_config_fr2en()

    datadir = open('datadir_qa', 'rb').readline().rstrip()

    # Original data
    config['src_original'] = datadir + '200k/' + 'q'
    config['trg_original'] = datadir + '200k/' + 'a'

    # Source and target vocabularies
    config['src_vocab'] = datadir + '200k/' + 'vocab.qa.q.pkl'
    config['trg_vocab'] = datadir + '200k/' + 'vocab.qa.a.pkl'

    # Source and target datasets
    config['src_data'] = datadir + '200k/' + 'q.tok.shuf'
    config['trg_data'] = datadir + '200k/' + 'a.tok.shuf'
    # Bleu script that will be used (moses multi-perl in this case)
    config['bleu_script'] = datadir + 'multi-bleu.perl'

    # Validation set source file
    config['val_set'] = datadir + 'valid/' + 'q.tok'

    # Validation set gold file
    config['val_set_grndtruth'] = datadir + 'valid/' + 'a.tok'

    # Test set
    config['test_set'] = datadir + 'test/' + 'q.tok'

    # Where to save model, this corresponds to 'prefix' in groundhog
    config['saveto'] = 'search_model_qa'

    return config

def get_config_ch_s2sa_charLevel():
    config = {}
    # Model related -----------------------------------------------------------
    # Sequences longer than this will be discarded
    config['seq_len'] = 100

    # Number of hidden units in encoder/decoder GRU
    config['enc_nhids'] = 1000
    config['dec_nhids'] = 1000

    # Dimension of the word embedding matrix in encoder/decoder
    config['enc_embed'] = 620
    config['dec_embed'] = 620

    # Where to save model, this corresponds to 'prefix' in groundhog
    config['saveto'] = 'D:\users\chxing\data\chinese\s2s_attention_input_bigdata\chinese_s2sa_charLevel'

    # Optimization related ----------------------------------------------------

    # Batch size
    config['batch_size'] = 256

    # This many batches will be read ahead and sorted?
    config['sort_k_batches'] = 12

    # Optimization step rule
    config['step_rule'] = 'AdaDelta'


    # Gradient clipping threshold?
    config['step_clipping'] = 1.

    # Std of weight initialization
    config['weight_scale'] = 0.01

    # Regularization related --------------------------------------------------

    # Weight noise flag for feed forward layers
    config['weight_noise_ff'] = False

    # Weight noise flag for recurrent layers
    config['weight_noise_rec'] = False

    # Dropout ratio, applied only after readout maxout
    config['dropout'] = 1.0

    # Vocabulary/dataset related ----------------------------------------------

    # Root directory for dataset
    datadir = open('D:\users\chxing\data\chinese\s2s_attention_input_bigdata\datadir_chinese_s2sa', 'rb').readline().rstrip()

    # Module name of the stream that will be used
    config['stream'] = 'stream'

    # Source and target vocabularies
    config['src_vocab'] = datadir + 'charLevel_query_vocab.pkl'
    config['trg_vocab'] = datadir + 'charLevel_response_vocab.pkl'

    # Source and target datasets
    config['src_data'] = datadir + 'charLevel_chinese_query.txt'
    config['trg_data'] = datadir + 'charLevel_chinese_response.txt'

    # Source and target vocabulary sizes, should include bos, eos, unk tokens
    config['src_vocab_size'] = 5462
    config['trg_vocab_size'] = 6904

    # Special tokens and indexes
    config['unk_id'] = 1
    config['bos_token'] = '<S>'
    config['eos_token'] = '</S>'
    config['unk_token'] = '<UNK>'
    # Validation set source file
    config['val_set'] = datadir + 'charLevel_chinese_query_vali.txt'

    # Validation set gold file
    config['val_set_grndtruth'] = datadir + 'charLevel_chinese_response_vali.txt'

    # Print validation output to file
    config['output_val_set'] = False

    # Validation output file
    config['val_set_out'] = config['saveto'] + '/validation_out.txt'

    # Test set
    config['test_set'] = datadir + 'charLevel_xiaoIceTest.txt'

    # Beam-size
    config['beam_size'] = 5
    # Timing/monitoring related -----------------------------------------------

    # Maximum number of updates
    config['finish_after'] = 1000000

    # Reload model from files if exist
    config['reload'] = True

    # Save model after this many updates
    config['save_freq'] = 50

    # Show samples from model after this many updates
    config['sampling_freq'] = 50

    # Show this many samples at each sampling
    config['hook_samples'] = 4

    config['bleu_script'] = None

    return config

def get_config_ch_s2sa():
    config = {}
    # Model related -----------------------------------------------------------
    # Sequences longer than this will be discarded
    config['seq_len'] = 100

    # Number of hidden units in encoder/decoder GRU
    config['enc_nhids'] = 1000
    config['dec_nhids'] = 1000

    # Dimension of the word embedding matrix in encoder/decoder
    config['enc_embed'] = 620
    config['dec_embed'] = 620

    # Where to save model, this corresponds to 'prefix' in groundhog
    config['saveto'] = 'D:\users\chxing\data\chinese\s2s_attention_input_bigdata\chinese_s2sa_normal'

    # Optimization related ----------------------------------------------------

    # Batch size
    config['batch_size'] = 256

    # This many batches will be read ahead and sorted?
    config['sort_k_batches'] = 12

    # Optimization step rule
    config['step_rule'] = 'AdaDelta'


    # Gradient clipping threshold?
    config['step_clipping'] = 1.

    # Std of weight initialization
    config['weight_scale'] = 0.01

    # Regularization related --------------------------------------------------

    # Weight noise flag for feed forward layers
    config['weight_noise_ff'] = False

    # Weight noise flag for recurrent layers
    config['weight_noise_rec'] = False

    # Dropout ratio, applied only after readout maxout
    config['dropout'] = 1.0

    # Vocabulary/dataset related ----------------------------------------------

    # Root directory for dataset
    datadir = open('D:\users\chxing\data\chinese\s2s_attention_input_bigdata\datadir_chinese_s2sa', 'rb').readline().rstrip()

    # Module name of the stream that will be used
    config['stream'] = 'stream'

    # Source and target vocabularies
    config['src_vocab'] = datadir + 'tokensized_chinese_query_vocab.pkl'
    config['trg_vocab'] = datadir + 'tokensized_chinese_response_vocab.pkl'

    # Source and target datasets
    config['src_data'] = datadir + 'tokenized_chinese_query.txt'
    config['trg_data'] = datadir + 'tokenized_chinese_response.txt'

    # Source and target vocabulary sizes, should include bos, eos, unk tokens
    config['src_vocab_size'] = 20000
    config['trg_vocab_size'] = 20000

    # Special tokens and indexes
    config['unk_id'] = 1
    config['bos_token'] = '<S>'
    config['eos_token'] = '</S>'
    config['unk_token'] = '<UNK>'
    # Validation set source file
    config['val_set'] = datadir + 'tokenized_xiaoIceTest-s2sa.txt.trans.out'

    # Validation set gold file
    config['val_set_grndtruth'] = datadir + 'tokenized_xiaoIceTest_dup50.txt'

    # Print validation output to file
    config['output_val_set'] = False

    # Validation output file
    config['val_set_out'] = config['saveto'] + '/validation_out.txt'

    # Test set
    config['test_set'] = datadir + 'tokenized_xiaoIceTest-s2sa.txt'

    # Beam-size
    config['beam_size'] = 50
    # Timing/monitoring related -----------------------------------------------

    # Maximum number of updates
    config['finish_after'] = 1000000

    # Reload model from files if exist
    config['reload'] = True

    # Save model after this many updates
    config['save_freq'] = 50

    # Show samples from model after this many updates
    config['sampling_freq'] = 50

    # Show this many samples at each sampling
    config['hook_samples'] = 4

    config['bleu_script'] = None

    return config

def get_config_ch_style():
    config = {}
    # Model related -----------------------------------------------------------
    # Sequences longer than this will be discarded
    config['seq_len'] = 100

    # Number of hidden units in encoder/decoder GRU
    config['enc_nhids'] = 1000
    config['dec_nhids'] = 1000

    # Dimension of the word embedding matrix in encoder/decoder
    config['enc_embed'] = 620
    config['dec_embed'] = 620

    # Where to save model, this corresponds to 'prefix' in groundhog
    config['saveto'] = 'D:\users\chxing\data\japanese\s2sa'

    # Optimization related ----------------------------------------------------

    # Batch size
    config['batch_size'] = 256

    # This many batches will be read ahead and sorted?
    config['sort_k_batches'] = 12

    # Optimization step rule
    config['step_rule'] = 'AdaDelta'


    # Gradient clipping threshold?
    config['step_clipping'] = 1.

    # Std of weight initialization
    config['weight_scale'] = 0.01

    # Regularization related --------------------------------------------------

    # Weight noise flag for feed forward layers
    config['weight_noise_ff'] = False

    # Weight noise flag for recurrent layers
    config['weight_noise_rec'] = False

    # Dropout ratio, applied only after readout maxout
    config['dropout'] = 1.0

    # Vocabulary/dataset related ----------------------------------------------

    # Root directory for dataset
    #datadir = open('D:\users\chxing\data\chinese\s2s_attention_input_bigdata\datadir_chinese_s2sa', 'rb').readline().rstrip()
    datadir='D:\users\chxing\data\japanese\\';
    # Module name of the stream that will be used
    config['stream'] = 'stream'

    # Source and target vocabularies
    config['src_vocab'] = datadir + 'charLevel_jap_query_vocab.pkl'
    config['trg_vocab'] = datadir + 'charLevel_jap_response_vocab.pkl'

    # Source and target datasets
    config['src_data'] = datadir + 'charLevel_jap_train.txt'
    config['trg_data'] = datadir + 'charLevel_jap_train_response.txt'

    # Source and target vocabulary sizes, should include bos, eos, unk tokens
    config['src_vocab_size'] = 5363
    config['trg_vocab_size'] = 5321

    # Special tokens and indexes
    config['unk_id'] = 1
    config['bos_token'] = '<S>'
    config['eos_token'] = '</S>'
    config['unk_token'] = '<UNK>'
    # Validation set source file
    config['val_set'] = datadir + 'charLevel_jap_vali.txt'

    # Validation set gold file
    config['val_set_grndtruth'] = datadir + 'charLevel_jap_vali_response.txt'

    # Print validation output to file
    config['output_val_set'] = False

    # Validation output file
    config['val_set_out'] = config['saveto'] + '/validation_out.txt'

    # Test set
    config['test_set'] = datadir + 'charLevel_jap_test.txt'

    # Beam-size
    config['beam_size'] = 5
    # Timing/monitoring related -----------------------------------------------

    # Maximum number of updates
    config['finish_after'] = 1000000

    # Reload model from files if exist
    config['reload'] = True

    # Save model after this many updates
    config['save_freq'] = 50

    # Show samples from model after this many updates
    config['sampling_freq'] = 50

    # Show this many samples at each sampling
    config['hook_samples'] = 4

    config['bleu_script'] = None

    return config
def get_config_ch_style_zhenhuan():
    config = {}
    # Model related -----------------------------------------------------------
    # Sequences longer than this will be discarded
    config['seq_len'] = 100

    # Number of hidden units in encoder/decoder GRU
    config['enc_nhids'] = 1000
    config['dec_nhids'] = 1000

    # Dimension of the word embedding matrix in encoder/decoder
    config['enc_embed'] = 620
    config['dec_embed'] = 620

    # Where to save model, this corresponds to 'prefix' in groundhog
    config['saveto'] = 'D:\users\chxing\data\style\zhenhuan_s2sa'

    # Optimization related ----------------------------------------------------

    # Batch size
    config['batch_size'] = 256

    # This many batches will be read ahead and sorted?
    config['sort_k_batches'] = 12

    # Optimization step rule
    config['step_rule'] = 'AdaDelta'


    # Gradient clipping threshold?
    config['step_clipping'] = 1.

    # Std of weight initialization
    config['weight_scale'] = 0.01

    # Regularization related --------------------------------------------------

    # Weight noise flag for feed forward layers
    config['weight_noise_ff'] = False

    # Weight noise flag for recurrent layers
    config['weight_noise_rec'] = False

    # Dropout ratio, applied only after readout maxout
    config['dropout'] = 1.0

    # Vocabulary/dataset related ----------------------------------------------

    # Root directory for dataset
    #datadir = open('D:\users\chxing\data\chinese\s2s_attention_input_bigdata\datadir_chinese_s2sa', 'rb').readline().rstrip()
    datadir='D:\users\chxing\data\style\\';
    # Module name of the stream that will be used
    config['stream'] = 'stream'

    # Source and target vocabularies
    config['src_vocab'] = datadir + 'zhenhuan_charLevel_vocab.pkl'
    config['trg_vocab'] = datadir + 'zhenhuan_charLevel_vocab.pkl'

    # Source and target datasets
    config['src_data'] = datadir + 'zhenhuan_split_sentences.txt'
    config['trg_data'] = datadir + 'zhenhuan_split_sentences.txt'

    # Source and target vocabulary sizes, should include bos, eos, unk tokens
    config['src_vocab_size'] = 1024
    config['trg_vocab_size'] = 1024

    # Special tokens and indexes
    config['unk_id'] = 1
    config['bos_token'] = '<S>'
    config['eos_token'] = '</S>'
    config['unk_token'] = '<UNK>'
    # Validation set source file
    config['val_set'] = datadir + 'zhenhuan_split_sentences.txt'

    # Validation set gold file
    config['val_set_grndtruth'] = datadir + 'zhenhuan_split_sentences.txt'

    # Print validation output to file
    config['output_val_set'] = False

    # Validation output file
    config['val_set_out'] = config['saveto'] + '/validation_out.txt'

    # Test set
    config['test_set'] = datadir + 'zhenhuan_split_sentences.txt'

    # Beam-size
    config['beam_size'] = 5
    # Timing/monitoring related -----------------------------------------------

    # Maximum number of updates
    config['finish_after'] = 1000000

    # Reload model from files if exist
    config['reload'] = True

    # Save model after this many updates
    config['save_freq'] = 50

    # Show samples from model after this many updates
    config['sampling_freq'] = 50

    # Show this many samples at each sampling
    config['hook_samples'] = 4

    config['bleu_script'] = None

    return config
def get_config_ch_topical():
    config = {}
    # Model related -----------------------------------------------------------
    # Sequences longer than this will be discarded
    config['seq_len'] = 100

    # Number of hidden units in encoder/decoder GRU
    config['enc_nhids'] = 1000
    config['dec_nhids'] = 1000

    # Dimension of the word embedding matrix in encoder/decoder
    config['enc_embed'] = 620
    config['dec_embed'] = 620

    # Where to save model, this corresponds to 'prefix' in groundhog
    config['saveto'] = 'D:\users\chxing\data\\topical\chinese\\100wpositive\\topical_tw_attention_debug'

    # Optimization related ----------------------------------------------------

    # Batch size
    config['batch_size'] = 128

    # This many batches will be read ahead and sorted?
    config['sort_k_batches'] = 12

    # Optimization step rule
    config['step_rule'] = 'AdaDelta'


    # Gradient clipping threshold?
    config['step_clipping'] = 1.

    # Std of weight initialization
    config['weight_scale'] = 0.01

    # Regularization related --------------------------------------------------

    # Weight noise flag for feed forward layers
    config['weight_noise_ff'] = False

    # Weight noise flag for recurrent layers
    config['weight_noise_rec'] = False

    # Dropout ratio, applied only after readout maxout
    config['dropout'] = 1.0

    # Vocabulary/dataset related ----------------------------------------------

    # Root directory for dataset
    datadir='D:\users\chxing\data\\topical\chinese\\100wpositive\\';
    # Module name of the stream that will be used
    config['stream'] = 'stream'

    # Source and target vocabularies
    config['src_vocab'] = datadir + '100wgeneration.2w960.source.vocab.pkl'
    config['trg_vocab'] = datadir + '100wgeneration.2w968.target.vocab.pkl'

    # Source and target datasets
    config['src_data'] = datadir + 'train100w.generation.source'
    config['trg_data'] = datadir + 'train100w.generation.target'

    # Source and target vocabulary sizes, should include bos, eos, unk tokens
    config['src_vocab_size'] = 20000
    config['trg_vocab_size'] = 20000

    # Special tokens and indexes
    config['unk_id'] = 1
    config['bos_token'] = '<S>'
    config['eos_token'] = '</S>'
    config['unk_token'] = '<UNK>'
    # Validation set source file
    config['val_set'] = datadir + 'dev100w.generation.source'

    # Validation set gold file
    config['val_set_grndtruth'] = datadir + 'dev100w.generation.target'

    # Print validation output to file
    config['output_val_set'] = False

    # Validation output file
    config['val_set_out'] = config['saveto'] + '/validation_out.txt'

    # Test set
    config['test_set'] = datadir + 'test100w.generation.source'
    config['topical_test_set']= datadir + 'test100w.generation.10topical'

    # Beam-size
    config['beam_size'] = 50
    # Timing/monitoring related -----------------------------------------------

    # Maximum number of updates
    config['finish_after'] = 1000000

    # Reload model from files if exist
    config['reload'] = True

    # Save model after this many updates
    config['save_freq'] = 50

    # Show samples from model after this many updates
    config['sampling_freq'] = 50

    # Show this many samples at each sampling
    config['hook_samples'] = 4

    config['bleu_script'] = None

    #topical related features--------------------------------------------------------
    config['topical_vocab_size']=1735

    config['topical_embedding_dim']=200

    config['topical_word_num']=10

    config['topical_embeddings']='D:\users\chxing\data\\topical\chinese\\tLDAtopic\word_topic_normalize.tt10.pkl'

    config['topical_vocab']='D:\users\chxing\data\\topical\chinese\\tLDAtopic\LDADic.tt10.pkl';

    config['topical_data']= datadir + 'train100w.generation.10topical'

    return config
def get_config_ch_topical_former_douban():
    config = {}
    # Model related -----------------------------------------------------------
    # Sequences longer than this will be discarded
    config['seq_len'] = 100

    # Number of hidden units in encoder/decoder GRU
    config['enc_nhids'] = 1000
    config['dec_nhids'] = 1000

    # Dimension of the word embedding matrix in encoder/decoder
    config['enc_embed'] = 620
    config['dec_embed'] = 620

    # Where to save model, this corresponds to 'prefix' in groundhog
    config['saveto'] = 'D:\users\chxing\data\\topical\chinese\\former_douban\\topical_tw_attention'

    # Optimization related ----------------------------------------------------

    # Batch size
    config['batch_size'] = 128

    # This many batches will be read ahead and sorted?
    config['sort_k_batches'] = 12

    # Optimization step rule
    config['step_rule'] = 'AdaDelta'


    # Gradient clipping threshold?
    config['step_clipping'] = 1.

    # Std of weight initialization
    config['weight_scale'] = 0.01

    # Regularization related --------------------------------------------------

    # Weight noise flag for feed forward layers
    config['weight_noise_ff'] = False

    # Weight noise flag for recurrent layers
    config['weight_noise_rec'] = False

    # Dropout ratio, applied only after readout maxout
    config['dropout'] = 1.0

    # Vocabulary/dataset related ----------------------------------------------

    # Root directory for dataset
    datadir='D:\users\chxing\data\\topical\chinese\\former_douban\\';
    # Module name of the stream that will be used
    config['stream'] = 'stream'

    # Source and target vocabularies
    config['src_vocab'] = datadir + 'tokensized_chinese_query_vocab.pkl'
    config['trg_vocab'] = datadir + 'tokensized_chinese_response_vocab.pkl'

    # Source and target datasets
    config['src_data'] = datadir + 'tokenized_chinese_query.txt'
    config['trg_data'] = datadir + 'tokenized_chinese_response.txt'

    # Source and target vocabulary sizes, should include bos, eos, unk tokens
    config['src_vocab_size'] = 20000
    config['trg_vocab_size'] = 20000

    # Special tokens and indexes
    config['unk_id'] = 1
    config['bos_token'] = '<S>'
    config['eos_token'] = '</S>'
    config['unk_token'] = '<UNK>'
    # Validation set source file
    config['val_set'] = datadir + 'tokenized_chinese_query_vali.txt'

    # Validation set gold file
    config['val_set_grndtruth'] = datadir + 'tokenized_chinese_response_vali.txt'

    # Print validation output to file
    config['output_val_set'] = False

    # Validation output file
    config['val_set_out'] = config['saveto'] + '/validation_out.txt'

    # Test set
    config['test_set'] = datadir + 'tokenized_chinese_query_test.s2sta_iter30.txt'
    config['topical_test_set']= datadir + 'test_former_douban.generation.10topical'

    # Beam-size
    config['beam_size'] = 50
    # Timing/monitoring related -----------------------------------------------

    # Maximum number of updates
    config['finish_after'] = 1000000

    # Reload model from files if exist
    config['reload'] = True

    # Save model after this many updates
    config['save_freq'] = 50

    # Show samples from model after this many updates
    config['sampling_freq'] = 50

    # Show this many samples at each sampling
    config['hook_samples'] = 4

    config['bleu_script'] = None

    #topical related features--------------------------------------------------------
    config['topical_vocab_size']=1735

    config['topical_embedding_dim']=200

    config['topical_word_num']=10

    config['topical_embeddings']='D:\users\chxing\data\\topical\chinese\\tLDAtopic\word_topic_normalize.tt10.pkl'

    config['topical_vocab']='D:\users\chxing\data\\topical\chinese\\tLDAtopic\LDADic.tt10.pkl';

    config['topical_data']= datadir + 'train_former_douban.generation.10topical'

    return config

if __name__ == '__main__':
    get_config_ch_topical_former_douban();
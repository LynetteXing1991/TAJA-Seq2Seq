from abc import ABCMeta, abstractmethod

from six import add_metaclass
from theano import tensor

from blocks.bricks import Initializable, Random, Bias, NDimensionalSoftmax
from blocks.bricks.base import application, Brick, lazy
from blocks.bricks.parallel import Fork, Merge
from blocks.bricks.lookup import LookupTable
from blocks.bricks.recurrent import recurrent
'''
from blocks.bricks.attention import (
    AbstractAttentionRecurrent, AttentionRecurrent)
'''
from attention_with_topicalq import AttentionRecurrent
from blocks.roles import add_role, COST
from blocks.utils import dict_union, dict_subset
from blocks.bricks.sequence_generators import BaseSequenceGenerator,FakeAttentionRecurrent



class SequenceGenerator(BaseSequenceGenerator):
    r"""A more user-friendly interface for :class:`BaseSequenceGenerator`.

    Parameters
    ----------
    readout : instance of :class:`AbstractReadout`
        The readout component for the sequence generator.
    transition : instance of :class:`.BaseRecurrent`
        The recurrent transition to be used in the sequence generator.
        Will be combined with `attention`, if that one is given.
    attention : object, optional
        The attention mechanism to be added to ``transition``,
        an instance of
        :class:`~blocks.bricks.attention.AbstractAttention`.
    add_contexts : bool
        If ``True``, the
        :class:`.AttentionRecurrent` wrapping the
        `transition` will add additional contexts for the attended and its
        mask.
    \*\*kwargs : dict
        All keywords arguments are passed to the base class. If `fork`
        keyword argument is not provided, :class:`.Fork` is created
        that forks all transition sequential inputs without a "mask"
        substring in them.

    """
    def __init__(self, readout, transition, topical_name,content_name,attention=None,topical_attention=None,
                 add_contexts=True, **kwargs):
        normal_inputs = [name for name in transition.apply.sequences
                         if 'mask' not in name]
        kwargs.setdefault('fork', Fork(normal_inputs))
        if attention:
            transition = AttentionRecurrent(
                transition, attention,topical_attention,topical_attended_name='topical_attended',topical_attended_mask_name='topical_attended_mask',content_name=content_name,topical_name=topical_name,
                add_contexts=add_contexts, name="att_trans")
        else:
            transition = FakeAttentionRecurrent(transition,
                                                name="with_fake_attention")
        self.topical_name=topical_name;
        self.content_name=content_name;
        self._topical_context_names=['topical_attended','topical_attended_mask'];
        #self._topical_glimpse_names=['topical_weighted_averages','topical_weights'];
        super(SequenceGenerator, self).__init__(
            readout, transition, **kwargs)

    @application
    def cost_matrix(self, application_call, outputs, mask=None, **kwargs):
        """Returns generation costs for output sequences.

        See Also
        --------
        :meth:`cost` : Scalar cost.

        """
        # We assume the data has axes (time, batch, features, ...)
        batch_size = outputs.shape[1]

        # Prepare input for the iterative part
        states = dict_subset(kwargs, self._state_names, must_have=False)
        # masks in context are optional (e.g. `attended_mask`)
        contexts = dict_subset(kwargs, self._context_names, must_have=False)
        topical_word_contexts=dict_subset(kwargs, self._topical_context_names)
        topical_embeddings=dict_subset(kwargs,[self.topical_name]);
        content_embeddings=dict_subset(kwargs,[self.content_name]);
        feedback = self.readout.feedback(outputs)
        inputs = self.fork.apply(feedback, as_dict=True)

        # Run the recurrent network
        results = self.transition.apply(
            mask=mask, return_initial_states=True, as_dict=True,
            **dict_union(inputs, states, contexts,topical_word_contexts,topical_embeddings,content_embeddings))#requires to be checked.

        # Separate the deliverables. The last states are discarded: they
        # are not used to predict any output symbol. The initial glimpses
        # are discarded because they are not used for prediction.
        # Remember, glimpses are computed _before_ output stage, states are
        # computed after.
        states = {name: results[name][:-1] for name in self._state_names}
        glimpses = {name: results[name][1:] for name in self._glimpse_names}
        glimpses_modified={'weighted_averages':glimpses['weighted_averages'],'weigths':glimpses['weights']}
        # Compute the cost
        feedback = tensor.roll(feedback, 1, 0)
        feedback = tensor.set_subtensor(
            feedback[0],
            self.readout.feedback(self.readout.initial_outputs(batch_size)))
        readouts = self.readout.readout(
            feedback=feedback, **dict_union(states, glimpses_modified, contexts))
        costs = self.readout.cost(readouts, outputs)
        if mask is not None:
            costs *= mask

        for name, variable in list(glimpses.items()) + list(states.items()):
            application_call.add_auxiliary_variable(
                variable.copy(), name=name)

        # This variables can be used to initialize the initial states of the
        # next batch using the last states of the current batch.
        for name in self._state_names:
            application_call.add_auxiliary_variable(
                results[name][-1].copy(), name=name+"_final_value")

        return costs

    @recurrent
    def generate(self, outputs, **kwargs):
        """A sequence generation step.

        Parameters
        ----------
        outputs : :class:`~tensor.TensorVariable`
            The outputs from the previous step.

        Notes
        -----
        The contexts, previous states and glimpses are expected as keyword
        arguments.

        """
        states = dict_subset(kwargs, self._state_names, must_have=False)
        # masks in context are optional (e.g. `attended_mask`)
        contexts = dict_subset(kwargs, self._context_names, must_have=False)
        topical_word_contexts=dict_subset(kwargs, self._topical_context_names)
        topical_embeddings=dict_subset(kwargs,[self.topical_name]);
        content_embeddings=dict_subset(kwargs,[self.content_name]);
        glimpses = dict_subset(kwargs, self._glimpse_names)



        next_glimpses = self.transition.take_glimpses(
            as_dict=True,
            **dict_union(
                states, glimpses,topical_embeddings,content_embeddings,contexts,topical_word_contexts));
        glimpses_modified={'weighted_averages':next_glimpses['weighted_averages'],'weigths':next_glimpses['weights']}
        '''
        next_glimpses = self.transition.take_glimpses(
            as_dict=True, **dict_union(states, glimpses, contexts,topical_embeddings))
        '''
        next_readouts = self.readout.readout(
            feedback=self.readout.feedback(outputs),
            **dict_union(states, glimpses_modified, contexts))
        next_outputs = self.readout.emit(next_readouts)
        next_costs = self.readout.cost(next_readouts, next_outputs)
        next_feedback = self.readout.feedback(next_outputs)
        next_inputs = (self.fork.apply(next_feedback, as_dict=True)
                       if self.fork else {'feedback': next_feedback})
        tmp=next_glimpses.values();
        next_states = self.transition.compute_states(
            as_list=True,
            **dict_union(next_inputs, states, next_glimpses, contexts,topical_word_contexts))#why adding contexts here,the do_apply version do not have contexts
        return (next_states + [next_outputs] +
                list(next_glimpses.values()) + [next_costs])

    @generate.delegate
    def generate_delegate(self):
        return self.transition.apply

    @generate.property('states')
    def generate_states(self):
        return self._state_names + ['outputs'] + self._glimpse_names#+self._topical_glimpse_names
    '''
    @generate.property('topical_word_contexts')
    def generate_topical_word_contexts(self):
        return self._topical_context_names
    '''

    @generate.property('outputs')
    def generate_outputs(self):
        return (self._state_names + ['outputs'] +
                self._glimpse_names+ ['costs'])

    def get_dim(self, name):
        if name in (self._state_names + self._context_names +
                    self._glimpse_names):#don't know whether this can achieve
            return self.transition.get_dim(name)
        elif name == 'outputs':
            return self.readout.get_dim(name)
        return super(BaseSequenceGenerator, self).get_dim(name)

    @application
    def initial_states(self, batch_size, *args, **kwargs):
        # TODO: support dict of outputs for application methods
        # to simplify this code.
        state_dict = dict(
            self.transition.initial_states(
                batch_size, as_dict=True, *args, **kwargs),
            outputs=self.readout.initial_outputs(batch_size))
        return [state_dict[state_name]
                for state_name in self.generate.states]

    @initial_states.property('outputs')
    def initial_states_outputs(self):
        return self.generate.states
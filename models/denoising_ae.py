__author__ = 'mateuszopala'

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from theano.sandbox.rng_mrg import MRG_RandomStreams as GPU_RandomStreams
import numpy as np


class DenoisingAutoEncoder(object):
    def __init__(self, n_visible, n_hidden, weights=None, hidden_bias=None, visible_bias=None, random_on_gpu=False,
                 seed=69, activation=T.nnet.sigmoid):
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        if random_on_gpu:
            self.t_rng = GPU_RandomStreams(seed)
        else:
            self.t_rng = RandomStreams(seed)

        # TODO 1
        if not weights:
            # weights should be initialized to gaussian with mean=0 and std=0.01
            pass
        self.ts_weights = theano.shared(value=weights, name='W', borrow=True)

        if not hidden_bias:
            hidden_bias = np.zeros(n_hidden, dtype=theano.config.floatX)

        self.ts_hidden_bias = theano.shared(value=hidden_bias, name='hb', borrow=True)

        if not visible_bias:
            visible_bias = np.zeros(n_visible, dtype=theano.config.floatX)

        self.ts_visible_bias = theano.shared(value=visible_bias, name='vb', borrow=True)

        self.x = T.matrix(name='x')

        self.activation = activation

        self.params = [self.ts_weights, self.ts_hidden_bias, self.ts_visible_bias]

    def get_corrupted_input(self, x, corruption_level):
        """

        :param x: training data
        :param corruption_level:
        :return: corrupted data with probability 1 - corruption_level
        """
        # TODO 2
        pass

    def hidden_values(self, x):
        """

        :param x: data
        :return: hidden layer activation
        """
        # TODO 3
        pass

    def reconstruction(self, hidden):
        """

        :param hidden: activation of hidden layer
        :return: data reconstruction (activation of output layer)
        """
        # TODO 4
        pass

    def get_cost_updates(self, corruption_level, learning_rate):
        corrupted_input = self.get_corrupted_input(self.x, corruption_level)
        hidden = self.hidden_values(corrupted_input)
        reconstruction = self.reconstruction(hidden)
        loss = -T.sum(self.x * T.log(reconstruction) + (1 - self.x) * T.log(1 - reconstruction), axis=1)
        cost = T.mean(loss)
        gparams = T.grad(cost, self.params)

        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - learning_rate * gparam))
        return cost, updates
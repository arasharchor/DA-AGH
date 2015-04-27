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

        if not weights:
            weights = np.asarray(
                np.random.normal(
                    scale=0.01,
                    size=(self.n_visible, self.n_hidden)),
                dtype=theano.config.floatX)
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
        return self.t_rng.binomial(size=x.shape, n=1, p=1 - corruption_level) * x

    def hidden_values(self, x):
        return self.activation(T.dot(x, self.ts_weights) + self.ts_hidden_bias)

    def reconstruction(self, hidden):
        return self.activation(T.dot(hidden, self.ts_weights.T) + self.ts_visible_bias)

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
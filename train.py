__author__ = 'mateuszopala'

from models.denoising_ae import DenoisingAutoEncoder
import cPickle
import numpy as np
import theano.tensor as T
import theano
import time
import matplotlib.pyplot as plt

with open('data/mnist.pkl', 'rb') as f:
    train_set, valid_set, test_set = cPickle.load(f)


def shared_dataset(data_xy):
    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX))
    shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX))
    return shared_x, T.cast(shared_y, 'int32')

train_data, train_labels = shared_dataset(train_set)
valid_data, valid_labels = shared_dataset(valid_set)
test_data, test_labels = shared_dataset(test_set)

index = T.lscalar()

n_visible = 784
# n_hidden = 3 * n_visible
n_hidden = 128
batch_size = 100
epochs = 10
n_train_batches = 500

corruption_level = 0.2
learning_rate = 0.1

da = DenoisingAutoEncoder(n_visible, n_hidden)
#
cost, updates = da.get_cost_updates(corruption_level, learning_rate)
#
train_da = theano.function([index], cost, updates=updates,
                           givens={da.x: train_data[index * batch_size:(index + 1) * batch_size]})

start = time.time()
# TRAINING
for epoch in xrange(epochs):
    c = []
    for batch_index in xrange(n_train_batches):
        cost = train_da(batch_index)
        c.append(cost)
    print "Training epoch %d, cost %f" % (epoch + 1, float(np.mean(c)))

end = time.time()

print 'Training time took %f minutes' % ((end - start) / 60.)

with open('data/params.pkl', 'w+') as f:
    cPickle.dump([param.get_value() for param in da.params], f)


def reconstruct(batch, params, activation):
    """

    :param batch: data (numpy array)
    :param params: tuple of weights, hidden bias, visible bias
    :param activation: activation function that takes one argument (numpy array)
    :return: reconstruction of batch
    """
    pass


def sigmoid(x):
    """

    :param x: data (numpy array)
    :return: value of sigmoid function
    """
    pass

reconstructed = reconstruct(valid_set[0][:30], [param.get_value() for param in da.params], sigmoid)

print reconstructed.shape

fig = plt.figure()

for i in xrange(30):
    a = fig.add_subplot(1, 2, 1)
    plt.imshow(reconstructed[i].reshape((28, 28)))



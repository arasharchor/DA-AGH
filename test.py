__author__ = 'mateuszopala'

import cPickle
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

with open("data/params.pkl", 'r') as f:
    params = cPickle.load(f)

with open("data/mnist.pkl", 'r') as f:
    train, val, test = cPickle.load(f)

test_data, test_labels = test

digits = []
for i in xrange(10, 20):
    digits.append((test_data[i].reshape((28, 28)), "Input %d" % test_labels[i]))


def reconstruct(batch, params, activation):
    w, hb, vb = params
    output = activation(batch.dot(w) + hb)
    output = activation(output.dot(w.T) + vb)
    return output


def sigmoid(x):
    return 1. / (1. + np.exp(-x))

print "Reconstructing images..."

reconstructed = []

for digit, label in digits:
    print "Digit %s" % label
    reconstruction = reconstruct(digit.reshape(1, 784), params, sigmoid).reshape((28, 28))
    label = label.split(' ')[1]
    reconstructed.append((reconstruction, "Output %s" % label))


# Plotting results

digits += reconstructed

print "Plotting results..."

fig, axes = plt.subplots(2, 10, figsize=(30, 10),
                         subplot_kw={'xticks': [], 'yticks': []})

for ax, (digit, label) in zip(axes.flat, digits):
    ax.imshow(digit, cmap='Greys')
    ax.set_title(label)

plt.show()
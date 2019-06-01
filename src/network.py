"""
A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.
"""

import random
import numpy as np
import pickle

class Network(object):

    def __init__(self, sizes):
        """Create the list of layers, sizes in each layer and initialize weights and 
        biases of the network by using random module"""

        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = []
        self.weights = []

        for y in sizes[1:]:
            self.biases.append(np.random.randn(y, 1))

        for x, y in zip(sizes[:-1], sizes[1:]):
            self.weights.append(np.random.randn(y, x))

    def feedforward(self, a):
        """ Return the output of the network by using output of the previous layer
        neuron and it's weights and biases"""  

        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """ Train the neural network using mini-batch stochastic gradient descent."""
        
        if test_data: n_test = len(test_data)
        n = len(training_data)

        for j in xrange(epochs):

            random.shuffle(training_data)

            mini_batches = []
            for k in xrange(0, n, mini_batch_size):
                mini_batches.append(training_data[k:k+mini_batch_size])

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            if test_data:
                print "Epoch {0}: {1} %  accurate".format(j, (float(self.evaluate(test_data)) / float(100)))
            else:
                print "Epoch {0} complete".format(j)

    def update_mini_batch(self, mini_batch, eta):
        """Update the value of weights and biases to train the network using
        backpropagation on mini_batches"""

        new_biases = [np.zeros(b.shape) for b in self.biases]
        new_weights = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_new_biases, delta_new_weights = self.backprop(x, y)

            new_biases = [nb+dnb for nb, dnb in zip(new_biases, delta_new_biases)]
            new_weights = [nw+dnw for nw, dnw in zip(new_weights, delta_new_weights)]

        self.weights = [w - ( eta/len(mini_batch) ) * nw for w, nw in zip(self.weights, new_weights)]
        self.biases = [b - ( eta/len(mini_batch) ) * nb for b, nb in zip(self.biases, new_biases)]

    def backprop(self, x, y):
        """ backprop is the main function that updates the value of weights and biases
        using cost function and activation values"""

        new_biases = [np.zeros(b.shape) for b in self.biases]
        new_weights = [np.zeros(w.shape) for w in self.weights]
        
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])

        new_biases[-1] = delta
        new_weights[-1] = np.dot(delta, activations[-2].transpose())
 
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            new_biases[-l] = delta
            new_weights[-l] = np.dot(delta, activations[-l-1].transpose())

        return (new_biases, new_weights)

    def evaluate(self, test_data):
    	""" This function checks whether the predicted output is equal to the
    	output provided in the test_data"""

        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives partial C_x partial a for the output 
        activations."""

        return (output_activations-y)

def sigmoid(z):
    """The sigmoid function."""

    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""

    return sigmoid(z)*(1-sigmoid(z))

def save_object(obj):
    """The file object that includes layers, sizes, weights and biases are dumped
    onto neural_net.pkl for further use"""

    file_name = "neural_net.pkl"
    file_object = open(file_name,'wb')
    pickle.dump(obj,file_object)
    file_object.close()

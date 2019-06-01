import cPickle
import gzip
import numpy as np

def load_data():
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.
    """

    f = gzip.open('../data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()

    return (training_data, validation_data, test_data)

def load_data_wrapper():
    """Return a tuple containing (training_data, validation_data, test_data)."""

    training_data2, validation_data2, test_data2 = load_data()

    training_inputs = [np.reshape(x, (784, 1)) for x in training_data2[0]]
    training_results = [vectorized_result(y) for y in training_data2[1]]

    training_data = zip(training_inputs, training_results)

    validation_inputs = [np.reshape(x, (784, 1)) for x in validation_data2[0]]
    validation_data = zip(validation_inputs, validation_data2[1])

    test_inputs = [np.reshape(x, (784, 1)) for x in test_data2[0]]
    test_data = zip(test_inputs, test_data2[1])

    return (training_data, validation_data, test_data)

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""

    output_result = np.zeros((10, 1))
    output_result[j] = 1.0
    return output_result

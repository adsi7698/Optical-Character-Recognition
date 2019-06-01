import mnist_loader
import network

def run_train():
	""" The network is trained by executing Network module """
	
	training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
	
	net = network.Network([784, 30, 10])
	net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
	network.save_object(net)

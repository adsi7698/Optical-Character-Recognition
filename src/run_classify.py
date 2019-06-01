import pickle
import network
import converter

def run_class():
	"""The trained neural network predicts the output for the input image
	containing handwritten characters"""

	file_name = "neural_net.pkl"
	file_object = open(file_name,'rb')

	net = pickle.load(file_object)
	file_object.close()

	input_from_image = converter.convert_image()
	output = list(net.feedforward(input_from_image))
	
	output_string = ""
	output_string += str(output.index(max(output)))
	output_string += ","
	output_string += str(float(max(output))*100)
	return output_string
	

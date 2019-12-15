import numpy as np


def sigmoid_activation(x):
	return 1/(1+np.exp(-x))


def sigmoid_derivative(x):
	return sigmoid_activation(x)*(1 - sigmoid_activation(x))


def relu_activation(x):
	return x.clip(min=0)


def relu_derivative(x):
	x[x < 0] = 0
	x[x > 0] = 1
	return x


# assuming stride to be one and no changes in the 2D dimension
def conv_block(conv_input, current_filters):
	# print conv_input.shape
	in_row, in_col, prev_f_no = conv_input.shape

	# print current_filters
	f_size_current, f_size_current, current_f_no = current_filters.shape

	output_wo_activation = np.zeros((in_row, in_col, current_f_no))
	current_output = np.zeros((in_row, in_col, current_f_no))

	input_channel_zero_padded = np.zeros((in_row + (f_size_current+1)/2, in_col + (f_size_current+1)/2, prev_f_no))

	for input_channel in range(prev_f_no):
		input_channel_zero_padded[:, :, input_channel] = \
			np.pad(conv_input[:, :, input_channel], (1), 'constant', constant_values=(0))

	for conv_filter in range(current_f_no):
		for row in range(in_row):
			for col in range(in_col):
				conv_sum = 0
				for input_channel in range(prev_f_no):
					current_filter = current_filters[:, :, conv_filter]
					patch = input_channel_zero_padded[row: row + f_size_current, col: col + f_size_current, input_channel]
					conv_sum = conv_sum + sum(sum(np.multiply(current_filter, patch)))

				output_wo_activation[row][col][conv_filter] = conv_sum #/ (prev_f_no*10)
				# current_output[row][col][conv_filter] = sigmoid_activation(output_wo_activation[row][col][conv_filter])
				current_output[row][col][conv_filter] = relu_activation(conv_sum)

	print "conv wo activation: ", output_wo_activation
	print "conv after activation: ", current_output

	return current_output, output_wo_activation

# current_filters, conv_input
# conv_input = np.zeros((2, 2, 2))
# conv_input[0][0][0] = 1
# conv_input[1][0][0] = 2
# conv_input[0][1][0] = 3
# conv_input[1][1][0] = 4

# conv_input[0][0][1] = 10
# conv_input[1][0][1] = 20
# conv_input[0][1][1] = 30
# conv_input[1][1][1] = 40

# current_filters = np.zeros((3, 3, 2))
# current_filters[0][0][0] = 1
# current_filters[0][1][0] = 2
# current_filters[0][2][0] = 3
# current_filters[1][0][0] = 4
# current_filters[1][1][0] = 5
# current_filters[1][2][0] = 6
# current_filters[2][0][0] = 7
# current_filters[2][1][0] = 8
# current_filters[2][2][0] = 9

# current_filters[0][0][1] = 10
# current_filters[0][1][1] = 20
# current_filters[0][2][1] = 30
# current_filters[1][0][1] = 40
# current_filters[1][1][1] = 50
# current_filters[1][2][1] = 60
# current_filters[2][0][1] = 70
# current_filters[2][1][1] = 80
# current_filters[2][2][1] = 90

# current_output = conv_block(conv_input, current_filters)

# print conv_input[:,:,0]
# print conv_input[:,:,1]
# print current_filters[:,:,0]

# print "------------------"
# print current_output[:,:,0]
# print current_output[:,:,1]



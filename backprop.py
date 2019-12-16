import numpy as np
import conv


def derivative_square_error(output, target):
	error = output - target
	squared_error = error * error
	return error


# Loss at the output layer
def compute_loss(output, target):
	out_row, out_col, out_channels = target.shape
	print "target shape", target.shape
	loss = np.zeros((out_row, out_col, out_channels))
	print "out_channels", out_channels
	for colour in range(out_channels):
		loss[:, :, colour] = derivative_square_error(output[:, :, colour], target[:, :, colour])
	return loss


# Back propagation
# Loss at layer l --> Backprop to layer l-1
# Update weights between layer l-1 and l
def backprop(delta_out, input, input_wo_act, weights, is_output_layer):
	alpha = 0.001
	input_sum = np.sum(input, axis = 2)
	input_sum_zero_padded = np.pad(input_sum, (1), 'constant', constant_values=(0))
	in_row, in_col = input_sum_zero_padded.shape

	# compute delta weights
	weight_dim = weights.shape
	print "weight dim: ", weight_dim

	delta_matrix = np.zeros(weight_dim)
	for row in range(weight_dim[0]):
		for col in range(weight_dim[1]):
			# print "delta_out: ", delta_out[:,:,0]
			# print "input_sum_zero_padded: ", input_sum_zero_padded[row: in_row - weight_dim[0] + 1, col: in_col - weight_dim[1] + 1]
			# print "x_cord_begin", row
			# print "x_cord", in_row - weight_dim[0] + 1
			# print "y_cord_begin", col
			# print "y_cord", in_col - weight_dim[1] + 1
			delta_w = sum(sum(np.multiply(delta_out[:, :, 0],
										  input_sum_zero_padded[row: row+in_row - weight_dim[0] + 1,
										  col: col+in_col - weight_dim[1] + 1])))
			delta_matrix[row][col][0] = delta_w

	# compute delta inputs
	input_dim = input.shape
	print "input dim: ", input_dim

	delta_x_matrix = np.zeros(input_dim)

	# flipped_weights = np.flipud(np.fliplr(weights))
	flipped_weights = np.zeros((weight_dim))
	for i in range(weight_dim[2]):
		flipped_weights[:, :, i] = np.flipud(np.fliplr(weights[:, :, i]))
		# print "weights ", weights[:, :, i]
		# print "flipped weights ", flipped_weights[:, :, i]

	print "input_wo_act in bp before derivative ", input_wo_act
	if is_output_layer:
		delta_out = delta_out * conv.relu_derivative(input_wo_act)
	else:
		# delta_out = delta_out * conv.sigmoid_derivative(input_wo_act)
		delta_out = delta_out * conv.relu_derivative(input_wo_act)
	print "delta_out ", delta_out
	_, delta_x_matrix = conv.conv_block(delta_out, flipped_weights)

	# delta_out_padded = np.pad(delta_out, (1), 'constant', constant_values=(0))
	# delta_shape = delta_out_padded.shape
	# for row in range(1, delta_shape[0]-1):
	# 	for col in range(1, delta_shape[1]-1):
	# 		# delta_x = sum(sum(sum(np.multiply(delta_out_padded[row - 1: row + 2, col - 1: col + 2], flipped_weights))))
	# 		print delta_x
	# 		delta_x_matrix[row-1][col-1][0] = delta_x

	# update the weights
	weights -= alpha * delta_w

	return delta_x_matrix


conv_input = np.zeros((2, 2, 1))
conv_input[0][0][0] = 1
conv_input[1][0][0] = 2
conv_input[0][1][0] = 3
conv_input[1][1][0] = 4

conv_target = np.zeros((2, 2, 1))
conv_target[0][0][0] = 70
conv_target[1][0][0] = 75
conv_target[0][1][0] = 60
conv_target[1][1][0] = 55

# conv_input[0][0][1] = 10
# conv_input[1][0][1] = 20
# conv_input[0][1][1] = 30
# conv_input[1][1][1] = 40

current_filters = np.zeros((3, 3, 1))
current_filters[0][0][0] = 1
current_filters[0][1][0] = 2
current_filters[0][2][0] = 3
current_filters[1][0][0] = 4
current_filters[1][1][0] = 5
current_filters[1][2][0] = 6
current_filters[2][0][0] = 7
current_filters[2][1][0] = 8
current_filters[2][2][0] = 9

# current_filters[0][0][1] = 10
# current_filters[0][1][1] = 20
# current_filters[0][2][1] = 30
# current_filters[1][0][1] = 40
# current_filters[1][1][1] = 50
# current_filters[1][2][1] = 60
# current_filters[2][0][1] = 70
# current_filters[2][1][1] = 80
# current_filters[2][2][1] = 90

# conv_output = conv.conv_block(conv_input, current_filters)
#
# loss = compute_loss(conv_output, conv_target)
# print loss[:, :, 0]
#
# delta_w = backprop(loss, conv_input, current_filters)
# print "----------"
# print conv_input[:,:,0]
# print "----------"
# print current_filters[:, :, 0]
# print "----------"
# print conv_output[:, :, 0]
# print "----------"
# print conv_target[:, :, 0]
# print "----------"
# print loss[:, :, 0]
# print "----------"
# print "delta_w"
# print delta_w[:, :, 0]

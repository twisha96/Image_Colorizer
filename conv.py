import numpy as np

# conv_input -> in_row X in_col X prev_f_no
# current filter size -> f_size_current
# number of current filters -> current_f_no
# current filters -> current_filters[]
# assuming stride to be one and no changes in the 2D dimension

current_output = out[in_row][in_col][current_f_no]

# do zero padding

for conv_filter in range(current_f_no):
	for row in range(in_row):
		for col in range(in_col):
			conv_sum = 0
			for input_channel in range(prev_f_no):
				current_filter = current_filters[:, :, conv_filter]
				patch = conv_input[row : row + f_size_current, col : col + f_size_current, input_channel]
				conv_sum = conv_sum + sum(sum(np.dot(current_filter, patch)))
			current_output[row][col][conv_filter] = activation_func(conv_sum)
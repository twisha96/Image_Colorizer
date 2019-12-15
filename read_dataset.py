# load and display an image with Matplotlib
from matplotlib import image
from matplotlib import pyplot
import conv as conv
from conv import conv_block
import backprop as bp
import numpy as np


def normalize_output_layer(output):
    num_layers = output.shape[2]
    normalized_output = np.zeros(output.shape)
    for l in range(num_layers):
        max_value = np.amax(output[:, :, l])
        min_value = np.amin(output[:, :, l])
        normalized_output[:, :, l] = (output[:, :, l] - min_value)/(max_value - min_value)
    return normalized_output


# load image as pixel array
data = image.imread('image.jpg')
# print(data)

# summarize shape of the pixel array
print("Shape of colour image: ", data.shape)

r = data[:, :, 0]/255.0
g = data[:, :, 1]/255.0
b = data[:, :, 2]/255.0
print r, g, b


# Gray(r, g, b) = 0.21r + 0.72g + 0.07b
grayscale = 0.21*r + 0.72*g + 0.07*b
print("Shape of B/W image: ", grayscale.shape)
# print(grayscale)

# display the array of pixels as an image
# pyplot.imshow(data)
# pyplot.show()

# display the array of pixels as an image
# pyplot.imshow(grayscale, cmap="gray")
# pyplot.show()


# Main function
iterations = 10
layers = 3

# print grayscale.shape[0], grayscale.shape[1]
curr_input = np.zeros((grayscale.shape[0], grayscale.shape[1], 1))
# print curr_input.shape
curr_input[:, :, 0] = grayscale

# print "input shape: ", curr_input.shape
W = [np.random.rand(3, 3, 3)*0.1, np.random.rand(3, 3, 3)*0.1, np.random.rand(3, 3, 3)*0.1]
layer_output = [curr_input]
layer_output_wo_activation = [curr_input]

for i in range(iterations):
    print "Iteration Number: ", i
    for l in range(layers):
        print "Forward pass layer number: ", l
        # if l == layers-1:
        #     curr_input = normalize_output_layer(curr_input)
        curr_input, without_activation = conv_block(curr_input, W[l])
        layer_output.append(curr_input)
        layer_output_wo_activation.append(without_activation)

    layer_output[-1] = conv.sigmoid_activation(layer_output_wo_activation[-1])

    target = np.zeros((grayscale.shape[0], grayscale.shape[1], 3))
    target[:, :, 0] = r
    target[:, :, 1] = g
    target[:, :, 2] = b

    loss = bp.compute_loss(layer_output_wo_activation[layers], target)
    # loss = bp.compute_loss(conv.sigmoid_activation(layer_output_wo_activation[layers]), target)
    # print sum(loss[:, :, 0])

    curr_loss = loss
    is_output_layer = True
    for l in range(0, layers):
        print "Backward pass layer number: ", l
        delta_x_matrix = bp.backprop(curr_loss, layer_output[layers-1-l],
                                     layer_output_wo_activation[layers-l], W[layers-1-l],
                                     is_output_layer)
        curr_loss = delta_x_matrix
        is_output_layer = False

    pyplot.imshow(layer_output[layers])
    pyplot.show()

pyplot.imshow(layer_output[layers - 1])
pyplot.show()

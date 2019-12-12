# load and display an image with Matplotlib
from matplotlib import image
from matplotlib import pyplot
import numpy as np
from conv import conv_block
import backprop as bp

# load image as pixel array
data = image.imread('image.jpg')
print(type(data))

# summarize shape of the pixel array
print("Shape of colour image: ", data.shape)

r = data[:, :, 0]
g = data[:, :, 1]
b = data[:, :, 2]

# Gray(r, g, b) = 0.21r + 0.72g + 0.07b
grayscale = 0.21*r + 0.72*g + 0.07*b
print("Shape of B/W image: ", grayscale.shape)
print(grayscale)

# display the array of pixels as an image
# pyplot.imshow(data)
# pyplot.show()

# display the array of pixels as an image
# pyplot.imshow(grayscale, cmap="gray")
# pyplot.show()

# Main function
iterations = 2
layers = 3

print grayscale.shape[0], grayscale.shape[1]
curr_input = np.zeros((grayscale.shape[0], grayscale.shape[1], 1))
print curr_input.shape
curr_input[:, :, 0] = grayscale

print "input shape: ", curr_input.shape
W = [np.random.rand(3, 3, 3), np.random.rand(3, 3, 3), np.random.rand(3, 3, 3)]
layer_output = [curr_input]
layer_output_wo_activation = [curr_input]

for i in range(iterations):
    print "Iteration Number: ", i
    for l in range(layers):
        print "Forward pass layer number: ", l
        curr_input, without_activation = conv_block(curr_input, W[l])
        layer_output.append(curr_input)
        layer_output_wo_activation.append(without_activation)

    target = np.zeros((grayscale.shape[0], grayscale.shape[1], 3))
    target[:, :, 0] = r
    target[:, :, 1] = g
    target[:, :, 2] = b
    loss = bp.compute_loss(layer_output[layers], target)
    print sum(loss[:, :, 0])

    curr_loss = loss
    for l in range(0, layers):
        print "Backward pass layer number: ", l
        delta_x_matrix = bp.backprop(curr_loss, layer_output[layers-2-l],
                                     layer_output_wo_activation[layers-1-l], W[layers-2-l])
        curr_loss = delta_x_matrix

# print layer_output[layers - 1]
pyplot.imshow(layer_output[layers])
pyplot.show()

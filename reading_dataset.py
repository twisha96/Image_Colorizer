# load and display an image with Matplotlib
from matplotlib import image
from matplotlib import pyplot
import numpy as np
import conv as conv
from conv import conv_block
import backprop as bp
import pdb
from PIL import Image

from PIL import Image
import numpy as np


def colorize_grayscale_image(r_matrix, g_matrix, b_matrix):
    pixels = []
    for row in range(len(r_matrix)):
        new = []
        for col in range(len(r_matrix[0])):
            pixel_rgb_value = (r_matrix[row][col], g_matrix[row][col], b_matrix[row][col])
            new.append(pixel_rgb_value)
            pixels.append(new)

    # Convert the pixels into an array using numpy
    array = np.array(pixels, dtype=np.uint8)

    # Use PIL to create an image from the new array of pixels
    new_image = Image.fromarray(array)
    new_image.show()
    new_image.save('new.png')


# r_matrix = [[54,232],[204,54]]
# g_matrix = [[54,23],[82,54]]
# b_matrix = [[54,93],[122,54]]
# colorize_grayscale_image(r_matrix, g_matrix, b_matrix)


def normalize_output_layer(output):
    num_layers = output.shape[2]
    normalized_output = np.zeros((output.shape))
    for l in range(num_layers):
        max_value = np.amax(output[:, :, l])
        min_value = np.amin(output[:, :, l])
        normalized_output[:, :, l] = (output[:, :, l] - min_value)/(max_value - min_value)
    return normalized_output


# load image as pixel array
data = image.imread('puppy_100.jpg')
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
iterations = 3
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

    #layer_output[-1] = conv.sigmoid_activation(layer_output_wo_activation[-1])
    layer_output[-1] = conv.relu_activation(layer_output_wo_activation[-1])

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

    # temp = layer_output[layers] * 255
    # print "temp  ", temp
    # newtemp = np.array(temp, dtype=np.uint8)
    # outt = Image.fromarray(newtemp)
    # outt.show()
    # filename = str("sigmoid_try3_iter_") + str(i) + str(".png")
    # outt.save(filename)


layer_output = []
layer_output_wo_activation = []
curr_input[:, :, 0] = grayscale
for i in range(1):
    for l in range(layers):
        print "Forward pass layer number: ", l
        # if l == layers-1:
        #     curr_input = normalize_output_layer(curr_input)
        curr_input, without_activation = conv_block(curr_input, W[l])
        layer_output.append(curr_input)
        layer_output_wo_activation.append(without_activation)

pyplot.imshow(layer_output[layers - 1])
pyplot.show()        
# print layer_output[layers - 1]
# normalized_output_layer = normalize_output_layer(layer_output[layers])
# print "normalised output: ", normalized_output_layer

# image = Image.open('image.jpg')
#
# temp = np.array(image)
# temp = temp/255.0

# temp = layer_output[layers]*255
# print "temp  ", temp
# newtemp = np.array(temp, dtype=np.uint8)
# outt = Image.fromarray(newtemp)
# outt.show()
# outt.save("try.png")

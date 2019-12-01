# load and display an image with Matplotlib
from matplotlib import image
from matplotlib import pyplot

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
pyplot.imshow(data)
pyplot.show()

# display the array of pixels as an image
pyplot.imshow(grayscale, cmap="gray")
pyplot.show()

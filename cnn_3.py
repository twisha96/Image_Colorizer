from keras.models import Sequential, model_from_json
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, UpSampling2D, Conv2DTranspose, InputLayer
from keras import losses
from skimage.color import rgb2lab, lab2rgb
from keras.preprocessing.image import img_to_array, load_img
from matplotlib import image
from matplotlib import pyplot
import numpy as np
import conv as conv
from conv import conv_block
import backprop as bp
import pdb
from PIL import Image
import numpy as np
import os 

def getData(dirname):
	count = 0
	for filename in os.listdir(dirname):
		if filename != ".DS_Store":
			count = count + 1
	X = np.zeros((count, 200, 200, 1))
	Y = np.zeros((count, 200, 200, 2))
	itr = 0
	for filename in os.listdir(dirname):
		if filename != ".DS_Store":
			image = img_to_array(load_img(dirname + '/' + filename))
			image = np.array(image, dtype=float)
			x = rgb2lab(1.0/255*image)[:,:,0]
			y = rgb2lab(1.0/255*image)[:,:,1:]
			y = y / 128
			x = x.reshape(200, 200, 1)
			y = y.reshape(200, 200, 2)
			X[itr,:,:,:] = x
			Y[itr,:,:,:] = y
			itr = itr + 1
	return X, Y


os.environ['KMP_DUPLICATE_LIB_OK']='True'

X, Y = getData("/Users/adityalakra/Desktop/Image_Colorizer/y_train")

# # Get images
# image = img_to_array(load_img('woman.png'))
# image = np.array(image, dtype=float)
# # Import map images into the lab colorspace
# X = rgb2lab(1.0/255*image)[:,:,0]
# Y = rgb2lab(1.0/255*image)[:,:,1:]
# Y = Y / 128
# X = X.reshape(1, 200, 200, 1)
# Y = Y.reshape(1, 200, 200, 2)

model = Sequential()
model.add(InputLayer(input_shape=(None, None, 1)))

# Building the neural network
model = Sequential()
model.add(InputLayer(input_shape=(200, 200, 1)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(2, (3, 3), activation='sigmoid', padding='same'))
model.add(UpSampling2D((2, 2)))

# Finish model
model.compile(optimizer='adam',loss='mse')

#Train the neural network
model.fit(x=X, y=Y, batch_size=40, epochs=5)
print(model.evaluate(X, Y, batch_size=1))

# Output colorizations
output = model.predict(X)
output = output * 128

for i in range(len(output)):
    cur = np.zeros((200, 200, 3))
    cur[:,:,0] = X[i][:,:,0]
    cur[:,:,1:] = output[i]
    pyplot.imsave("result/img_"+str(i)+".png", lab2rgb(cur))

# canvas = np.zeros((200, 200, 3))
# canvas[:,:,0] = X[0][:,:,0]
# canvas[:,:,1:] = output[0]
# pyplot.imsave("img_result.png", lab2rgb(cur))
# pyplot.imsave("img_gray_scale.png", rgb2gray(lab2rgb(cur)))
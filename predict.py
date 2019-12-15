from keras.models import Sequential, model_from_json
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, UpSampling2D, Conv2DTranspose, InputLayer
from keras import losses
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


def getData(dirname, channel):
	count = 0
	for filename in os.listdir(dirname):
		if filename != ".DS_Store":
			count = count + 1

	x = np.zeros((count, 200, 200, channel))
	itr = 0
	for filename in os.listdir(dirname):
		if filename != ".DS_Store":
			img = Image.open(dirname + '/' + filename)
			data = np.asarray(img, dtype=np.uint8)
			data = data/255.
			if channel == 1:
				x[itr,:,:,0] = data
			else:
				x[itr,:,:,:] = data	
			itr = itr + 1
	return x

def showpredict(x_test, no_of_samples):
	out = model.predict(x_test)
	for i in range(no_of_samples):
		output = out[i,:,:,:]
		temp = output*255
		newtemp = np.array(temp, dtype=np.uint8)
		img_f = Image.fromarray(newtemp)
		img_f.save('/Users/adityalakra/Desktop/Image_Colorizer/y_test_out/' + str(i) + '.jpg')

height = 200
width = 200


with open('model.txt', 'r') as file:
    json_string = file.read()
model = model_from_json(json_string)

# model = Sequential()
# model.add(InputLayer(input_shape=(200, 200, 1)))
# model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
# model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=2))
# model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
# model.add(Conv2D(128, (3, 3), activation='relu', padding='same', strides=2))
# model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
# model.add(Conv2D(256, (3, 3), activation='relu', padding='same', strides=2))
# model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
# model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
# model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
# model.add(UpSampling2D((2, 2)))
# model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
# model.add(UpSampling2D((2, 2)))
# model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
# model.add(Conv2D(2, (3, 3), activation='sigmoid', padding='same'))
# model.add(UpSampling2D((2, 2)))

# model.compile(optimizer='adam', loss=losses.mean_squared_error, metrics=['mse'])

model.load_weights('model_weights.txt')


X_test = getData('/Users/adityalakra/Desktop/Image_Colorizer/x_test', 1)
no_of_samples,_,_,_ = X_test.shape
showpredict(X_test, no_of_samples)
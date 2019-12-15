from keras.models import Sequential, model_from_json, save_model
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


def save_model(model):
	json_string = model.to_json()
	text_file = open("model.txt", "w")
	text_file.write(json_string)
	text_file.close()

	model.save_weights("model_weights.txt")


def showpredict(x_test, no_of_samples):
	out = model.predict(x_test)
	for i in range(no_of_samples):
		output = out[i,:,:,:]
		temp = output*255
		newtemp = np.array(temp, dtype=np.uint8)
		img_f = Image.fromarray(newtemp)
		img_f.show()


def getData(dirname, channel):
	count = 0
	for filename in os.listdir(dirname):
		if filename != ".DS_Store":
			count = count + 1
	# count = 1
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
			# break
	return x		

os.environ['KMP_DUPLICATE_LIB_OK']='True'

width = 200
height = 200

# img = Image.open('flowerpot.jpg')
# data = np.asarray(img, dtype=np.uint8)
# data = data/255.0
# print data.shape
# y_train = np.zeros((1,height,width,3))
# y_train[0,:,:,:] = data

# r = data[:, :, 0]
# g = data[:, :, 1]
# b = data[:, :, 2]
# train = np.zeros((height,width,1))
# train[:,:,0] = 0.21*r + 0.72*g + 0.07*b
# X_train = np.zeros((1,height,width,1))
# X_train[0,:,:,:] = train

X_train = getData('x_train', 1)
y_train = getData('y_train', 3)

no_of_samples,_, _, _, = X_train.shape
print X_train.shape

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
model.add(Conv2D(3, (3, 3), activation='sigmoid', padding='same'))
model.add(UpSampling2D((2, 2)))

model.compile(optimizer='adam', loss=losses.mean_squared_error, metrics=['mse'])
model.fit(X_train, y_train, epochs=6, steps_per_epoch=40)

save_model(model)

X_test = getData('x_test', 1)
no_of_samples, _, _, _ = X_test.shape
showpredict(X_test, no_of_samples)

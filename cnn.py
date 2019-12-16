from keras.models import Sequential, model_from_json
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, UpSampling2D, Conv2DTranspose
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
	count = 1
	x = np.zeros((count, height, width, channel))
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
			break
	return x		

width = 367
height = 500

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

X_train = getData('/Users/adityalakra/Desktop/Image_Colorizer/im_f', 1)
y_train = getData('/Users/adityalakra/Desktop/Image_Colorizer/im', 3)

no_of_samples,_, _, _, = X_train.shape
print X_train.shape

model = Sequential()
model.add(Conv2D(3, kernel_size=3, padding='same', activation='relu', input_shape=(height,width,1)))
# model.add(MaxPooling2D())
model.add(Conv2D(5, kernel_size=5, padding='same', activation='relu'))
# model.add(UpSampling2D())
# model.add(Conv2D(3, kernel_size=3, padding='same', activation='relu'))
# model.add(Conv2DTranspose(3, kernel_size=5, padding='same', activation='relu', input_shape=(height/2,width/2,1), out_shape=(height,width,1)))
model.add(Conv2D(3, kernel_size=3, padding='same', activation='relu'))

model.compile(optimizer='adam', loss=losses.mean_squared_error, metrics=['mse'])
model.fit(X_train, y_train, epochs=500, steps_per_epoch = no_of_samples)

save_model(model)

X_test = getData('/Users/adityalakra/Desktop/Image_Colorizer/im_f', 1)
no_of_samples,_,_,_ = X_test.shape
showpredict(X_test, 1)
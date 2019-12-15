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

def save_model(model, i):
	json_string = model.to_json()
	text_file = open("model" + str(i) + ".txt", "w")
	text_file.write(json_string)
	text_file.close()

	model.save_weights("model_weights.txt")

def showpredict(x_test, no_of_samples):
	out1 = model1.predict(x_test)
	out2 = model2.predict(x_test)
	out3 = model3.predict(x_test)
	output = np.zeros((200,200,3))
	for i in range(no_of_samples):
		output[:,:,0] = out1[i,:,:,0]
		output[:,:,1] = out2[i,:,:,0]
		output[:,:,2] = out3[i,:,:,0]
		temp = output*255
		newtemp = np.array(temp, dtype=np.uint8)
		img_f = Image.fromarray(newtemp)
		img_f.show()

def getData(dirname, color):
	count = 0
	for filename in os.listdir(dirname):
		if filename != ".DS_Store":
			count = count + 1
	x = np.zeros((count, 200, 200, 1))
	itr = 0
	for filename in os.listdir(dirname):
		if filename != ".DS_Store":
			img = Image.open(dirname + '/' + filename)
			data = np.asarray(img, dtype=np.uint8)
			data = data/255.
			if color == -1:
				x[itr,:,:,0] = data
			else:
				x[itr,:,:,0] = data[:,:,color]	
			itr = itr + 1
	return x		

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

X_train = getData('/Users/adityalakra/Desktop/Image_Colorizer/x_train', -1)
y_train = getData('/Users/adityalakra/Desktop/Image_Colorizer/y_train', 0)

no_of_samples,_,_,_ = X_train.shape
print X_train.shape
print y_train.shape







model1 = Sequential()
model1.add(Conv2D(128, kernel_size=3, padding='same', activation='relu', input_shape=(height,width,1)))
# model1.add(MaxPooling2D())
model1.add(Conv2D(256, kernel_size=5, padding='same', activation='relu'))
model1.add(Conv2D(64, kernel_size=3, padding='same', activation='relu'))
# model1.add(UpSampling2D())
# model1.add(Conv2D(3, kernel_size=3, padding='same', activation='relu'))
# model1.add(Conv2DTranspose(3, kernel_size=5, padding='same', activation='relu', input_shape=(height/2,width/2,1), out_shape=(height,width,1)))
model1.add(Conv2D(1, kernel_size=3, padding='same', activation='sigmoid'))

model1.compile(optimizer='adam', loss=losses.mean_squared_error, metrics=['mse'])
model1.fit(X_train, y_train, epochs=10, steps_per_epoch = 1)

save_model(model1, 1)




y_train = getData('/Users/adityalakra/Desktop/Image_Colorizer/y_train', 1)
model2 = Sequential()
model2.add(Conv2D(128, kernel_size=3, padding='same', activation='relu', input_shape=(height,width,1)))
# model2.add(MaxPooling2D())
model2.add(Conv2D(256, kernel_size=5, padding='same', activation='relu'))
model2.add(Conv2D(64, kernel_size=3, padding='same', activation='relu'))
# model2.add(UpSampling2D())
# model2.add(Conv2D(3, kernel_size=3, padding='same', activation='relu'))
# model2.add(Conv2DTranspose(3, kernel_size=5, padding='same', activation='relu', input_shape=(height/2,width/2,1), out_shape=(height,width,1)))
model2.add(Conv2D(1, kernel_size=3, padding='same', activation='sigmoid'))

model2.compile(optimizer='adam', loss=losses.mean_squared_error, metrics=['mse'])
model2.fit(X_train, y_train, epochs=10, steps_per_epoch = 1)

save_model(model2, 2)






y_train = getData('/Users/adityalakra/Desktop/Image_Colorizer/y_train', 2)
model3 = Sequential()
model3.add(Conv2D(128, kernel_size=3, padding='same', activation='relu', input_shape=(height,width,1)))
# model3.add(MaxPooling2D())
model3.add(Conv2D(256, kernel_size=5, padding='same', activation='relu'))
model3.add(Conv2D(64, kernel_size=3, padding='same', activation='relu'))
# model3.add(UpSampling2D())
# model3.add(Conv2D(3, kernel_size=3, padding='same', activation='relu'))
# model3.add(Conv2DTranspose(3, kernel_size=5, padding='same', activation='relu', input_shape=(height/2,width/2,1), out_shape=(height,width,1)))
model3.add(Conv2D(1, kernel_size=3, padding='same', activation='sigmoid'))

model3.compile(optimizer='adam', loss=losses.mean_squared_error, metrics=['mse'])
model3.fit(X_train, y_train, epochs=10, steps_per_epoch = 1)

save_model(model3, 3)














X_test = getData('/Users/adityalakra/Desktop/Image_Colorizer/x_test', -1)
no_of_samples,_,_,_ = X_test.shape
showpredict(X_test, no_of_samples)
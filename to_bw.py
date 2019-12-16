from PIL import Image
import os
import numpy as np


for filename in os.listdir('/Users/adityalakra/Desktop/Image_Colorizer/im/'):
	if filename != ".DS_Store":
		print filename
		img = Image.open('/Users/adityalakra/Desktop/Image_Colorizer/im/' + filename)

		data = np.asarray(img, dtype=np.uint8)
		r = data[:, :, 0]
		g = data[:, :, 1]
		b = data[:, :, 2]
		newdata = 0.21*r + 0.72*g + 0.07*b
		newtemp = np.array(newdata, dtype=np.uint8)
		img_f = Image.fromarray(newtemp)

		img_f.save('/Users/adityalakra/Desktop/Image_Colorizer/im_f/' + filename)



# img = Image.open('flowerpot.jpg')
# data = np.asarray(img, dtype=np.uint8)
# data = data/255.0
# print data.shape
# width = 367
# height = 500
# y_train = np.zeros((1,height,width,3))
# y_train[0,:,:,:] = data

# r = data[:, :, 0]
# g = data[:, :, 1]
# b = data[:, :, 2]
# train = np.zeros((height,width,1))
# train[:,:,0] = 0.21*r + 0.72*g + 0.07*b
# X_train = np.zeros((1,height,width,1))
# X_train[0,:,:,:] = train

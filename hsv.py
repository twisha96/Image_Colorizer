from skimage.color import rgb2lab, lab2rgb, rgb2hsv, hsv2rgb
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

img = img_to_array(load_img("woman.jpg"))
img = np.array(img, dtype=float)
x = rgb2hsv(img)[:,:,0]
y = rgb2hsv(img)[:,:,1]
z = rgb2hsv(img)[:,:,2]

print np.min(np.array(x))
print np.max(np.array(x))
print np.min(np.array(y))
print np.max(np.array(y))

print np.min(np.array(z))
print np.max(np.array(z))

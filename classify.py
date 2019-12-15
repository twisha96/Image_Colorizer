from PIL import Image
import os

basic_colors = [[0.0, 0.0, 0.0], [255.0, 255.0, 255.0], [127.5, 127.5, 127.5],
                [255.0, 0.0, 0.0], [0.0, 255.0, 0.0], [0.0, 0.0, 255.0],
                [255.0, 255.0, 0.0], [255.0, 0.0, 255.0], [0.0, 255.0, 255.0],
                [0.0, 127.5, 0.0], [127.5, 0.0, 0.0], [0.0, 0.0, 127.5],
                [127.5, 127.5, 0.0], [127.5, 0.0, 127.5], [0.0, 127.5, 127.5]]

bc = np.array(basic_colors)

for filename in os.listdir('/Users/adityalakra/Desktop/Image_Colorizer/y_train'):
	if filename.endswith(.jpg) or filename.endswith(.jpeg):
		print filename
		img = Image.open('/Users/adityalakra/Desktop/Image_Colorizer/y_train/' + filename)
		for i in range(200):
			for j in range(200):
				val = np.array([bc[i, j, 0]], [bc[i, j, 1]], [bc[i, j, 2]])
				val = closest(val)
		img.save('/Users/adityalakra/Desktop/Image_Colorizer/class_input/' + filename)
from PIL import Image
import os

for filename in os.listdir('/Users/adityalakra/Desktop/Image_Colorizer/raw_input'):
	if filename == "woman.png":
		print filename
		img = Image.open('/Users/adityalakra/Desktop/Image_Colorizer/raw_input/' + filename)
		newsize = (400, 400) 
		img = img.resize(newsize) 
		img.save('/Users/adityalakra/Desktop/Image_Colorizer/Input/' + filename)
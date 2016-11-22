import numpy as np
import os
import re
import sys
from scipy import ndimage
from scipy import misc

def getPNG(directory):
	file_list = os.listdir(directory)
	png_list = []
	for file in file_list:
		match = re.search('.png',file)
		if match:
			png_list.append(directory + '/' + file)
	return png_list

def resizeGray(filename):
	img = ndimage.imread(filename,mode='L')
	img = misc.imresize(img,size=(32,32))
	return img

# Starting preprocess
directory = os.getcwd() + '/' + sys.argv[1]
files = getPNG(directory)
M = np.ndarray((0,1024))
label = 0
Labeled = False
try:
	label = int(sys.argv[2])
	Labeled = True
except IndexError:
	pass

for f in files:
	print(f)
	img = resizeGray(f)
	array = np.reshape(img.astype(float),(1,1024))
	array /= 255.0
	M = np.vstack((M,array))

if Labeled == True:
	y = label * np.ones((np.size(M,0),1))
	M = np.hstack((M,y))

print('Appending to process.txt')
file = open('process.txt','a')
np.savetxt(file,M,delimiter=' ')

print('Done!')
import numpy as np
import os
import re
import sys
from scipy import ndimage
from scipy import misc

# Lay danh sach cac file .png trg thu muc truyen vao
def getPNG(directory):
	file_list = os.listdir(directory) # Danh sach file trg directory
	png_list = []
	for file in file_list:
		match = re.search('.png',file) # Ktra file co chua .png
		if match:
			png_list.append(directory + '/' + file) # Them file vao danh sach
	return png_list

# Dua anh ve kich co 32x32 dang den trang
def resizeGray(filename):
	img = ndimage.imread(filename,mode='L') # mode='L': doc dinh dang den trang
	img = misc.imresize(img,size=(32,32))
	return img

# Starting preprocess
directory = os.getcwd() + '/' + sys.argv[1] # Lay duong dan tuyet doi cua thu muc dua vao
files = getPNG(directory) # Lay ds file
M = np.ndarray((0,1024)) # Ma tran chua du lieu + nhan (neu co)
label = 0
Labeled = False
try:
	label = int(sys.argv[2]) #Ktra co nhan dc truyen vao
	Labeled = True
except IndexError:
	pass

for f in files:
	print(f)
	img = resizeGray(f)
	array = np.reshape(img.astype(float),(1,1024))
	array /= 255.0 # Dua gia tri pixel ve khoang 0-1
	M = np.vstack((M,array)) # Them vao cuoi M

# Gan them nhan neu co
if Labeled == True:
	y = label * np.ones((np.size(M,0),1))
	M = np.hstack((M,y))

print('Appending to process.txt')
file = open('process.txt','a')
np.savetxt(file,M,delimiter=' ')

print('Done!')

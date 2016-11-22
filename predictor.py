import nnlib as nn
import numpy as np
import matplotlib.pyplot as plt
import sys

inp_name = sys.argv[1]
param_name = sys.argv[2]
num_labels = int(sys.argv[3])

X = np.loadtxt(inp_name)
row = np.size(X,0)
X = np.hstack((np.ones((row,1)),X))
col = np.size(X,1)

params = np.loadtxt(param_name)

input_size = col-1
hidden_size1 = nn.hidden_size1
hidden_size2 = nn.hidden_size2

pred = nn.predict(X,params,input_size,hidden_size1,hidden_size2,num_labels)
pred = pred.flatten()

c = 1
plt.ion()
for i in pred:
	character = ''
	if(i<26):
		character = chr(int(i) + ord('A'))
	else:
		character = chr(int(i) + 6 + ord('A'))
	print('Result at input #' + str(c) + ': label ' + str(int(i)) + ' - ' + character)
	img = np.reshape(X[c-1,1:np.size(X,1)],(32,32))
	plt.imshow(img,cmap='Greys_r')
	c += 1
	try:
		input('Press Enter to continue...')
	except SyntaxError:
		pass
print('End of input')
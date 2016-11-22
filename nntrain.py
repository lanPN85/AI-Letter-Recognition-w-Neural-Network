import sys
import numpy as np
import scipy.optimize
import warnings
import math

def sigmoid(z):
    s = 1.0 + np.exp(-z)
    return 1.0/s

def sigmoidGrad(z):
	return sigmoid(z) * (1.0-sigmoid(z))

def feedForward(X,theta0,theta1,theta2):
	a0 = X
	a1 = sigmoid(np.dot(theta0,a0.T))
	a1 = np.hstack((np.ones((np.size(a1,1),1)),a1.T))
	a2 = sigmoid(np.dot(theta1,a1.T))
	a2 = np.hstack((np.ones((np.size(a2,1),1)),a2.T))
	h = sigmoid(np.dot(theta2,a2.T))
	return h

def costFunction(params, input_size, hidden_size1, hidden_size2, num_labels, X, y, lamb):
	m = np.size(X,0)
	Y = y.T
	theta0 = np.reshape(params[0:((input_size+1)*hidden_size1)],(hidden_size1,input_size+1))
	theta1 = np.reshape(params[((input_size+1)*hidden_size1): (hidden_size1+1)*hidden_size2 + (input_size+1)*hidden_size1],(hidden_size2,hidden_size1+1))
	theta2 = np.reshape(params[((hidden_size1+1)*hidden_size2 + (input_size+1)*hidden_size1): params.size],(num_labels,hidden_size2+1))
	
	h = feedForward(X,theta0,theta1,theta2)

	J = (1.0/m) * np.sum( np.log(h)*(-Y) - np.log(1.0-h)*(1-Y) )
	reg = (lamb/2.0/m) * ( np.sum(theta0[:,1:theta0.size]**2) + np.sum(theta1[:,1:theta1.size]**2) + np.sum(theta1[:,1:theta1.size]**2) )
	J += reg
	return J

def gradient(params, input_size, hidden_size1, hidden_size2, num_labels, X, y, lamb):
	m = np.size(X,0)
	Y = y.T
	theta0 = np.reshape(params[0:((input_size+1)*hidden_size1)],(hidden_size1,input_size+1))
	theta1 = np.reshape(params[((input_size+1)*hidden_size1): (hidden_size1+1)*hidden_size2 + (input_size+1)*hidden_size1],(hidden_size2,hidden_size1+1))
	theta2 = np.reshape(params[((hidden_size1+1)*hidden_size2 + (input_size+1)*hidden_size1): params.size],(num_labels,hidden_size2+1))

	theta0_grad = np.zeros_like(theta0)
	theta1_grad = np.zeros_like(theta1)
	theta2_grad = np.zeros_like(theta2)

	i = 0
	while i<m:
		a0 = X[i,:].transpose()
		a0 = a0.reshape((a0.size,1))
		a1 = np.vstack((np.ones((1,1)),sigmoid(np.dot(theta0,a0))))
		a2 = np.vstack((np.ones((1,1)),sigmoid(np.dot(theta1,a1))))
		a3 = sigmoid(np.dot(theta2,a2))

		delta3 = a3 - Y[:,i].reshape((-1,1))
		delta2 = np.dot(theta2.T,delta3)
		delta2 = delta2[1:delta2.size].reshape((delta2.size-1,1))		
		delta2 = delta2 * sigmoidGrad(np.dot(theta1,a1))
		delta1 = np.dot(theta1.T,delta2)
		delta1 = delta1[1:delta1.size] * sigmoidGrad(np.dot(theta0,a0))

		theta0_grad += np.dot(delta1,a0.T)
		theta1_grad += np.dot(delta2,a1.T)
		theta2_grad += np.dot(delta3,a2.T)

		i+=1

	theta0_grad /= m
	theta0_grad[:,1:np.size(theta0_grad,1)] += ((lamb/m)*theta0)[:,1:theta0.size]
	theta1_grad /= m
	theta1_grad[:,1:np.size(theta1_grad,1)] += ((lamb/m)*theta1)[:,1:theta1.size]
	theta2_grad /= m
	theta2_grad[:,1:np.size(theta2_grad,1)] += ((lamb/m)*theta2)[:,1:theta2.size]

	grad = np.vstack((np.reshape(theta0_grad,(theta0_grad.size,1)),np.reshape(theta1_grad,(theta1_grad.size,1)),np.reshape(theta2_grad,(theta2_grad.size,1)))).flatten()
	return grad

def predict(X,pr0,pr1,pr2):
	h0 = sigmoid(np.dot(X,pr0.T))
	h0 = np.hstack((np.ones((np.size(h0,0),1)),h0))
	h1 = sigmoid(np.dot(h0,pr1.T))
	h1 = np.hstack((np.ones((np.size(h1,0),1)),h1))
	h2 = sigmoid(np.dot(h1,pr2.T))

	p = np.zeros((np.size(h2,0),1))
	i = 0
	while i<np.size(h2,0):
		j = 0; mx = -100; ind = 0
		while j<np.size(h2,1):
			if h2[i,j] > mx:
				ind = j
				mx = h2[i,j]
			j += 1
		p[i] = ind
		i += 1
	return p

# Fetch data
inp_name = sys.argv[1]
inp = open(inp_name,"r",buffering=1)

row = int(sys.argv[2])
col = int(sys.argv[3])
X = np.ones((row, col))
y = np.zeros((row, 1))
num_labels = 1

i = 0
for line in inp:
    n = line.split()
    j = 1
    while j<col:
        X[i][j] = float(n[j-1])
        j += 1
    y[i][0] = float(n[col-1])
    if y[i][0] > num_labels:
        num_labels = y[i][0]
    i += 1

# Convert labels
Y = np.zeros((row,num_labels+1))
i = 0
while i<row:
    Y[i,y[i,0]] = 1
    i += 1
# Set up neural network
lamb = 0
eps = 0.15
input_size = col-1
hidden_size1 = 5
hidden_size2 = 5
theta0 = np.random.rand(hidden_size1,input_size+1)
#eps = math.sqrt(6.0) / math.sqrt(float(hidden_size1+input_size+1))
theta0 = theta0 * 2 * eps - eps
theta1 = np.random.rand(hidden_size2,hidden_size1+1)
#eps = math.sqrt(6.0) / math.sqrt(float(hidden_size2+hidden_size1+1))
theta1 = theta1 * 2 * eps - eps
theta2 = np.random.rand(num_labels+1,hidden_size2+1	)
#eps = math.sqrt(6.0) / math.sqrt(float(num_labels+hidden_size2+2))
theta2 = theta2 * 2 * eps - eps
nn_params = np.vstack((np.reshape(theta0,(theta0.size,1)),np.reshape(theta1,(theta1.size,1)),np.reshape(theta2,(theta2.size,1)))).flatten()
# Training
print("Training neural network...")
warnings.filterwarnings("ignore")
print("Initial cost: "); print(costFunction(nn_params,input_size,hidden_size1,hidden_size2,num_labels+1,X,Y,lamb));

result = scipy.optimize.fmin_l_bfgs_b(costFunction,nn_params,fprime=gradient,args=(input_size, hidden_size1, hidden_size2, num_labels+1,X,Y,lamb),disp = 1, maxiter = 120, factr = 1e5)
#result = scipy.optimize.fmin_bfgs(costFunction,nn_params,fprime=gradient,args=(input_size, hidden_size1, hidden_size2, num_labels+1,X,Y,lamb))
nn_params = result[0]

# Training prediction
pr0 = np.reshape(nn_params[0:((input_size+1)*hidden_size1)],(hidden_size1,input_size+1))
pr1 = np.reshape(nn_params[((input_size+1)*hidden_size1): (hidden_size1+1)*hidden_size2 + (input_size+1)*hidden_size1],(hidden_size2,hidden_size1+1))
pr2 = np.reshape(nn_params[((hidden_size1+1)*hidden_size2 + (input_size+1)*hidden_size1): nn_params.size],(num_labels+1,hidden_size2+1))

i = 0
pred = predict(X,pr0,pr1,pr2)
diff = pred - y
correct = 0
while i<row:
	if diff[i]==0:
		correct += 1
	i += 1

#print(diff)
print("Training accuracy: " + (float(correct)/float(row)).__str__())
def assess(X,y,pr0,pr1,pr2,row):
	i = 0
	pred = predict(X,pr0,pr1,pr2)
	diff = pred - y
	correct = 0
	while i<row:
		if diff[i]==0:
			correct += 1
		i += 1
	acc = float(correct)/float(row)
	return acc
print("Training accuracy: " + str(assess(X,y,pr0,pr1,pr2,row)))
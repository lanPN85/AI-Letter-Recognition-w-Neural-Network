import numpy as np
import math

# ----Network parameters----
hidden_size1 = 40
hidden_size2 = 60
lamb_list = (5.4,6.0)
eps = 0.15
max_iter = 3600

# ----Functions used for a 2-layer Neural Network----
def sigmoid(z):
    s = 1.0 + np.exp(-z)
    return 1.0/s

def sigmoidGrad(z):
	s = sigmoid(z)
	return s * (1.0-s)

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

def predict(X,params,input_size,hidden_size1,hidden_size2,num_labels):
	pr0 = np.reshape(params[0:((input_size+1)*hidden_size1)],(hidden_size1,input_size+1))
	pr1 = np.reshape(params[((input_size+1)*hidden_size1): (hidden_size1+1)*hidden_size2 + (input_size+1)*hidden_size1],(hidden_size2,hidden_size1+1))
	pr2 = np.reshape(params[((hidden_size1+1)*hidden_size2 + (input_size+1)*hidden_size1): params.size],(num_labels,hidden_size2+1))

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

def assess(X,y,params,row,input_size,hidden_size1,hidden_size2,num_labels):
	i = 0
	pred = predict(X,params,input_size,hidden_size1,hidden_size2,num_labels)
	diff = pred - y
	correct = 0
	while i<row:
		if diff[i]==0.0:
			correct += 1
		i += 1
	acc = float(correct)/float(row)
	return acc

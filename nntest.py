import sys
import numpy as np
import scipy.optimize
import warnings
import math
import nnlib as nn

# ----- Neural Network /w 2 Hidden Layers ------
# Fetch training data
inp_name = sys.argv[1]
M = np.loadtxt(inp_name)
row = np.size(M,0)
col = np.size(M,1)

X = np.hstack((np.ones((row,1)),M[:,0:col-1]))
y = np.reshape(M[:,col-1],(row,1))

# Fetch cross-validation data
inp_name = sys.argv[2]
M = np.loadtxt(inp_name)
cv_row = np.size(M,0)

Xcv = np.hstack((np.ones((cv_row,1)),M[:,0:col-1]))
ycv = np.reshape(M[:,col-1],(cv_row,1))

# Fetch test data
inp_name = sys.argv[3]
M = np.loadtxt(inp_name)
test_row = np.size(M,0)

Xtest = np.hstack((np.ones((test_row,1)),M[:,0:col-1]))
ytest = np.reshape(M[:,col-1],(test_row,1))

# Convert labels
num_labels = int(sys.argv[4])
Y = np.zeros((row,num_labels))
Ycv = np.zeros((cv_row,num_labels))
i = 0
while i<row:
    Y[i,y[i,0]] = 1
    i += 1
i = 0
while i<cv_row:
    Ycv[i,ycv[i,0]] = 1
    i += 1

# Set up neural network
lamb = 0
lamb_list = nn.lamb_list
eps = nn.eps
input_size = col-1
hidden_size1 = nn.hidden_size1
hidden_size2 = nn.hidden_size2

# Training + Cross Validating
warnings.filterwarnings("ignore")
theta0 = np.random.rand(hidden_size1,input_size+1)
theta1 = np.random.rand(hidden_size2,hidden_size1+1)
theta2 = np.random.rand(num_labels,hidden_size2+1	)
best_params = np.vstack((np.reshape(theta0,(theta0.size,1)),np.reshape(theta1,(theta1.size,1)),np.reshape(theta2,(theta2.size,1)))).flatten()
best_acc = -1.0
best_jcv = 1000.0
best_cv_acc = -1.0
best_lamb = 0

for lamb in lamb_list:
	theta0 = np.random.rand(hidden_size1,input_size+1)
	theta0 = theta0 * 2 * eps - eps
	theta1 = np.random.rand(hidden_size2,hidden_size1+1)
	theta1 = theta1 * 2 * eps - eps
	theta2 = np.random.rand(num_labels,hidden_size2+1)
	theta2 = theta2 * 2 * eps - eps
	nn_params = np.vstack((np.reshape(theta0,(theta0.size,1)),np.reshape(theta1,(theta1.size,1)),np.reshape(theta2,(theta2.size,1)))).flatten()

	print("Training neural network...")
	print("lambda = " + str(lamb))
	print("Initial cost: "+ str(nn.costFunction(nn_params,input_size,hidden_size1,hidden_size2,num_labels,X,Y,lamb)))

	result = scipy.optimize.fmin_l_bfgs_b(nn.costFunction,nn_params,fprime=nn.gradient,args=(input_size, hidden_size1, hidden_size2, num_labels,X,Y,lamb),disp = 1, maxiter = nn.max_iter)
	train_params = result[0]
	
	jcv = nn.costFunction(train_params,input_size,hidden_size1,hidden_size2,num_labels,Xcv,Ycv,lamb)
	print("Cost at cross-validation: " + str(jcv))
	cv_acc = nn.assess(Xcv,ycv,train_params,cv_row,input_size,hidden_size1,hidden_size2,num_labels)
	print("Cross-validation accuracy: " + str(cv_acc))

	if jcv<best_jcv:
		best_jcv = jcv
		best_cv_acc = cv_acc
		best_acc = nn.assess(X,y,train_params,row,input_size,hidden_size1,hidden_size2,num_labels)
		best_params = train_params
		best_lamb = lamb

# Test results
print("\n---Training complete---")
print("Optimal lambda: " + str(best_lamb))
print("Best cost at cross-validation: " + str(best_jcv))
print("Best cross-validation accuracy: " + str(best_cv_acc))
print("Training accuracy: " + str(best_acc))
print("Test accuracy: " + str(nn.assess(Xtest,ytest,best_params,test_row,input_size,hidden_size1,hidden_size2,num_labels)))

# Save parameters
out_name = "params.dat"
print("\nSaving parameters to " + out_name + " ...")
np.savetxt(out_name,best_params,delimiter=" ")
print("Done!")
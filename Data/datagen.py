import numpy as np
import math

# ----Randomize and split processed data into a training set, a cross-validation set
# ----and a test set with a 70:15:15 ratio
print('Loading from process.txt...')
M = np.loadtxt('process.txt',delimiter=' ')
print('Done!')
np.random.shuffle(M) # Tron ngau nhien cac hang trg M
rows = np.size(M,0)

r1 = math.ceil(float(rows)*0.7) # Tinh so hang cua tap train
Train = M[0:r1,:]
r2 = math.floor(float(rows-r1)*0.5) # Tinh so hang cua tap cv & test
Cv = M[r1:r1+r2,:]
Test = M[r1+r2:rows,:]

# Luu vao file
print('Generating data...')
np.savetxt('train.txt',Train,delimiter=' ')
np.savetxt('cv.txt',Cv,delimiter=' ')
np.savetxt('test.txt',Test,delimiter=' ')
print('Done!')

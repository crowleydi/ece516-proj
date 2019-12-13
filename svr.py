import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import svm
from sklearn.model_selection import cross_val_score
from graddesc import graddesc

# set to True for SVR
# set to False for NuSVR
svr=True

# read the data file
df = pd.read_csv("data.csv", header=None)

# break the data file up into components as numpy arrays
#
#ftest = df.iloc[:,0].to_numpy()
#ftrain = df.iloc[:,1].to_numpy()
Xtest = df.iloc[:,2:21].to_numpy()
Xtrain = df.iloc[:,21:40].to_numpy()
ytest = df.iloc[:,40].to_numpy()
ytrain = df.iloc[:,41].to_numpy()

fevals = 0;
def loss_func(x):
	global svr, Xtrain, ytrain, fevals
	fevals = fevals + 1
	C = 10**x[0]
	eps = 10**x[1]
	nu = x[1]
	
	if nu < 0.1:
		nu = 0.1
	elif nu > 0.9:
		nu = 0.9

	if svr:
		clf = svm.SVR(gamma='auto',C=C,epsilon=eps)
	else:
		clf = svm.NuSVR(gamma='auto',C=C,nu=nu)
	
	scores = cross_val_score(clf, Xtrain, ytrain, cv=5,scoring='neg_mean_squared_error')
	print("scores="+str(scores))
	return -10*scores.mean()


#
# Generate a 3D surface plot of the loss_func function
#
X = np.arange(-1, 2, 0.1)
if svr:
	Y = np.arange(-3, -1, 2/40)
else:
	Y = np.arange(0.1,0.9,0.8/40)

X, Y = np.meshgrid(X, Y)
Z = X * 0.0

for i in range(X.shape[0]):
	for j in range(X.shape[1]):
		Z[i,j] = loss_func([X[i,j],Y[i,j]])

fig = plt.figure()
ax = fig.gca(projection='3d')

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=plt.cm.hot, alpha=0.7)
# linewidth=0, antialiased=False)

# function to observe the gradient descent
X = []
Y = []
Z = []

def myobs(fx, x, g, alpha):
	global X, Y, Z
	X.append(x[0])
	Y.append(x[1])
	Z.append(fx)

# calculate a random starting point
C = np.random.uniform(low=-1,high=2)
if svr:
	eps = np.random.uniform(low=-3,high=-1)
	x0=np.array([C, eps])
else:
	nu = np.random.uniform(low=0.1,high=0.9)
	x0=np.array([C, nu])

dx = np.array([0.01,0.01])

#print("loss_func(x0)="+str(loss_func(x0)))
#print("loss_func(x0)="+str(loss_func(x0)))
fevals = 0
x = graddesc(loss_func,x0,dx,obs=myobs,alpha=0.25)
print("final solution="+str(x))
print("function evals="+str(fevals))
ax.plot(X,Y,Z,'bo-')
ax.plot([X[0]],[Y[0]],[Z[0]],'go-')
ax.plot([X[-1]],[Y[-1]],[Z[-1]],'ro-')
plt.show()

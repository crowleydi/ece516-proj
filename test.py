from graddesc import graddesc
import numpy as np

fevals = 0

def myfunc(xx):
	global fevals
	x = xx[0]
	y = xx[1]
	# rosenbrock
	z = (1-x)**2+100*(y-x*x)**2
	#z = (x+7)**2+(y-2)**2
	fevals = fevals + 1
	return z

def myobs(fx, x, g, alpha):
	print("["+str(fx)+","+str(x)+","+str(g)+","+str(alpha)+"]")

x0 = np.array([-1,1]);
dx = np.array([1e-6,1e-6]);
x = graddesc(myfunc,x0,dx,obs=myobs)

print("Total fevals: "+str(fevals))

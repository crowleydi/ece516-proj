import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

X = np.arange(-1, 2, 0.1)
#if svr:
Y = np.arange(-3, -1, 2/40)
#else:
	#Y = np.arange(0.1,0.9,0.8/40)

X, Y = np.meshgrid(X, Y)
# Plot the surface.
surffile='surffile.npy'
Z = np.load(surffile)

fname = sys.argv[1]
#
# Generate a 3D surface plot of the loss_func function
#
#plt.show()
hist = np.load(fname)
x = hist[0,:]
y = hist[1,:]
z = hist[2,:]
zlow = z[0]
ilow = 0
for i in range(len(z)):
	if z[i] < zlow:
		zlow = z[i]
		ilow = i

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap=plt.cm.hot, alpha=0.7)
# linewidth=0, antialiased=False)
ax.plot(x,y,z,'bo-')
ax.plot([x[0]],[y[0]],[z[0]],'go-')
ax.plot([x[-1]],[y[-1]],[z[-1]],'ro-')
ax.plot([x[ilow]],[y[ilow]],[z[ilow]],'yo-')
plt.draw()
plt.suptitle("Gradient descent of hyperparamters")
title="f(x0)={:5.3f}, f(low)={:5.3f}, f(final)={:5.3f}".format(z[0],zlow,z[-1])
print(title)
plt.title(title)
fig.savefig(fname+".png")

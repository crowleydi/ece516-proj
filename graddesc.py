import numpy as np

def gradapprox(f, fx, x, dx):
	g = x * 0.0
	ddx = dx*0.0
	for i in range(len(x)):
		ddx[i] = dx[i]
		fxdx = f(x + ddx)
		g[i] = (fxdx-fx)/ddx[i]
		ddx[i] = 0
	return g

def graddesc(f, x0, dx, T=30, alpha=1, gtol=1e-2, tol=1e-6, obs=None):
	nstep = 0
	v = x0 * 0.0;
	x = x0;
	fx = f(x)
	for t in range(T):
		g = -gradapprox(f, fx, x, dx)
		if (obs != None):
			obs(fx, x, g, alpha)
		gnorm = np.linalg.norm(g)

		x1 = x+alpha*g
		fx1 = f(x1)

		nstep = nstep + 1

		while (fx1 > fx):
			alpha = alpha / 2.0
			nstep = 0
			x1 = x+alpha*g
			fx1 = f(x1)

			if (obs != None):
				obs(fx1, x1, g, alpha)

			diff = abs(fx-fx1)
			if (diff < tol):
				break

		fx = fx1
		x = x1

		if (nstep == 4):
			alpha = alpha * 2.0
			nstep = 0

		if (diff < tol):
			break

		if(gnorm < gtol):
			break

	if (obs != None):
		obs(fx, x, g, alpha)

	return x


import math

def _norm(X):
	# calculate L2 norm of vector X
	total = 0;
	for x in X:
		total = total + x*x
	return math.sqrt(total)

def gradapprox(f, fx, x, dx):
	# f - function
	# fx - function value evaluated at x
	# x - location to calculate gradient
	# dx - vector of lengths to wiggle in each dimension

	# initialize gradient vector
	g = x * 0.0
	# vector of zeros for wiggle vector
	ddx = dx * 0.0
	for i in range(len(x)):
		# put a value in the wiggle vector
		ddx[i] = dx[i]
		# calculate function with wiggle
		fxdx = f(x + ddx)
		# calculate gradient in wiggle dimension
		g[i] = (fxdx-fx)/ddx[i]
		# reset wiggle to zero
		ddx[i] = 0
	
	# return approximate gradient
	return g

def graddesc(f, x0, dx, T=30, alpha=1, gtol=1e-2, tol=1e-6, obs=None):
	nstep = 0
	v = x0 * 0.0;
	x = x0;

	# calulate function value at x0
	fx = f(x)

	for t in range(T):

		# calculate gradient at x
		g = -gradapprox(f, fx, x, dx)

		# call the observer
		if (obs != None):
			obs(fx, x, g, alpha)

		#gnorm = _norm(g)

		# calculate the next point along the gradient direction
		x1 = x+alpha*g
		fx1 = f(x1)
		diff = abs(fx-fx1)
		nstep = nstep + 1

		#
		# in this loop we basically do a line search
		# along the gradient line until we find a
		# location that goes down or only goes up
		# a little bit (10%)
		loop = 4
		while (fx1 > fx*1.1 and loop > 0):
			loop = loop - 1

			# need to slow down and not move as far
			alpha = alpha / 2.0
			nstep = 0
			x1 = x+alpha*g
			fx1 = f(x1)

			if (obs != None):
				obs(fx1, x1, g, alpha)

			#diff = abs(fx-fx1)
			#if (diff < tol):
				#break

		# move
		fx = fx1
		x = x1

		# if we are on a roll speed up
		if (nstep > 4):
			alpha = alpha * 2.0

		# exit when the flatness tolerance
		# is reached
		if (diff < tol):
			print("exiting tolerance")
			break

		#if(gnorm < gtol):
			#print("exiting gradient")
			#break

	if (obs != None):
		obs(fx, x, g, alpha)

	return x



import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import solver


l = 1.0


# real part x and imaginary part y of complex z = x + iy
def Hopf (S, t):
	
	# ~ l = 0.5
	w = 1.0
	g = 0.0
	
	x, y = S
	
	zz = np.square(x) + np.square(y)
	
	return (np.array([(l - zz) * x - w * y + g * zz * y, 
					  (l - zz) * y + w * x - g * zz * x]))


def numInt (Ttrans, Teval, dt, X0):
	
	s         = solver.ODE_Solver()
	time, sol = s.solveODE(Hopf, X0, Ttrans, Teval, dt, 1, 
						   "explicitRungeKutta4")
	
	X = sol[0, :]
	Y = sol[1, :]
	
	return (time, X, Y)


def plotTimeSeries ():
	
	Ttrans = 0.0
	Teval  = 30.0
	dt     = 1e-2
	X0     = [-1.0, -1.0]
	
	time, X, Y = numInt(Ttrans, Teval, dt, X0)
	
	# parameters for streamplot
	global l
	w = 1.0
	g = 0.0
	
	fig, ax = plt.subplots()
	fig.set_size_inches(5.9, 5.9)
	plt.subplots_adjust(top=0.98, bottom=0.13, right=0.98)
	ax.plot(time, X, color="red", label=r"$x(t)$")
	ax.plot(time, Y, color="blue", label=r"$y(t)$")
	plt.text(21.5, -0.525, r"$\lambda=$" + str(l), fontsize=20, bbox={"facecolor":"white", "edgecolor":"none", "alpha":0.95})
	ax.set_aspect(30.0 / 2.0)
	ax.set_xlabel(r"time $t$", fontsize=20)
	ax.set_ylim(-1.1, 1.1)
	ax.tick_params(axis="both", labelsize=20)
	plt.setp(ax, xticks=[0.0, 10.0, 20.0, 30.0], yticks=[-1.0, 0.0, 1.0])
	ax.legend(fancybox=True, framealpha=0.95, loc="lower right", fontsize=20)
	plt.grid(color="lightgray")
	
	fig, ax = plt.subplots()
	fig.set_size_inches(5.9, 5.9)
	plt.subplots_adjust(top=0.98, bottom=0.13,right=0.98)
	ax.plot(X, Y, color="black")
	y, x = np.mgrid[-1.5:1.5:100j, -1.5:1.5:100j]
	u = (l - np.square(x) - np.square(y)) * x - w * y + g * (np.square(x) + np.square(y)) * y
	v = (l - np.square(x) - np.square(y)) * y + w * x - g * (np.square(x) + np.square(y)) * x
	ax.streamplot(x, y, u, v, color="lightgray")
	ax.set_aspect(1.0)
	ax.set_xlabel(r"$x$", fontsize=20)
	ax.set_ylabel(r"$y$", fontsize=20)
	ax.plot(X[0], Y[0], color="gray", marker="o", markersize=8)
	if l < 0.0:
		ax.plot(X[-1], Y[-1], color="black", marker="o", markersize=8)
	ax.text(-0.28, 1.175, r"$\lambda=$" + str(l), fontsize=20)
	plt.setp(ax, xticks=[-1.0, 0.0, 1.0], yticks=[-1.0, 0.0, 1.0])
	ax.tick_params(axis="both", labelsize=20)
	
	# ~ plt.figure()
	# ~ ax = plt.axes(projection="3d")
	# ~ ax.plot3D(time, X, Y, color="black")
	# ~ ax.set_xlabel(r"time $t$")
	# ~ ax.set_ylabel(r"Re($z$)")
	# ~ ax.set_zlabel(r"Im($z$)")
	
	plt.show()


def plotHopfBifurcationDiagram ():
	
	myfontsize = 18.0
	
	Ttrans = 100.0
	Teval  = 10.0
	dt     = 1e-2
	X0     = [1e-3, 0.0]
	
	nSteps = int(Teval / dt)
	
	lList = np.arange(-1.0, 1.0 + 1e-5, 0.025)
	
	X = np.zeros((nSteps, lList.size))
	Y = np.zeros((nSteps, lList.size))
	
	i = 0
	
	global l
	
	for L in lList:
		
		l = L
		
		time, x, y = numInt(Ttrans, Teval, dt, X0)
		
		X[:, i] = x
		Y[:, i] = y
		
		i += 1
		
		print(i, " of ", lList.size, " simulations")
	
	
	one = np.ones(X[:, 0].shape)
	
	fig = plt.figure()
	ax  = fig.add_subplot(111, projection="3d")
	fig.set_size_inches(5.9, 4.8)
	plt.rcParams.update({"font.size": myfontsize})
	plt.subplots_adjust(top=1.05, bottom=0.13, left=0.01, right=0.99)
	
	for i in np.arange(lList.size):
		ax.scatter(lList[i] * one, X[:, i], Y[:, i], color="black", s=0.1)
	
	ax.view_init(elev=30.0, azim=-120.0)
	ax.zaxis.set_rotate_label(False)
	ax.set_xlabel(r"bifurcation parameter $\lambda$", fontsize=myfontsize)
	ax.set_ylabel(r"$x$", fontsize=myfontsize)
	ax.set_zlabel(r"$y$", fontsize=myfontsize, rotation=90.0)
	ax.xaxis.set_tick_params(labelsize=myfontsize)
	ax.yaxis.set_tick_params(labelsize=myfontsize)
	ax.zaxis.set_tick_params(labelsize=myfontsize)
	ax.xaxis.labelpad=18
	ax.yaxis.labelpad=9
	ax.zaxis.labelpad=4
	ax.set_xticks([-1.0, 0.0, 1.0])
	ax.set_yticks([-1.0, 0.0, 1.0])
	ax.set_zticks([-1.0, 0.0, 1.0])
	
	plt.show()


plotTimeSeries()
plotHopfBifurcationDiagram()


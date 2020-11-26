import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import solver


b = 1.01


def sniper (S, t):
	
	x, y = S
	
	zz = np.square(x) + np.square(y)
	
	return (np.array([x * (1 - zz) + y * (x - b), 
					  y * (1 - zz) - x * (x - b)]))


def numInt (Ttrans, Teval, dt, X0):
	
	s         = solver.ODE_Solver()
	time, sol = s.solveODE(sniper, X0, Ttrans, Teval, dt, 1, 
						   "explicitRungeKutta4")
	
	X = sol[0, :]
	Y = sol[1, :]
	
	return (time, X, Y)


def plotTimeSeries ():
	
	Ttrans = 0.0
	Teval  = 200.0
	dt     = 1e-2
	X0     = [0.0, 1e-3]
	
	global b
	
	time, X, Y = numInt(Ttrans, Teval, dt, X0)
	
	fig, ax = plt.subplots()
	fig.set_size_inches(5.9, 5.9)
	ax.set_aspect(200.0 / 2.0)
	plt.subplots_adjust(top=0.99, bottom=0.13, left=0.15, right=0.99)
	plt.plot(time, X, color="red", label=r"$x(t)$")
	plt.plot(time, Y, color="blue", label=r"$y(t)$")
	plt.text(142.0, -0.525, r"$b=$" + str(b), fontsize=20, bbox={"facecolor":"white", "edgecolor":"none", "alpha":0.95})
	plt.xlabel(r"time $t$", fontsize=20)
	plt.xticks([0.0, 50.0, 100.0, 150.0, 200.0], fontsize=20)
	plt.yticks([-1.0, 0.0, 1.0], fontsize=20)
	plt.legend(fancybox=True, framealpha=1.0, loc="lower right", fontsize=20)
	plt.grid(color="lightgray")
	
	fig, ax = plt.subplots()
	fig.set_size_inches(5.9, 5.9)
	ax.set_aspect(1.0)
	plt.subplots_adjust(top=0.99, bottom=0.13, left=0.15, right=0.99)
	y, x = np.mgrid[-1.5:1.5:100j, -1.5:1.5:100j]
	u = x * (1 - np.square(x) - np.square(y)) + y * (x - b)
	v = y * (1 - np.square(x) - np.square(y)) - x * (x - b)
	plt.streamplot(x, y, u, v, color="lightgray")
	plt.plot(X, Y, color="black")
	plt.plot(X[0], Y[0], color="gray", marker="o", markersize=8)
	if b <= 1.0:
		plt.plot(X[-1], Y[-1], color="black", marker="o", markersize=8)
	plt.text(-0.28, 1.175, r"$b=$" + str(b), fontsize=20)
	plt.xlabel(r"$x$", fontsize=20)
	plt.ylabel(r"$y$", fontsize=20)
	plt.xticks([-1.0, 0.0, 1.0], fontsize=20)
	plt.yticks([-1.0, 0.0, 1.0], fontsize=20)
	plt.grid(color="lightgray")
	
	plt.show()


def plotSNIPERbifurcationDiagram ():
	
	myfontsize = 18.0
	
	Ttrans = 100.0
	Teval  = 100.0
	dt     = 1e-2
	X0     = [0.0, 1e-3]
	
	nSteps = int(Teval / dt)
	
	bList = np.arange(0.025, 2.0 + 1e-5, 0.025)
	
	X = np.zeros((nSteps, bList.size))
	Y = np.zeros((nSteps, bList.size))
	
	i = 0
	
	global b
	
	for B in bList:
		
		b = B
		
		time, x, y = numInt(Ttrans, Teval, dt, X0)
		
		X[:, i] = x
		Y[:, i] = y
		
		i += 1
		
		print(i, " of ", bList.size, " simulations")
	
	
	one = np.ones(X[:, 0].shape)
	
	fig = plt.figure()
	ax  = fig.add_subplot(111, projection="3d")
	fig.set_size_inches(5.9, 4.8)
	plt.rcParams.update({"font.size": myfontsize})
	plt.subplots_adjust(top=1.05, bottom=0.13, left=0.01, right=0.99)
	
	for i in np.arange(bList.size):
		ax.scatter(bList[i] * one, X[:, i], Y[:, i], color="black", s=0.1)
	
	ax.view_init(elev=30.0, azim=-120.0)
	ax.zaxis.set_rotate_label(False)
	ax.set_xlabel(r"bifurcation parameter $b$", fontsize=myfontsize)
	ax.set_ylabel(r"$x$", fontsize=myfontsize)
	ax.set_zlabel(r"$y$", fontsize=myfontsize, rotation=90.0)
	ax.set_xlim(-0.1, 2.1)
	ax.xaxis.set_tick_params(labelsize=myfontsize)
	ax.yaxis.set_tick_params(labelsize=myfontsize)
	ax.zaxis.set_tick_params(labelsize=myfontsize)
	ax.xaxis.labelpad=18 #19
	ax.yaxis.labelpad=9  #15
	ax.zaxis.labelpad=4  #7
	ax.set_xticks([0.0, 1.0, 2.0])
	ax.set_yticks([-1.0, 0.0, 1.0])
	ax.set_zticks([-1.0, 0.0, 1.0])
	
	plt.show()


plotTimeSeries()
plotSNIPERbifurcationDiagram()


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import solver


mu = -0.8645

def homoclinic (S, t):
	
	x, y = S
	
	return (np.array([y, 
					  mu * y + x - np.square(x) + x * y]))


def numInt (Ttrans, Teval, dt, X0):
	
	s         = solver.ODE_Solver()
	time, sol = s.solveODE(homoclinic, X0, Ttrans, Teval, dt, 1, 
						   "explicitRungeKutta4")
	
	X = sol[0, :]
	Y = sol[1, :]
	
	return (time, X, Y)


def plotTimeSeries ():
	
	myfontsize = 20.0
	dt         = 1e-2
	
	Ttrans1 = 0.0
	Teval1  = 8.0
	X0_1    = [0.0, 1.0]
	
	Ttrans2 = 0.0
	Teval2  = 47.0
	X0_2    = [0.3263, 0.0]
	
	global mu
	
	time1, X1, Y1 = numInt(Ttrans1, Teval1, dt, X0_1)
	time2, X2, Y2 = numInt(Ttrans2, Teval2, dt, X0_2)
	
	'''
	fig, ax = plt.subplots()
	fig.set_size_inches(5.9, 5.9)
	# ~ ax.set_aspect(4.4 / 3.4)
	plt.subplots_adjust(top=0.99, bottom=0.13, left=0.15, right=0.99)
	plt.plot(time2, X2, color="red", label=r"$x(t)$")
	plt.plot(time2, Y2, color="blue", label=r"$y(t)$")
	plt.text(21.5, -0.13, r"$\mu=$" + str(mu), fontsize=20, bbox={"facecolor":"white", "edgecolor":"none", "alpha":0.95})
	plt.xlabel(r"time $t$", fontsize=myfontsize)
	plt.xticks([0.0, 10.0, 20.0, 30.0], fontsize=myfontsize)
	plt.yticks([-0.5, 0.0, 0.5, 1.0, 1.5], fontsize=myfontsize)
	plt.legend(fancybox=True, framealpha=1.0, loc="lower right", fontsize=myfontsize)
	plt.grid(color="lightgray")
	'''
	
	
	fig, ax = plt.subplots()
	fig.set_size_inches(5.9, 5.9)
	ax.set_aspect(1)
	plt.subplots_adjust(top=0.99, bottom=0.13, left=0.15, right=1.00)
	y, x = np.mgrid[-2.5:2.5:75j, -2.5:2.5:75j]
	u = y
	v = mu * y + x - np.square(x) + x * y
	plt.streamplot(x, y, u, v, color="lightgray")
	plt.plot(X1, Y1, color="black", linestyle="--")
	plt.plot(X1[0], Y1[0], color="gray", marker="o", markersize=8)
	plt.plot(X2, Y2, color="black")
	plt.plot(X2[0], Y2[0], color="gray", marker="o", markersize=8)
	plt.xlim(-2.2, 2.2)
	plt.ylim(-2.2, 2.2)
	plt.text(-0.69, 1.55, r"$\mu=$" + str(mu), fontsize=myfontsize, bbox={"facecolor":"white", "edgecolor":"none", "alpha":0.95})
	plt.xlabel(r"$x$", fontsize=myfontsize)
	plt.ylabel(r"$y$", fontsize=myfontsize)
	plt.xticks([-2.0, -1.0, 0.0, 1.0, 2.0], fontsize=myfontsize)
	plt.yticks([-2.0, -1.0, 0.0, 1.0, 2.0], fontsize=myfontsize)
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
# ~ plotSNIPERbifurcationDiagram()


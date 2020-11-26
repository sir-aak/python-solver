import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import solver


def lorenz (S, t):
    
    s = 10.0
    r = 28.0
    b = 8 / 3
    
    x, y, z = S
    
    return (np.array([-s * x + s * y, 
                      r * x - y - x * z, 
                      x * y - b * z]))


def numInt ():
    
    Ttrans = 0.0
    Teval  = 50.0
    dt     = 1e-3
    y0     = [0.1, 0.0, 0.0]
    
    s         = solver.ODE_Solver()
    time, sol = s.solveODE(lorenz, y0, Ttrans, Teval, dt, 1, "explicitRungeKutta4")
    
    X = sol[0, :]
    Y = sol[1, :]
    Z = sol[2, :]
    
    return (time, X, Y, Z)


def view ():
    
    time, X, Y, Z = numInt()
    
    fig = plt.figure()
    fig.suptitle("Lorenz system", fontsize=18)
    fig.set_size_inches(15, 7)
    fig.subplots_adjust(wspace=0.3)
    
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(time, X, label=r"$x(t)$")
    ax1.plot(time, Y, label=r"$y(t)$")
    ax1.plot(time, Z, label=r"$z(t)$")
    ax1.set_title("time series", fontsize=16)
    ax1.set_xlabel(r"time $t$")
    ax1.legend(loc="lower left")
    ax1.grid(color="lightgray")
    
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.plot3D(X, Y, Z, lw=1, zorder=-1)
    ax2.scatter(X[0], Y[0], Z[0], color="orange", marker="o")
    ax2.set_title("phase space", fontsize=16)
    ax2.set_xlabel(r"$x$", labelpad=10)
    ax2.set_ylabel(r"$y$", labelpad=10)
    ax2.set_zlabel(r"$z$", labelpad=10)


def animation (fps, speed=300):
    
    s  = solver.ODE_Solver()            # instantiate ode-solver object
    
    y  = [0.1, 0.0, 0.0]
    dt = 1e-3
    X  = []
    Y  = []
    Z  = []
    
    i = 0
    t = 0.0
    
    fig   = plt.figure(1)
    fig.suptitle("Lorenz system: phase space", fontsize=18)
    fig.set_size_inches(12, 9)
    
    ax    = plt.axes(projection="3d")
    line  = ax.plot3D(X, Y, Z, lw=1, zorder=-1)[0]
    point = ax.scatter(X, Y, Z, color="orange", marker="o")
    
    ax.set_xlim3d(-20, 20)
    ax.set_ylim3d(-30, 30)
    ax.set_zlim3d(0, 50)
    
    ax.set_xlabel(r"$x$", labelpad=10)
    ax.set_ylabel(r"$y$", labelpad=10)
    ax.set_zlabel(r"$z$", labelpad=10)
    
    while plt.fignum_exists(1) and i <= 1e5:
        
        i += 1
        t += dt
        y  = s.explicitRungeKutta4(lorenz, y, t, dt)
        X.append(y[0])
        Y.append(y[1])
        Z.append(y[2])
        
        if i % speed == 0:
            ax.scatter([X[-1]], [Y[-1]], [Z[-1]], color="orange")
            ax.plot3D(X, Y, Z, lw=1, color="cornflowerblue", zorder=-1)
            plt.pause(1 / fps)


view()
# animation(24)


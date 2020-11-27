import numpy as np
import matplotlib.pyplot as plt
import solver


def instableFocus (S, t):
    
    l = 0.5
    w = np.pi
    
    x, y = S
    
    return np.array([l * x + w * y, 
                     -w * x + l * y])


def numInt ():
    
    Ttrans = 0.0
    Teval  = 10.0
    dt     = 1e-2
    y0     = [0.1, 0.0]
    
    s         = solver.ODE_Solver()
    time, sol = s.solveODE(instableFocus, y0, Ttrans, Teval, dt, 1, 
                           "explicitRungeKutta4")
    
    X = sol[0, :]
    Y = sol[1, :]
    
    return (time, X, Y)


def view ():
    
    time, X, Y = numInt()
    
    fig = plt.figure()
    fig.suptitle("normal form of instable focus", fontsize=18)
    fig.set_size_inches(14, 6)
    fig.subplots_adjust(wspace=0.3)
    
    plt.subplot(1, 2, 1)
    plt.plot(time, X, label=r"$x(t)$")
    plt.plot(time, Y, label=r"$y(t)$")
    plt.legend(loc="upper left")
    plt.title("time series", fontsize=16)
    plt.xlabel(r"time $t$")
    plt.grid(color="lightgray")
    
    plt.subplot(1, 2, 2)
    plt.plot(X, Y, color="black", zorder=2)
    plt.plot(X[0], Y[0], color="red", marker="o", markersize=6, zorder=3)
    plt.title("phase space", fontsize=16)
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.grid(color="lightgray")


view()


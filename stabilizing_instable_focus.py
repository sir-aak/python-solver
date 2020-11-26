import numpy as np
import matplotlib.pyplot as plt
import solver


def instableFocusControlled (S, Sd, t):
    
    l = 0.5
    w = np.pi
    K = 0.25
    
    x, y   = S
    xd, yd = Sd
    
    return (np.array([l * x + w * y - K * (x - xd), 
                      -w * x + l * y - K * (y - yd)]))


def numInt ():
    
    Ttrans = 0.0
    Teval  = 150.0
    tau    = 1.0
    dt     = 1e-3
    n_tau  = int(tau / dt)
    
    history     = -1e-2 * np.ones((2, n_tau + 1))
    derivatives = np.zeros((2, n_tau + 1))
    
    s         = solver.DDE_Solver()
    time, sol = s.solveDDE(instableFocusControlled, history, derivatives, 
                           Ttrans, Teval, tau, dt, 1, "explicitRungeKutta4")
    
    X = sol[0, :]
    Y = sol[1, :]
    
    return (time, X, Y)


def view():
    
    time, X, Y = numInt()
    
    fig = plt.figure()
    fig.suptitle("delay controlled instable focus", fontsize=18)
    fig.set_size_inches(15, 7)
    fig.subplots_adjust(wspace=0.3)
    
    plt.subplot(1, 2, 1)
    plt.plot(time, X, label=r"$x(t)$")
    plt.plot(time, Y, label=r"$y(t)$")
    plt.legend(loc="upper left")
    plt.title("time series", fontsize=16)
    plt.xlabel(r"time $t$")
    plt.grid(color="lightgray")
    
    plt.subplot(1, 2, 2)
    plt.plot(X, Y, zorder=2)
    plt.plot(X[0], Y[0], color="orange", marker="o", markersize=6, zorder=3)
    plt.title("phase space", fontsize=16)
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.grid(color="lightgray")


view()


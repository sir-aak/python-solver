import numpy as np
import matplotlib.pyplot as plt
import solver


def linear_dde (y, yd, t):
    
    a = 1.0
    b = -2.0
    
    return (-a * y + b * yd)


def numInt ():
    
    Ttrans = 0.0
    Teval  = 100.0
    tau    = 1.0
    dt     = 1e-3
    n_tau  = int(tau / dt)
    
    history     = 0.1 * np.ones((1, n_tau + 1))
    derivatives = np.zeros((1, n_tau + 1))
    
    s         = solver.DDE_Solver()
    time, sol = s.solveDDE(linear_dde, history, derivatives, 
                           Ttrans, Teval, tau, dt, 1, "explicitRungeKutta4")
    
    Y = sol[0, :]
    
    return (time, Y)


def view ():
    
    time, Y = numInt()

    plt.figure(figsize=(10, 5.6))
    plt.suptitle("linear 1D dde: time series")
    plt.title(r"$f(y(t), y(t-\tau), t) = -ay(t) + by(t - \tau)$", 
                 fontsize=14)
    plt.plot(time, Y)
    plt.xlabel(r"time $t$")
    plt.ylabel(r"$y(t)$")
    plt.grid(color="lightgray")


view()


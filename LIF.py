import numpy as np
import matplotlib.pyplot as plt
import solver


def LIF (u, t):
    
    I = 0.9
    D = 0.075
    
    f_det   = I - u
    f_stoch = D
    
    return (f_det, f_stoch)


def numInt ():
    
    s = solver.SDE_Solver()                                         # instantiate sde-solver object
    
    dt            = 1e-2                                            # numerical time step
    sqrt_dt       = np.sqrt(dt)                                     # square root of time step is needed for stochastic integration
    Ito_dt        = np.sqrt(np.power(dt, 3) / 3.0)
    T             = 100.0                                           # integration time
    numberOfSteps = int(T / dt)
    
    t             = 0.0                                             # time at beginning of computation
    time          = np.zeros(numberOfSteps)                         # initialize time vector
    time[0]       = t
    
    u0            = 0.0                                             # initial condition
    U             = np.zeros((numberOfSteps, np.array(u0).size))    # initialize solution matrix
    U[0]          = u0
    
    u = u0
    
    for i in range(1, numberOfSteps):
        
        t      += dt
        time[i] = t
        
        # membrane potential reset condition
        if u >= 1.0:
            u = 0.0
        
        u = s.explicitStrongOrder15(LIF, u, t, dt, sqrt_dt, Ito_dt)
        
        U[i] = u
    
    return (time, U)


def view ():
    
    time, U = numInt()

    plt.figure(figsize=(8, 6))
    plt.plot(time, U)
    plt.suptitle("leaky integrate-and-fire model: time series")
    plt.xlabel(r"time $t$")
    plt.ylabel(r"membrane potential $u(t)$")
    plt.grid(color="lightgray")


view()


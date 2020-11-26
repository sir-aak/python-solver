import numpy as np
import matplotlib.pyplot as plt
import solver


# right hand side of linear sde model equation dX = aX dt + bX dW
# returns deterministic and stochastic part of right hand side
def linearSDE (X, t):
    
    # model parameters
    a = 5.0
    b = 1.0
    
    f_det   = np.array([a * X])
    f_stoch = np.array([b * X])
    
    return (f_det, f_stoch)


# returns analytic solution to linear sde dX = aX dt + bX dW
def X_analytic (t, Wt):
    
    # model parameters
    a  = 5.0
    b  = 1.0
    
    # initial state
    x0 = 1.0
    
    return (x0 * np.exp((a - np.power(b, 2) / 2.0) * t + b * Wt))


# returns time vector and numerical solution of sde
def numInt (x0, Ttrans, Teval, dt, sqrt_dt, Ito_dt, method):
    
    s       = solver.SDE_Solver()
    time, X = s.solveSDE(linearSDE, x0, Ttrans, Teval, dt, 1, method)
    X       = X[0, :]
    
    return (time, X)


# plots time series for a single realization
def plotSingle ():
    
    # initial state
    x0 = 1.0
    
    # parameters for numerical integration
    Ttrans  = 0.0
    Teval   = 1.0
    dt      = 1e-3
    sqrt_dt = np.sqrt(dt)                     # stdev for Wiener process
    Ito_dt  = np.sqrt(np.power(dt, 3) / 3.0)  # stdev for Itô integral
    method  = "explicitStrongOrder1"
    
    # generate data
    time, X = numInt(x0, Ttrans, Teval, dt, sqrt_dt, Ito_dt, method)
    
    # plot data
    plt.figure(figsize=(8, 6))
    plt.suptitle("time series")
    plt.title(r"linear SDE $dX_t = aX_t\,dt + bX_t\,dW_t$", fontsize=14)
    plt.plot(time, X)
    plt.xlabel(r"time $t$")
    plt.ylabel(r"$X(t)$")
    plt.grid(color="lightgray")
    plt.show()
    

# plots averaged time series and global error with respect to the analytic solution
def plotMulti ():
    
    # initial state
    x0 = 1.0
    
    # parameters for numerical integration
    Ttrans  = 0.0
    Teval   = 1.0
    dt      = 1e-3
    sqrt_dt = np.sqrt(dt)                     # stdev for Wiener process
    Ito_dt  = np.sqrt(np.power(dt, 3) / 3.0)  # stdev for Itô integral
    nSteps  = int((Ttrans + Teval) / dt)
    nSims   = 10
    
    # initialize solution arrays
    Wt      = np.zeros((nSteps, nSims))
    Ana     = np.zeros((nSteps, nSims))
    Euler   = np.zeros((nSteps, nSims))
    RuKu    = np.zeros((nSteps, nSims))
    SO1     = np.zeros((nSteps, nSims))
    SO15    = np.zeros((nSteps, nSims))
    
    # generate gaussian noise for analytic solution through integration 
    # of Wiener increments dW
    for i in range(1, nSteps):
        Wt[i, :] = Wt[i-1, :] + np.random.normal(loc=0.0, scale=sqrt_dt, size=nSims)
    
    # generate data
    for i in range(nSims):
        
        print(i+1, " of ", nSims, " simulations")
        
        # generate data
        time, X_Eu   = numInt(x0, Ttrans, Teval, dt, sqrt_dt, Ito_dt, "explicitEulerMaruyama")
        time, X_RuKu = numInt(x0, Ttrans, Teval, dt, sqrt_dt, Ito_dt, "explicitRungeKutta")
        time, X_SO1  = numInt(x0, Ttrans, Teval, dt, sqrt_dt, Ito_dt, "explicitStrongOrder1")
        time, X_SO15 = numInt(x0, Ttrans, Teval, dt, sqrt_dt, Ito_dt, "explicitStrongOrder15")
        
        Ana[:, i]    = X_analytic(time, Wt[:, i])
        RuKu[:, i]   = X_RuKu
        Euler[:, i]  = X_Eu
        SO1[:, i]    = X_SO1
        SO15[:, i]   = X_SO15
    
    # calculation of averaged time series
    AnaMean   = np.mean(Ana, axis=1)
    EulerMean = np.mean(Euler, axis=1)
    RuKuMean  = np.mean(RuKu, axis=1)
    SO1Mean   = np.mean(SO1, axis=1)
    SO15Mean  = np.mean(SO15, axis=1)
    
    # calculation of global deviation from analytic solution
    errEuler  = np.sqrt(np.sum(np.square(AnaMean - EulerMean)))
    errRuKu   = np.sqrt(np.sum(np.square(AnaMean - RuKuMean)))
    errSO1    = np.sqrt(np.sum(np.square(AnaMean - SO1Mean)))
    errSO15   = np.sqrt(np.sum(np.square(AnaMean - SO15Mean)))
    
    # plot data
    plt.figure(figsize=(12, 9))
    plt.suptitle("averaged time series for " + str(nSims) + " simulations, " + "dt = " + str(dt))
    plt.title(r"linear SDE $dX_t = aX_t\,dt + bX_t\,dW_t$", fontsize=14)    
    
    #~ for i in range(nSims):
        #~ plt.plot(time, Ana[:, i], color="blue", alpha=0.05)
        #~ plt.plot(time, Euler[:, i], color="orange", alpha=0.05)
        #~ plt.plot(time, SO1[:, i], color="green", alpha=0.05)
        #~ plt.plot(time, SO15[:, i], color="red", alpha=0.05)
    
    plt.plot(time, AnaMean, color="darkblue", label=r"analytic: $X_t = X_0\,\exp[(a - 0.5\,b^2)t + b\,W_t]$")
    plt.plot(time, EulerMean, color="darkorange", label="Euler-Maruyama, rms = " + str(errEuler))
    plt.plot(time, RuKuMean, color="purple", label="Runge-Kutta, rms = " + str(errRuKu))
    plt.plot(time, SO1Mean, color="darkgreen", label="strong order 1.0, rms = " + str(errSO1))
    plt.plot(time, SO15Mean, color="darkred", label="strong order 1.5, rms = " + str(errSO15))
    
    plt.xlabel(r"time $t$")
    plt.ylabel(r"$X(t)$")
    plt.legend(loc="upper left")
    plt.grid(color="lightgray")
    
    plt.show()


plotSingle()
plotMulti()


import numpy as np
import matplotlib.pyplot as plt
import solver


# returns right hand side of damped harmonic oscillator model
def dampedLinearOscillator (X, t):
    
    # parameters
    D     = 2e-3        # coefficient for stochastic noise
    gamma = 0.1         # damping coefficient
    w0    = 1.0         # frequency
    
    # extract current position x and velocity v from state vector X
    x, v    = X
    f_det   = np.array([v, -np.power(w0, 2) * x - gamma * v])
    f_stoch = np.array([D, D])
    
    return (f_det, f_stoch)


# performs numerical integration of the damped harmonic oscillator model
# returns solution arrays of time, of position X and of velocity V
def generateData ():
    
    # parameters for numerical integration
    Ttrans = 0.0                # transient time -- will not be evaluated
    Teval  = 100.0              # evaluation time -- will be evaluated
    dt     = 1e-2               # numerical time step
    X0     = [0.1, 0.0]         # initial condition for sde
    
    # numerical integration
    s         = solver.SDE_Solver()
    time, sol = s.solveSDE(dampedLinearOscillator, X0, Ttrans, Teval, dt, 1, 
                           "explicitStrongOrder15")
    
    X = sol[0, :]
    Y = sol[1, :]
    
    return (time, X, Y, dt)


# plots time series and phase space for damped harmonic oscillator
def view ():
    
    # load data
    time, X, V, dt = generateData()
    
    fig = plt.figure()
    fig.set_size_inches(14, 6)
    fig.suptitle("damped linear oscillator", fontsize=14)
    plt.subplots_adjust(wspace=0.35)
    
    ax1 = fig.add_subplot(1, 2, 1)
    plt.title("time series")
    ax1.plot(time, X, label=r"position $x(t)$")
    ax1.plot(time, V, label=r"velocity $v(t)$")
    ax1.legend(loc="upper right")
    plt.xlabel(r"time $t$")
    ax1.grid(color="lightgray")
    
    ax2 = plt.subplot(1, 2, 2)
    ax2.set_axisbelow(True)
    plt.title("phase space")
    ax2.plot(X, V, zorder=5)
    ax2.plot(X[0], V[0], color="orange", marker="o", markersize=6, zorder=6)
    plt.xlabel(r"position $x$")
    plt.ylabel(r"velocity $v$")
    ax2.grid(color="lightgray")


# returns estimated new state / new best estimate
# X - previous state / previous best estimate
# S - covariance matrix of current state
# P - prediction matrix
# U - uncertainty from the environment
def predict (X, S, P, U):
    
    X = P@X
    S = P@(S@P.T) + U
    
    return (X, S)


# returns estimated new state refined with measurement data from sensor
# X  - new best estimate
# S  - covariance matrix of new best estimate
# H  - scaling matrix of sensor
# Z  - sensor data
# Sn - covariance matrix of sensor reading / sensor noise
def correct (X, S, H, Z, Sn):
    
    # Kalman gain
    arg  = H@(S@H.T) + Sn
    K    = S@(H.T)@(np.linalg.inv(arg))
    
    Xnew = X + K@(Z - H@(X))
    Snew = S - K@(H@(S))
    
    return (Xnew, Snew)


# computes estimation of position Xest and of velocity Vest using Kalman filter
# returns estimation data Xest, Vest and sensor data time, X, V and dt
def predictionWithKalmanFilter ():
    
    # load sensor data
    time, X, V, dt = generateData()
    
    # number of time steps
    n  = time.size
    
    # current state's / best estimate's uncertainty
    S  = np.array([[6.25e-6, 0.0], 
                   [0.0, 6.25e-6]])
    
    # prediction rule
    P  = np.array([[1.0, dt], 
                   [0.0, 1.0]])
    
    # uncertainty from the environment
    U  = np.array([[6.25e-6, 6.25e-8], 
                   [6.25e-8, 6.25e-6]])
    
    # scaling matrix
    H  = np.array([[1.0, 0.0], 
                   [0.0, 1.0]])
    
    # sensor data
    Z  = np.column_stack((X, V))
    
    # sensor noise
    Sn = np.array([[6.25e-6, 0.0], 
                   [0.0, 6.25e-6]])
    
    # initialize estimated state matrix
    Zest = np.zeros((n, 2))
    
    # fill estimated state matrix
    for i in range(n):
        Zest[i], S = predict(Z[i], S, P, U)
        Zest[i], S = correct(Zest[i], S, H, Z[i], Sn)
    
    # extract estimated position Xest and velocity Vest from state vector Zest
    Xest = Zest[:, 0]
    Vest = Zest[:, 1]
        
    return (Xest, Vest, time, X, V, dt)


def plotKalmanEstimation ():
    
    # load estimation data Xest, Vest and sensor data time, X, V, dt
    Xest, Vest, time, X, V, dt = predictionWithKalmanFilter()
    
    fig = plt.figure()
    fig.set_size_inches(14, 6)
    fig.suptitle("linear Kalman filter estimation of damped linear oscillator", fontsize=14)
    plt.subplots_adjust(wspace=0.35)
    
    ax1 = fig.add_subplot(1, 2, 1)
    plt.title("time series")
    ax1.plot(time, X, label=r"actual position $x(t)$", color="blue")
    ax1.plot(time, V, label=r"actual velocity $v(t)$", color="orange")
    ax1.plot(time, Xest, label=r"estimated position $x_{est}(t)$", color="red", linestyle=":")
    ax1.plot(time, Vest, label=r"estimated velocity $v_{est}(t)$", color="green", linestyle=":")
    ax1.legend(loc="upper right")
    plt.xlabel(r"time $t$")
    ax1.grid(color="lightgray")
    
    ax2 = plt.subplot(1, 2, 2)
    ax2.set_axisbelow(True)
    plt.title("phase space")
    ax2.plot(X, V, color="blue", label="measurement")
    ax2.plot(X[0], V[0], color="orange", marker="o", markersize=6, zorder=6)
    ax2.plot(Xest, Vest, color="red", linestyle=":", label="estimation")
    ax2.legend(loc="upper right")
    plt.xlabel(r"position $x$")
    plt.ylabel(r"velocity $v$")
    ax2.grid(color="lightgray")
    
    print("X-error: ", np.sqrt(np.sum(np.square(X - Xest))))
    print("V-error: ", np.sqrt(np.sum(np.square(V - Vest))))


view()
plotKalmanEstimation()


import numpy as np

# version 03.10.2019


# solveODE(f, y0, Ttrans, Teval, dt, outSteps, method) solves ODEs for their solution sol
#
# arguments:
# f        : right hand side of ode, function object, returns float or numpy array
# y0       : initial values,
#            for onedimensional odes int or float
#            for n-dimensional odes list or numpy array
# Ttrans   : transient time, float
# Teval    : evaluation time, float
# dt       : integration time step, float
# outSteps : store every outSteps-th step in sol, integer
# method   : method for numerical integration, string


class ODE_Solver:
    
    # constructor
    def __init__ (self):
        pass
    
    
    # numerical method functions perform integration step for given right-hand side f of ode
    
    # performs explicit Euler step
    # convergence order 1
    def explicitEuler (self, f, y, t, dt):
        return (y + f(y, t) * dt)
    
    
    # performs implicit Euler step with fixed point iteration
    # convergence order 1
    def implicitEulerFPI (self, f, y, t, dt, tol = 1e-10):
        
        x      = y
        x_prev = x + 2.0 * tol
        j      = 0
        
        while np.linalg.norm(x - x_prev) >= tol and j < 15:       # raise error
            
            j     += 1
            x_prev = x
            x      = y + f(x, t) * dt
            
        return (x)
    
    
    # performs explicit midpoint step
    # convergence order 2
    def explicitMidpoint (self, f, y, t, dt):
        
        k1 = f(y, t)
        k2 = f(y + k1 * dt / 2.0, t + dt / 2.0)
        
        return (y + k2 * dt)
    
    
    # performs explicit Runge-Kutta step of stage 2
    # convergence order 2
    def explicitHeun (self, f, y, t, dt):
        
        k1 = f(y, t)
        k2 = f(y + k1 * dt, t + dt)
        
        return (y + (k1 + k2) * dt / 2.0)
    
    
    # performs explicit Runge-Kutta step of stage 4
    # convergence order 4
    def explicitRungeKutta4 (self, f, y, t, dt):
    
        k1 = f(y, t)
        k2 = f(y + k1 * dt / 2.0, t + dt / 2.0)
        k3 = f(y + k2 * dt / 2.0, t + dt / 2.0)
        k4 = f(y + k3 * dt, t + dt)
    
        return (y + (k1 + 2.0 * (k2 + k3) + k4) * dt / 6.0)
    
    
    def RungeKutta54_coefficients ():
        
        # stage
        s = 7
        
        c = np.array([0, 1.0 / 5.0, 3.0, 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0, 1.0])
        
        A = np.matrix([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                       [1.0 / 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                       [3.0 / 40.0, 9.0 / 40.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                       [44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0, 0.0, 0.0, 0.0, 0.0], 
                       [19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0, 0.0, 0.0, 0.0], 
                       [9017.0 / 3168.0, -355.0 / 33.0, 46732.0 / 5247.0, 49.0 / 176.0, -5103.0 / 18656.0, 0.0, 0.0], 
                       [35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0, 0.0]])
        
        # niederordrig
        b1 = np.array([5179.0 / 57600.0, 0.0, 7571.0 / 16695.0, 393.0 / 640.0, -92097.0 / 339200.0, 187.0 / 2100.0, 1.0 / 40.0])
        
        # höherordrig
        b2 = np.array([35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0, 0.0])
        
        return (s, A, b1, b2, c)
    
    
#    def RungeKutta4 (self, f, y, t, dt):
    
    
    
    # returns time vector Tvec and solution matrix sol for solved ode
    def solveODE (self, f, y0, Ttrans, Teval, dt = 1e-4, outSteps = 1, 
                  method = "explicitEuler"):
        
        if Ttrans + Teval <= dt:
            raise ValueError("Time step dt should be greater than zero and smaller than integration time.")
        
        numberOfSteps = int((Ttrans + Teval) / dt)
        skipSteps     = int(Ttrans / dt)
        
        # initialize time vector Tvec
        Tvec = np.arange(0.0, Ttrans + Teval, dt)
        
        # time at beginning of computation
        t = Tvec[0]
        
        # initial condition is casted to numpy-array
        y0 = np.array(y0)
        
        # initialize solution matrix sol
        sol = np.zeros((y0.size, numberOfSteps))
        
        # write initial condition into solution matrix
        sol[:, 0] = y0
        
        y = y0
        
        # numerical integration
        
        if method == "RungeKutta54":
            
            raise ValueError("Dormand-Prince methods are not implemented yet.")
#            s, A, b1, b2, c = self.RungeKutta54_coefficients()
#            
#            k1    = np.zeros(s)
#            k1[0] = f(y, t)
#            k2    = k1
#            
#            for i in range(1, k1):
#                k1[i] = f(y + dt * A[i, :i] @ k1[:i], t + c[i] * dt)
#            
#            y1 = y + dt * b1 @ k1
#            
#            for i in range(1, k2):
#                k2[i] = f(y + dt * A[i, :i] @ k2[:i], t + c[i] * dt)
#            
#            y2 = y + dt * b2 @ k2
        
        else:
            
            for i in range(1, numberOfSteps):
                
                if method == "explicitEuler":
                    y = self.explicitEuler(f, y, t, dt)
                
                elif method == "implicitEulerFPI":
                    y = self.implicitEulerFPI(f, y, t, dt, 1e-10)
                
                elif method == "explicitMidpoint":
                    y = self.explicitMidpoint(f, y, t, dt)
                
                elif method == "explicitHeun":
                    y = self.explicitHeun(f, y, t, dt)
                
                elif method == "explicitRungeKutta4":
                    y = self.explicitRungeKutta4(f, y, t, dt)
                
                else:
                    raise ValueError("Choose numerical integration method from {explicitEuler, implicitEulerFPI, explicitMidpoint, explicitHeun, explicitRungeKutta4}")
                
                sol[:, i] = y
                
                t = Tvec[i]
        
        return (Tvec[skipSteps::outSteps], sol[:, skipSteps::outSteps])



# solveDDE(f, history, Ttrans, Teval, tau, dt, outSteps, method) solves ddes for their solution sol
#
# arguments:
# f           : right hand side of dde, function object, returns numpy array
# history     : history array on the interval [-tau, 0], numpy array
# derivatives : array with derivative information on the interval [-tau, 0], numpy array
# Ttrans      : transient time, float
# Teval       : evaluation time, float
# tau         : delay time, float
# dt          : integration time step, float
# outSteps    : store every outSteps-th step in sol, integer
# method      : method for numerical integration, string


class DDE_Solver ():
    
    # constructor
    def __init__ (self):
        pass
    
    
    # numerical method functions perform integration step for given right-hand side f of dde
    
    # performs explicit Euler step
    # convergence order 1
    def explicitEuler (self, f, y, yd, t, dt):
        return (y + dt * f(y, yd, t))
    
    
    # performs explicit Runge-Kutta step of stage 4
    # convergence order 4
    def explicitRungeKutta4 (self, f, y, p0, p1, m0, m1, t, dt):
        
        # p05 is evaluated cubic spline interpolation between p0 and p1
        p05 = 0.5 * (p0 + p1) + 0.125 * (m0 - m1)
        
        k1 = f(y, p0, t)
        k2 = f(y + dt * k1 / 2.0, p05, t + dt / 2.0)
        k3 = f(y + dt * k2 / 2.0, p05, t + dt / 2.0)
        k4 = f(y + dt * k3, p1, t + dt)
        
        return (y + dt * (k1 + 2.0 * (k2 + k3) + k4) / 6.0)
    
    
    
    # returns time vector Tvec and solution matrix sol for solved dde
    def solveDDE (self, f, history, derivatives, Ttrans, Teval, tau, dt = 1e-4, 
                  outSteps = 1, method = "explicitEuler"):
        
        if Ttrans + Teval <= dt:
            raise ValueError("Time step dt should be greater than zero and smaller than integration time.")
        
        if Ttrans + Teval <= tau:
            raise ValueError("Integration time should be greater than delay time tau.")
        
        numberOfSteps = int((Ttrans + Teval) / dt)
        skipSteps     = int(Ttrans / dt)    
        n_tau         = int(tau / dt)
        
        # get system dimension
        n = history.shape[0]
        
        # initialize time vector Tvec
        Tvec = np.zeros(n_tau + numberOfSteps)
        
        # fill Tvec while t is in [-tau, 0]
        for i in range(n_tau):
            Tvec[i] = -tau + i * dt
        
         # time at beginning of computation
        t = Tvec[-1]
        
        # fill Tvec while t is in [0, Ttrans + Teval]
        Tvec[n_tau:] = np.arange(0.0, Ttrans + Teval, dt)
        
        # initialize solution matrix sol and derivatives deriv
        sol   = np.zeros((n, n_tau + numberOfSteps))
        deriv = np.zeros((n, n_tau + numberOfSteps))
        
        # fill sol with history and deriv with derivatives information
        sol[:, :n_tau + 1]   = history
        deriv[:, :n_tau + 1] = derivatives
        
        y = history[:, -1]
        
        # numerical integration
        for i in range(n_tau, numberOfSteps + n_tau):
            
            t  = Tvec[i]
            p0 = sol[:, i - n_tau]
            
            if method == "explicitEuler":
                y = self.explicitEuler(f, y, p0, t, dt)
            
            elif method == "explicitRungeKutta4":
                
                p1 = sol[:, i - n_tau + 1]
                m0 = deriv[:, i - n_tau]
                m1 = deriv[:, i - n_tau + 1]
                
                y = self.explicitRungeKutta4(f, y, p0, p1, m0, m1, t, dt)
                deriv[:, i] = f(y, p1, t)
            
            else:
                raise ValueError("Choose numerical integration method from {explicitEuler, explicitRungeKutta4}")
            
            sol[:, i] = y
        
        return (Tvec[skipSteps::outSteps], sol[:, skipSteps::outSteps])



# solveSDE(f, y0, Ttrans, Teval, dt, outSteps, method) solves SDEs for their solution sol
#
# arguments:
# f        : right hand side of sde, function object, returns float or numpy array
# y0       : initial values, list or numpy array
# Ttrans   : transient time, float
# Teval    : evaluation time, float
# dt       : integration time step, float
# outSteps : store every outSteps-th step in sol, integer
# method   : choose method for numerical integration, string


class SDE_Solver ():
    
    # constructor
    def __init__ (self):
        pass
    
    
    # Wiener process, second argument of normal() is standard deviation
    # n = 1 for scalar noise
    def dW (self, sqrt_dt, n = 1):
        return (np.random.normal(loc=0.0, scale=sqrt_dt, size=n))
    
    
    # Itô integral, second argument of normal() is standard deviation
    # Ito_dt = sqrt(dt³/3), n = 1 for scalar noise
    def dZ (self, Ito_dt, n = 1):
        return (np.random.normal(loc=0.0, scale=Ito_dt, size=n))
    
    
    # numerical method functions perform integration step for given right-hand side f of sde
    
    # performs explicit Euler-Maruyama step
    # strong convergence order 1/2, weak convergence order 1
    def explicitEulerMaruyama (self, f, y, t, dt, sqrt_dt, n = 1):
        
        a, b = f(y, t)
        
        if (n == 1):
            return (y + a * dt + b * self.dW(sqrt_dt))
        
        else:
            return (y + a * dt + b @ self.dW(sqrt_dt, n))
    
    
    # performs explicit sde Runge Kutta step
    # strong convergence order 1, weak convergence order 1
    def explicitRungeKutta (self, f, y, t, dt, sqrt_dt):
        
        dW = self.dW(sqrt_dt)
        
        a, b         = f(y, t)
        y_bar        = y + a * dt + b * sqrt_dt
        a_bar, b_bar = f(y_bar, t)
        
        return (y + a * dt + b * dW 
              + (b_bar - b) * (np.power(dW, 2) - dt) / (2.0 * sqrt_dt))
    
    
    # performs explicit strong order 1 step
    # strong convergence order 1, weak convergence order ???
    def explicitStrongOrder1 (self, f, y, t, dt, sqrt_dt):
        
        dW = self.dW(sqrt_dt)
        
        a, b         = f(y, t)
        y_bar        = y + a * dt + b * dW
        a_bar, b_bar = f(y_bar, t)
        
        return (y + (a_bar + a) * dt / 2.0 + b * dW)
    
    
    # performs explicit strong order 1.5 step
    # strong convergence order 1.5, weak convergence order ???
    def explicitStrongOrder15 (self, f, y, t, dt, sqrt_dt, Ito_dt):
        
        dW = self.dW(sqrt_dt)
        dZ = self.dZ(Ito_dt)
        
        a, b             = f(y, t)
        
        y_p              = y + a * dt + b * sqrt_dt
        y_m              = y + a * dt - b * sqrt_dt
        
        a_p, b_p         = f(y_p, t)
        a_m, b_m         = f(y_m, t)
        
        phi_p            = y_p + b_p * sqrt_dt
        phi_m            = y_p - b_p * sqrt_dt
        
        a_phi_p, b_phi_p = f(phi_p, t)
        a_phi_m, b_phi_m = f(phi_m, t)
        
        return (y + b * dW + (a_p - a_m) * dZ / (2.0 * sqrt_dt) 
              + (a_p + 2.0 * a + a_m) * dt / 4.0 
              + (b_p - b_m) * (np.power(dW, 2) - dt) / (4.0 * sqrt_dt) 
              + (b_p - 2.0 * b + b_m) * (dW * dt - dZ) / (2.0 * dt) 
              + (b_phi_p - b_phi_m - b_p + b_m) 
              * (np.power(dW, 2) / 3.0 - dt) * dW / (4.0 * dt))
    
    
    
    # returns time vector Tvec and solution matrix sol for solved sde
    def solveSDE (self, f, y0, Ttrans, Teval, dt = 1e-3, outSteps = 1, 
                  method = "explicitEulerMaruyama", scalarNoise = True):
        
        if Ttrans + Teval <= dt:
            raise ValueError("Time step dt should be greater than zero and smaller than integration time.")
        
        numberOfSteps = int((Ttrans + Teval) / dt)
        skipSteps     = int(Ttrans / dt)
        
        # square root of time step sqrt_dt and Ito_dt = sqrt(dt³/3) 
        # are needed for stochastic integration
        sqrt_dt = np.sqrt(dt)
        Ito_dt  = np.sqrt(np.power(dt, 3) / 3.0)
        
        # initialize time vector Tvec
        Tvec = np.arange(0.0, Ttrans + Teval, dt)
        
        # time at beginning of computation
        t = Tvec[0]
        
        # initial condition is casted to numpy-array
        y0 = np.array(y0)
        
        # get system dimension
        n = y0.size
        
        # initialize solution matrix sol
        sol = np.zeros((n, numberOfSteps))
        
        # write initial condition into solution matrix
        sol[:, 0] = y0
        
        y = y0
        
        for i in range(1, numberOfSteps):
            
            if (scalarNoise == True):
                
                if method == "explicitEulerMaruyama":
                    y = self.explicitEulerMaruyama(f, y, t, dt, sqrt_dt)
                
                elif method == "explicitRungeKutta":
                    y = self.explicitRungeKutta(f, y, t, dt, sqrt_dt)
                
                elif method == "explicitStrongOrder1":
                    y = self.explicitStrongOrder1(f, y, t, dt, sqrt_dt)
                
                elif method == "explicitStrongOrder15":
                    y = self.explicitStrongOrder15(f, y, t, dt, sqrt_dt, Ito_dt)
                
                else:
                    raise ValueError("Choose numerical integration method from {explicitEulerMaruyama, explicitRungeKutta, explicitStrongOrder1, explicitStrongOrder15}")
            
            else:
                
                if method == "explicitEulerMaruyama":
                    y = self.explicitEulerMaruyama(f, y, t, dt, sqrt_dt, n)
                
                else:
                    raise ValueError("for non-scalar noise, the only implemented method is explicitEulerMaruyama so far")
            
            sol[:, i] = y
            
            t = Tvec[i]
        
        return (Tvec[skipSteps::outSteps], sol[:, skipSteps::outSteps])


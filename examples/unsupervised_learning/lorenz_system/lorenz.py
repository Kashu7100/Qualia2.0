from scipy.integrate import odeint
import numpy as np

def lorenz(u, t):
    x, y, z = u
    sigma = 10.0
    beta = 8.0/3.0
    rho = 28.0
        
    dxdt = sigma*(y-x)
    dydt = x*(rho-z)-y
    dzdt = x*y-beta*z
    return np.array([dxdt,dydt,dzdt])

dt = 0.01   
t = np.arange(0,25, dt)
u0 = np.array([-8.0, 7.0, 27])
u = odeint(lorenz, u0, t)

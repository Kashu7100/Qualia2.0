from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.gridspec as gridspec

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
u0 = np.array([1, 1, 1])
u = odeint(lorenz, u0, t)

def colorline3d(ax, x, y, z, cmap):
    N = len(x)
    skip = int(0.01*N)
    for i in range(0,N,skip):
        ax.plot(x[i:i+skip+1], y[i:i+skip+1], z[i:i+skip+1], color=cmap(int(255*i/N)))
        
def plot3d(u, learned_u):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.axis('off')
    
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=0.95, bottom=0.1, left=0.0, right=0.90, wspace=0.15)
    
    ax = plt.subplot(gs0[:, 0:1], projection='3d')
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    colorline3d(ax, u[:,0], u[:,1], u[:,2], cmap = plt.cm.YlOrBr)
    ax.grid(False)
    ax.set_xlim([-20,20])
    ax.set_ylim([-50,50])
    ax.set_zlim([0,50])
    ax.set_xticks([-20,0,20])
    ax.set_yticks([-40,0,40])
    ax.set_zticks([0,25,50])
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$z$')
    ax.set_title('Exact Dynamics', fontsize = 10)

    ax = plt.subplot(gs0[:, 1:2], projection='3d')
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))    
    colorline3d(ax, learned_u[:,0], learned_u[:,1], learned_u[:,2], cmap = plt.cm.YlOrBr)
    ax.grid(False)
    ax.set_xlim([-20,20])
    ax.set_ylim([-50,50])
    ax.set_zlim([0,50])
    ax.set_xticks([-20,0,20])
    ax.set_yticks([-40,0,40])
    ax.set_zticks([0,25,50])
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$z$')
    ax.set_title('Learned Dynamics', fontsize = 10)

    plt.show()
    
def plot2d(u, learned_u, t):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.axis('off')

    gs0 = gridspec.GridSpec(3, 1)
    gs0.update(top=0.95, bottom=0.15, left=0.1, right=0.95, hspace=0.5)
    
    ax = plt.subplot(gs0[0:1, 0:1])
    ax.plot(t,u[:,0],'r-')
    ax.plot(t,learned_u[:,0],'k--')
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    
    ax = plt.subplot(gs0[1:2, 0:1])
    ax.plot(t,u[:,1],'r-')
    ax.plot(t,learned_u[:,1],'k--')
    ax.set_xlabel('$t$')
    ax.set_ylabel('$y$')
    
    ax = plt.subplot(gs0[2:3, 0:1])
    ax.plot(t,u[:,2],'r-',label='Exact Dynamics')
    ax.plot(t,learned_u[:,2],'k--',label='Learned Dynamics')
    ax.set_xlabel('$t$')
    ax.set_ylabel('$z$')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.4), ncol=2, frameon=False)

    plt.show()

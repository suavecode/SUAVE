## @defgroup Visualization-Performance
# Propeller_Plots.py
#
# Created:  Mar 2021, R. Erhard
# Modified: Feb 2022, R. Erhard
#           Nov 2022, R. Erhard

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
from SUAVE.Core import Units

import matplotlib.patches as patches
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

## @defgroup Visualization-Performance  
def plot_rotor_disc_performance(prop,outputs,i=0,title=None,save_figure=False):
    """
    Inputs
         prop          SUAVE Propeller
         outputs       outputs from spinning propeller
         i             control point to plot results from
         
    """
    # Now plotting:
    psi  = outputs.disc_azimuthal_distribution[i,:,:]
    r    = outputs.disc_radial_distribution[i,:,:]
    psi  = np.append(psi,np.atleast_2d(np.ones_like(psi[:,0])).T*2*np.pi,axis=1)
    r    = np.append(r,np.atleast_2d(r[:,0]).T,axis=1)
    
    T    = outputs.disc_thrust_distribution[i]
    Q    = outputs.disc_torque_distribution[i]
    alf  = (outputs.disc_effective_angle_of_attack[i])/Units.deg
    va   = outputs.disc_axial_induced_velocity[i]
    vt   = outputs.disc_tangential_induced_velocity[i]
        
    
    T    = np.append(T,np.atleast_2d(T[:,0]).T,axis=1)
    Q    = np.append(Q,np.atleast_2d(Q[:,0]).T,axis=1)
    alf  = np.append(alf,np.atleast_2d(alf[:,0]).T,axis=1)
    
    va = np.append(va, np.atleast_2d(va[:,0]).T, axis=1)
    vt = np.append(vt, np.atleast_2d(vt[:,0]).T, axis=1)
    
    lev = 101
    cm  = 'jet'
    
    # plot the grid point velocities
    fig0 = plt.figure(figsize=(4,4))
    ax0  = fig0.add_subplot(111, polar=True)
    p0   = ax0.contourf(psi, r, T,lev,cmap=cm)
    ax0.set_title('Thrust Distribution',pad=15)      
    ax0.set_rorigin(0)
    ax0.set_yticklabels([])
    plt.colorbar(p0, ax=ax0)
    
    fig1 = plt.figure(figsize=(4,4)) 
    ax1  = fig1.add_subplot(111, polar=True)   
    p1   = ax1.contourf(psi, r, Q,lev,cmap=cm) 
    ax1.set_title('Torque Distribution',pad=15) 
    ax1.set_rorigin(0)
    ax1.set_yticklabels([])    
    plt.colorbar(p1, ax=ax1)
    
    fig2 = plt.figure(figsize=(4,4)) 
    ax2  = fig2.add_subplot(111, polar=True)       
    p2   = ax2.contourf(psi, r, alf,lev,cmap=cm) 
    ax2.set_title('Local Blade Angle (deg)',pad=15) 
    ax2.set_rorigin(0)
    ax2.set_yticklabels([])
    plt.colorbar(p2, ax=ax2)

    fig3 = plt.figure(figsize=(4,4)) 
    ax3  = fig3.add_subplot(111, polar=True)       
    p3   = ax3.contourf(psi, r, va,lev,cmap=cm) 
    ax3.set_title('Va',pad=15) 
    ax3.set_rorigin(0)
    ax3.set_yticklabels([])
    plt.colorbar(p3, ax=ax3)    
    
    fig4 = plt.figure(figsize=(4,4)) 
    ax4  = fig4.add_subplot(111, polar=True)       
    p4   = ax4.contourf(psi, r, vt,lev,cmap=cm) 
    ax4.set_title('Vt',pad=15) 
    ax4.set_rorigin(0)
    ax4.set_yticklabels([])
    plt.colorbar(p4, ax=ax4)       
 
 
    return fig0, fig1, fig2, fig3, fig4
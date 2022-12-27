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
def plot_rotor_disc_inflow(prop,velocities, grid_points):
    
    u = velocities.u_velocities
    v = velocities.v_velocities
    w = velocities.w_velocities
    vtot = np.sqrt(u**2 + v**2 + w**2)
    
    # plot the velocities at propeller
    y = grid_points.ymesh
    z = grid_points.zmesh

    R  = prop.tip_radius
    Rh = prop.hub_radius
    psi_360 = np.linspace(0,2*np.pi,40)
    
    vmin = round(np.min([u,v,w,vtot]),3)
    vmax = round(np.max([u,v,w,vtot]),3)
    levels = np.linspace(vmin,vmax, 21)
    
    # plot the grid point velocities
    fig  = plt.figure(figsize=(10,8))
    ax1 = plt.subplot(221)
    ax2 = plt.subplot(222)
    ax3 = plt.subplot(223)
    ax4 = plt.subplot(224)
    
    c1 = ax1.tricontourf(y,z, u, levels=levels, vmax=vmax, vmin=vmin, cmap='seismic')
    plt.colorbar(c1, ax=ax1)#, orientation="horizontal")           
                               
    c2 = ax2.tricontourf(y,z, v, levels=levels, vmax=vmax, vmin=vmin, cmap='seismic')
    plt.colorbar(c2, ax=ax2)#, orientation="horizontal")           
                               
    c3 = ax3.tricontourf(y,z, w, levels=levels, vmax=vmax, vmin=vmin, cmap='seismic')
    plt.colorbar(c3, ax=ax3)#, orientation="horizontal")
    
    c4 = ax4.tricontourf(y,z, vtot, levels=levels, vmax=vmax, vmin=vmin, cmap='seismic')
    plt.colorbar(c4, ax=ax4)#, orientation="horizontal")    
    
    # plot the propeller radius
    ax1.plot(R*np.cos(psi_360), R*np.sin(psi_360), 'k')
    ax2.plot(R*np.cos(psi_360), R*np.sin(psi_360), 'k')
    ax3.plot(R*np.cos(psi_360), R*np.sin(psi_360), 'k')
    ax4.plot(R*np.cos(psi_360), R*np.sin(psi_360), 'k')
    
    # plot the propeller hub
    ax1.plot(Rh*np.cos(psi_360), Rh*np.sin(psi_360), 'k')
    ax2.plot(Rh*np.cos(psi_360), Rh*np.sin(psi_360), 'k')
    ax3.plot(Rh*np.cos(psi_360), Rh*np.sin(psi_360), 'k')
    ax4.plot(Rh*np.cos(psi_360), Rh*np.sin(psi_360), 'k')
    
    # plot rotation direction
    style = "Simple, tail_width=0.5, head_width=4, head_length=8"
    kw    = dict(arrowstyle=style,color="k")
    
    if prop.rotation==1:
        # Rotation direction is ccw
        arrow1 = patches.FancyArrowPatch((-0.8*R,-0.8*R),(0.8*R,-0.8*R), connectionstyle="arc3,rad=0.4", **kw)
        arrow2 = patches.FancyArrowPatch((-0.8*R,-0.8*R),(0.8*R,-0.8*R), connectionstyle="arc3,rad=0.4", **kw)
        arrow3 = patches.FancyArrowPatch((-0.8*R,-0.8*R),(0.8*R,-0.8*R), connectionstyle="arc3,rad=0.4", **kw)
        arrow4 = patches.FancyArrowPatch((-0.8*R,-0.8*R),(0.8*R,-0.8*R), connectionstyle="arc3,rad=0.4", **kw)
    elif prop.rotation==-1:
        # Rotation direction is cw
        arrow1 = patches.FancyArrowPatch((0.8*R,-0.8*R),(-0.8*R,-0.8*R), connectionstyle="arc3,rad=-0.4", **kw)
        arrow2 = patches.FancyArrowPatch((0.8*R,-0.8*R),(-0.8*R,-0.8*R), connectionstyle="arc3,rad=-0.4", **kw)
        arrow3 = patches.FancyArrowPatch((0.8*R,-0.8*R),(-0.8*R,-0.8*R), connectionstyle="arc3,rad=-0.4", **kw)
        arrow4 = patches.FancyArrowPatch((0.8*R,-0.8*R),(-0.8*R,-0.8*R), connectionstyle="arc3,rad=-0.4", **kw) 
    
    ax1.add_patch(arrow1)
    ax2.add_patch(arrow2)
    ax3.add_patch(arrow3)
    ax4.add_patch(arrow4)
    
    ax1.set_aspect('equal', 'box')
    ax2.set_aspect('equal', 'box')
    ax3.set_aspect('equal', 'box')
    ax4.set_aspect('equal', 'box')
    ax1.set_xlabel('y')
    ax1.set_ylabel("z")
    ax2.set_xlabel('y')
    ax2.set_ylabel("z")
    ax3.set_xlabel('y')
    ax3.set_ylabel("z")
    ax4.set_xlabel('y')
    ax4.set_ylabel("z")
    ax1.set_title("Axial Velocity, u")        
    ax2.set_title("Spanwise Velocity, v")
    ax3.set_title("Downwash Velocity, w")
    ax4.set_title("Total Velocity")
    
    fig.suptitle("Induced Velocities at Propeller")
    
    return
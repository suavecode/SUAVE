## @ingroup Plots
# Propeller_Plots.py
#
# Created:  Mar 2021, R. Erhard
# Modified: Feb 2022, R. Erhard

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
from SUAVE.Core import Units

import matplotlib.patches as patches
import pylab as plt
import numpy as np

## @ingroup Plots
def plot_propeller_disc_inflow(prop,velocities, grid_points):
    
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

def plot_propeller_disc_performance(prop,outputs,i=0,title=None):
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
    
    # NORMALIZED PLOTS
    #cmap = matplotlib.cm.jet
    #norm = matplotlib.colors.Normalize()#vmin=0, vmax=1.0)   
    #fig0.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),ax=ax0, orientation='horizontal')

    

    fig1 = plt.figure(figsize=(4,4)) 
    ax1  = fig1.add_subplot(111, polar=True)   
    p1   = ax1.contourf(psi, r, Q,lev,cmap=cm) 
    ax1.set_title('Torque Distribution',pad=15) 
    ax1.set_rorigin(0)
    ax1.set_yticklabels([])    
    plt.colorbar(p1, ax=ax1)
    
    # NORMALIZED PLOTS
    #cmap = matplotlib.cm.jet
    #norm = matplotlib.colors.Normalize()#vmin=0, vmax=0.035) 
    #fig1.colorbar() #matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax1, orientation='horizontal')
    
    
    
    fig2 = plt.figure(figsize=(4,4)) 
    ax2  = fig2.add_subplot(111, polar=True)       
    p2   = ax2.contourf(psi, r, alf,lev,cmap=cm) 
    ax2.set_title('Local Blade Angle (deg)',pad=15) 
    ax2.set_rorigin(0)
    ax2.set_yticklabels([])
    plt.colorbar(p2, ax=ax2)

    # NORMALIZED PLOTS    
    #cmap = matplotlib.cm.jet
    #norm = matplotlib.colors.Normalize()#vmin=-5, vmax=5) 
    #fig2.colorbar() #matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax2, orientation='horizontal')
    
    
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
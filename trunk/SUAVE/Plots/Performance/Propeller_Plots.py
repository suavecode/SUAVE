## @ingroup Plots
# Propeller_Plots.py
#
# Created: Mar 2021, R. Erhard

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
from SUAVE.Core import Units

import matplotlib.patches as patches
import pylab as plt
import numpy as np
import matplotlib

## @ingroup Plots
def plot_propeller_performance(prop,outputs,conditions):
    """This plots local velocities, blade angles, and blade loading
    
    Assumptions:
    None
    
    Source:
    None
    
    Inputs:
    
       
    Outputs: 
    Plots
    
    Properties Used:
    N/A	
    """ 
    # Setting Latex Font style
    font = {'family' : 'normal',
            'weight' : 'normal',
            'size'   :  22}
    
    matplotlib.rc('font', **font)
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'
    matplotlib.rc('lines', lw=3)
    
    # plot results corresponding to the ith control point
    i = 0
    
    # extracting outputs for plotting
    psi         = outputs.disc_azimuthal_distribution[i,:,:]
    r           = outputs.disc_radial_distribution[i,:,:]
    Va_V_disc   = outputs.disc_axial_induced_velocity[i]/outputs.velocity[i][0]
    Vt_V_disc   = outputs.disc_tangential_induced_velocity[i]/outputs.velocity[i][0]
    thrust_disc = outputs.disc_thrust_distribution[i]
    torque_disc = outputs.disc_torque_distribution[i]
    
    delta_alpha = (outputs.disc_effective_angle_of_attack[i] - 
                   conditions.aerodynamics.angle_of_attack[i,0])/Units.deg  
    
    # completing the full revolution for the disc
    psi          = np.append(psi,np.array([np.ones_like(psi[0])*2*np.pi]),axis=0)
    r            = np.append(r,np.array([r[0]]),axis=0)
    Va_V_disc    = np.append(Va_V_disc,np.array([Va_V_disc[0]]),axis=0)
    Vt_V_disc    = np.append(Vt_V_disc,np.array([Vt_V_disc[0]]),axis=0)
    thrust_disc  = np.append(thrust_disc,np.array([thrust_disc[0]]),axis=0)
    torque_disc  = np.append(torque_disc,np.array([torque_disc[0]]),axis=0)
    delta_alpha  = np.append(delta_alpha,np.array([delta_alpha[0]]),axis=0)
    
    # adjusting so that the hub is included in the plot:
    rh = prop.hub_radius
    
    fig0, axis0 = plt.subplots(subplot_kw=dict(projection='polar'))
    CS_0 = axis0.contourf(psi, r, Va_V_disc,20,cmap=plt.cm.jet)
    cbar0 = plt.colorbar(CS_0, ax=axis0, format=matplotlib.ticker.FormatStrFormatter('%.2f'))
    cbar0.ax.set_ylabel('$\dfrac{V_a}{V_\infty}$',rotation=0,labelpad=25)
    axis0.set_title('Axial Velocity of Propeller',pad=15) 
    thetaticks = np.arange(0,360,45)
    axis0.set_thetagrids(thetaticks)   
    axis0.set_rorigin(-rh)
    axis0.set_yticklabels([])
    
    fig0, axis0 = plt.subplots(subplot_kw=dict(projection='polar'))
    CS_0 = axis0.contourf(psi, r, Vt_V_disc,20,cmap=plt.cm.jet)
    cbar0 = plt.colorbar(CS_0, ax=axis0,format=matplotlib.ticker.FormatStrFormatter('%.2f'))
    cbar0.ax.set_ylabel('$\dfrac{V_t}{V_\infty}$',rotation=0,labelpad=25)
    axis0.set_title('Tangential Velocity of Propeller',pad=15)   
    thetaticks = np.arange(0,360,45)
    axis0.set_thetagrids(thetaticks)
    axis0.set_rorigin(-rh)
    axis0.set_yticklabels([])
    
    fig0, axis0 = plt.subplots(subplot_kw=dict(projection='polar'))
    CS_0 = axis0.contourf(psi, r, thrust_disc,20,cmap=plt.cm.jet)
    cbar0 = plt.colorbar(CS_0, ax=axis0, format=matplotlib.ticker.FormatStrFormatter('%.2f'))
    cbar0.ax.set_ylabel('Thrust (N)',labelpad=25)
    axis0.set_title('Thrust Distribution of Propeller',pad=15)  
    axis0.set_rorigin(-rh)
    axis0.set_yticklabels([])
    
    fig0, axis0 = plt.subplots(subplot_kw=dict(projection='polar'))
    CS_0 = axis0.contourf(psi, r, torque_disc,20,cmap=plt.cm.jet)
    cbar0 = plt.colorbar(CS_0, ax=axis0, format=matplotlib.ticker.FormatStrFormatter('%.2f'))
    cbar0.ax.set_ylabel('Torque (Nm)',labelpad=25)
    axis0.set_title('Torque Distribution of Propeller',pad=15) 
    axis0.set_rorigin(-rh)
    axis0.set_yticklabels([])
    
    fig0, axis0 = plt.subplots(subplot_kw=dict(projection='polar'))
    CS_0 = axis0.contourf(psi, r, delta_alpha,10,cmap=plt.cm.jet)
    cbar0 = plt.colorbar(CS_0, ax=axis0, format=matplotlib.ticker.FormatStrFormatter('%.2f'))
    cbar0.ax.set_ylabel('Angle of Attack (deg)',labelpad=25)
    axis0.set_title('Blade Local Effective Angle of Attack',pad=15) 
    axis0.set_rorigin(-rh)
    axis0.set_yticklabels([])  
    
    
    
    
    delta_u = (Va_V_disc**2 + Vt_V_disc**2)**0.5
    Cp = -2*delta_u/np.linalg.norm(outputs.velocity)
    
    fig0, axis0 = plt.subplots(subplot_kw=dict(projection='polar'))
    CS_0 = axis0.contourf(psi, r, Cp,10,cmap=plt.cm.jet)
    cbar0 = plt.colorbar(CS_0, ax=axis0, format=matplotlib.ticker.FormatStrFormatter('%.3f'))
    cbar0.ax.set_ylabel('$C_p$',labelpad=25)
    axis0.set_title('Pressure Distribution',pad=15) 
    axis0.set_rorigin(-rh)
    axis0.set_yticklabels([])      
    
    return    


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
    
    if prop.rotation[0]==1:
        # Rotation direction is ccw
        arrow1 = patches.FancyArrowPatch((-0.8*R,-0.8*R),(0.8*R,-0.8*R), connectionstyle="arc3,rad=0.4", **kw)
        arrow2 = patches.FancyArrowPatch((-0.8*R,-0.8*R),(0.8*R,-0.8*R), connectionstyle="arc3,rad=0.4", **kw)
        arrow3 = patches.FancyArrowPatch((-0.8*R,-0.8*R),(0.8*R,-0.8*R), connectionstyle="arc3,rad=0.4", **kw)
        arrow4 = patches.FancyArrowPatch((-0.8*R,-0.8*R),(0.8*R,-0.8*R), connectionstyle="arc3,rad=0.4", **kw)
    elif prop.rotation[0]==-1:
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
    
    T    = np.append(T,np.atleast_2d(T[:,0]).T,axis=1)
    Q    = np.append(Q,np.atleast_2d(Q[:,0]).T,axis=1)
    alf  = np.append(alf,np.atleast_2d(alf[:,0]).T,axis=1)
    
    rh  = prop.hub_radius
    lev = 21
    cm  = 'jet'
    
    # plot the grid point velocities
    fig = plt.figure(figsize=(8,4))
    ax0 = fig.add_subplot(131, polar=True)
    ax1 = fig.add_subplot(132, polar=True)
    ax2 = fig.add_subplot(133, polar=True)
    

    CS_0 = ax0.contourf(psi, r, T,lev,cmap=cm)
    plt.colorbar(CS_0, ax=ax0, orientation='horizontal')
    ax0.set_title('Thrust Distribution',pad=15)  
    ax0.set_rorigin(0)
    
    CS_1 = ax1.contourf(psi, r, Q,lev,cmap=cm) 
    plt.colorbar(CS_1, ax=ax1, orientation='horizontal')
    ax1.set_title('Torque Distribution',pad=15) 
    ax1.set_rorigin(0)
    
    CS_2 = ax2.contourf(psi, r, alf,lev,cmap=cm) 
    plt.colorbar(CS_2, ax=ax2, orientation='horizontal')
    ax2.set_title('Local Blade Angle (deg)',pad=15) 
    ax2.set_rorigin(0)
    fig.suptitle(title)
    
 
 
    return
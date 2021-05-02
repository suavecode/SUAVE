# Propeller_Plots.py
#
# Created: Mar 2021, R. Erhard
# Modified:

from SUAVE.Core import Units
import pylab as plt
import numpy as np
import matplotlib


def plot_propeller_performance(prop,outputs,conditions):
    # Plots local velocities, blade angles, and blade loading
    
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
    Va_V_disc   = outputs.disc_axial_velocity[i]/outputs.velocity[i][0]
    Vt_V_disc   = outputs.disc_tangential_velocity[i]/outputs.velocity[i][0]
    thrust_disc = outputs.disc_thrust_distribution[i]
    torque_disc = outputs.disc_torque_distribution[i]
    
    delta_alpha = (outputs.disc_local_angle_of_attack[i] - 
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
    axis0.set_title('Blade Local Angle of Attack',pad=15) 
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
    
    vmin = round(np.min([u,v,w]),3)
    vmax = round(np.max([u,v,w]),3)
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
    
    fig.suptitle("Inflow to Downstream Propeller")
    
    return

def plot_propeller_disc_performance(prop,outputs,i=0):
    """
    Inputs
         prop          SUAVE Propeller
         outputs       outputs from spinning propeller
         i             control point to plot results from
         
    """
    # Now plotting:
    psi  = outputs.disc_azimuthal_distribution[i,:,:]
    r    = outputs.disc_radial_distribution[i,:,:]
    psi  = np.append(psi,np.array([np.ones_like(psi[0])*2*np.pi]),axis=0)
    r    = np.append(r,np.array([r[0]]),axis=0)
    
    T    = outputs.disc_thrust_distribution[i]
    Q    = outputs.disc_torque_distribution[i]
    alf = (outputs.disc_local_angle_of_attack[i])/Units.deg
    
    T     = np.append(T,np.array([T[0]]),axis=0)
    Q     = np.append(Q,np.array([Q[0]]),axis=0)
    alf  = np.append(alf,np.array([alf[0]]),axis=0)
    
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
    ax0.set_rorigin(-rh)
    
    CS_1 = ax1.contourf(psi, r, Q,lev,cmap=cm) 
    plt.colorbar(CS_1, ax=ax1, orientation='horizontal')
    ax1.set_title('Torque Distribution',pad=15) 
    ax1.set_rorigin(-rh)
    
    CS_2 = ax2.contourf(psi, r, alf,lev,cmap=cm) 
    plt.colorbar(CS_2, ax=ax2, orientation='horizontal')
    ax2.set_title('Local Blade Angle (deg)',pad=15) 
    ax2.set_rorigin(-rh)
 
    return
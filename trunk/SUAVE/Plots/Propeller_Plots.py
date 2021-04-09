# Propeller_Plots.py
#
# Created: Mar 2021, R. Erhard
# Modified:

from SUAVE.Core import Units
import pylab as plt
import numpy as np
import matplotlib


def plot_propeller_performance(prop,outputs,conditions):
    '''
    
    
    
    
    
    
    
    
    
    
    
    '''
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



## ------------------------------------------------------------------
##   Rotor/Propeller Acoustics
## ------------------------------------------------------------------
#def plot_propeller_noise_contour(conditions, save_figure = False, save_filename = "Noise_Level"):
    #""" 
    #TEXT 
        
    #"""       
    #prop_outputs         = conditions.noise.sources['propeller'].acoustic_outputs  
    #dim_mic              = int(np.sqrt(len(conditions.noise.total_SPL_dBA[0,:])))
    #dim_ctrl_pts         = len(conditions.noise.microphone_locations[:,0,0])
    #levals               = np.linspace(0,130,50) 
    #ctrl_pt_idx          = 0  
    
    #for i in range(dim_ctrl_pts):
        #ctrl_pt_idx += 1  
        #SPL    = conditions.noise.total_SPL_dBA[i,:].reshape(dim_mic,dim_mic)
        #x_vals = conditions.noise.microphone_locations[i,:,0].reshape(dim_mic,dim_mic)
        #y_vals = conditions.noise.microphone_locations[i,:,1].reshape(dim_mic,dim_mic) 
        
        #tag    = save_filename + '_Cp' + str(prop_outputs[i,0])
        #fig    = plt.figure(tag)
        #fig.set_size_inches(8, 8) 
        #axes   = fig.add_subplot(1,1,1)  
        #CS     = axes.contourf(x_vals,y_vals, SPL, cmap = 'jet',levels=levals,extend='both') 
        #CS     = axes.contourf(x_vals,-y_vals, SPL, cmap = 'jet',levels=levals,extend='both') 
        #cbar   = plt.colorbar(CS)
        #cbar.ax.set_ylabel('SPL')        
    
        #if save_figure:
            #plt.savefig(tag + ".png")   

    #return 

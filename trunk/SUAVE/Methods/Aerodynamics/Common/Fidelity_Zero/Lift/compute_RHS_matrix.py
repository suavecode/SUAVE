## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
# compute_RHS_matrix.py
# 
# Created:  Aug 2018, M. Clarke
#           Apr 2020, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# package imports
import numpy as np
import pylab as plt
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.generate_propeller_wake_distribution import generate_propeller_wake_distribution
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.compute_wake_induced_velocity import compute_wake_induced_velocity

## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
def compute_RHS_matrix(n_sw,n_cw,delta,phi,conditions,geometry,sur_flag,wake_model):     
    """ This computes the right hand side matrix for the VLM. In this
    function, induced velocites from propeller wake are also included 
    when relevent and where specified    

    Assumptions:
    Slipstream effect is not a function of time. 
    Axial Variation of slipstream does not include swirl 

    Source:  
    Stone, R. Hugh. "Aerodynamic modeling of the wing-propeller 
    interaction for a tail-sitter unmanned air vehicle." Journal 
    of Aircraft 45.1 (2008): 198-210.

    Inputs:
    geometry
        propulsors                               [Unitless]  
        vehicle vortex distribution              [Unitless] 
    conditions.
        aerodynamics.angle_of_attack             [radians] 
        freestream.velocity                      [m/s]
    n_sw        - number_panels_spanwise         [Unitless]
    n_cw        - number_panels_chordwise        [Unitless]
    sur_flag    - use_surrogate flag             [Unitless]
    slipstream  - include_slipstream_effect flag [Unitless] 
    delta, phi  - flow tangency angles           [radians]
       
    Outputs:                                   
    RHS                                        [Unitless] 

    Properties Used:
    N/A
    """  

    # unpack  
    VD               = geometry.vortex_distribution
    aoa              = conditions.aerodynamics.angle_of_attack 
    aoa_distribution = np.repeat(aoa, VD.n_cp, axis = 1) 
    V_inf            = conditions.freestream.velocity
    V_distribution   = np.repeat(V_inf , VD.n_cp, axis = 1)
    Vx_ind_total     = np.zeros_like(V_distribution)    
    Vz_ind_total     = np.zeros_like(V_distribution)    
    m                = len(aoa) # number of control points      
    
    for propulsor in geometry.propulsors:  
        #-------------------------------------------------------------------------------------------------------
        # PROPELLER SLIPSTREAM MODEL
        #-------------------------------------------------------------------------------------------------------         
        if ((sur_flag == False) and 'propeller' in propulsor.keys()):  
            if wake_model == 'Stream_Tube':
                prop =  propulsor.propeller 
                
                # loop through propellers on aircraft to get combined effect of slipstreams
                num_prop   = len(prop.origin)
                for i in range(num_prop): 
                    vt = np.mean(prop.outputs.tangential_induced_velocity_2d, axis=1)  # induced velocitied averaged around the azimuth
                    va = np.mean(prop.outputs.axial_induced_velocity_2d     , axis=1)  # induced velocitied averaged around the azimuth
                    R0 = prop.hub_radius 
                    
                    # optain propeller and slipstream properties
                    R_p       = prop.tip_radius   
                    vt_old    = np.concatenate((vt , - vt[::-1]), axis=1)  # induced tangential velocity at propeller disc using propeller discretization
                    va_old    = np.concatenate((-va, - va[::-1]), axis=1)  # induced axial velocity at propeller disc  using propeller discretization
                    n_old     = len(prop.chord_distribution)               # spanwise discretization of propeller
                    r_old     = np.linspace(R0,R_p,n_old) 
                    d_old     = np.concatenate((-r_old[::-1], r_old) , axis=0)   
                
                    # compute slipstream development factor from propeller disc to control point on wing
                    s                 = VD.XC - prop.origin[i][0]                     # dimension of vortex distribution
                    Kd                = np.atleast_2d(1 + s/(np.sqrt(s**2 + R_p**2))) # dimension of vortex distribution   
                
                    Vx                = np.tile(np.atleast_2d(prop.outputs.velocity[:,0]).T, (1, n_old)) # dimension of propeller distretization
                    prop_dif          = np.atleast_3d(va[:,1:] +  va[:,:-1])
                    prop_dif          = np.repeat (prop_dif, VD.n_cp, axis = 2)
                    VX                = np.repeat(np.atleast_3d(Vx[:,1:]), VD.n_cp, axis = 2) # dimension (num control points X propeller distribution X vortex distribution )
                    Kv                = (2*VX + prop_dif) /(2*VX + Kd*prop_dif) # dimension (num control points X propeller distribution X vortex distribution )
                
                    r_diff            = np.ones((m,n_old-1))*(r_old[1:]**2 - r_old[:-1]**2 )
                    r_diff            = np.repeat(np.atleast_3d(r_diff), VD.n_cp, axis = 2)
                    r_prime           = np.zeros((m,n_old,VD.n_cp))                
                    r_prime[:,0,:]    = R0
                    
                    for j in range(n_old-1):      
                        r_prime[:,1 + j,:]   = np.sqrt(r_prime[:,j,:]**2 + (r_diff*Kv)[:,j,:]) 
                    
                    r_div_r_prime_val = np.repeat(np.atleast_3d(r_old), VD.n_cp, axis = 2) /r_prime                    
                    r_div_r_prime_old = np.concatenate((r_div_r_prime_val[:,::-1], r_div_r_prime_val), axis=1)    
                
                    # determine if slipstream wake interacts with wing in the z directions
                    locations   = np.where( ((prop.origin[i][2] + r_div_r_prime_old[0,-1,:]*R0) - VD.ZC > 0.0 ) &  ((prop.origin[i][2] - r_div_r_prime_old[0,-1,:]*R0) - VD.ZC < 0.0) \
                                           & ((prop.origin[i][1] + r_div_r_prime_old[0,-1,:]*R0) - VD.YC > 0.0 ) &  (VD.YC - (prop.origin[i][1] - r_div_r_prime_old[0,-1,:]*R0) > 0.0) )        
                
                    if len(locations[0]) > 0:
                        prop_y_discre = prop.origin[i][1] + d_old
                        vt            = np.zeros((m,VD.n_cp))           
                        va            = np.zeros((m,VD.n_cp))
                        r_div_r_prime = np.zeros((m,VD.n_cp)) 
                
                        Y_vals = VD.YC[locations]
                        for k in range(m):   
                            vt[k,locations] = np.interp(Y_vals, prop_y_discre, vt_old[k,:]) # induced tangential velocity at propeller disc using wing discretization
                            va[k,locations] = np.interp(Y_vals, prop_y_discre, va_old[k,:]) # induced axial velocity at propeller disc using wing discretization            
                            for ii in range(len(locations[0])):
                                r_div_r_prime[k,locations[0][ii]] = np.interp(Y_vals[ii], prop_y_discre, r_div_r_prime_old[k,:,locations[0][ii]])   
                
                        # adjust axial and tangential components if propeller is off centered 
                        va_prime       = Kd*va*np.sqrt(1 - (abs(prop.origin[i][2] - VD.ZC)/R_p))
                        vt_prime       = 2*vt*r_div_r_prime*np.sqrt(1 - (abs(prop.origin[i][2] - VD.ZC)/R_p)) 
                
                        # adjust for clockwise/counter clockwise rotation
                        if (prop.rotation != None) and (prop.rotation[i] == -1):
                            for j in range(m): 
                                vt_prime[j,locations] = vt_prime[j,locations][:,::-1]
                                va_prime[j,locations] = va_prime[j,locations][:,::-1]
                                
                        # compute new components of freestream
                        Vx             = V_inf*np.cos(aoa) - va_prime   
                        Vz             = V_inf*np.sin(aoa) - vt_prime   
                        modified_V_inf = np.sqrt(Vx**2 + Vz**2 )                    
                        modified_aoa   = np.arctan(Vz/Vx)          
                
                        # modifiy air speed distribution behind propeller 
                        V_distribution[:,locations]   = modified_V_inf[:,locations] 
                        aoa_distribution[:,locations] = modified_aoa[:,locations]
                        
                RHS = np.sin(aoa_distribution - delta )*np.cos(phi)
                        
                return  RHS 
            
            elif wake_model == 'DVE_Fixed_Wake':        
                # extract the propeller data struction 
                prop = propulsor.propeller 
            
                # generate the geometry of the propeller helical wake
                wake_distribution, ts,B,N = generate_propeller_wake_distribution(prop,m,VD)
            
                # compute the induced velocity
                V_wake_ind = compute_wake_induced_velocity(wake_distribution,VD,m,ts,B,N)
            
                # update the total induced velocity distribution 
                Vx_ind_total = Vx_ind_total + V_wake_ind[:,:,0]
                Vz_ind_total = Vz_ind_total + V_wake_ind[:,:,2]
            
            
                Vx                = V_inf*np.cos(aoa) - Vx_ind_total 
                Vz                = V_inf*np.sin(aoa) - Vz_ind_total 
                V_distribution    = np.sqrt(Vx**2 + Vz**2 )                    
                aoa_distribution  = np.arctan(Vz/Vx)     
            
                RHS = np.sin(aoa_distribution - delta )*np.cos(phi)   
                
                
                fig        = plt.figure('Slipstream_Influence_Velocities')	
                axes_1     = fig.add_subplot(2, 2, 1)   
                axes_2     = fig.add_subplot(2, 2, 2)
                axes_3     = fig.add_subplot(2, 2, 3) 
                x_max      = max(VD.XC)
                y_max      = max(VD.YC)
                axes_1.set_ylim(7.7, 0)
                axes_2.set_ylim(7.7, 0)
                axes_3.set_ylim(7.7, 0) 
                axes_1.set_xlim(-y_max, y_max)            
                axes_2.set_xlim(-y_max, y_max) 
                axes_3.set_xlim(-y_max, y_max)  
                fig.set_size_inches(12, 8)         	 
                for i in range(VD.n_w-1):
                    x_pts = np.reshape(np.atleast_2d(VD.XC[i*(n_sw*n_cw):(i+1)*(n_sw*n_cw)]).T, (n_sw,-1))
                    y_pts = np.reshape(np.atleast_2d(VD.YC[i*(n_sw*n_cw):(i+1)*(n_sw*n_cw)]).T, (n_sw,-1))
                    z_pts_1 = np.reshape(np.atleast_2d(Vx_ind_total[0][i*(n_sw*n_cw):(i+1)*(n_sw*n_cw)]).T, (n_sw,-1))    
                    z_pts_2 = np.reshape(np.atleast_2d(Vz_ind_total[0][i*(n_sw*n_cw):(i+1)*(n_sw*n_cw)]).T, (n_sw,-1)) 
                    z_pts_3 = np.reshape(np.atleast_2d(V_distribution[0][i*(n_sw*n_cw):(i+1)*(n_sw*n_cw)]).T, (n_sw,-1))  
                    CS_1 = axes_1.contourf( y_pts, x_pts,z_pts_1 , cmap = 'jet'  )   
                    CS_2 = axes_2.contourf( y_pts, x_pts,z_pts_2 , cmap = 'jet'  ) 
                    CS_3 = axes_3.contourf( y_pts, x_pts,z_pts_3 , cmap = 'jet'  ) 
                    
                # Set Color bar	
                cbar_1 = fig.colorbar(CS_1, ax=axes_1)
                cbar_2 = fig.colorbar(CS_2, ax=axes_2)
                cbar_3 = fig.colorbar(CS_3, ax=axes_3)
                cbar_1.ax.set_ylabel('$V_{X_{induced}}$', rotation =  0, labelpad= 30 ) 
                cbar_2.ax.set_ylabel('$V_{Z_{induced}}$', rotation =  0, labelpad= 30 )
                cbar_3.ax.set_ylabel('$V_{total}$' , rotation =  0, labelpad= 30 ) 
                axes_1.set_title('$V_{X_{induced}}$')
                axes_2.set_title('$V_{Z_{induced}}$')
                axes_3.set_title('$V_{total}$')
                axes_1.axis('off')	
                axes_2.axis('off')
                axes_3.axis('off')  
                return  RHS 
            
            else:
                pass
             
    RHS = np.sin(aoa_distribution - delta )*np.cos(phi)
    
    return RHS 
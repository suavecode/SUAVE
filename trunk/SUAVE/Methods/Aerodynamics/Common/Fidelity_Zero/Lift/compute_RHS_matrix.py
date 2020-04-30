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

## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
def compute_RHS_matrix(n_sw,n_cw,delta,phi,conditions,geometry,sur_flag,slipstream):     
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
    propulsors       = geometry.propulsors
    VD               = geometry.vortex_distribution
    aoa              = conditions.aerodynamics.angle_of_attack 
    aoa_distribution = np.repeat(aoa, VD.n_cp, axis = 1) 
    V_inf            = conditions.freestream.velocity
    V_distribution   = np.repeat(V_inf , VD.n_cp, axis = 1)
    m                = len(aoa) # number of control points      
    
    #-------------------------------------------------------------------------------------------------------
    # PROPELLER SLIPSTREAM MODEL
    #------------------------------------------------------------------------------------------------------- 
    if ((sur_flag == False) and ('propulsor' in propulsors)) and slipstream:
        prop =  propulsors['propulsor'].propeller 
    
        # loop through propellers on aircraft to get combined effect of slipstreams
        num_prop   = len(prop.origin)
        for i in range(num_prop): 
            vt = prop.outputs.tangential_velocity_distribution
            va = prop.outputs.axial_velocity_distribution
            R0 = prop.hub_radius 
            
            # optain propeller and slipstream properties
            R_p       = prop.tip_radius   
            vt_old    = np.concatenate((vt , - vt[::-1]), axis=1)  # induced tangential velocity at propeller disc using propeller discretization
            va_old    = np.concatenate((-va, - va[::-1]), axis=1)  # induced axial velocity at propeller disc  using propeller discretization
            n_old     = len(prop.chord_distribution)                                                                          # spanwise discretization of propeller
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
            r_prime[:,1:,:]   = np.sqrt(r_prime[:,:-1,:]**2 + r_diff*Kv)    
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
                Vy             = V_inf*np.sin(aoa) - vt_prime   
                modified_V_inf = np.sqrt(Vx**2 + Vy**2 )                    
                modified_aoa   = np.arctan(Vy/Vx)          
    
                # modifiy air speed distribution behind propeller 
                V_distribution[:,locations]   = modified_V_inf[:,locations] 
                aoa_distribution[:,locations] = modified_aoa[:,locations]
    
    RHS = np.sin(aoa_distribution - delta )*np.cos(phi)
    
    return RHS 
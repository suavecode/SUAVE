## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
# compute_PVW_inflow_velocities.py
#
# Created:  Sep 2021, R. Erhard
# Modified: Jan 2022, R. Erhard

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
from SUAVE.Core import Data
from SUAVE.Methods.Propulsion.Rotor_Wake.Fidelity_One.compute_wake_induced_velocity import compute_wake_induced_velocity

# package imports
import numpy as np
from scipy.interpolate import interp1d


def compute_PVW_inflow_velocities( wake, prop, WD ):
    """
    Assumptions:
        None

    Source:
        N/A
    Inputs:
        prop - rotor instance
    Outputs:
        Va   - axial velocity array of shape (ctrl_pts, Nr, Na)        [m/s]
        Vt   - tangential velocity array of shape (ctrl_pts, Nr, Na)   [m/s]
    """
    
    VD                       = prop.vortex_distribution
    omega                    = prop.inputs.omega
    time                     = wake.wake_settings.wake_development_time
    init_timestep_offset     = wake.wake_settings.initial_timestep_offset
    number_of_wake_timesteps = wake.wake_settings.number_of_wake_timesteps

    # use results from prior bemt iteration
    prop_outputs  = prop.outputs
    cpts          = len(prop_outputs.velocity)
    Na            = prop.number_azimuthal_stations
    Nr            = len(prop.chord_distribution)
    r             = prop.radius_distribution
    rot           = prop.rotation


    try:
        props = prop.propellers_in_network
    except:
        props=Data()
        props.propeller = prop

    # compute radial blade section locations based on initial timestep offset
    dt   = time/number_of_wake_timesteps
    t0   = dt*init_timestep_offset

    # set shape of velocitie arrays
    Va = np.zeros((cpts,Nr,Na))
    Vt = np.zeros((cpts,Nr,Na))
    
    for i in range(Na):
        # increment blade angle to new azimuthal position
        blade_angle   = rot*(omega[0]*t0 + i*(2*np.pi/(Na)))  # Positive rotation, positive blade angle

        # update wake geometry
        prop.start_angle = blade_angle

        start_angle = prop.start_angle
        angles = np.linspace(0,2*np.pi,Na+1)[:-1]
        start_angle_idx = np.where(np.isclose(abs(start_angle),angles))[0][0]        
    
        #----------------------------------------------------------------
        #Compute the wake-induced velocities at propeller blade
        #----------------------------------------------------------------
        #set the evaluation points in the vortex distribution: (ncpts, nblades, Nr, Ntsteps)
        r = prop.radius_distribution 
        Yb   = wake.Wake_VD.Yblades_cp[start_angle_idx,0,0,:,0] 
        Zb   = wake.Wake_VD.Zblades_cp[start_angle_idx,0,0,:,0] 
        Xb   = wake.Wake_VD.Xblades_cp[start_angle_idx,0,0,:,0] 
        
        VD.YC = (Yb[1:] + Yb[:-1])/2
        VD.ZC = (Zb[1:] + Zb[:-1])/2
        VD.XC = (Xb[1:] + Xb[:-1])/2
         
        VD.n_cp = np.size(VD.YC)

        # Compute induced velocities at blade from the helical fixed wake
        VD.Wake_collapsed = WD
        
        V_ind   = compute_wake_induced_velocity(WD, VD, cpts, azi_start_idx=i)
        
        # put into body frame
        u       = V_ind[0,:,0]   # velocity in vehicle x-frame
        v       = V_ind[0,:,1]   # velocity in vehicle y-frame
        w       = V_ind[0,:,2]   # velocity in vehicle z-frame
        
        # rotate to prop frame:
        alpha = prop.orientation_euler_angles[1]
        uprop = u*np.cos(alpha) + w*np.sin(alpha)
        vprop = v
        wprop = u*np.sin(alpha) + w*np.cos(alpha)        
        
        # interpolate to get values at rotor radial stations
        r_midpts = (r[1:] + r[:-1])/2
        u_r = interp1d(r_midpts, uprop, fill_value="extrapolate")
        v_r = interp1d(r_midpts, vprop, fill_value="extrapolate")
        w_r = interp1d(r_midpts, wprop, fill_value="extrapolate")
        
        up = u_r(r)
        vp = v_r(r)
        wp = w_r(r)       

        # Update velocities at the disc
        Va[:,:,i]  = up
        Vt[:,:,i]  = -rot*(vp*(np.cos(blade_angle)) + wp*(np.sin(blade_angle)) )
        
    prop.vortex_distribution = VD
    

    return Va, Vt




def compute_PVW_inflow_velocities_VECTORIZED( wake, prop, WD ):
    """
    Assumptions:
        None

    Source:
        N/A
    Inputs:
        prop - rotor instance
    Outputs:
        Va   - axial velocity array of shape (ctrl_pts, Nr, Na)        [m/s]
        Vt   - tangential velocity array of shape (ctrl_pts, Nr, Na)   [m/s]
    """
    
    VD                       = prop.vortex_distribution
    omega                    = prop.inputs.omega
    time                     = wake.wake_settings.wake_development_time
    init_timestep_offset     = wake.wake_settings.initial_timestep_offset
    number_of_wake_timesteps = wake.wake_settings.number_of_wake_timesteps

    # use results from prior bemt iteration
    prop_outputs  = prop.outputs
    cpts          = len(prop_outputs.velocity)
    Na            = prop.number_azimuthal_stations
    Nr            = len(prop.chord_distribution)
    r             = prop.radius_distribution


    try:
        props = prop.propellers_in_network
    except:
        props=Data()
        props.propeller = prop

    # compute radial blade section locations based on initial timestep offset
    dt   = time/number_of_wake_timesteps
    t0   = dt*init_timestep_offset

    # set shape of velocity arrays
    Va = np.zeros((cpts,Nr,Na))
    Vt = np.zeros((cpts,Nr,Na))
    
    prop.wake_skew_angle = WD.wake_skew_angle
    
    # ----------------------------------------------------------------
    # Compute the wake-induced velocities at propeller blade
    # ----------------------------------------------------------------
    # set the evaluation points in the vortex distribution: (ncpts, nblades, Nr, Ntsteps)
    r = prop.radius_distribution 
    Yb   = wake.Wake_VD.Yblades_cp[:,0,0,:,0] 
    Zb   = wake.Wake_VD.Zblades_cp[:,0,0,:,0] 
    Xb   = wake.Wake_VD.Xblades_cp[:,0,0,:,0] 
    

    VD.YC = (Yb[:,1:] + Yb[:,:-1])/2
    VD.ZC = (Zb[:,1:] + Zb[:,:-1])/2
    VD.XC = (Xb[:,1:] + Xb[:,:-1])/2
    

    VD.n_cp = len(VD.YC[0,:])

    # Compute induced velocities at blade from the helical fixed wake
    VD.Wake_collapsed = WD
    
    V_ind   = compute_wake_induced_velocity(WD, VD, cpts)
    
    
    # put into body frame
    u       = -V_ind[:,0,:,0]   # velocity in vehicle x-frame
    v       = V_ind[:,0,:,1]    # velocity in vehicle y-frame
    w       = -V_ind[:,0,:,2]   # velocity in vehicle z-frame
    
    r = prop.radius_distribution 
    r_midpts = (r[1:] + r[:-1])/2
    alpha = prop.orientation_euler_angles[1]
    uprop = u*np.cos(alpha) + w*np.sin(alpha)
    vprop = v
    wprop = u*np.sin(alpha) + w*np.cos(alpha)       
    u_r = interp1d(r_midpts, uprop, fill_value="extrapolate")
    v_r = interp1d(r_midpts, vprop, fill_value="extrapolate")
    w_r = interp1d(r_midpts, wprop, fill_value="extrapolate")
    
    up = u_r(r)
    vp = v_r(r)
    wp = w_r(r)      
    
    for i in range(Na):
        blade_angle   = (omega[0]*t0 + i*(2*np.pi/(Na))) * prop.rotation  # Positive rotation, positive blade angle

        # Update velocities at the disc
        Va[0,:,i]  = up[i,:]
        Vt[0,:,i]  = -prop.rotation*(vp[i,:]*(np.cos(blade_angle)) + wp[i,:]*(np.sin(blade_angle)) )


    return Va, Vt
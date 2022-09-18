## @ingroup Methods-Propulsion-Rotor_Wake-Fidelity_One
# compute_fidelity_one_inflow_velocities.py
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

## @ingroup Methods-Propulsion-Rotor_Wake-Fidelity_One
def compute_fidelity_one_inflow_velocities( wake, prop ):
    """
    Assumptions:
        None

    Source:
        N/A
    Inputs:
        wake - rotor wake
        prop - rotor instance
        WD   - wake vortex distribution
    Outputs:
        Va   - axial velocity, shape (ctrl_pts, Nr, Na); axis 2 in direction of rotation       [m/s]
        Vt   - tangential velocity, shape (ctrl_pts, Nr, Na); axis 2 in direction of rotation    [m/s]
    """
    
    VD                       = wake.vortex_distribution
    omega                    = prop.inputs.omega
    init_timestep_offset     = wake.wake_settings.initial_timestep_offset

    # use results from prior bevw iteration
    prop_outputs  = prop.outputs
    cpts          = len(prop_outputs.velocity)
    Na            = prop.number_azimuthal_stations
    Nr            = len(prop.chord_distribution)
    r             = prop.radius_distribution
    rot           = prop.rotation
    #WD            = wake.vortex_distribution


    try:
        props = prop.propellers_in_network
    except:
        props = Data()
        props.propeller = prop

    # compute radial blade section locations based on initial timestep offset
    azi_step = 2*np.pi/(Na+1)
    dt       = azi_step/omega[0][0]
    t0       = dt*init_timestep_offset

    # set shape of velocitie arrays
    Va = np.zeros((cpts,Nr,Na))
    Vt = np.zeros((cpts,Nr,Na))
    
    for i in range(Na):
        # increment blade angle to new azimuthal position 
        blade_angle   = -rot*(omega[0]*t0 + i*(2*np.pi/(Na)))  # axial view of rotor, negative rotation --> positive blade angle
    
        #----------------------------------------------------------------
        #Compute the wake-induced velocities at propeller blade
        #----------------------------------------------------------------
        #set the evaluation points in the vortex distribution: (ncpts, nblades, Nr, Ntsteps)
        r    = prop.radius_distribution 
        Yb   = wake.vortex_distribution.reshaped_wake.Yblades_cp[i,0,0,:,0]
        Zb   = wake.vortex_distribution.reshaped_wake.Zblades_cp[i,0,0,:,0]
        Xb   = wake.vortex_distribution.reshaped_wake.Xblades_cp[i,0,0,:,0]
        
        VD.YC = (Yb[1:] + Yb[:-1])/2
        VD.ZC = (Zb[1:] + Zb[:-1])/2
        VD.XC = (Xb[1:] + Xb[:-1])/2
         
        VD.n_cp = np.size(VD.YC)

        # Compute induced velocities at blade from the helical fixed wake
        #VD.Wake_collapsed = WD
        
        V_ind   = compute_wake_induced_velocity(VD, VD, cpts, azi_start_idx=i)
        
        # velocities in vehicle frame
        u       = V_ind[:,:,0]   # velocity in vehicle x-frame
        v       = V_ind[:,:,1]    # velocity in vehicle y-frame
        w       = V_ind[:,:,2]    # velocity in vehicle z-frame
        
        # rotate from vehicle to prop frame:
        rot_to_prop = prop.vec_to_prop_body()
        uprop       = u*rot_to_prop[:,0,0][:,None] + w*rot_to_prop[:,0,2][:,None]
        vprop       = v
        wprop       = u*rot_to_prop[:,2,0][:,None] + w*rot_to_prop[:,2,2][:,None]     
        
        # interpolate to get values at rotor radial stations
        r_midpts = (r[1:] + r[:-1])/2
        u_r      = interp1d(r_midpts, uprop, fill_value="extrapolate")
        v_r      = interp1d(r_midpts, vprop, fill_value="extrapolate")
        w_r      = interp1d(r_midpts, wprop, fill_value="extrapolate")
        
        up = u_r(r)
        vp = v_r(r)
        wp = w_r(r)       

        # Update velocities at the disc
        Va[:,:,i]  = up
        Vt[:,:,i]  = -rot*(vp*(np.cos(blade_angle)) - wp*(np.sin(blade_angle)) )  # velocity component in direction of rotation     
    
    #prop.Wake.vortex_distribution = VD
    

    return Va, Vt


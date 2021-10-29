## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
# compute_HFW_inflow_velocities.py
#
# Created:  Sep 2021, R. Erhard
# Modified:    

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
from SUAVE.Core import Data
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.generate_propeller_wake_distribution import generate_propeller_wake_distribution
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.compute_wake_induced_velocity import compute_wake_induced_velocity

# package imports
import numpy as np
from scipy.interpolate import interp1d

def compute_HFW_inflow_velocities( prop ):
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
    
    VD                       = Data()
    omega                    = prop.inputs.omega
    time                     = prop.wake_settings.wake_development_time
    init_timestep_offset     = prop.wake_settings.init_timestep_offset
    number_of_wake_timesteps = prop.wake_settings.number_of_wake_timesteps

    # use results from prior bemt iteration
    prop_outputs  = prop.outputs
    cpts          = len(prop_outputs.velocity)
    Na            = prop.number_azimuthal_stations
    Nr            = len(prop.chord_distribution)
    r             = prop.radius_distribution

    conditions = Data()
    conditions.noise = Data()
    conditions.noise.sources = Data()
    conditions.noise.sources.propellers = Data()
    conditions.noise.sources.propellers.propeller = prop_outputs

    props=Data()
    props.propeller = prop
    identical=False

    # compute radial blade section locations based on initial timestep offset
    dt   = time/number_of_wake_timesteps
    t0   = dt*init_timestep_offset

    # set shape of velocitie arrays
    Va = np.zeros((cpts,Nr,Na))
    Vt = np.zeros((cpts,Nr,Na))
    for i in range(Na):
        # increment blade angle to new azimuthal position
        blade_angle   = (omega[0]*t0 + i*(2*np.pi/(Na))) * prop.rotation  # Positive rotation, positive blade angle

        # update wake geometry
        init_timestep_offset = blade_angle/(omega * dt)

        # generate wake distribution using initial circulation from BEMT
        WD, _, _, _, _  = generate_propeller_wake_distribution(props,identical,cpts,VD,
                                                               init_timestep_offset, time,
                                                               number_of_wake_timesteps,conditions )


        # ----------------------------------------------------------------
        # Compute the wake-induced velocities at propeller blade
        # ----------------------------------------------------------------
        # set the evaluation points in the vortex distribution: (ncpts, nblades, Nr, Ntsteps)
        r = prop.radius_distribution 
        Yb   = prop.Wake_VD.Yblades_cp[0,0,:,0] 
        Zb   = prop.Wake_VD.Zblades_cp[0,0,:,0] 
        Xb   = prop.Wake_VD.Xblades_cp[0,0,:,0] 
        

        VD.YC = (Yb[1:] + Yb[:-1])/2
        VD.ZC = (Zb[1:] + Zb[:-1])/2
        VD.XC = (Xb[1:] + Xb[:-1])/2
         
        
        VD.n_cp = np.size(VD.YC)

        # Compute induced velocities at blade from the helical fixed wake
        VD.Wake_collapsed = WD

        V_ind   = compute_wake_induced_velocity(WD, VD, cpts)
        u       = V_ind[0,:,0]   # velocity in vehicle x-frame
        v       = V_ind[0,:,1]   # velocity in vehicle y-frame
        w       = V_ind[0,:,2]   # velocity in vehicle z-frame
        
        # interpolate to get values at rotor radial stations
        r_midpts = (r[1:] + r[:-1])/2
        u_r = interp1d(r_midpts, u, fill_value="extrapolate")
        v_r = interp1d(r_midpts, v, fill_value="extrapolate")
        w_r = interp1d(r_midpts, w, fill_value="extrapolate")
        
        up = u_r(r)
        vp = v_r(r)
        wp = w_r(r)       

        # Update velocities at the disc
        Va[:,:,i]  = -up
        Vt[:,:,i]  = (-vp*np.cos(blade_angle) + wp*np.sin(blade_angle)) 


    prop.vortex_distribution = VD

    return Va, Vt
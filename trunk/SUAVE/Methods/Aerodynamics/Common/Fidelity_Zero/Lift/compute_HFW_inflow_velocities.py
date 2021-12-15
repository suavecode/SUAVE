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

import copy
from SUAVE.Input_Output.VTK.save_vehicle_vtk import save_vehicle_vtks

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
    if prop.system_vortex_distribution is not None:
        vehicle = copy.deepcopy(prop.vehicle)

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
    conditions.noise.sources.propellers.propeller2 = prop_outputs

    try:
        props = prop.propellers_in_network
    except:
        print("No other rotors appended to this rotor network. Using single rotor wake induced velocities.")
        props=Data()
        props.propeller = prop

    # compute radial blade section locations based on initial timestep offset
    dt   = time/number_of_wake_timesteps
    t0   = dt*init_timestep_offset

    # set shape of velocitie arrays
    Va = np.zeros((cpts,Nr,Na))
    Vt = np.zeros((cpts,Nr,Na))
    print("Compute HFW inflow velocities...")
    for i in range(Na):
        # increment blade angle to new azimuthal position
        blade_angle   = (omega[0]*t0 + i*(2*np.pi/(Na))) * prop.rotation  # Positive rotation, positive blade angle

        # update wake geometry
        init_timestep_offset = blade_angle/(omega * dt)
        
            
        # generate wake distribution using initial circulation from BEMT
        WD, _, _, _, _  = generate_propeller_wake_distribution(props,cpts,VD,
                                                               init_timestep_offset, time,
                                                               number_of_wake_timesteps,conditions )

        prop.wake_skew_angle = WD.wake_skew_angle
    
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
        
        # put into body fram
        u       = -V_ind[0,:,0]   # velocity in vehicle x-frame
        v       = V_ind[0,:,1]   # velocity in vehicle y-frame
        w       = -V_ind[0,:,2]   # velocity in vehicle z-frame
        
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
        Vt[:,:,i]  = -(vp*np.cos(blade_angle) + wp*np.sin(blade_angle)) 



        #====================================================================================
        #======DEBUG: STORE VTKS AFTER NEW WAKE GENERATION===================================
        #====================================================================================       
        debug = False
        if debug:
            print("\nStoring VTKs...")
            vehicle = prop.vehicle
            Results = Data()
            Results.all_prop_outputs = Data()
            Results.all_prop_outputs.propeller = Data()
            Results.identical = True
            
            conditions=None
            save_vehicle_vtks(vehicle, conditions, Results, time_step=i,save_loc="/Users/rerha/Desktop/Test_SBS_VTKs/A60/")         
    

        #====================================================================================   
        #====================================================================================                     


        prop.vortex_distribution = VD

    return Va, Vt
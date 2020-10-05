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
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.generate_propeller_wake_distribution import generate_propeller_wake_distribution
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.compute_wake_induced_velocity import compute_wake_induced_velocity

## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift 
def compute_RHS_matrix(n_sw,n_cw,delta,phi,conditions,geometry,propeller_wake_model,initial_timestep_offset,wake_development_time):     
    """ This computes the right hand side matrix for the VLM. In this
    function, induced velocites from propeller wake are also included 
    when relevent and where specified     

    Source:  
    None

    Inputs:
    geometry
        propulsors                               [Unitless]  
        vehicle vortex distribution              [Unitless] 
    conditions.
        aerodynamics.angle_of_attack             [radians] 
        freestream.velocity                      [m/s]
    n_sw        - number_spanwise_vortices       [Unitless]
    n_cw        - number_chordwise_vortices      [Unitless]
    sur_flag    - use_surrogate flag             [Unitless]
    slipstream  - propeller_wake_model flag      [Unitless] 
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
    dt               = 0 
    Vz_ind_total     = np.zeros_like(V_distribution)    
    num_ctrl_pts     = len(aoa) # number of control points      
    
    for propulsor in geometry.propulsors:  
        #-------------------------------------------------------------------------------------------------------
        # PROPELLER SLIPSTREAM MODEL
        #-------------------------------------------------------------------------------------------------------         
        if propeller_wake_model and (('propeller' in propulsor.keys()) or ('rotor' in propulsor.keys())):   
        
            # extract the propeller data struction 
            try:
                prop = propulsor.propeller 
            except:
                prop = propulsor.rotor 
            
            # generate the geometry of the propeller helical wake
            wake_distribution, dt,time_steps,num_blades, num_radial_stations = generate_propeller_wake_distribution(prop,propulsor.thrust_angle,num_ctrl_pts,\
                                                                                                                    VD,initial_timestep_offset,wake_development_time)
            
            # compute the induced velocity
            V_wake_ind = compute_wake_induced_velocity(wake_distribution,VD,num_ctrl_pts)
            
            # update the total induced velocity distribution 
            Vx_ind_total = Vx_ind_total + V_wake_ind[:,:,0]
            Vz_ind_total = Vz_ind_total + V_wake_ind[:,:,2] 
            
            Vx                = V_inf*np.cos(aoa) - Vx_ind_total 
            Vz                = V_inf*np.sin(aoa) - Vz_ind_total 
            V_distribution    = np.sqrt(Vx**2 + Vz**2 )                    
            aoa_distribution  = np.arctan(Vz/Vx)
            
            RHS = np.sin(aoa_distribution - delta )*np.cos(phi)   
            
            return  RHS ,Vx_ind_total , Vz_ind_total , V_distribution , dt 
         
    RHS = np.sin(aoa_distribution - delta )*np.cos(phi)
    
    return RHS ,Vx_ind_total , Vz_ind_total , V_distribution , dt 
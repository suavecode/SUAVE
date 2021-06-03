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
from SUAVE.Core import Data
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.generate_propeller_wake_distribution import generate_propeller_wake_distribution
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.compute_wake_induced_velocity import compute_wake_induced_velocity

## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift 
def compute_RHS_matrix(n_sw,n_cw,delta,phi,conditions,geometry,propeller_wake_model,initial_timestep_offset,wake_development_time,number_of_wake_timesteps):     
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
    PSI              = conditions.aerodynamics.sideslip_angle
    PSI_distribution = np.repeat(PSI, VD.n_cp, axis = 1)
    V_inf            = conditions.freestream.velocity
    V_distribution   = np.repeat(V_inf , VD.n_cp, axis = 1)
    
    rot_V_wake_ind   = np.zeros((len(aoa), VD.n_cp,3))
    prop_V_wake_ind  = np.zeros((len(aoa), VD.n_cp,3))
    Vx_ind_total     = np.zeros_like(V_distribution)
    dt               = 0 
    Vz_ind_total     = np.zeros_like(V_distribution)    
    num_ctrl_pts     = len(aoa) # number of control points      
    
    for propulsor in geometry.propulsors:
            if propeller_wake_model:
                if 'propeller' in propulsor.keys():
                    # extract the propeller data structure
                    prop = propulsor.propeller

                    # generate the geometry of the propeller helical wake
                    wake_distribution, dt,time_steps,num_blades, num_radial_stations = generate_propeller_wake_distribution(prop,num_ctrl_pts,\
                                                                                                                            VD,initial_timestep_offset,wake_development_time,\
                                                                                                                            number_of_wake_timesteps)

                    # compute the induced velocity
                    prop_V_wake_ind = compute_wake_induced_velocity(wake_distribution,VD,num_ctrl_pts)

                if 'rotor' in propulsor.keys():

                    # extract the propeller data structure
                    rot = propulsor.rotor

                    # generate the geometry of the propeller helical wake
                    wake_distribution, dt,time_steps,num_blades, num_radial_stations = generate_propeller_wake_distribution(rot,num_ctrl_pts,\
                                                                                                                            VD,initial_timestep_offset,wake_development_time,\
                                                                                                                            number_of_wake_timesteps)

                    # compute the induced velocity
                    rot_V_wake_ind = compute_wake_induced_velocity(wake_distribution,VD,num_ctrl_pts)


                # update the total induced velocity distribution
                Vx_ind_total = Vx_ind_total + prop_V_wake_ind[:,:,0] + rot_V_wake_ind[:,:,0]
                Vz_ind_total = Vz_ind_total + prop_V_wake_ind[:,:,2] + rot_V_wake_ind[:,:,2]

                Vx                = V_inf*np.cos(aoa)*np.cos(PSI) - Vx_ind_total
                Vz                = V_inf*np.sin(aoa) - Vz_ind_total
                V_distribution    = np.sqrt(Vx**2 + Vz**2 )
                aoa_distribution  = np.arctan(Vz/Vx)

                rhs = build_RHS(VD, conditions, n_sw, n_cw, aoa_distribution, delta, phi, PSI_distribution,
                                Vx_ind_total, Vz_ind_total, V_distribution, dt)
                return  rhs
    
    rhs = build_RHS(VD, conditions, n_sw, n_cw, aoa_distribution, delta, phi, PSI_distribution,
                    Vx_ind_total, Vz_ind_total, V_distribution, dt)    
    return rhs



def build_RHS(VD, conditions, n_sw, n_cw, aoa_distribution, delta, phi, PSI_distribution,
              Vx_ind_total, Vz_ind_total, V_distribution, dt):
    #VORLAX subroutine = BOUNDY
    
    #unpack conditions 
    ALFA   = aoa_distribution
    PSIRAD = PSI_distribution
    PITCHQ = conditions.stability.dynamic.pitch_rate              
    ROLLQ  = conditions.stability.dynamic.roll_rate             
    YAWQ   = conditions.stability.dynamic.yaw_rate 
    VINF   = conditions.freestream.velocity

    SINALF = np.sin(ALFA)
    COSIN  = np.cos(ALFA) * np.sin(PSIRAD)
    COSCOS = np.cos(ALFA) * np.cos(PSIRAD)
    PITCH = PITCHQ /VINF
    ROLL = ROLLQ /VINF
    YAW = YAWQ /VINF     
    
    #unpack/rename variables from VD
    X      = VD.XCH
    YY     = VD.YCH
    ZZ     = VD.ZCH 
    XBAR   = VD.XBAR
    ZBAR   = VD.ZBAR

    CHORD  = VD.CHORD[0,:]
    DELTAX = 0.5/VD.RNMAX 
    
    # LOCATE VORTEX LATTICE CONTROL POINT WITH RESPECT TO THE
    # ROTATION CENTER (XBAR, 0, ZBAR). THE RELATIVE COORDINATES
    # ARE XGIRO, YGIRO, AND ZGIRO.
    XGIRO = X + CHORD*DELTAX - np.repeat(XBAR, n_cw)
    YGIRO = YY 
    ZGIRO = ZZ - np.repeat(ZBAR, n_cw)  
    
    # VX, VY, VZ ARE THE FLOW ONSET VELOCITY COMPONENTS AT THE LEADING
    # EDGE (STRIP MIDPOINT). VX, VY, VZ AND THE ROTATION RATES ARE
    # REFERENCED TO THE FREE STREAM VELOCITY.    
    VX = (COSCOS - PITCH*ZGIRO + YAW  *YGIRO)
    VY = (COSIN  - YAW  *XGIRO + ROLL *ZGIRO)
    VZ = (SINALF - ROLL *YGIRO + PITCH*XGIRO)

    # CCNTL AND SCNTL ARE DIRECTION COSINE PARAMETERS OF TANGENT TO
    # CAMBERLINE AT LEADING EDGE.
    # SLE is slope at leading edge only     
    SLE = np.repeat(VD.SLE, n_cw)
    CCNTL = 1. / np.sqrt(1.0 + SLE**2) 
    SCNTL = SLE *CCNTL
    COD = np.cos(phi)
    SID = np.sin(phi)

    # COMPUTE ONSET FLOW COMPONENT ALONG THE OUTWARD NORMAL TO
    # THE SURFACE AT THE CONTROL POINT, ALOC.
    ALOC = VX *SCNTL + VY *CCNTL *SID - VZ *CCNTL *COD    
    
    # COMPUTE VELOCITY COMPONENT ALONG X-AXIS INDUCED BY THE RIGID
    # BODY ROTATION, ONSET.    
    ONSET = - PITCH *ZGIRO + YAW *YGIRO
    
    #pack RHS
    rhs = Data()
    rhs.RHS            = ALOC
    rhs.ONSET          = ONSET
    rhs.Vx_ind_total   = Vx_ind_total
    rhs.Vz_ind_total   = Vz_ind_total
    rhs.V_distribution = V_distribution
    rhs.dt             = dt
    
    #these values will be used later to calculate EFFINC
    rhs.YGIRO  = YGIRO
    rhs.ZGIRO  = ZGIRO
    rhs.VX     = VX   
    rhs.SCNTL  = SCNTL
    rhs.CCNTL  = CCNTL
    rhs.COD    = COD  
    rhs.SID    = SID  
    
    return rhs
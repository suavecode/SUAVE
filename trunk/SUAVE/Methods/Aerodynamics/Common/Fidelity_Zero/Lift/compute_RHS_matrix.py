## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
# compute_RHS_matrix.py
#
# Created:  Aug 2018, M. Clarke
# Modified: Apr 2020, M. Clarke
#           Jun 2021, R. Erhard
#           Jul 2021, A. Blaufox
#           Jul 2021, E. Botero
#           Jul 2021, R. Erhard
#           Feb 2022, R. Erhard
#           May 2022, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# package imports
#import numpy as np
import jax.numpy as jnp
from jax import lax, jit
from SUAVE.Core import Data, to_jnumpy

## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
#@jit
def compute_RHS_matrix(delta,phi,conditions,settings,geometry,propeller_wake_model):

    """ This computes the right hand side matrix for the VLM. In this
    function, induced velocites from propeller wake are also included
    when relevent and where specified

    Source:
    1. Low-Speed Aerodynamics, Second Edition by Joseph Katz, Allen Plotkin
    Pgs. 331-338

    2. VORLAX Source Code

    Inputs:
    geometry
        networks                                 [Unitless]
        vehicle vortex distribution              [Unitless]

    conditions.aerodynamics.angle_of_attack      [radians]
    conditions.aerodynamics.side_slip_angle      [radians]
    conditions.freestream.velocity               [m/s]
    conditions.stability.dynamic.pitch_rate      [radians/s]
    conditions.stability.dynamic.roll_rate       [radians/s]
    conditions.stability.dynamic.yaw_rate        [radians/s]

    sur_flag    - use_surrogate flag             [Unitless]
    slipstream  - propeller_wake_model flag      [Unitless]
    delta, phi  - flow tangency angles           [radians]

    Outputs:
    rhs.
        RHS
        ONSET
        Vx_ind_total
        Vz_ind_total
        V_distribution
        dt
        YGIRO
        ZGIRO
        VX
        SCNTL
        CCNTL
        COD
        SID

    Properties Used:
    N/A
    """

    # unpack
    VD               = geometry.vortex_distribution
    n_cp             = VD.XAH.shape[0]

    aoa              = conditions.aerodynamics.angle_of_attack
    aoa_distribution = jnp.repeat(aoa, n_cp, axis = 1)
    PSI              = conditions.aerodynamics.side_slip_angle
    PSI_distribution = jnp.repeat(PSI, n_cp, axis = 1)
    V_inf            = conditions.freestream.velocity
    V_distribution   = jnp.repeat(V_inf, n_cp, axis = 1)

    Vx_ind_total     = jnp.zeros_like(V_distribution)
    Vy_ind_total     = jnp.zeros_like(V_distribution)
    Vz_ind_total     = jnp.zeros_like(V_distribution)

    dt               = 0
    num_ctrl_pts     = len(aoa) # number of control points
    num_eval_pts     = len(VD.XC)

    r_p_V_wake_ind   = jnp.zeros((num_ctrl_pts,num_eval_pts,3))
    
    # If we have a propeller wake attached
    #wake_model_false = lambda _: r_p_V_wake_ind 
    #cond_inputs      = to_jnumpy((geometry,num_ctrl_pts, r_p_V_wake_ind))
    #r_p_V_wake_ind   = lax.cond(propeller_wake_model,wake_model_true,wake_model_false,cond_inputs)
                
    # update the total induced velocity distribution
    Vx_ind_total = Vx_ind_total + r_p_V_wake_ind[:,:,0]
    Vy_ind_total = Vy_ind_total + r_p_V_wake_ind[:,:,1]
    Vz_ind_total = Vz_ind_total + r_p_V_wake_ind[:,:,2]


    rhs = build_RHS(VD, conditions, settings, aoa_distribution, delta, phi, PSI_distribution,
                    Vx_ind_total, Vy_ind_total, Vz_ind_total, V_distribution, dt)
    
    return rhs



def build_RHS(VD, conditions, settings, aoa_distribution, delta, phi, PSI_distribution,
              Vx_ind_total, Vy_ind_total, Vz_ind_total, V_distribution, dt):
    """ This uses freestream conditions, induced wake velocities, and rotation
    rates (pitch, roll, yaw) to find the boundary conditions (RHS) needed to
    compute gamma. The function defaults to a classical boundary condition
    equation, RHS = V dot N, where V is the unit velocity vector only the panel
    and N is the panel unit normal. However, the user may also define
    settings.use_VORLAX_matrix_calculation to use the boundary condition equation
    from VORLAX, the code on which this VLM is based. This is useful for future
    developers who need to compare numbers 1:1 with VORLAX.

    Note if using VORLAX boundary conditions:
    VORLAX does not take induced wake velocity into account. Additionally,
    VORLAX uses the camber, twist, and dihedral values of a strip's leading
    edge panel for every panel in that strip when calculating panel normals.

    Source:
    1. Low-Speed Aerodynamics, Second Edition by Joseph Katz, Allen Plotkin Pgs. 331-338

    2. VORLAX Source Code

    Inputs:
    settings.use_VORLAX_matrix_calculation  - RHS equation switch               [boolean]
    conditions.stability.dynamic.pitch_rate -                                   [radians/s]
    conditions.stability.dynamic.roll_rate  -                                   [radians/s]
    conditions.stability.dynamic.yaw_rate   -                                   [radians/s]
    aoa_distribution                        - angle of attack                   [radians]
    PSI_distribution                        - sideslip angle                    [radians]
    V[]_ind_total                           - component induced wake velocities [m/s]
    V_distribution                          - freestream velocity magnitude     [m/s]
    delta, phi                              - flow tangency angles              [radians]

    Outputs:
    rhs  -  a Data object used to hold several values including RHS

    Properties Used:
    N/A
    """
    LE_ind       = VD.leading_edge_indices
    RNMAX        = VD.panels_per_strip
    panel_strips = VD.stripwise_panels_per_strip


    # VORLAX frame RHS calculation---------------------------------------------------------
    #VORLAX subroutine = BOUNDY
    #unpack conditions
    ALFA   = aoa_distribution
    PSIRAD = PSI_distribution
    PITCHQ = conditions.stability.dynamic.pitch_rate
    ROLLQ  = conditions.stability.dynamic.roll_rate
    YAWQ   = conditions.stability.dynamic.yaw_rate
    VINF   = conditions.freestream.velocity

    SINALF = jnp.sin(ALFA)
    COSIN  = jnp.cos(ALFA) * jnp.sin(PSIRAD)
    COSCOS = jnp.cos(ALFA) * jnp.cos(PSIRAD)
    PITCH  = PITCHQ / VINF
    ROLL   = ROLLQ  / VINF
    YAW    = YAWQ   / VINF

    #unpack/rename variables from VD
    X      = VD.XCH
    YY     = VD.YCH
    ZZ     = VD.ZCH
    XBAR   = VD.XBAR
    ZBAR   = VD.ZBAR

    CHORD  = VD.chord_lengths[0,:]
    DELTAX = 0.5/RNMAX

    # LOCATE VORTEX LATTICE CONTROL POINT WITH RESPECT TO THE
    # ROTATION CENTER (XBAR, 0, ZBAR). THE RELATIVE COORDINATES
    # ARE XGIRO, YGIRO, AND ZGIRO.
    XGIRO = X + CHORD*DELTAX - jnp.repeat(XBAR, panel_strips)
    YGIRO = YY
    ZGIRO = ZZ - jnp.repeat(ZBAR, panel_strips)

    # VX, VY, VZ ARE THE FLOW ONSET VELOCITY COMPONENTS AT THE LEADING
    # EDGE (STRIP MIDPOINT). VX, VY, VZ AND THE ROTATION RATES ARE
    # REFERENCED TO THE FREE STREAM VELOCITY.
    VX = (COSCOS - PITCH*ZGIRO + YAW  *YGIRO)
    VY = (COSIN  - YAW  *XGIRO + ROLL *ZGIRO)
    VZ = (SINALF - ROLL *YGIRO + PITCH*XGIRO)
    
    #COMPUTE DIRECTION COSINES.
    SCNTL  = VD.SLOPE/jnp.sqrt(1. + VD.SLOPE **2)
    CCNTL  = 1. / jnp.sqrt(1.0 + SCNTL**2)
    phi_LE = phi# jnp.repeat(phi[:,LE_ind]  , panel_strips, axis=1)
    COD    = jnp.cos(phi_LE)
    SID    = jnp.sin(phi_LE)

    # COMPUTE ONSET FLOW COMPONENT ALONG THE OUTWARD NORMAL TO
    # THE SURFACE AT THE CONTROL POINT, ALOC.
    ALOC  = VX *SCNTL + VY *CCNTL *SID - VZ *CCNTL *COD

    # COMPUTE VELOCITY COMPONENT ALONG X-AXIS INDUCED BY THE RIGID
    # BODY ROTATION, ONSET.
    ONSET = - PITCH *ZGIRO + YAW *YGIRO

    # Body-Frame RHS calculation----------------------------------------------------------
    # Add wake and rotation effects to the freestream
    Vx_rotation       = -PITCHQ*ZGIRO + YAWQ  *YGIRO
    Vy_rotation       = -YAWQ  *XGIRO + ROLLQ *ZGIRO
    Vz_rotation       = -ROLLQ *YGIRO + PITCHQ*XGIRO

    Vx                = V_distribution*jnp.cos(aoa_distribution)*jnp.cos(PSI_distribution) + Vx_rotation + Vx_ind_total
    Vy                = V_distribution*jnp.cos(aoa_distribution)*jnp.sin(PSI_distribution) + Vy_rotation + Vy_ind_total
    Vz                = V_distribution*jnp.sin(aoa_distribution)                          + Vz_rotation + Vz_ind_total    
    
    aoa_distribution  = jnp.arctan(Vz/ jnp.sqrt(Vx**2 + Vy**2) )
    PSI_distribution  = jnp.arctan(Vy / Vx)

    # compute RHS: dot(v, panel_normals)
    V_unit_vector    = (jnp.array([Vx,Vy,Vz])/V_distribution).T
    panel_normals    = VD.normals[:,jnp.newaxis,:]
    RHS_from_normals = jnp.sum(V_unit_vector*panel_normals, axis=2).T    

    #pack values--------------------------------------------------------------------------
    use_VORLAX_RHS = settings.use_VORLAX_matrix_calculation

    rhs = Data()
    rhs.RHS            = RHS_from_normals if not use_VORLAX_RHS else ALOC
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

def wake_model_true(cond_inputs):

    # Unpacks
    geometry, num_ctrl_pts, r_p_V_wake_ind = cond_inputs[0], cond_inputs[1], cond_inputs[2]

    nets             = list(geometry.networks.keys())
    n_nets           = len(nets)    

    for ii in range(0,n_nets):
        network = geometry.networks[nets[ii]]

        # include the propeller and rotor wake effect on the wing
        rotor_props = Data()

        try: # Add in the propellers
            rotor_props.update(network.propellers)
        except:
            pass

        try:  # Add in the rotors
            rotor_props.update(network.lift_rotors)
        except:
            pass

        # Loop over all the rotors and props
        for r_p in rotor_props:
            r_p_V_wake_ind += r_p.Wake.evaluate_slipstream(r_p,geometry,num_ctrl_pts)

    return r_p_V_wake_ind

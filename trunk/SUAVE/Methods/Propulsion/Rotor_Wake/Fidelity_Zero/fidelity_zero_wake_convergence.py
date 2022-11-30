## @defgroup Methods-Propulsion-Rotor_Wake-Fidelity_Zero
# fidelity_zero_wake_convergence.py
#
# Created:  Feb 2022, R. Erhard
# Modified: 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.BET_calculations import compute_airfoil_aerodynamics,compute_inflow_and_tip_loss
import jax.numpy as jnp
from jax import jacobian, jit
from jax.lax import while_loop
from SUAVE.Methods.Propulsion.Rotor_Wake.Common import simple_newton

# ----------------------------------------------------------------------
# Wake Convergence
# ----------------------------------------------------------------------

## @defgroup Methods-Propulsion-Rotor_Wake-Fidelity_Zero
@jit
def fidelity_zero_wake_convergence(wake,rotor,wake_inputs):
    """
    Wake evaluation is performed using a simplified vortex wake method for Fidelity Zero, 
    following Helmholtz vortex theory.
    
    Assumptions:
    None

    Source:
    Drela, M. "Qprop Formulation", MIT AeroAstro, June 2006
    http://web.mit.edu/drela/Public/web/qprop/qprop_theory.pdf

    Inputs:
       self         - rotor wake
       rotor        - SUAVE rotor
       wake_inputs.
          Ua        - Axial velocity
          Ut        - Tangential velocity
          r         - radius distribution
       
       
    Outputs:
       va  - axially-induced velocity from rotor wake
       vt  - tangentially-induced velocity from rotor wake
    
    Properties Used:
    None
    
    """        
    
    # Setup
    PSI   = jnp.ones_like(wake_inputs.velocity_total).flatten()
    limit = wake.maximum_convergence_iteration
    
    # Solve!       
    PSI_final, ii = simple_newton(iteration,jacobian_iteration,PSI,while_loop,args=(wake_inputs,rotor),limit=limit)

    # Calculate the velocities given PSI
    va, vt = va_vt(PSI_final, wake_inputs, rotor)

    return va, vt


## @defgroup Methods-Propulsion-Rotor_Wake-Fidelity_Zero
def iteration(PSI, wake_inputs, rotor):
    """
    Computes the BEVW iteration.

    Assumptions:
    N/A

    Source:
    N/A

    Inputs:
       B                          number of rotor blades                          [-]
       beta                       blade twist distribution                        [-]
       r                          radius distribution                             [m]
       R                          tip radius                                      [m]
       Wt                         tangential velocity                             [m/s]
       Wa                         axial velocity                                  [m/s]
       U                          total velocity                                  [m/s]
       Ut                         tangential velocity                             [m/s]
       Ua                         axial velocity                                  [m/s]
       cos_psi                    cosine of the inflow angle PSI                  [-]
       sin_psi                    sine of the inflow angle PSI                    [-]

    Outputs:
       dR_dpsi                    derivative of residual wrt inflow angle         [-]

    """    
    
    # Unpack inputs to rotor wake fidelity zero
    U               = wake_inputs.velocity_total
    Ua              = wake_inputs.velocity_axial
    Ut              = wake_inputs.velocity_tangential
    beta            = wake_inputs.twist_distribution
    c               = wake_inputs.chord_distribution
    r               = wake_inputs.radius_distribution
    a               = wake_inputs.speed_of_sounds
    nu              = wake_inputs.dynamic_viscosities
    ctrl_pts        = wake_inputs.ctrl_pts
    Nr              = wake_inputs.Nr
    Na              = wake_inputs.Na

    # Unpack rotor data        
    R        = rotor.tip_radius
    B        = rotor.number_of_blades    
    tc       = rotor.thickness_to_chord
    a_geo    = rotor.airfoil_geometry
    a_loc    = rotor.airfoil_polar_stations
    cl_sur   = rotor.airfoil_cl_surrogates
    cd_sur   = rotor.airfoil_cd_surrogates   
    RE_data  = rotor.RE_data 
    aoa_data = rotor.aoa_data

    PSI      = jnp.reshape(PSI,jnp.shape(U))

    # compute velocities
    sin_psi      = jnp.sin(PSI)
    cos_psi      = jnp.cos(PSI)
    Wa           = 0.5*Ua + 0.5*U*sin_psi
    Wt           = 0.5*Ut + 0.5*U*cos_psi
    vt           = Ut - Wt

    # compute blade airfoil forces and properties
    Cl, Cdval, alpha, Ma, W = compute_airfoil_aerodynamics(beta,c,r,R,B,Wa,Wt,a,nu,a_loc,a_geo,RE_data, aoa_data,cl_sur,cd_sur,ctrl_pts,Nr,Na,tc)

    # compute inflow velocity and tip loss factor
    lamdaw, F, _ = compute_inflow_and_tip_loss(r,R,Wa,Wt,B)

    # compute Newton residual on circulation
    Gamma       = vt*(4.*jnp.pi*r/B)*F*(1.+(4.*lamdaw*R/(jnp.pi*B*r))*(4.*lamdaw*R/(jnp.pi*B*r)))**0.5
    Rsquiggly   = Gamma - 0.5*W*c*Cl

    return Rsquiggly.flatten()

@jacobian
def jacobian_iteration(PSI, wake_inputs, rotor):
    return iteration(PSI, wake_inputs, rotor)

## @defgroup Methods-Propulsion-Rotor_Wake-Fidelity_Zero
@jit
def va_vt(PSI, wake_inputs, rotor):
    """
    Computes the inflow velocities from the inflow angle

    Assumptions:
    N/A

    Source:
    N/A

    Inputs:
       U                          total velocity                                  [m/s]
       Ut                         tangential velocity                             [m/s]
       Ua                         axial velocity                                  [m/s]
       PSI                        inflow angle PSI                                [Rad]


    Outputs:
       va                         axially-induced velocity from rotor wake        [m/s]
       vt                         angentially-induced velocity from rotor wake    [m/s]

    """    
    
    # Unpack inputs to rotor wake fidelity zero
    U               = wake_inputs.velocity_total
    Ua              = wake_inputs.velocity_axial
    Ut              = wake_inputs.velocity_tangential
    
    # Reshape PSI because the solver gives it flat
    PSI    = jnp.reshape(PSI,jnp.shape(U))
    
    # compute velocities
    sin_psi      = jnp.sin(PSI)
    cos_psi      = jnp.cos(PSI)
    Wa           = 0.5*Ua + 0.5*U*sin_psi
    Wt           = 0.5*Ut + 0.5*U*cos_psi
    va           = Wa - Ua
    vt           = Ut - Wt

    return va, vt



    


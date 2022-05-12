## @defgroup Methods-Propulsion-Rotor_Wake-Fidelity_Zero
# fidelity_zero_wake_convergence.py
#
# Created:  Feb 2022, R. Erhard
# Modified: 

from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.BET_calculations import compute_airfoil_aerodynamics,compute_inflow_and_tip_loss
import numpy as np
import jax.numpy as jnp
from jax import jacobian, lax

## @defgroup Methods-Propulsion-Rotor_Wake-Fidelity_Zero
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
    
    # Unpack some wake inputs
    U      = wake_inputs.velocity_total

    PSI    = jnp.ones_like(U).flatten()
    
    jac = jacobian(iteration)
    
    PSI_final, ii = simple_newton(iteration,jac,PSI,args=(wake_inputs,rotor))
    
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
       piece                      output of a step in tip loss calculation        [-]

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

    PSI    = jnp.reshape(PSI,jnp.shape(U))

    # compute velocities
    sin_psi      = jnp.sin(PSI)
    cos_psi      = jnp.cos(PSI)
    Wa           = 0.5*Ua + 0.5*U*sin_psi
    Wt           = 0.5*Ut + 0.5*U*cos_psi
    vt           = Ut - Wt

    # compute blade airfoil forces and properties
    Cl, Cdval, alpha, Ma, W = compute_airfoil_aerodynamics(beta,c,r,R,B,Wa,Wt,a,nu,a_loc,a_geo,cl_sur,cd_sur,ctrl_pts,Nr,Na,tc)

    # compute inflow velocity and tip loss factor
    lamdaw, F, piece = compute_inflow_and_tip_loss(r,R,Wa,Wt,B)

    # compute Newton residual on circulation
    Gamma       = vt*(4.*np.pi*r/B)*F*(1.+(4.*lamdaw*R/(np.pi*B*r))*(4.*lamdaw*R/(np.pi*B*r)))**0.5
    Rsquiggly   = Gamma - 0.5*W*c*Cl

    return Rsquiggly.flatten()

## @defgroup Methods-Propulsion-Rotor_Wake-Fidelity_Zero
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


def simple_newton(function,jac,intial_x,tol=1e-8,args=()):
    """
    Performs the inside of the while loop in a newton iteration

    Assumptions:
    N/A

    Source:
    N/A

    Inputs:


    Outputs:


    """             
    
    # Set initials and pack into list
    ii   = 1
    Xn   = intial_x.flatten()
    Xnp1 = intial_x.flatten()
    damping_factor = 1.
    R    = 1.
    Full_vector = [Xn,Xnp1,R,ii,damping_factor]
    
    cond_fun = lambda Full_vector:cond(Full_vector,tol,function,jac,*args)
    inner_newton_fun = lambda Full_vector:inner_newton(Full_vector,function,jac,*args)
    
    Full_vector = lax.while_loop(cond_fun, inner_newton_fun, Full_vector)

    # Unpack the final versioon
    Xnp1 = Full_vector[1]
    ii   = Full_vector[3]

    return Xnp1, ii


def cond(Full_vector,tol,function,jac,*args):
    """
    Performs the inside of the while loop in a newton iteration

    Assumptions:
    N/A

    Source:
    N/A

    Inputs:


    Outputs:


    """      
    
    Full_vector = inner_newton(Full_vector,function,jac,*args)
    R           = Full_vector[2]
    
    return R>tol
    

def inner_newton(Full_vector,function,jac,*args):
    """
    Performs the inside of the while loop in a newton iteration

    Assumptions:
    N/A

    Source:
    N/A

    Inputs:


    Outputs:


    """       
    
    # Unpack the full vector
    df = Full_vector[4] # damping factor
    Xn = Full_vector[1] # The newest one!
    ii = Full_vector[3] # number of iterations
    
    # Calculate the functional value and the derivative
    f    = jnp.array(function(Xn,*args)).flatten()
    fp   = jnp.diagonal(jnp.array(jac(Xn,*args))).flatten()
    
    # Update to the new point
    Xnp1 = Xn - df*f/fp

    # Take the residual
    R  = jnp.max(jnp.abs(Xnp1-Xn))
    
    # Update the state
    true_fun  = lambda df: df/2
    false_fun = lambda df: df
    cond1     = R<1e-4
    #cond2     = ii>8
    #conds     = cond1 or cond2
    df = lax.cond(cond1, true_fun, false_fun, df)

    ii+=1    
    
    # Pack the full vector
    Full_vector = [Xn,Xnp1,R,ii,df]
    
    return Full_vector
    
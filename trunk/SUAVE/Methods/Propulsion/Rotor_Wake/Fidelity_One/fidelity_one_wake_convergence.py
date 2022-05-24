## @ingroup Methods-Propulsion-Rotor_Wake-Fidelity_One
# fidelity_one_wake_convergence.py
#
# Created:  Feb 2022, R. Erhard
# Modified: 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------


from SUAVE.Methods.Propulsion.Rotor_Wake.Fidelity_One.compute_fidelity_one_inflow_velocities import compute_fidelity_one_inflow_velocities
from SUAVE.Methods.Propulsion.Rotor_Wake.Fidelity_One.generate_fidelity_one_wake_shape import generate_fidelity_one_wake_shape
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.BET_calculations import compute_inflow_and_tip_loss
import jax.numpy as jnp
from jax import lax, jacobian
from SUAVE.Methods.Propulsion.Rotor_Wake.Common import simple_newton

# ----------------------------------------------------------------------
# Wake Convergence
# ----------------------------------------------------------------------

## @ingroup Methods-Propulsion-Rotor_Wake-Fidelity_One
def fidelity_one_wake_convergence(wake,rotor,wake_inputs):
    """
    This converges on the wake shape for the fidelity-one rotor wake.
    
    Assumptions:
    None
    
    Source:
    N/A
    
    Inputs:
    wake        - rotor wake
    rotor       - rotor
    wake_inputs - inputs passed from the BET rotor spin function
    
    Outputs:
    None
    
    Properties Used:
    None
    """    
    
    # converge on va for a semi-prescribed wake method
    
    # Pull out the newton end conditions
    tol   = wake.axial_velocity_convergence_tolerance
    limit = lax.cond(wake.semi_prescribed_converge,lambda : wake.maximum_convergence_iteration, lambda : 1.) 
        
    # assume a va by running once
    va, vt, F = va_vt(wake, wake_inputs, rotor)
    Fva       = F*va

    # Take the jacobian of the iteration loop
    jac = jacobian(iteration)
    
    # Solve!
    Fva_final, ii = simple_newton(iteration,jac,Fva, tol=tol, limit=limit, args=(wake,wake_inputs,rotor))  
    
    rotor.outputs.disc_axial_induced_velocity = Fva_final
        
    # save converged wake:
    WD  = generate_fidelity_one_wake_shape(wake_inputs,rotor)
    
    # Use the converged solution
    va, vt, F = va_vt(wake, wake_inputs, rotor)
    
    return WD, va, vt


# ----------------------------------------------------------------------
# Iteration Function
# ----------------------------------------------------------------------

## @defgroup Methods-Propulsion-Rotor_Wake-Fidelity_One
def iteration(Fva,wake,wake_inputs,rotor):
    """
    Computes the BEVW iteration.

    Assumptions:
    N/A

    Source:
    N/A

    Inputs:


    Outputs:

    """    
    
    # update the axial disc velocity based on new va from HFW
    rotor.outputs.disc_axial_induced_velocity = F*va     
    
    # Unpack some things
    Ua = wake_inputs.velocity_axial
    Ut = wake_inputs.velocity_tangential
    r  = wake_inputs.radius_distribution
    
    R  = rotor.tip_radius
    B  = rotor.number_of_blades      

    # generate wake geometry for rotor
    WD  = generate_fidelity_one_wake_shape(wake,rotor)
    
    # compute axial wake-induced velocity (a byproduct of the circulation distribution which is an input to the wake geometry)
    va, vt = compute_fidelity_one_inflow_velocities(wake,rotor, WD)

    # compute new blade velocities
    Wa   = va + Ua
    Wt   = Ut - vt

    _, F, _ = compute_inflow_and_tip_loss(r,R,Wa,Wt,B)

    va_diff = F*va - rotor.outputs.disc_axial_induced_velocity
    
    return va_diff


## @defgroup Methods-Propulsion-Rotor_Wake-Fidelity_Zero
def va_vt(wake, wake_inputs, rotor):
    """
    Computes the inflow velocities from the inflow angle

    Assumptions:
    N/A

    Source:
    N/A

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
       F   - tip loss factor                                       

    """    
    
    # Unpack some things
    Ua = wake_inputs.velocity_axial
    Ut = wake_inputs.velocity_tangential
    r  = wake_inputs.radius_distribution
    
    R  = rotor.tip_radius
    B  = rotor.number_of_blades      

    # generate wake geometry for rotor
    WD  = generate_fidelity_one_wake_shape(wake,rotor)
    
    # compute axial wake-induced velocity (a byproduct of the circulation distribution which is an input to the wake geometry)
    va, vt = compute_fidelity_one_inflow_velocities(wake,rotor, WD)
    
    # compute new blade velocities
    Wa   = va + Ua
    Wt   = Ut - vt

    _, F, _ = compute_inflow_and_tip_loss(r,R,Wa,Wt,B)    
    
    return va, vt, F
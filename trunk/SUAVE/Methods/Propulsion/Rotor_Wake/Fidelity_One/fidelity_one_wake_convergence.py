## @ingroup Methods-Propulsion-Rotor_Wake-Fidelity_One
# fidelity_one_wake_convergence.py
#
# Created:  Feb 2022, R. Erhard
# Modified: 

from SUAVE.Methods.Propulsion.Rotor_Wake.Fidelity_One.compute_fidelity_one_inflow_velocities import compute_fidelity_one_inflow_velocities
from SUAVE.Methods.Propulsion.Rotor_Wake.Fidelity_One.generate_fidelity_one_wake_shape import generate_fidelity_one_wake_shape
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.BET_calculations import compute_inflow_and_tip_loss

import numpy as np

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
    # Unpack inputs
    Ua = wake_inputs.velocity_axial
    Ut = wake_inputs.velocity_tangential
    r  = wake_inputs.radius_distribution
    
    R  = rotor.tip_radius
    B  = rotor.number_of_blades    
    
    # converge on va for a semi-prescribed wake method
    va_diff, ii = 1, 0
    tol = wake.axial_velocity_convergence_tolerance
    if wake.semi_prescribed_converge:
        if wake.verbose:
            print("\tConverging on semi-prescribed wake shape...")
        ii_max = wake.maximum_convergence_iteration
    else:
        if wake.verbose:
            print("\tGenerating fully-prescribed wake shape...")
        ii_max = 1
        

    while va_diff > tol:  
        # generate wake geometry for rotor
        WD  = generate_fidelity_one_wake_shape(wake,rotor)
        
        # compute axial wake-induced velocity (a byproduct of the circulation distribution which is an input to the wake geometry)
        va, vt = compute_fidelity_one_inflow_velocities(wake,rotor, WD)
    
        # compute new blade velocities
        Wa   = va + Ua
        Wt   = Ut - vt
    
        lamdaw, F, _ = compute_inflow_and_tip_loss(r,R,Wa,Wt,B)
    
        va_diff = np.max(abs(F*va - rotor.outputs.disc_axial_induced_velocity))
    
        # update the axial disc velocity based on new va from HFW
        rotor.outputs.disc_axial_induced_velocity = F*va 
        
        ii+=1
        if ii>=ii_max and va_diff>tol:
            if wake.semi_prescribed_converge and wake.verbose:
                print("Semi-prescribed vortex wake did not converge on axial inflow used for wake shape.")
            break
    
        
    return WD, va, vt
# compute_aircraft_lift.py
# 
# Modified: from Fidelity_Zero.Lift.compute_aircraft_lift by Tim Momose, February 2015

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import SUAVE
# suave imports
from SUAVE.Structure import Data

from SUAVE.Attributes.Results import Result

from SUAVE.Methods.Aerodynamics.Fidelity_Zero.Lift import weissinger_vortex_lattice
from SUAVE.Methods.Aerodynamics.Fidelity_Zero.Lift import vortex_lift
from SUAVE.Methods.Flight_Dynamics.Static_Stability.Approximations.datcom import datcom

# package imports
import numpy as np


# ----------------------------------------------------------------------
#  The Function
# ----------------------------------------------------------------------

def compute_aircraft_lift(conditions,configuration,geometry):
    """ SUAVE.Methods.Aerodynamics.compute_aircraft_lift(conditions,configuration,geometry)
        computes the lift associated with an aircraft based on a surrogate from AVL
        
        Inputs:
            conditions - data dictionary with fields:
                mach_number - float or 1D array of freestream mach numbers
                angle_of_attack - floar or 1D array of angle of attacks
                
            configuration - data dictionary with fields:
                surrogate_models.lift_coefficient - a callable function or class 
                    with inputs of angle of attack and outputs of lift coefficent
                    
            geometry - the geometry being analyzed, including the field,
                geometry.Wings['Main Wing']
            
        
        Outputs:
            CL - float or 1D array of lift coefficients of the total aircraft
        
        Updates:
            conditions.lift_breakdown - stores results here
            
        Assumptions:
            surrogate model returns total incompressible lift due to wings
            compressibility correction is based on the DATCOM lift-curve-slope
            formula, assuming that the AoA for zero-lift is not significantly 
            affectd by flight Mach number.        
    """    
   
    # unpack
    Mc             = conditions.freestream.mach_number
    AoA            = conditions.aerodynamics.angle_of_attack
    
    # the lift surrogate model for wings only
    wings_lift_model = configuration.surrogate_models.lift_coefficient
    
    # pack for interpolate
    X_interp = AoA
    
    vortex_cl = np.array([[0.0]] * len(Mc))    
    
    # interpolate
    wings_lift = wings_lift_model(X_interp)  
    
    wing = geometry.wings['Main Wing']
    if wing.vortex_lift is True:
        vortex_cl = vortex_lift(X_interp,configuration,wing) # This was initialized at 0.0
        wings_lift = wings_lift + vortex_cl   
    
    # compressibility correction
    #OLD Prandtl-Glauert Correction: compress_corr = 1./(np.sqrt(1.-Mc**2.))
    compress_corr = datcom(wing,Mc)/datcom(wing,[0.0])
    
    # correct lift
    wings_lift_comp = wings_lift * compress_corr
    
    # total lift, accounting one fuselage
    aircraft_lift_total = wings_lift_comp
    
    # store results
    lift_results = Result(
        total                = aircraft_lift_total ,
        incompressible_wings = wings_lift          ,
        compressible_wings   = wings_lift_comp     ,
        compressibility_correction_factor = compress_corr  ,
        fuselage_correction_factor        = 1.0 ,
    )
    conditions.aerodynamics.lift_breakdown.update( lift_results )    #update
    
    conditions.aerodynamics.lift_coefficient= aircraft_lift_total

    return aircraft_lift_total


if __name__ == '__main__':   
    #test()
    raise RuntimeError , 'module test failed, not implemented'

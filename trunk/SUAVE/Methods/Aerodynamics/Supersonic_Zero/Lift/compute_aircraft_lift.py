# compute_aircraft_lift.py
# 
# Created:  Dec 2013, A. Variyar,
# Modified: Feb 2014, A. Variyar, T. Lukaczyk, T. Orra 
#           Apr 2014, A. Variyar  
#           Aug 2014, T. Macdonald
#           Jan 2016, E. Botero     

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Analyses import Results
from SUAVE.Methods.Aerodynamics.Supersonic_Zero.Lift.vortex_lift import vortex_lift

import numpy as np

# ----------------------------------------------------------------------
#  The Function
# ----------------------------------------------------------------------

def compute_aircraft_lift(conditions,configuration,geometry):
    """ SUAVE.Methods.Aerodynamics.compute_aircraft_lift(conditions,configuration,geometry)
        computes the lift associated with an aircraft 
        
        Inputs:
            conditions - data dictionary with fields:
                mach_number - float or 1D array of freestream mach numbers
                angle_of_attack - floar or 1D array of angle of attacks
                
            configuration - data dictionary with fields:
                surrogate_models.lift_coefficient - a callable function or class 
                    with inputs of angle of attack and outputs of lift coefficent
                fuselage_lift_correction - the correction to fuselage contribution to lift
                    
            geometry - used for wing
            
        
        Outputs:
            CL - float or 1D array of lift coefficients of the total aircraft
        
        Updates:
            conditions.lift_breakdown - stores results here
            
        Assumptions:
            surrogate model returns total incompressible lift due to wings
            prandtl-glaurert compressibility correction on this
            fuselage contribution to lift correction as a factor
        
    """    
   
    # unpack
    fus_correction = configuration.fuselage_lift_correction
    Mc             = conditions.freestream.mach_number
    AoA            = conditions.aerodynamics.angle_of_attack
    
    # pack for interpolate
    X_interp = AoA
    
    wings_lift          = np.array([[0.0]] * len(Mc))
    wings_lift_comp     = np.array([[0.0]] * len(Mc))
    compress_corr       = np.array([[0.0]] * len(Mc))
    aircraft_lift_total = np.array([[0.0]] * len(Mc))
    vortex_cl           = np.array([[0.0]] * len(Mc))

    wing = geometry.wings['main_wing']
    
    # Subsonic setup
    wings_lift = conditions.aerodynamics.lift_breakdown.inviscid_wings_lift.total
    compress_corr[Mc < 0.95] = 1./(np.sqrt(1.-Mc[Mc < 0.95]**2.))
    compress_corr[Mc >= 0.95] = 1./(np.sqrt(1.-0.95**2)) # Values for Mc > 1.05 are update after this assignment

    if wing.vortex_lift is True:
        vortex_cl[Mc < 1.0] = vortex_lift(X_interp[Mc < 1.0],configuration,wing) # This was initialized at 0.0
    wings_lift[Mc <= 1.05] = wings_lift[Mc <= 1.05] + vortex_cl[Mc <= 1.05]
    
    # Supersonic setup
    compress_corr[Mc > 1.05] = 1./(np.sqrt(Mc[Mc > 1.05]**2.-1.))

    wings_lift_comp = wings_lift * compress_corr    
        
    aircraft_lift_total = wings_lift_comp * fus_correction
    
    # store results
    lift_results = Results(
        total                = aircraft_lift_total ,
        incompressible_wings = wings_lift          ,
        compressible_wings   = wings_lift_comp     ,
        compressibility_correction_factor = compress_corr  ,
        fuselage_correction_factor        = fus_correction ,
        vortex                            = vortex_cl ,
    )
    conditions.aerodynamics.lift_breakdown.update( lift_results )    #update
        
    conditions.aerodynamics.lift_coefficient = aircraft_lift_total

    return aircraft_lift_total
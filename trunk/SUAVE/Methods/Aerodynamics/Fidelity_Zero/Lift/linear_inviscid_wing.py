# compute_aircraft_lift.py
# 
# Created:  Anil V., Dec 2013
# Modified: Anil, Trent, Tarik, Feb 2014 
# Modified: Anil  April 2014 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import SUAVE
# suave imports

from SUAVE.Core import Data
from SUAVE.Core import Units

from SUAVE.Core import Results

from SUAVE.Methods.Aerodynamics.Fidelity_Zero.Lift import weissinger_vortex_lattice
from SUAVE.Methods.Aerodynamics.Fidelity_Zero.Lift import vortex_lift

# python imports
import os, sys, shutil
from copy import deepcopy
from warnings import warn

# package imports
import numpy as np
import scipy as sp


# ----------------------------------------------------------------------
#  The Function
# ----------------------------------------------------------------------

def linear_inviscid_wing(state,settings,geometry):
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
                    
            geometry - Not used
            
        
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
    fus_correction = settings.fuselage_lift_correction
    Mc             = state.conditions.freestream.mach_number
    AoA            = state.conditions.aerodynamics.angle_of_attack
    
    wings_lift = state.conditions.aerodynamics.lift_breakdown.inviscid_wings_lift
    
    #wing = geometry.wings[0]
    #if wing.vortex_lift is True:
        #vortex_cl = vortex_lift(X_interp,configuration,wing) # This was initialized at 0.0
        #wings_lift = wings_lift + vortex_cl   
    
    ## compressibility correction
    #compress_corr = 1./(np.sqrt(1.-Mc**2.))
    
    ## correct lift
    #wings_lift_comp = wings_lift * compress_corr
    
    ## total lift, accounting one fuselage
    #aircraft_lift_total = wings_lift_comp * fus_correction 
    
    ## store results
    #lift_results = Results(
        #total                = aircraft_lift_total ,
        #incompressible_wings = wings_lift          ,
        #compressible_wings   = wings_lift_comp     ,
        #compressibility_correction_factor = compress_corr  ,
        #fuselage_correction_factor        = fus_correction ,
    #)
    #conditions.aerodynamics.lift_breakdown.update( lift_results )    #update
    
    conditions.aerodynamics.lift_coefficient= wings_lift

    return wings_lift


if __name__ == '__main__':   
    #test()
    raise RuntimeError , 'module test failed, not implemented'

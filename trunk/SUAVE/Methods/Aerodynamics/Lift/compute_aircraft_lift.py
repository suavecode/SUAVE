# compute_aircraft_lift.py
# 
# Created:  Anil V., Dec 2013
# Modified: Anil, Trent, Tarik, Feb 2014 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
from SUAVE.Attributes.Results import Result

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

def compute_aircraft_lift(conditions,configuration,geometry=None):
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
    fus_correction = configuration.fuselage_lift_correction
    Mc             = conditions.mach_number
    AoA            = conditions.angle_of_attack
    
    # the lift surrogate model for wings only
    wings_lift_model = configuration.surrogate_models.lift_coefficient
    
    # pack for interpolate
    X_interp = np.array([AoA]).T
    
    # interpolate
    wings_lift = wings_lift_model(X_interp)  
    
    # compressibility correction
    compress_corr = 1./(np.sqrt(1.-Mc**2.))
    
    # correct lift
    wings_lift_comp = wings_lift * compress_corr
    
    # total lift, accounting one fuselage
    aircraft_lift_total = wings_lift_comp * fus_correction 
    
    # store to results
    lift_results = Result(
        total                = aircraft_lift_total ,
        incompressible_wings = wings_lift          ,
        compressible_wings   = wings_lift_comp     ,
        compressibility_correction_factor = compress_corr  ,
        fuselage_correction_factor        = fus_correction ,
    )
    conditions.lift_breakdown.update( lift_results )
    
    # done!
    return aircraft_lift_total


# ----------------------------------------------------------------------
#   Module Tests
# ----------------------------------------------------------------------
# this will run from command line, put simple tests for your code here
if __name__ == '__main__':    
    raise RuntimeError , 'module test failed, not implemented'
# compute_aircraft_drag.py
# 
# Created:  Anil V., Dec 2013
# Modified: Anil, Trent, Tarik, Feb 2014 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
from SUAVE.Core import Results

from SUAVE.Methods.Aerodynamics.Fidelity_Zero.Drag import \
     parasite_drag_aircraft, induced_drag_aircraft, compressibility_drag_wing, \
     miscellaneous_drag_aircraft

from parasite_drag_aircraft import parasite_drag_aircraft
from induced_drag_aircraft import induced_drag_aircraft
from compressibility_drag_wing import compressibility_drag_wing
from miscellaneous_drag_aircraft_ESDU import miscellaneous_drag_aircraft_ESDU

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

def compute_aircraft_drag(conditions,configuration,geometry=None):
    """ SUAVE.Methods.Aerodynamics.compute_aircraft_drag(conditions,configuration,geometry)
        computes the lift associated with an aircraft 
        
        Inputs:
            conditions - data dictionary with fields:
                mach_number - float or 1D array of freestream mach numbers
                angle_of_attack - floar or 1D array of angle of attacks
                
            configuration - data dictionary with fields:
                surrogate_models.lift_coefficient - a callable function or class 
                    with inputs of angle of attack and outputs of lift coefficent
                fuselage_lift_correction - the correction to fuselage contribution to lift
                    
            geometry - the aircraft geoemtry with fields:
            
        
        Outputs:
            CD - float or 1D array of drag coefficients of the total aircraft
        
        Updates:
            conditions.drag_breakdown - stores results here
            
        Assumptions:
            
            
    """    
    
    # unpack inputs
    trim_correction_factor     = configuration.trim_drag_correction_factor
    drag_coefficient_increment = configuration.drag_coefficient_increment
    drag_breakdown             = conditions.aerodynamics.drag_breakdown

    # various drag components
    parasite_total        = parasite_drag_aircraft           (conditions,configuration,geometry)
    induced_total         = induced_drag_aircraft            (conditions,configuration,geometry)
    compressibility_total = compressibility_drag_wing        (conditions,configuration,geometry)
    miscellaneous_drag    = miscellaneous_drag_aircraft_ESDU (conditions,configuration,geometry)

    # untrimmed drag
    aircraft_untrimmed = parasite_total        \
                       + induced_total         \
                       + compressibility_total \
                       + miscellaneous_drag

    # start additional corrections
    aircraft_total_drag = aircraft_untrimmed

    # trim correction
    aircraft_total_drag *= trim_correction_factor
    drag_breakdown.miscellaneous.trim_correction_factor = trim_correction_factor

    # add drag_coefficient_increment
    aircraft_total_drag += drag_coefficient_increment
    drag_breakdown.drag_coefficient_increment = drag_coefficient_increment

    # store to results
    drag_breakdown.total     = aircraft_total_drag
    drag_breakdown.untrimmed = aircraft_untrimmed
    
    # done!
    return aircraft_total_drag


# ----------------------------------------------------------------------
#   Module Tests
# ----------------------------------------------------------------------
# this will run from command line, put simple tests for your code here
if __name__ == '__main__':    
    raise RuntimeError , 'module test failed, not implemented'


#--------------test this case as well--------------
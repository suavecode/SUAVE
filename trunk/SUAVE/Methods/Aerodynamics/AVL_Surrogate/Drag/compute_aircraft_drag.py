# compute_aircraft_drag.py (for AVL_Surrogate)
# 
# Modified from SUAVE.Methods.Aerodynamics.Fidelity_Zero.Drag.compute_aircraft_drag
#    February 2015

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
from SUAVE.Attributes.Results import Result
# Fidelity_Zero methods
from SUAVE.Methods.Aerodynamics.Fidelity_Zero.Drag.parasite_drag_aircraft import parasite_drag_aircraft
from SUAVE.Methods.Aerodynamics.Fidelity_Zero.Drag.compressibility_drag_wing import compressibility_drag_wing
from SUAVE.Methods.Aerodynamics.Fidelity_Zero.Drag.miscellaneous_drag_aircraft import miscellaneous_drag_aircraft

# local imports
from induced_drag_aircraft import induced_drag_aircraft

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
    """ SUAVE.Methods.Aerodynamics.AVL_Surrogate.Drag.compute_aircraft_drag(conditions,configuration,geometry)
        computes the drag associated with an aircraft 
        
        Inputs:
            conditions - data dictionary with fields:
                mach_number - float or 1D array of freestream mach numbers
                angle_of_attack - floar or 1D array of angle of attacks
                
            configuration - data dictionary with fields:
                surrogate_models.lift_coefficient - a callable function or class 
                    with inputs of angle of attack and outputs of lift coefficent
                fuselage_lift_correction - the correction to fuselage contribution to lift
                    
            geometry - the aircraft geoemtry with fields:
                geometry.wings['Main Wing'].aspect_ratio
        
        Outputs:
            CD - float or 1D array of drag coefficients of the total aircraft
        
        Updates:
            conditions.drag_breakdown - stores results here
            
        Assumptions:
            assumes compute_aircraft_lift has already been run, so an updated
            compressibility correction factor and lift coefficient are
            available in conditions:
             -conditions.aerodynamics.lift_coefficient
             -conditions.aerodynamics.lift_breakdown.compressibility_correction_factor
    """    
    
    # unpack inputs
    trim_correction_factor     = configuration.trim_drag_correction_factor
    drag_coefficient_increment = configuration.drag_coefficient_increment
    drag_breakdown             = conditions.aerodynamics.drag_breakdown

    # various drag components
    parasite_total        = parasite_drag_aircraft     (conditions,configuration,geometry)
    induced_total         = induced_drag_aircraft      (conditions,configuration,geometry)
    compressibility_total = compressibility_drag_wing  (conditions,configuration,geometry)
    miscellaneous_drag    = miscellaneous_drag_aircraft(conditions,configuration,geometry)

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
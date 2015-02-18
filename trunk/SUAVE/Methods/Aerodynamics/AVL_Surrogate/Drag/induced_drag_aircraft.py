
# induced_drag_aircraft.py
# 
# Created:  Tim Momose, Feb 2015
# (Modified from Fidelity_Zero version)         

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
from SUAVE.Attributes.Results import Result
from SUAVE.Methods.Aerodynamics.AVL_Surrogate.Lift.compute_aircraft_lift import compute_aircraft_lift

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

def induced_drag_aircraft(conditions,configuration,geometry):
    """ SUAVE.Methods.induced_drag_aircraft(conditions,configuration,geometry)
        computes the induced drag associated with a wing 
        
        Inputs:
        
        Outputs:
        
        Assumptions:
            based on a set of fits
            assumes compute_aircraft_lift has already been run, so an updated
            compressibility correction factor and lift coefficient are
            available in conditions:
             -conditions.aerodynamics.lift_coefficient
             -conditions.aerodynamics.lift_breakdown.compressibility_correction_factor
    """

    # unpack inputs
    wing = geometry.wings['Main Wing']
    ar   = wing.aspect_ratio
    AoA  = conditions.aerodynamics.angle_of_attack
    Mc   = conditions.freestream.mach_number
    
    aircraft_lift   = conditions.aerodynamics.lift_coefficient
    mach_correction = conditions.aerodynamics.lift_breakdown.compressibility_correction_factor
    
    # unpack surrogate model and pack AoA for interpolate
    induced_drag_model  = configuration.surrogate_models.induced_drag_coefficient
    X_interp            = AoA

    # interpolate
    total_induced_drag = induced_drag_model(X_interp) * mach_correction**2.

    e = aircraft_lift**2 / (np.pi*ar*total_induced_drag) # ANOTHER OPTION: Make a surrogate model for e directly from AVL results
    
    # store data
    conditions.aerodynamics.drag_breakdown.induced = Result(
        total             = total_induced_drag ,
        efficiency_factor = e                  ,
        aspect_ratio      = ar                 ,
    )
    
    # done!

    return total_induced_drag
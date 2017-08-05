## @ingroup Methods-Aerodynamics-Supersonic_Zero-Drag
# compute_aircraft_drag.py
# 
# Created:  Dec 2013, A. Variyar
# Modified: Aug 2014, T. MacDonald
#           Jan 2016, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Methods.Aerodynamics.Supersonic_Zero.Drag import \
     parasite_drag_aircraft, compressibility_drag_total, \
     miscellaneous_drag_aircraft

# ----------------------------------------------------------------------
#  The Function
# ----------------------------------------------------------------------

## @ingroup Methods-Aerodynamics-Supersonic_Zero-Drag
def compute_aircraft_drag(conditions,configuration,geometry=None):
    """ Unused function
    """    
    
    # Unpack inputs
    trim_correction_factor = configuration.trim_drag_correction_factor
    drag_breakdown = conditions.aerodynamics.drag_breakdown
    
    # Various drag components
    parasite_total        = parasite_drag_aircraft(conditions,configuration,geometry) 
    induced_total         = induced_drag_aircraft(conditions,configuration,geometry)
    compressibility_total = compressibility_drag_total(conditions,configuration,geometry) 
    miscellaneous_drag    = miscellaneous_drag_aircraft(conditions,configuration,geometry)
    
    
    # Untrimmed drag
    aircraft_untrimmed = parasite_total        \
                       + induced_total         \
                       + compressibility_total \
                       + miscellaneous_drag 
    
    # Trim correction
    aircraft_total_drag = aircraft_untrimmed * trim_correction_factor
    drag_breakdown.miscellaneous.trim_correction_factor = trim_correction_factor
    
    # Store to results
    drag_breakdown.total     = aircraft_total_drag
    drag_breakdown.untrimmed = aircraft_untrimmed       
    
    return aircraft_total_drag
# spoiler_drag.py
#
# Created:  Jan 2014, A. Variyar
# Modified: Jan 2016, E. Botero 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
# SUAVE imports
from SUAVE.Analyses import Results

# ----------------------------------------------------------------------
#  Adds the spoiler drag
# ----------------------------------------------------------------------
def spoiler_drag(state,settings,geometry):
    
    # unpack inputs
    conditions     = state.conditions
    configuration  = settings
    drag_breakdown = conditions.aerodynamics.drag_breakdown

    # various drag components
    spoiler_drag = settings.spoiler_drag_increment

    # untrimmed drag
    conditions.aerodynamics.drag_breakdown.spoiler_drag = spoiler_drag
    
    return spoiler_drag

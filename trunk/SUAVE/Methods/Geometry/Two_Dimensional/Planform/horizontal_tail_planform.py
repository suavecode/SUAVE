# horizontal_tail_planform.py
#
# Created:  Mar 2013, SUAVE Team
# Modified: Jan 2016, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from wing_planform import wing_planform

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------
def horizontal_tail_planform(Wing):
    """ results = SUAVE.Methods.Geometry.horizontal_tail_planform(Wing)
    
        see SUAVE.Methods.Geometry.wing_planform()
    """
    wing_planform(Wing)
    
    return 0
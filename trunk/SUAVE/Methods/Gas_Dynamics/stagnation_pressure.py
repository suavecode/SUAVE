""" stagnation_pressure.py: compute stagnation pressure for a 1D ideal gas """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from f_of_M import f_of_M

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

def stagnation_pressure(g,p,M):

    """  pt = stagnation_pressure(g,p,M): compute stagnation pressure for a 1D ideal gas
    
         Inputs:    g = ratio of specific heats     (required)  (float)
                    p = absolute pressure           (required)  (floats)    
                    M = Mach number                 (required)  (floats)

         Outputs:   pt = stagnation pressure                    (floats)

    """

    n = g/(g-1)

    return p*f_of_M(g,M)**n
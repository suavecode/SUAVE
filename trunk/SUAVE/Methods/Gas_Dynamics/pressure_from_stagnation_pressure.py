""" pressure_from_stagnation_pressure.py: compute pressure from stagnation pressure and Mach for a 1D ideal gas """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from f_of_M import f_of_M

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

def pressure_from_stagnation_pressure(g,pt,M):

    """  p = pressure_from_stagnation_pressure(g,pt,M): compute pressure from stagnation pressure and Mach for a 1D ideal gas
    
         Inputs:    g = ratio of specific heats     (required)  (float)
                    pt = stagnation pressure        (required)  (floats)    
                    M = Mach number                 (required)  (floats)

         Outputs:   p = absolute pressure                       (floats)

        """

    n = g/(g-1)

    return pt/(f_of_M(g,M)**n)
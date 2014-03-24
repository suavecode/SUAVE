""" temperature_from_stagnation_temperature.py: compute temperature from stagnation temperature and Mach for a 1D ideal gas """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from f_of_M import f_of_M

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

def temperature_from_stagnation_temperature(g,T,M):

    """  T = stagnation_temperature(g,Tt,M): compute temperature from stagnation temperature and Mach for a 1D ideal gas
    
         Inputs:    g = ratio of specific heats     (required)  (float)
                    Tt = stagnation temperature     (required)  (floats)    
                    M = Mach number                 (required)  (floats)

         Outputs:   T = absolute temperature                    (floats)

        """

    return Tt/f_of_M(g,M)
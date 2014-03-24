""" area_ratio_from_mach.py: get A/A* from M and gamma for a 1D ideal gas flow """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from f_of_M import f_of_M

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

def area_ratio_from_mach(g,M):

    """  M = mach_from_area_ratio(g,area_ratio,branch="supersonic"): get M from A/A* and gamma for a 1D ideal gas flow
    
         Inputs:    g = ratio of specific heats     (required)  (float)    
                    M = Mach number                 (required)  (float)

         Outputs:   area_ratio = A/A*                           (float)

        """

    n = (g+1)/(2*(g-1))
    c = ((g+1)/2)**n

    return 1/(c*M/f_of_M(g,M)**n)
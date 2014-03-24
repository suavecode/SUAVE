""" f_of_M.py: helper function that computes f(M) for a 1D ideal gas """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# None

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

def f_of_M(g,M):

    """  f = f_of_M(g,M): helper function that computes f(M) for a 1D ideal gas
    
         Inputs:    g = ratio of specific heats     (required)  (float)   
                    M = Mach number                 (required)  (floats)

         Outputs:   f = f(M) (see docs)                         (floats)

    """

    return 1 + ((g-1)/2)*M*M
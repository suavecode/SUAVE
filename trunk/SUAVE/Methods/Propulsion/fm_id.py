## @ingroup Methods-Propulsion
# fm_id.py
# 
# Created:  ### ####, SUAVE Team
# Modified: Feb 2016, E. Botero

# ----------------------------------------------------------------------
#  fm_id
# ----------------------------------------------------------------------

## @ingroup Methods-Propulsion
def fm_id(M):
    """
    Function that takes in the Mach number, and outputs a function fm
    that's commonly used in compressible flow calculations
    
    Inputs:
    M       [dimensionless]
    
    Outputs:
    fm
    
    """
    R  = 287.87
    g  = 1.4
    
    m0 = (g+1)/(2*(g-1))
    m1 = ((g+1)/2)**m0
    m2 = (1+(g-1)/2*M**2)**m0
    
    fm = m1*M/m2
    
    return fm
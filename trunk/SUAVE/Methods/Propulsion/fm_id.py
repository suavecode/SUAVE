## @ingroup Methods-Propulsion
# fm_id.py
#
# Created:  ### ####, SUAVE Team
# Modified: Feb 2016, E. Botero
#           Dec 2017, W. Maier
# ----------------------------------------------------------------------
#  fm_id
# ----------------------------------------------------------------------

## @ingroup Methods-Propulsion
def fm_id(M,gamma):
    """Function that takes in the Mach number and isentropic expansion factor,
    and outputs a value for f(M) that's commonly used in compressible flow
    calculations.

    Inputs:
    M       [dimensionless]
    gamma       [dimensionless]

    Outputs:
    fm
    """

    m0 = (gamma+1)/(2*(gamma-1))
    m1 = ((gamma+1)/2)**m0
    m2 = (1+(gamma-1)/2*M**2)**m0

    fm = m1*M/m2

    return fm

""" mach_from_area_ratio.py: get M from A/A* and gamma for a 1D ideal gas flow """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from f_of_M import f_of_M
from scipy.optimize import newton

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

def mach_from_area_ratio(g,area_ratio,branch="supersonic"):

    """  M = mach_from_area_ratio(g,area_ratio,branch="supersonic"): get M from A/A* and gamma for a 1D ideal gas flow
    
         Inputs:    g = ratio of specific heats                                     (required)                                  (float)    
                    area_ratio = A/A*                                               (required)                                  (float)
                    branch = "subsonic" or "supersonic", which solution to return   (required, default = "supersonic")          (string)

         Outputs:   M = Mach number                                                                                             (float)

        """

    if branch.lower() == "supersonic":
        M0 = 2.5
    elif branch.lower() == "subsonic":
        M0 = 0.5
    else:
        print "Error: branch must either be subsonic or supersonic"
        return 0.0

    n = (g+1)/(2*(g-1))
    c = ((g+1)/2)**n

    return newton(R, M0, fprime=dRdM, args=(g,area_ratio,c,n), tol=1.0e-08, maxiter=10000)

def R(M,g,area_ratio,c,n):

    return 1/area_ratio - c*M/f_of_M(g,M)**n

def dRdM(M,g,area_ratio,c,n):

    f = f_of_M(g,M)

    return -c*(-n*M*(g-1)*M/(f**(n+1)) + 1/f**n)
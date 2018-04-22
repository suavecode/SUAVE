## @ingroup Methods-Aerodynamics-Common-Gas_Dynamics
# oblique_shock.py
#
# Created:  Jan 2018, W. Maier
# Modified:  
#           

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np

# ----------------------------------------------------------------------
#  Oblique Shock Relations
# ----------------------------------------------------------------------

## @ingroup Methods-Aerodynamics-Common-Gas_Dynamics
def oblique_shock_relations(M0,gamma,theta,beta):
    """Computes flow quatities/ratios after undergoing 
    an oblique shock

    Assumptions:
    None

    Source:
    Chapter 12 of:
    https://web.stanford.edu/~cantwell/AA210A_Course_Material/AA210A_Course_Notes/

    https://arc.aiaa.org/doi/pdf/10.2514/2.2349
    
    Inputs:
    Mach, M0                             [-]
    Isentropic Expansion Factor, gamma   [-]
    Flow turn angle, theta               [rad]
    Shock Angle, beta                    [rad]

    Outputs:
    Mach, M1                             [-]
    Static Temperature Ratio, Tr         [-]
    Static Pressure Ratio, Pr            [-]
    Stagnation Pressure Ratio, Ptr       [-]
    
    Properties Used:
    N/A
    """

    # determine normal component of machs 
    M0_n = M0*np.sin(beta)
    M1_n = np.sqrt(((gamma-1.)*M0_n*M0_n+2.)/(2.*gamma*M0_n*M0_n-(gamma-1.)))
    
    # determine flow quantaties and ratios
    M1   = M1_n/np.sin(beta-theta)
    Pr   = (2.*gamma*M0_n*M0_n-(gamma-1.))/(gamma+1.)
    Tr   = Pr*((((gamma-1.)*M0_n*M0_n)+2.)/((gamma+1.)*M0_n*M0_n))
    Ptr  = ((((gamma+1.)*M0_n*M0_n)/((gamma-1.)*M0_n*M0_n+2))**(gamma/(gamma-1.)))*((gamma+1.)/(2.*gamma*M0_n*M0_n-(gamma-1.)))**(1./(gamma-1.))
    
    return M1,Pr,Tr,Ptr

def theta_beta_mach(M0,gamma,theta,n=0):
    """Computes shock angle of an oblique shock
    
        Assumptions:
        None
    
        Source:
        Chapter 12 of:
        https://web.stanford.edu/~cantwell/AA210A_Course_Material/AA210A_Course_Notes/
    
        Inputs:
        Mach, M0                             [-]
        Isentropic Expansion Factor, gamma   [-]
        Flow turn angle, theta               [rad]
        Strong Shock (0 = weak), delta       [-]
    
        Outputs:
        Shock Angle, Beta                    [rad]
        
        Properties Used:
        N/A
        """
    
    # Calculate wave angle
    mu   = np.arcsin(1./M0)
    c    = np.tan(mu)*np.tan(mu)
    
    # Calculate shock angle
    a    = ((gamma-1.)/2.+(gamma+1.)*c/2.)*np.tan(theta)
    b    = ((gamma+1.)/2.+(gamma+3.)*c/2.)*np.tan(theta)
    d    = np.sqrt(((4.*(1.-3.*a*b)**3.)/(27.*a*a*c+9.*a*b-2.)**2)-1.)    
    
    beta =np.arctan((b+9.*a*c)/(2.*(1.-3.*a*b))-(d*(27.*a**2.*c+9.*a*b-2))/(6.*a*(1.-3.*a*b))*np.tan(n*np.pi/3.+1./3.*np.arctan(1./d)))
    
    return beta
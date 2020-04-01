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

def get_invisc_press_recov(theta_r, M):
    """The throat inviscid total pressure recovery from the ramp angle and local mach number

    Assumptions:
    None

    Source:
    Appendix E of Nicolai's Fundamentals of Aircraft and Airship Design, Volume 1 – Aircraft Design

    
    Inputs:
    Ramp Angle, theta_r                                              [deg]
    Local Mach Number, M                                             [-]

    Outputs:
    Throat Inviscid Total Pressure Recovery, P_ratio_invis           [-]

    
    Properties Used:
    N/A
    """

    # Coefficients for the polynomial fits of polynomial fits
    c1_list = [0.00014406243134474503, -0.00402027976055483, 0.04997108211210769, -0.3641540374980741, 1.7221308514339708, -5.5198877047561865, 12.138552714211727, -18.075189970810253, 17.434838699326207, -9.833148027905457, 2.461594508027205]
    c2_list = [-0.0055604282744676865, 0.1554556560524434, -1.9358764676331295, 14.134280131336867, -66.97487006079216, 215.11380170359934, -474.0683987114353, 707.5320453091963, -684.1215258586642, 386.8426376649763, -97.1134804177477]
    c3_list = [0.06710627189319379, -1.8750228675835392, 23.33776147596419, -170.32732125312387, 806.8840287507015, -2591.361327364612, 5711.42833037374, -8526.82178094426, 8249.170339508777, -4668.126751998226, 1173.0152250284932]
    c4_list = [-0.03623016285315659, 1.1065604088229157, -14.941929141888329, 117.52156587518051, -596.4009759574767, 2040.7701521287534, -4768.554852377715, 7512.932136536898, -7637.931425765242, 4524.11039496336, -1184.4185922078136]

    # Get the coefficients for the specified mach number
    c1 = np.polyval(c1_list, M)
    c2 = np.polyval(c2_list, M)
    c3 = np.polyval(c3_list, M)
    c4 = np.polyval(c4_list, M)

    # Use coefficients on theta_r to get the pressure recovery
    fit = [c1, c2, c3, c4]
    P_ratio_invis = np.polyval(fit, theta_r)
    
    return P_ratio_invis
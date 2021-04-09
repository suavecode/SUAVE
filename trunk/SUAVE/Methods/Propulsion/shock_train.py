## @ingroup Methods-Propulsion
# shock_train.py
#
# Created:  Sep 2017, P Goncalves
# Modified: Jan 2018, W. Maier

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import numpy as np
from SUAVE.Methods.Aerodynamics.Common.Gas_Dynamics.Oblique_Shock import theta_beta_mach,oblique_shock_relations

# ----------------------------------------------------------------------
#  Shock Train Calculations
# ----------------------------------------------------------------------
## @ingroup Methods-Propulsion
def shock_train(M0, gamma, nbr_shocks, theta):
    """Function that takes in Mach,gamma, number of expected oblique , and 
    wedge angle of inlet and calculates the flow properties after under going said
    oblique shicks.  It verifies if all oblique shocks actually take place, depending
    on inlet condition.
    
    Assumptions:
    No shock interactions
    
    Inputs:
    M           [-]
    gamma       [-]
    nbr_shocks  [-]
    theta       [rad]
    
    Outputs:
    Tr          [-]
    Ptr         [-]
    
    Source:
    https://web.stanford.edu/~cantwell/AA210A_Course_Material/AA210A_Course_Notes/
    
    """
   
    # initializing count and ratios
    shock_count = 0
    T_ratio     = 1.0
    Pt_ratio    = 1.0
    P_ratio     = 1.0
    
    # undergo shocks 
    while shock_count<nbr_shocks:
    
        beta         = theta_beta_mach(M0,gamma,theta[shock_count])
        M1,Pr,Tr,Ptr = oblique_shock_relations(M0,gamma,theta[shock_count],beta)
        T_ratio      = T_ratio*Tr
        Pt_ratio     = Pt_ratio*Ptr
        P_ratio      = P_ratio*Pr
        M0           = M1
        shock_count  = shock_count+1;
        
    return T_ratio, Pt_ratio
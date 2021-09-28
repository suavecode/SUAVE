## @ingroup Methods-Power-Battery-Cell_Cycle_Models 
# LiNCA_cell_cycle_model.py
# 
# Created: Sep 2021, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import numpy as np  

# ----------------------------------------------------------------------
#  Compute NCA cell state variables
# ----------------------------------------------------------------------

## @ingroup Methods-Power-Battery-Cell_Cycle_Models 
def compute_NCA_cell_state_variables(battery_data,SOC,T):
    """This computes the electrical state variables of a lithium ion 
    battery cell with a  lithium-nickel-cobalt-aluminum oxide cathode 
    chemistry from look-up tables 
     
    Assumtions: 
    N/A
    
    Source:  
    N/A 
     
    Inputs:
        SOC           - state of charge of cell     [unitless]
        battery_data  - look-up data structure      [unitless]
        T             - battery cell temperature    [Kelvin]
    
    Outputs:  
        V_oc         - open-circuit voltage         [Volts]
        R_Th         - thevenin resistance          [Ohms]
        C_Th         - thevenin capacitance         [Coulombs]
        R_0          - phmic resistance             [Ohms]
        
    """
    V_oc = np.zeros_like(SOC)
    R_Th = np.zeros_like(SOC)
    C_Th = np.zeros_like(SOC)
    R_0  = np.zeros_like(SOC)
    SOC[SOC<0.] = 0.
    SOC[SOC>1.] = 1.
    for i in range(len(SOC)): 
        V_oc[i] = battery_data.V_oc_interp(T[i], SOC[i])[0]
        C_Th[i] = battery_data.C_Th_interp(T[i], SOC[i])[0]
        R_Th[i] = battery_data.R_Th_interp(T[i], SOC[i])[0]
        R_0[i]  = battery_data.R_0_interp(T[i], SOC[i])[0]  
    
    return V_oc,C_Th,R_Th,R_0
 
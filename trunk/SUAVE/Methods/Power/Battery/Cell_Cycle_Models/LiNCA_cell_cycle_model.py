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

    SOC[SOC<0.] = 0.
    SOC[SOC>1.] = 1.
        
    V_oc = battery_data.V_oc_interp(T, SOC,grid=False)    
    C_Th = battery_data.C_Th_interp(T, SOC,grid=False)  
    R_Th = battery_data.R_Th_interp(T, SOC,grid=False)   
    R_0  = battery_data.R_0_interp(T, SOC,grid=False)
    
    return V_oc,C_Th,R_Th,R_0
 
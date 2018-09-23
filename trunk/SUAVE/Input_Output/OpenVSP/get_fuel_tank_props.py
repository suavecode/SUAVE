## @ingroup Input_Output-OpenVSP
# write_vsp_fea.py
# 
# Created:  Mar 2018, T. MacDonald
# Modified: 

try:
    import vsp_g as vsp
except ImportError:
    pass # This allows SUAVE to build without OpenVSP
import numpy as np

## @ingroup Input_Output-OpenVSP
def get_fuel_tank_props(geometry,tag,fuel_tank_set_ind):
    """
    
    Assumptions:
    None

    Source:
    N/A

    Inputs:

    Outputs:                             

    Properties Used:
    N/A
    """      
    
    # Reset OpenVSP to avoid including a previous vehicle
    vsp.ClearVSPModel()    
    vsp.ReadVSPFile(tag + '.vsp3')
    
    num_slices = 100
    vsp.ComputeMassProps(fuel_tank_set_ind, num_slices)

    
    pass   
    
if __name__ == '__main__':
    tag = '/home/tim/Documents/SUAVE/regression/scripts/concorde/fuel_tank_test'
    import sys
    sys.path.append('/home/tim/Documents/SUAVE/regression/scripts/Vehicles')
    from Concorde import vehicle_setup, configs_setup
    vehicle = vehicle_setup()
    get_fuel_tank_props(vehicle,tag,3)
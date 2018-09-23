## @ingroup Input_Output-OpenVSP
# get_fuel_tank_props.py
# 
# Created:  Sep 2018, T. MacDonald
# Modified: 

try:
    import vsp_g as vsp
except ImportError:
    pass # This allows SUAVE to build without OpenVSP
import numpy as np
from SUAVE.Core import Data

## @ingroup Input_Output-OpenVSP
def get_fuel_tank_props(vehicle,tag,fuel_tank_set_ind):
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
    
    fuel_tanks = get_fuel_tank_tags(vehicle)
    
    num_slices = 100
    mass_props_output_file = tag + '_mass_props.txt'
    vsp.SetComputationFileName(vsp.MASS_PROP_TXT_TYPE,mass_props_output_file)
    print('Computing Fuel Tank Mass Properties... ',end='')
    vsp.ComputeMassProps(fuel_tank_set_ind, num_slices)
    print('Done')
    
    fo = open(mass_props_output_file)
    for line in fo:
        prop_list = line.split()
        try:
            if prop_list[0] in fuel_tanks:
                cg_x = float(prop_list[2])
                cg_y = float(prop_list[3])
                cg_z = float(prop_list[4])
                mass = float(prop_list[1])
                vol  = float(prop_list[-1])
                if 'center_of_gravity' not in fuel_tanks[prop_list[0]]: # assumes at most two identical tank names
                    fuel_tanks[prop_list[0]].center_of_gravity = np.array([[cg_x,cg_y,cg_z]])
                    fuel_tanks[prop_list[0]].full_fuel_mass    = mass
                    fuel_tanks[prop_list[0]].volume            = vol
                else:
                    fuel_tanks[prop_list[0]].center_of_gravity = \
                        (fuel_tanks[prop_list[0]].center_of_gravity+np.array([[cg_x,cg_y,cg_z]]))/2.
                    fuel_tanks[prop_list[0]].full_fuel_mass   += mass
                    fuel_tanks[prop_list[0]].volume           += vol                    
                    
        except IndexError:  # in case line is empty
            pass

    vehicle = apply_properties(vehicle, fuel_tanks)
    
    
    return vehicle

def apply_properties(vehicle,fuel_tanks):
    if 'wings' in vehicle:
        for wing in vehicle.wings:
            if 'Fuel_Tanks' in wing:
                for tank in wing.Fuel_Tanks:
                    tank.mass_properties.center_of_gravity = fuel_tanks[tank.tag].center_of_gravity
                    tank.mass_properties.full_fuel_mass    = fuel_tanks[tank.tag].full_fuel_mass
                    tank.mass_properties.full_fuel_volume  = fuel_tanks[tank.tag].volume
                    
    if 'fuselages' in vehicle:
        for fuse in vehicle.fuselages:
            if 'Fuel_Tanks' in fuse:
                for tank in fuse.Fuel_Tanks:
                    tank.mass_properties.center_of_gravity = fuel_tanks[tank.tag].center_of_gravity
                    tank.mass_properties.full_fuel_mass    = fuel_tanks[tank.tag].full_fuel_mass
                    tank.mass_properties.full_fuel_volume  = fuel_tanks[tank.tag].volume    
                    
    return vehicle
    

def get_fuel_tank_tags(vehicle):
    fuel_tanks = Data()
    
    if 'wings' in vehicle:
        for wing in vehicle.wings:
            if 'Fuel_Tanks' in wing:
                for tank in wing.Fuel_Tanks:
                    fuel_tanks[tank.tag] = Data()
                    
    if 'fuselages' in vehicle:
        for fuse in vehicle.fuselages:
            if 'Fuel_Tanks' in fuse:
                for tank in fuse.Fuel_Tanks:
                    fuel_tanks[tank.tag] = Data()
                    
    return fuel_tanks
    
if __name__ == '__main__':
    tag = '/home/tim/Documents/SUAVE/regression/scripts/concorde/fuel_tank_test'
    import sys
    sys.path.append('/home/tim/Documents/SUAVE/regression/scripts/Vehicles')
    from Concorde import vehicle_setup, configs_setup
    vehicle = vehicle_setup()
    get_fuel_tank_props(vehicle,tag,3)
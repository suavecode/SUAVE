## @ingroup Input_Output-OpenVSP
# get_fuel_tank_props.py
# 
# Created:  Sep 2018, T. MacDonald
# Modified: Oct 2018, T. MacDonald

try:
    import vsp_g as vsp
except ImportError:
    pass # This allows SUAVE to build without OpenVSP
import numpy as np
from SUAVE.Core import Data

## @ingroup Input_Output-OpenVSP
def get_fuel_tank_props(vehicle,tag,fuel_tank_set_index=3):
    """This function computes the center of gravity, total possible fuel mass,
    the available volume of each fuel tank in the vehicle through a mass
    properties computation in OpenVSP.
    
    Assumptions:
    Fuel tanks exists in the fuselage and wings only
    All fuel tanks have unique names

    Source:
    N/A

    Inputs:
    vehicle.fuselages.*.Fuel_Tanks.*.tag     [-]
    vehicle.wings.*.Fuel_Tanks.*.tag         [-]

    Outputs:    
    vehicle.fuselages.*.Fuel_Tanks.*.mass_properties.
      center_of_gravity                      [m]
      full_fuel_mass                         [kg]
      full_fuel_volume                       [m^3]
    vehicle.wings.*.Fuel_Tanks.*.mass_properties.
      center_of_gravity                      [m]
      full_fuel_mass                         [kg]
      full_fuel_volume                       [m^3]
      

    Properties Used:
    N/A
    """      
    
    # Reset OpenVSP to avoid including a previous vehicle
    vsp.ClearVSPModel()    
    vsp.ReadVSPFile(tag + '.vsp3')
    
    # Extract fuel tanks from vehicle
    fuel_tanks = get_fuel_tank_tags(vehicle)
    
    num_slices = 100 # Slices used to estimate mass distribution from areas in OpenVSP
    mass_props_output_file = tag + '_mass_props.txt'
    vsp.SetComputationFileName(vsp.MASS_PROP_TXT_TYPE,mass_props_output_file)
    print('Computing Fuel Tank Mass Properties... ')
    vsp.ComputeMassProps(fuel_tank_set_index, num_slices)
    print('Done')
    
    # Extract full tank mass properties from OpenVSP output file
    fo = open(mass_props_output_file)
    for line in fo:
        prop_list = line.split()
        try:
            if prop_list[0] in fuel_tanks:
                # Indices based on position in OpenVSP output (may change in the future)
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

    # Apply fuel tank properties to the vehicle
    vehicle = apply_properties(vehicle, fuel_tanks)
    
    
    return vehicle

## @ingroup Input_Output-OpenVSP
def apply_properties(vehicle,fuel_tanks):
    """Apply fuel tank properties from OpenVSP to the SUAVE vehicle.
    
    Assumptions:
    Fuel tanks exists in the fuselage and wings only

    Source:
    N/A

    Inputs:
    vehicle.fuselages.*.Fuel_Tanks.*.tag     [-]
    vehicle.wings.*.Fuel_Tanks.*.tag         [-]
    fuel_tanks.
      tag                                    [-]
      center_of_gravity                      [m]
      full_fuel_mass                         [kg]
      volume                                 [m^3]
      

    Outputs:    
    vehicle.fuselages.*.Fuel_Tanks.*.mass_properties.
      center_of_gravity                      [m]
      full_fuel_mass                         [kg]
      full_fuel_volume                       [m^3]
    vehicle.wings.*.Fuel_Tanks.*.mass_properties.
      center_of_gravity                      [m]
      full_fuel_mass                         [kg]
      full_fuel_volume                       [m^3]
      

    Properties Used:
    N/A
    """      
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
    
## @ingroup Input_Output-OpenVSP
def get_fuel_tank_tags(vehicle):
    """Creates a data structure with fuel tanks based on 
    fuel tanks present in the vehicle
    
    Assumptions:
    Fuel tanks exists in the fuselage and wings only

    Source:
    N/A

    Inputs:
    vehicle.fuselages.*.Fuel_Tanks.*.tag     [-]
    vehicle.wings.*.Fuel_Tanks.*.tag         [-]

    Outputs:    
    fuel_tanks.tag                           [-]

    Properties Used:
    N/A
    """       
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
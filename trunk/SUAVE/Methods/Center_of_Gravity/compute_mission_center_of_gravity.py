
#computes the CG for the vehicle based on the mzfw cg of the vehicle, and an assigned fuel
#Created:M. Vegh Nov. 2015



def compute_mission_center_of_gravity(vehicle, mission_fuel_weight):
    #computes the aircraft center of gravity from the vehicle max zero fuel weight and an assigned fuel weight
    
    
    mzf_cg    =vehicle.mass_properties.zero_fuel_center_of_gravity
    mzf_weight=vehicle.mass_properties.max_zero_fuel
    fuel      =vehicle.fuel
    fuel_cg   =vehicle.fuel.mass_properties.center_of_gravity
    cg        =((mzf_cg)*mzf_weight+(fuel_cg+fuel.origin)*mission_fuel_weight)/(mission_fuel_weight+mzf_weight)

   
    return cg

## @ingroup Methods-Weights-Buildups-eVTOL
# converge_evtol_weight.py

# Created: Aug 2022, M. Clarke

#-------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------- 
from MARC.Methods.Weights.Buildups.eVTOL.empty import empty
from MARC.Core import Data

#-------------------------------------------------------------------------------
# Empty
#-------------------------------------------------------------------------------

## @ingroup Methods-Weights-Buildups-eVTOL 
def converge_evtol_weight(vehicle, 
                          settings,                        
                          print_iterations              = False,
                          contingency_factor            = 1.1,
                          speed_of_sound                = 340.294,
                          max_tip_mach                  = 0.65,
                          disk_area_factor              = 1.15,
                          safety_factor                 = 1.5,
                          max_thrust_to_weight_ratio    = 1.1,
                          max_g_load                    = 3.8,
                          motor_efficiency              = 0.85 * 0.98):
    '''Converges the maximum takeoff weight of an aircraft using the eVTOL 
    weight buildup routine.  
    
    Source: 
    Vehicle ompoments i.e. rotors and motors are not resized 
    
    Assumptions:
    None
    
    Inputs:
    vehicle                     MARC Config Data Stucture
    print_iterations            Boolean Flag      
    contingency_factor          Factor capturing uncertainty in vehicle weight [Unitless]
    speed_of_sound:             Local Speed of Sound                           [m/s]
    max_tip_mach:               Allowable Tip Mach Number                      [Unitless]
    disk_area_factor:           Inverse of Disk Area Efficiency                [Unitless]
    max_thrust_to_weight_ratio: Allowable Thrust to Weight Ratio               [Unitless]
    safety_factor               Safety Factor in vehicle design                [Unitless]
    max_g_load                  Maximum g-forces load for certification        [UNitless]
    motor_efficiency:           Motor Efficiency                               [Unitless]
    
    Outputs:
    None
    
    Properties Used:
    N/A
    '''
    breakdown      = empty(vehicle,settings,contingency_factor,
                           speed_of_sound,max_tip_mach,disk_area_factor,
                           safety_factor,max_thrust_to_weight_ratio,
                           max_g_load,motor_efficiency) 
    build_up_mass  = breakdown.total    
    diff           = vehicle.mass_properties.max_takeoff - build_up_mass
    iterations     = 0
    
    while(abs(diff)>1):
        vehicle.mass_properties.max_takeoff = vehicle.mass_properties.max_takeoff - diff
        
        # Note that 'diff' will be negative if buildup mass is larger than MTOW, so subtractive
        # iteration always moves MTOW toward buildup mass
        
        breakdown      = empty(vehicle,settings,contingency_factor,
                           speed_of_sound,max_tip_mach,disk_area_factor,
                           safety_factor,max_thrust_to_weight_ratio,
                           max_g_load,motor_efficiency)
        build_up_mass  = breakdown.total    
        diff           = vehicle.mass_properties.max_takeoff - build_up_mass 
        iterations     += 1
        if print_iterations:
            print(round(diff,3))
        if iterations == 100:
            print('Weight convergence failed!')
            return False  
    
    return True 

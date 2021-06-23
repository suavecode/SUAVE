## @ingroup Time_Accurate
# advance_vehicle_in_time.py
# 
# Created:  Jun 2021, R. Erhard


## @ingroup Time_Accurate
def advance_vehicle_in_time(vehicle, dx, t0):
    """ This advances the vehicle in time, adjusting for translation
    and rotation.

    Assumptions:  
       None
    
    Source:  
       N/A
    
    Inputs: 
       vehicle - SUAVE vehicle
       dx      - helical wake distribution points               [Unitless] 
       t0      - vortex distribution points on lifting surfaces [Unitless] 

    """     
    
    # Translate the vehicle forward in time
    translate_vehicle(vehicle, dx, t0, tiltwing=False)
    
    # Translate the propeller wake in time
    translate_wake(vehicle, dx, t0)
    
    # Rotate the propeller forward in time
    rotate_propellers(vehicle,dx,t0)
    
    return


def translate_wake(vehicle, dx,t0):
    #-------------------------------------------
    # Translation
    #-------------------------------------------
    # move all wake vertices forward in space
    vehicle.vortex_distribution.Wake.XA1 = vehicle.vortex_distribution.Wake.XA1 - dx
    vehicle.vortex_distribution.Wake.XA2 = vehicle.vortex_distribution.Wake.XA2 - dx
    vehicle.vortex_distribution.Wake.XB1 = vehicle.vortex_distribution.Wake.XB1 - dx
    vehicle.vortex_distribution.Wake.XB2 = vehicle.vortex_distribution.Wake.XB2 - dx    
  
    return



def translate_vehicle(vehicle, dx, t0, tiltwing=False):
    #-------------------------------------------
    # Translation
    #-------------------------------------------
    if len(vehicle.wings) !=0:
        # move all wing vertices forward in space
        vehicle.vortex_distribution.XA1 = vehicle.vortex_distribution.XA1 - dx
        vehicle.vortex_distribution.XA2 = vehicle.vortex_distribution.XA2 - dx
        vehicle.vortex_distribution.XB1 = vehicle.vortex_distribution.XB1 - dx
        vehicle.vortex_distribution.XB2 = vehicle.vortex_distribution.XB2 - dx 
    
    # move propeller location forward in space
    propeller = vehicle.propulsors.battery_propeller.propeller
    n_propellers = int(vehicle.propulsors.battery_propeller.number_of_engines)
    
    for i in range(n_propellers):
        propeller.origin[i][0] = propeller.origin[i][0] - dx
    
    
    return


def rotate_propellers(vehicle,dx,t0):
    #-------------------------------------------
    # Propeller rotation
    #------------------------------------------- 
    propeller = vehicle.propulsors.battery_propeller.propeller
    
    # Update propeller offset
    propeller.azimuthal_offset = propeller.inputs.omega[0][0]*t0  
    
    return 
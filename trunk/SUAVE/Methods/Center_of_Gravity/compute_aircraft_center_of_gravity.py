#computes the CG for the vehicle from the assigned vehicle mass properties and locations
#Created:M. Vegh Oct. 2015

import numpy as np
import copy



def compute_aircraft_center_of_gravity(vehicle, nose_load_fraction=.06):
    
    #unpack components
    wing= vehicle.wings['main_wing']
    h_tail=vehicle.wings['horizontal_stabilizer']
    v_tail=vehicle.wings['vertical_stabilizer']
    control_systems=vehicle.control_systems
    fuselage=vehicle.fuselages['fuselage']
    landing_gear=vehicle.landing_gear
    turbo_fan=vehicle.propulsors['turbo_fan']
    electrical_systems=vehicle.electrical_systems
    avionics=vehicle.avionics
    furnishings=vehicle.furnishings
    passenger_weights=vehicle.passenger_weights
    air_conditioner=vehicle.air_conditioner
    fuel=vehicle.fuel
    apu=vehicle.apu
    hydraulics=vehicle.hydraulics
    optionals=vehicle.optionals
    
    #compute moments from each component about the nose of the aircraft
    wing_moment              =(wing.origin+wing.mass_properties.center_of_gravity)*wing.mass_properties.mass
    h_tail_moment            =(h_tail.origin+h_tail.mass_properties.center_of_gravity)*h_tail.mass_properties.mass
    v_tail_moment            =(v_tail.origin+v_tail.mass_properties.center_of_gravity)*(v_tail.mass_properties.mass+v_tail.rudder.mass_properties.mass)
    control_systems_moment   =(control_systems.origin+control_systems.mass_properties.center_of_gravity)*control_systems.mass_properties.mass
    fuselage_moment          =(fuselage.origin+fuselage.mass_properties.center_of_gravity)*fuselage.mass_properties.mass
    turbo_fan_moment         =(turbo_fan.origin+turbo_fan.mass_properties.center_of_gravity)*turbo_fan.mass_properties.mass
    electrical_systems_moment=(electrical_systems.origin+electrical_systems.mass_properties.center_of_gravity)*electrical_systems.mass_properties.mass
    avionics_moment          =(avionics.origin+avionics.mass_properties.center_of_gravity)*avionics.mass_properties.mass
    furnishings_moment       =(furnishings.origin+furnishings.mass_properties.center_of_gravity)*furnishings.mass_properties.mass
    passengers_moment        =(passenger_weights.origin+passenger_weights.mass_properties.center_of_gravity)*passenger_weights.mass_properties.mass
    ac_moment                =(air_conditioner.origin+air_conditioner.mass_properties.center_of_gravity)*air_conditioner.mass_properties.mass
    fuel_moment              =(fuel.origin+fuel.mass_properties.center_of_gravity)*fuel.mass_properties.mass
    apu_moment               =(apu.origin+apu.mass_properties.center_of_gravity)*apu.mass_properties.mass
    hydraulics_moment        =(hydraulics.origin+hydraulics.mass_properties.center_of_gravity)*hydraulics.mass_properties.mass
    optionals_moment         =(optionals.origin+optionals.mass_properties.center_of_gravity)*optionals.mass_properties.mass
    
    #optionals_moment         =np.array([fuselage.lengths.total*.51,0,0])*optionals.mass_properties.mass 
    
    
    center_of_gravity_guess=[0,0,0]
    error =1
    aft_gear_location=np.zeros(3) #guess
    
    landing_gear.origin=copy.copy(fuselage.origin)#front gear location
    landing_gear.origin[0]=fuselage.origin[0]+fuselage.lengths.nose
    count=0 #don't allow this iteration to break the code
    
   
    while np.abs(error)>.01 and count<10:
        landing_gear.mass_properties.center_of_gravity=aft_gear_location*2./3.
        landing_gear_moment=(landing_gear.origin+landing_gear.mass_properties.center_of_gravity)*landing_gear.mass_properties.mass
        vehicle.mass_properties.center_of_gravity=(wing_moment+h_tail_moment+v_tail_moment+control_systems_moment+\
        fuselage_moment+turbo_fan_moment+electrical_systems_moment+avionics_moment+furnishings_moment+passengers_moment+\
        ac_moment+fuel_moment+apu_moment+landing_gear_moment+ hydraulics_moment+optionals_moment  )/(vehicle.mass_properties.max_takeoff)
        
        aft_gear_location[0]=(vehicle.mass_properties.center_of_gravity[0]-landing_gear.origin[0])*\
        nose_load_fraction/(1-nose_load_fraction)+vehicle.mass_properties.center_of_gravity[0]-landing_gear.origin[0]
        error=(center_of_gravity_guess[0]-vehicle.mass_properties.center_of_gravity[0])/((center_of_gravity_guess[0]+vehicle.mass_properties.center_of_gravity[0])/2.)
        center_of_gravity_guess=copy.copy(vehicle.mass_properties.center_of_gravity)
        count+=1 
    vehicle.mass_properties.center_of_gravity[1]=0 #symmetric aircraft
    
    
    return vehicle.mass_properties.center_of_gravity
#computes the CG for the vehicle from the assigned vehicle mass properties and locations
#Created:M. Vegh Oct. 2015

import numpy as np
import copy



def compute_aircraft_center_of_gravity(vehicle, nose_load_fraction=.06):
    
    #unpack components
    wing               = vehicle.wings['main_wing']
    h_tail             =vehicle.wings['horizontal_stabilizer']
    v_tail             =vehicle.wings['vertical_stabilizer']
    control_systems    =vehicle.control_systems
    fuselage           =vehicle.fuselages['fuselage']
    landing_gear       =vehicle.landing_gear
    turbo_fan          =vehicle.propulsors['turbo_fan']
    electrical_systems=vehicle.electrical_systems
    avionics           =vehicle.avionics
    furnishings        =vehicle.furnishings
    passenger_weights  =vehicle.passenger_weights
    air_conditioner    =vehicle.air_conditioner
    fuel               =vehicle.fuel
    apu                =vehicle.apu
    hydraulics         =vehicle.hydraulics
    optionals          =vehicle.optionals
    
    #compute moments from each component about the nose of the aircraft
    wing_cg                  =wing.mass_properties.center_of_gravity
    wing_moment              =(wing.origin+wing_cg)*wing.mass_properties.mass
    
    h_tail_cg                =h_tail.mass_properties.center_of_gravity
    h_tail_moment            =(h_tail.origin+h_tail_cg)*h_tail.mass_properties.mass
    
    v_tail_cg                 =v_tail.mass_properties.center_of_gravity
    v_tail_moment            =(v_tail.origin+ v_tail_cg)*(v_tail.mass_properties.mass+v_tail.rudder.mass_properties.mass)
    
    control_systems_cg       =control_systems.mass_properties.center_of_gravity
    control_systems_moment   =(control_systems.origin+control_systems_cg )*control_systems.mass_properties.mass
    
    fuselage_cg              =fuselage.mass_properties.center_of_gravity
    fuselage_moment          =(fuselage.origin+fuselage_cg)*fuselage.mass_properties.mass
    
    
    turbo_fan_cg             =turbo_fan.mass_properties.center_of_gravity
    turbo_fan_moment         =(turbo_fan.origin+turbo_fan_cg)*turbo_fan.mass_properties.mass
    
    electrical_systems_cg    =electrical_systems.mass_properties.center_of_gravity
    electrical_systems_moment=(electrical_systems.origin+electrical_systems_cg)*electrical_systems.mass_properties.mass
    
    avionics_cg              =avionics.mass_properties.center_of_gravity
    avionics_moment          =(avionics.origin+avionics_cg)*avionics.mass_properties.mass
    
    furnishings_cg           =furnishings.mass_properties.center_of_gravity
    furnishings_moment       =(furnishings.origin+furnishings_cg )*furnishings.mass_properties.mass
    
    passengers_cg            =passenger_weights.mass_properties.center_of_gravity
    passengers_moment        =(passenger_weights.origin+passengers_cg)*passenger_weights.mass_properties.mass
    
    ac_cg                    =air_conditioner.mass_properties.center_of_gravity
    ac_moment                =(air_conditioner.origin+ac_cg)*air_conditioner.mass_properties.mass
    
    fuel_cg                  =fuel.mass_properties.center_of_gravity
    fuel_moment              =(fuel.origin+fuel_cg)*fuel.mass_properties.mass
    
    apu_cg                   =apu.mass_properties.center_of_gravity
    apu_moment               =(apu.origin+apu_cg)*apu.mass_properties.mass
    
    hydraulics_cg            =hydraulics.mass_properties.center_of_gravity
    hydraulics_moment        =(hydraulics.origin+hydraulics_cg)*hydraulics.mass_properties.mass
    
    optionals_cg             =optionals.mass_properties.center_of_gravity
    optionals_moment         =(optionals.origin+optionals_cg)*optionals.mass_properties.mass
    
    #optionals_moment         =np.array([fuselage.lengths.total*.51,0,0])*optionals.mass_properties.mass 
    
    
    
    landing_gear.origin      =1*(fuselage.origin)#front gear location
    landing_gear.origin[0]   =fuselage.origin[0]+fuselage.lengths.nose
    
    
#find moment of every object other than landing gear to find aft gear location, then cg
    sum_moments              =(wing_moment+h_tail_moment+v_tail_moment+control_systems_moment+\
        fuselage_moment+turbo_fan_moment+electrical_systems_moment+\
        avionics_moment+furnishings_moment+passengers_moment+ac_moment+\
        fuel_moment+apu_moment+ hydraulics_moment+optionals_moment  )
            
    #took some algebra to get this
    aft_gear_location                             =sum_moments/(vehicle.mass_properties.takeoff+landing_gear.mass_properties.mass/(1-nose_load_fraction))
    aft_gear_fraction                             =.9  #assume that nose landing gear is 10% of landing gear weight (rough estimate based on data in Roskam, Airplaine Design:Part V:Component Weight Estimation)
    landing_gear_cg                               =aft_gear_location*aft_gear_fraction
    
    landing_gear.mass_properties.center_of_gravity=landing_gear_cg      
    
    landing_gear_moment                           =(landing_gear.origin+landing_gear_cg)*landing_gear.mass_properties.mass
   
    vehicle.mass_properties.center_of_gravity     =(sum_moments+landing_gear_moment)/vehicle.mass_properties.max_takeoff

    vehicle.mass_properties.center_of_gravity[1]  =0 #symmetric aircraft


    sum_moments_less_fuel   = sum_moments-fuel_moment
    vehicle.mass_properties.zero_fuel_center_of_gravity=(sum_moments_less_fuel+landing_gear_moment)/vehicle.mass_properties.max_zero_fuel

    return vehicle.mass_properties.center_of_gravity 

## @ingroup Methods-Center_of_Gravity
#compute_aircraft_center_of_gravity.py
# 
# Created:  Oct 2015, M. Vegh
# Modified: Jan 2016, E. Botero
# Modified: May 2021, M. Clarke 


# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np

# ----------------------------------------------------------------------
#  Computer Aircraft Center of Gravity
# ----------------------------------------------------------------------

## @ingroup Methods-Center_of_Gravity
def compute_aircraft_center_of_gravity(vehicle, nose_load_fraction=.06):
    """ This computes the CG for the vehicle from the assigned vehicle mass 
    properties and locations

    Assumptions:
    None

    Source:
    N/A

    Inputs:
    vehicle
    nose_load_fraction

    Outputs:
    vehicle.mass_properties.center_of_gravity      [meters]

    Properties Used:
    N/A
    """   
    # Wing Moments 
    wing_moments      = 0.
    propulsor_moments = 0.
    bat_moment        = 0.
    fuel_moment       = 0.
    reference_mass    = 0 
    
    for wing in vehicle.wings: 
        wing_cg         = wing.mass_properties.center_of_gravity
        wing_moments   += (wing.origin+wing_cg)*wing.mass_properties.mass 
        reference_mass += wing.mass_properties.mass 
    # Propulsor Moments   
    for propulsor in vehicle.propulsors:  
        for i in range(len(propulsor.origin)):
            propulsor_cg      = propulsor.mass_properties.center_of_gravity 
            propulsor_moments+= (propulsor.origin[i]+propulsor_cg)*propulsor.mass_properties.mass  
            reference_mass   += propulsor.mass_properties.mass  
            
        if 'rotor' in propulsor.keys(): 
            for j in range(len(propulsor.rotor.origin)):
                rotor_cg          = propulsor.rotor.mass_properties.center_of_gravity 
                propulsor_moments+= (propulsor.rotor.origin[j]+rotor_cg)*propulsor.rotor.mass_properties.mass  
                reference_mass   += propulsor.rotor.mass_properties.mass  
                
        if 'propeller' in propulsor.keys():    
            for j in range(len(propulsor.propeller.origin)):
                propeller_cg      = propulsor.propeller.mass_properties.center_of_gravity 
                propulsor_moments+= (propulsor.propeller.origin[j]+propeller_cg)*propulsor.propeller.mass_properties.mass   
                reference_mass   += propulsor.propeller.mass_properties.mass   
                
        if 'battery' in propulsor.keys():   
            bat_cg      = propulsor.battery.mass_properties.center_of_gravity
            bat_moment += (propulsor.battery.origin + bat_cg) * propulsor.battery.mass_properties.mass  
            
    # ---------------------------------------------------------------------------------
    # configurations with fuselages (BWB, Tube and Wing)  
    # ---------------------------------------------------------------------------------
    if vehicle.fuselages.keys() != []:  
        landing_gear              = vehicle.landing_gear.main 
        electrical_systems        = vehicle.systems.electrical_systems
        avionics                  = vehicle.systems.avionics
        furnishings               = vehicle.systems.furnishings
        passengers                = vehicle.payload.passengers 
        baggage                   = vehicle.payload.baggage  
        cargo                     = vehicle.payload.cargo 
        air_conditioner           = vehicle.systems.air_conditioner
        apu                       = vehicle.systems.apu
        hydraulics                = vehicle.systems.hydraulics
        optionals                 = vehicle.systems.optionals     
        control_systems           = vehicle.systems.control_systems 

        # Control Sytems               
        control_systems_cg        = control_systems.mass_properties.center_of_gravity
        control_systems_moment    = (control_systems.origin+control_systems_cg )*control_systems.mass_properties.mass

        # Fuselage & Energy Store
        fuse_key                  = list(vehicle.fuselages.keys())[0] #['fuselage']
        fuselage                  = vehicle.fuselages[fuse_key]

        fuselage_cg               = fuselage.mass_properties.center_of_gravity
        fuselage_moment           = (fuselage.origin+fuselage_cg)*fuselage.mass_properties.mass
         
        if list(fuselage.Fuel_Tanks.keys()) != []:
            fuel         = vehicle.fuel
            fuel_cg      = fuel.mass_properties.center_of_gravity
            fuel_moment += (fuel.origin + fuel_cg) * fuel.mass_properties.mass  
            
        # Furnishings
        furnishings_cg            = furnishings.mass_properties.center_of_gravity
        furnishings_moment        = (furnishings.origin+furnishings_cg )*furnishings.mass_properties.mass

        # Passengers
        passengers_cg             = passengers.mass_properties.center_of_gravity
        passengers_moment         = (passengers.origin+passengers_cg)*passengers.mass_properties.mass
        reference_mass            += passengers.mass_properties.mass
        
        # Baggage
        baggage_cg                = baggage.mass_properties.center_of_gravity
        baggage_moment            = (baggage.origin+baggage_cg)*baggage.mass_properties.mass
        
        # Cargo
        cargo_cg                  = cargo.mass_properties.center_of_gravity
        cargo_moment              = (cargo.origin+cargo_cg)*cargo.mass_properties.mass
        
        # Air conditioning
        ac_cg                     = air_conditioner.mass_properties.center_of_gravity
        ac_moment                 = (air_conditioner.origin+ac_cg)*air_conditioner.mass_properties.mass

        # APU
        apu_cg                    = apu.mass_properties.center_of_gravity
        apu_moment                = (apu.origin+apu_cg)*apu.mass_properties.mass

        # Optionals
        optionals_cg              = optionals.mass_properties.center_of_gravity
        optionals_moment          = (optionals.origin+optionals_cg)*optionals.mass_properties.mass


        # Electrical system
        electrical_systems_cg     = electrical_systems.mass_properties.center_of_gravity
        electrical_systems_moment = (electrical_systems.origin+electrical_systems_cg)*electrical_systems.mass_properties.mass

        # Hydraulics
        hydraulics_cg             = hydraulics.mass_properties.center_of_gravity
        hydraulics_moment         = (hydraulics.origin+hydraulics_cg)*hydraulics.mass_properties.mass

        # Avionics
        avionics_cg               = avionics.mass_properties.center_of_gravity
        avionics_moment           = (avionics.origin+avionics_cg)*avionics.mass_properties.mass

        # Landing Gear
        landing_gear.origin       = 1*(fuselage.origin)  
        landing_gear.origin[0][0] = fuselage.origin[0][0]+fuselage.lengths.nose 

        #find moment of every object other than landing gear to find aft gear location, then cg
        sum_moments               = (wing_moments+control_systems_moment+baggage_moment+cargo_moment+\
                                     fuselage_moment+propulsor_moments+electrical_systems_moment+\
                                     avionics_moment+furnishings_moment+passengers_moment+ac_moment+\
                                     apu_moment+ hydraulics_moment+optionals_moment+fuel_moment+bat_moment) 
 

        # assume that nose landing gear is 10% of landing gear weight (rough estimate based on data in Roskam,Airplaine Design:Part V:Component Weight Estimation)
        aft_gear_location                              = sum_moments /(reference_mass+landing_gear.mass_properties.mass/(1-nose_load_fraction)) 
        aft_gear_fraction                              = .9  
        landing_gear_cg                                = aft_gear_location*aft_gear_fraction
        landing_gear.mass_properties.center_of_gravity = landing_gear_cg      
        landing_gear_moment                            = (landing_gear.origin+landing_gear_cg)*landing_gear.mass_properties.mass 
        vehicle.mass_properties.center_of_gravity      = (sum_moments+landing_gear_moment)/reference_mass
        vehicle.mass_properties.center_of_gravity[0,1] = 0 # symmetric aircraft

        if list(fuselage.Fuel_Tanks.keys()) != []: 
            sum_moments_less_fuel                               = sum_moments-fuel_moment 
            vehicle.mass_properties.zero_fuel_center_of_gravity = (sum_moments_less_fuel+landing_gear_moment)/vehicle.mass_properties.max_zero_fuel

    # ---------------------------------------------------------------------------------        
    # Electric UAV Configurations without Fuselages/Landing Gear/Fuel
    # ---------------------------------------------------------------------------------
    else:    
        sum_moments                                             = (wing_moments+ propulsor_moments) 
        vehicle.mass_properties.center_of_gravity               = (sum_moments)/vehicle.mass_properties.max_takeoff
        vehicle.mass_properties.zero_fuel_center_of_gravity     = vehicle.mass_properties.center_of_gravity
        
    return vehicle.mass_properties.center_of_gravity


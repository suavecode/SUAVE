## @ingroup Methods-Center_of_Gravity
# compute_component_centers_of_gravity.py
#
# Created:  Oct 2015, M. Vegh
# Modified: Jan 2016, E. Botero
# Mofified: Jun 2017, M. Clarke
#           Apr 2020, M. Clarke
#           May 2020, E. Botero


# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
from SUAVE.Methods.Geometry.Three_Dimensional.compute_span_location_from_chord_length import compute_span_location_from_chord_length
from SUAVE.Methods.Geometry.Three_Dimensional.compute_chord_length_from_span_location import compute_chord_length_from_span_location
from SUAVE.Components.Fuselages import Elliptical_Fuselage
from SUAVE.Methods.Flight_Dynamics.Static_Stability.Approximations.Supporting_Functions.convert_sweep import convert_sweep

import SUAVE

# ----------------------------------------------------------------------
#  Computer Aircraft Center of Gravity
# ----------------------------------------------------------------------

## @ingroup Methods-Center_of_Gravity
def compute_component_centers_of_gravity(vehicle, nose_load = 0.06):
    """ computes the CG of all of the vehicle components based on correlations 
    from AA241

    Assumptions:
    None

    Source:
    AA 241 Notes

    Inputs:
    vehicle

    Outputs:
    None

    Properties Used:
    N/A
    """  
    
    C =  SUAVE.Components
    
    # Go through all wings
    for wing in vehicle.wings:
    
        if wing.sweeps.leading_edge == None:
            wing.sweeps.leading_edge = convert_sweep(wing,old_ref_chord_fraction = 0.25 ,new_ref_chord_fraction = 0.0)
        
        if isinstance(wing,C.Wings.Main_Wing):
                wing.mass_properties.center_of_gravity[0][0] = .05*wing.chords.mean_aerodynamic +wing.aerodynamic_center[0]             
                
            
        elif isinstance(wing,C.Wings.Horizontal_Tail):
            chord_length_h_tail_35_percent_semi_span  = compute_chord_length_from_span_location(wing,.35*wing.spans.projected*.5)
            h_tail_35_percent_semi_span_offset        = np.tan(wing.sweeps.quarter_chord)*.35*.5*wing.spans.projected   
            wing.mass_properties.center_of_gravity[0][0] = .3*chord_length_h_tail_35_percent_semi_span + \
                                                                          h_tail_35_percent_semi_span_offset            

    # computes the CG of propulsors. If origin not specified in vehicle set up, change compute_propulsor_origin boolean to True
    propulsor_name                                              = list(vehicle.propulsors.keys())[0]
    propulsor                                                   = vehicle.propulsors[propulsor_name]   
    
    if compute_propulsor_origin == True:
        propulsor.origin = [[0,0,0]]
        propulsor.origin[0][0] = wing.origin[0] + mac_le_offset/2.-(3./4.)*propulsor.engine_length
        propulsor.origin[0][1] = 0.
        propulsor.origin[0][2] = 0.
        
    propulsor.mass_properties.center_of_gravity[0]              = propulsor.engine_length*.5
 
   
    # ---------------------------------------------------------------------------------
    # configurations with fuselages (BWB, Tube and Wing)  
    # ---------------------------------------------------------------------------------
    if vehicle.fuselages.keys() != []:

        fuse_key                                                = list(vehicle.fuselages.keys())[0]
        fuselage                                                = vehicle.fuselages[fuse_key]

        if fuselage.Fuel_Tanks.keys() != []:
            fuel                                                = vehicle.fuel
            fuel.origin                                         = wing.origin
            fuel.mass_properties.center_of_gravity              = wing.mass_properties.center_of_gravity

        if fuselage.Batteries.keys()!= []:
            bat_key                                             = list(fuselage.Batteries.keys())[0]
            battery                                             = fuselage.Batteries[bat_key]

        control_systems                                         = vehicle.control_systems
        control_systems.origin                                  = wing.origin
        control_systems.mass_properties.center_of_gravity[0]    = .4*wing.chords.mean_aerodynamic+mac_le_offset
        electrical_systems                                      = vehicle.electrical_systems
        landing_gear                                            = vehicle.landing_gear
        avionics                                                = vehicle.avionics
        furnishings                                             = vehicle.furnishings
        passenger_weights                                       = vehicle.passenger_weights
        air_conditioner                                         = vehicle.air_conditioner
        apu                                                     = vehicle.apu
        hydraulics                                              = vehicle.hydraulics
        optionals                                               = vehicle.optionals

        if isinstance(fuselage, Elliptical_Fuselage):
            # Assumptions reflected here are unsourced

            fuselage.mass_properties.center_of_gravity[0]           = 0.375 * fuselage.lengths.total
            avionics.origin                                         = fuselage.origin
            avionics.mass_properties.center_of_gravity[0]           = 0.1 * fuselage.lengths.total
            electrical_systems.origin                               = battery.origin
            electrical_systems.mass_properties.center_of_gravity[0] = battery.mass_properties.center_of_gravity[0]
            air_conditioner.origin                                  = fuselage.origin
            air_conditioner.mass_properties.center_of_gravity[0]    = 0.125 * fuselage.lengths.total

        else:
            fuselage.mass_properties.center_of_gravity[0]           = .45*fuselage.lengths.total
            avionics.origin                                         = fuselage.origin
            avionics.mass_properties.center_of_gravity[0]           = .4 * fuselage.lengths.nose
            electrical_systems.mass_properties.center_of_gravity[0] = .75*(fuselage.origin[0][0]+  .5*fuselage.lengths.total)+.25*(propulsor.origin[0][0]+propulsor.mass_properties.center_of_gravity[0])
            air_conditioner.origin = fuselage.origin
            air_conditioner.mass_properties.center_of_gravity[0] = fuselage.lengths.nose
            hydraulics.origin = fuselage.origin
            hydraulics.mass_properties.center_of_gravity = .75 * (wing.origin + wing.mass_properties.center_of_gravity) + .25 * (propulsor.origin[0] + propulsor.mass_properties.center_of_gravity)

        furnishings.origin                                      = fuselage.origin
        furnishings.mass_properties.center_of_gravity[0]        = .51*fuselage.lengths.total
        
        passenger_weights.origin                                = fuselage.origin
        passenger_weights.mass_properties.center_of_gravity[0]  = .51*fuselage.lengths.total
        
        #assumption that it's at 90% of fuselage length (not from notes)
        apu.origin                                              = fuselage.origin
        apu.mass_properties.center_of_gravity[0]                = .9*fuselage.lengths.total 
        elif isinstance(wing,C.Wings.Vertical_Tail):
            chord_length_v_tail_35_percent_semi_span  = compute_chord_length_from_span_location(wing,.35*wing.spans.projected)
            v_tail_35_percent_semi_span_offset        = np.tan(wing.sweeps.quarter_chord)*.35*.5*wing.spans.projected
            wing.mass_properties.center_of_gravity[0][0] = .3*chord_length_v_tail_35_percent_semi_span + \
                                                                        v_tail_35_percent_semi_span_offset
        else:
            span_location_mac = compute_span_location_from_chord_length(wing, wing.chords.mean_aerodynamic)
            mac_le_offset     = np.tan(wing.sweeps.leading_edge)*span_location_mac
            
            wing.mass_properties.center_of_gravity[0][0] = .3*wing.chords.mean_aerodynamic + mac_le_offset
            
            
    # Go through all the propulsors
    propulsion_moment = 0.
    propulsion_mass   = 0. 
    for prop in vehicle.propulsors:
            prop.mass_properties.center_of_gravity[0][0] = prop.engine_length*.5
            propulsion_mass                              += prop.mass_properties.mass         
            propulsion_moment                            += propulsion_mass*(prop.engine_length*.5+prop.origin[0][0])
            
    if propulsion_mass!= 0.:
        propulsion_cg = propulsion_moment/propulsion_mass
    else:
        propulsion_cg = 0.

    # Go through all the fuselages
    for fuse in vehicle.fuselages:
        fuse.mass_properties.center_of_gravity[0][0]   = .45*fuse.lengths.total

    #---------------------------------------------------------------------------------
    # All other components
    #---------------------------------------------------------------------------------
     
    # Select a length scale depending on what kind of vehicle this is
    length_scale = 1.
    nose_length  = 0.
     
    # Check if there is a fuselage
    if len(vehicle.fuselages) == 0.:
        for wing in vehicle.wings:
            if isinstance(wing,C.Wings.Main_Wing):
                b = wing.chords.root
                if b>length_scale:
                    length_scale = b
                    nose_length  = 0.25*b
    else:
        for fuse in vehicle.fuselages:
            nose   = fuse.lengths.nose
            length = fuse.lengths.total
            if length > length_scale:
                length_scale = length
                nose_length  = nose
                
    # unpack all components:
    avionics                                                = vehicle.systems.avionics
    furnishings                                             = vehicle.systems.furnishings
    apu                                                     = vehicle.systems.apu
    passengers                                              = vehicle.payload.passengers
    baggage                                                 = vehicle.payload.baggage
    cargo                                                   = vehicle.payload.cargo
    air_conditioner                                         = vehicle.systems.air_conditioner
    optionals                                               = vehicle.systems.optionals  
    fuel                                                    = vehicle.systems.fuel 
    control_systems                                         = vehicle.systems.control_systems
    electrical_systems                                      = vehicle.systems.electrical_systems
    main_gear                                               = vehicle.landing_gear.main    
    nose_gear                                               = vehicle.landing_gear.nose 
    hydraulics                                              = vehicle.systems.hydraulics
        

    avionics.origin[0][0]                                      = 0.4 * nose_length
    avionics.mass_properties.center_of_gravity[0][0]           = 0.0
    
    furnishings.origin[0][0]                                   = 0.51 * length_scale
    furnishings.mass_properties.center_of_gravity[0][0]        = 0.0
    
    #assumption that it's at 90% of fuselage length (not from notes)
    apu.origin[0][0]                                           = 0.9 * length_scale   
    apu.mass_properties.center_of_gravity[0][0]                = 0.0
    
    passengers.origin[0][0]                                    = 0.51 * length_scale  
    passengers.mass_properties.center_of_gravity[0][0]         = 0.0
    
    baggage.origin[0][0]                                       = 0.51 * length_scale  
    baggage.mass_properties.center_of_gravity[0][0]            = 0.0
    
    cargo.origin[0][0]                                         = 0.51 * length_scale  
    cargo.mass_properties.center_of_gravity[0][0]              = 0.0    
    
    air_conditioner.origin[0][0]                               = nose_length
    air_conditioner.mass_properties.center_of_gravity[0][0]    = 0.0
    
    optionals.origin[0][0]                                     = 0.51 * length_scale  
    optionals.mass_properties.center_of_gravity[0][0]          = 0.0   
        
    fuel.origin[0][0]                                          = vehicle.wings.main_wing.origin[0][0] 
    fuel.mass_properties.center_of_gravity                     = vehicle.wings.main_wing.mass_properties.center_of_gravity
    
    control_systems.origin[0][0]                               = vehicle.wings.main_wing.origin[0][0] 
    control_systems.mass_properties.center_of_gravity[0][0]    = vehicle.wings.main_wing.mass_properties.center_of_gravity[0][0] + \
        .1*vehicle.wings.main_wing.chords.mean_aerodynamic
    
    
    electrical_systems.origin[0][0]                            = .75*(.5*length_scale) + propulsion_cg*.25
    electrical_systems.mass_properties.center_of_gravity[0][0] = 0.0
    
    hydraulics.origin[0][0]                                    = .75*(vehicle.wings.main_wing.origin[0][0] + \
                                                                      wing.mass_properties.center_of_gravity[0][0]) + 0.25* \
                                                                     length_scale*.95
    hydraulics.mass_properties.center_of_gravity[0][0]         = 0.0       
    
    # Now the landing gear
    
    # Nose gear
    nose_gear.origin[0][0]                                     = 0.25*nose_length
    nose_gear.mass_properties.center_of_gravity[0][0]          = 0.0   
    
    # Main gear
    moment_sans_main = vehicle.center_of_gravity()[0][0]*(vehicle.sum_mass()-main_gear.mass_properties.mass)
    
    main_gear_location = moment_sans_main/(vehicle.mass_properties.takeoff-main_gear.mass_properties.mass)/(1-nose_load)
    main_gear.origin[0][0]                                     = main_gear_location
    main_gear.mass_properties.center_of_gravity                = 0.0
    
    return

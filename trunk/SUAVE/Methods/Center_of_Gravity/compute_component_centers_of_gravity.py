#computes the CG of all of the vehicle components based on correlations from AA241

#Created:M. Vegh Oct. 2015
import SUAVE
from SUAVE.Core import Units, Data
import numpy as np
from SUAVE.Methods.Geometry.Three_Dimensional.compute_span_location_from_chord_length import compute_span_location_from_chord_length
from SUAVE.Methods.Geometry.Three_Dimensional.compute_chord_length_from_span_location import compute_chord_length_from_span_location


def compute_component_centers_of_gravity(vehicle):
    
    wing              = vehicle.wings['main_wing']
    h_tail            =vehicle.wings['horizontal_stabilizer']
    v_tail            =vehicle.wings['vertical_stabilizer']
    control_systems   =vehicle.control_systems
    fuselage          =vehicle.fuselages['fuselage']
    landing_gear      =vehicle.landing_gear
    turbo_fan         =vehicle.propulsors['turbo_fan']
    electrical_systems=vehicle.electrical_systems
    avionics          =vehicle.avionics
    furnishings       =vehicle.furnishings
    passenger_weights =vehicle.passenger_weights
    air_conditioner   =vehicle.air_conditioner
    fuel              =vehicle.fuel
    apu               =vehicle.apu
    hydraulics        =vehicle.hydraulics
    optionals         =vehicle.optionals
    
    span_location_mac                          =compute_span_location_from_chord_length(wing, wing.chords.mean_aerodynamic)
    mac_le_offset                              =.8*np.sin(wing.sweep)*span_location_mac  #assume that 80% of the chord difference is from leading edge sweep
    chord_length_h_tail_35_percent_semi_span   =compute_chord_length_from_span_location(h_tail,.35*h_tail.spans.projected*.5)
    chord_length_v_tail_35_percent_semi_span   =compute_chord_length_from_span_location(v_tail,.35*v_tail.spans.projected*.5)
    
    h_tail_35_percent_semi_span_offset         =.8*np.sin(h_tail.sweep)*.35*.5*h_tail.spans.projected             #x distance from leading edge of root chord to leading edge of aerodynamic center
    v_tail_35_percent_semi_span_offset         =.8*np.sin(v_tail.sweep)*.35*.5*v_tail.spans.projected
    
    wing.mass_properties.center_of_gravity[0]  =.3*wing.chords.mean_aerodynamic+mac_le_offset
    h_tail.mass_properties.center_of_gravity[0]=.3*chord_length_h_tail_35_percent_semi_span+h_tail_35_percent_semi_span_offset
    #h_tail.mass_properties.center_of_gravity[1]=.35*h_tail.spans.projected
    v_tail.mass_properties.center_of_gravity[0]=.3*chord_length_v_tail_35_percent_semi_span+v_tail_35_percent_semi_span_offset
    
 
    
    
    control_systems.origin                                 = wing.origin
    control_systems.mass_properties.center_of_gravity[0]   = .4*wing.chords.mean_aerodynamic+mac_le_offset
    fuselage.mass_properties.center_of_gravity[0]          = .45*fuselage.lengths.total
    turbo_fan.origin[0]                                    = wing.origin+mac_le_offset/2.-(3./4.)*turbo_fan.engine_length
    turbo_fan.mass_properties.center_of_gravity[0]         = turbo_fan.engine_length*.5
    electrical_systems.mass_properties.center_of_gravity[0]= .75*(fuselage.origin[0]+\
                .5*fuselage.lengths.total)+.25*(turbo_fan.origin[0]+turbo_fan.mass_properties.center_of_gravity[0])
    
    
    avionics.origin                                       =fuselage.origin
    avionics.mass_properties.center_of_gravity[0]         =.4*fuselage.lengths.nose
    furnishings.origin                                    =fuselage.origin
    furnishings.mass_properties.center_of_gravity[0]      =.51*fuselage.lengths.total
    passenger_weights.origin                              =fuselage.origin
    passenger_weights.mass_properties.center_of_gravity[0]=.51*fuselage.lengths.total
    air_conditioner.origin                                =fuselage.origin
    air_conditioner.mass_properties.center_of_gravity[0]  =fuselage.lengths.nose
    fuel.origin                                           =wing.origin
    fuel.mass_properties.center_of_gravity                =wing.mass_properties.center_of_gravity #assume fuel cg is wing cg (not from notes)
    
    apu.origin                                            =fuselage.origin
    apu.mass_properties.center_of_gravity[0]              =.9*fuselage.lengths.total #assumption that it's at 90% of fuselage length (not from notes)
    hydraulics.origin                                     =fuselage.origin
    hydraulics.mass_properties.center_of_gravity          =.75*(wing.origin+wing.mass_properties.center_of_gravity)+.25*(turbo_fan.origin+turbo_fan.mass_properties.center_of_gravity)
    
    optionals_origin                                      =fuselage.origin
    optionals.mass_properties.center_of_gravity[0]        =.51*fuselage.lengths.total
    
    
  
    
    
    
    
    
    
    
    
    
    return 0

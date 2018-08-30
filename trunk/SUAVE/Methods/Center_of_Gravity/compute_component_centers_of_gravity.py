## @ingroup Methods-Center_of_Gravity
# compute_component_centers_of_gravity.py
#
# Created:  Oct 2015, M. Vegh
# Modified: Jan 2016, E. Botero
# Mofified: Jun 2017, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
from SUAVE.Methods.Geometry.Three_Dimensional.compute_span_location_from_chord_length import compute_span_location_from_chord_length
from SUAVE.Methods.Geometry.Three_Dimensional.compute_chord_length_from_span_location import compute_chord_length_from_span_location

# ----------------------------------------------------------------------
#  Computer Aircraft Center of Gravity
# ---------------origin-------------------------------------------------------

## @ingroup Methods-Center_of_Gravity
def compute_component_centers_of_gravity(vehicle, compute_propulsor_origin = False):
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
    
    wing                                                        = vehicle.wings['main_wing']
    span_location_mac                                           = compute_span_location_from_chord_length(wing, wing.chords.mean_aerodynamic)
    
    #assume that 80% of the chord difference is from leading edge sweep
    #x distance from leading edge of root chord to leading edge of aerodynamic center
    mac_le_offset                                               = .8*np.sin(wing.sweeps.leading_edge)*span_location_mac
    wing.mass_properties.center_of_gravity[0]                   = .3*wing.chords.mean_aerodynamic + mac_le_offset
    
    if 'horizontal_stabilizer' in vehicle.wings:
        h_tail                                                  = vehicle.wings['horizontal_stabilizer']
        chord_length_h_tail_35_percent_semi_span                = compute_chord_length_from_span_location(h_tail,.35*h_tail.spans.projected*.5)
        h_tail_35_percent_semi_span_offset                      =.8*np.sin(h_tail.sweeps.quarter_chord)*.35*.5*h_tail.spans.projected   
        h_tail.mass_properties.center_of_gravity[0]             = .3*chord_length_h_tail_35_percent_semi_span + \
                                                                      h_tail_35_percent_semi_span_offset
    else: 
        print("no horizontal stabilizer")
        
    if 'vertical_stabilizer' in vehicle.wings:
        v_tail                                                  = vehicle.wings['vertical_stabilizer']
        chord_length_v_tail_35_percent_semi_span                = compute_chord_length_from_span_location(v_tail,.35*v_tail.spans.projected*.5)
        v_tail_35_percent_semi_span_offset                      =.8*np.sin(v_tail.sweeps.quarter_chord)*.35*.5*v_tail.spans.projected
        v_tail.mass_properties.center_of_gravity[0]             = .3*chord_length_v_tail_35_percent_semi_span + \
                                                                    v_tail_35_percent_semi_span_offset

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
        
        fuel                                                    = vehicle.fuel        
        fuel.origin                                             = wing.origin
        fuel.mass_properties.center_of_gravity                  = wing.mass_properties.center_of_gravity  
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
        
        fuse_key                                                = list(vehicle.fuselages.keys())[0] 
        fuselage                                                = vehicle.fuselages[fuse_key]
        
        fuselage.mass_properties.center_of_gravity[0]           = .45*fuselage.lengths.total
        electrical_systems.mass_properties.center_of_gravity[0] = .75*(fuselage.origin[0][0]+  .5*fuselage.lengths.total)+.25*(propulsor.origin[0][0]+propulsor.mass_properties.center_of_gravity[0])      
        avionics.origin                                         = fuselage.origin
        avionics.mass_properties.center_of_gravity[0]           = .4*fuselage.lengths.nose
        
        furnishings.origin                                      = fuselage.origin
        furnishings.mass_properties.center_of_gravity[0]        = .51*fuselage.lengths.total
        
        passenger_weights.origin                                = fuselage.origin
        passenger_weights.mass_properties.center_of_gravity[0]  = .51*fuselage.lengths.total
        
        air_conditioner.origin                                  = fuselage.origin
        air_conditioner.mass_properties.center_of_gravity[0]    = fuselage.lengths.nose
        
        #assumption that it's at 90% of fuselage length (not from notes)
        apu.origin                                              = fuselage.origin
        apu.mass_properties.center_of_gravity[0]                = .9*fuselage.lengths.total 
                
        optionals.origin                                        = fuselage.origin
        optionals.mass_properties.center_of_gravity[0]          = .51*fuselage.lengths.total
        
        hydraulics.origin                                       = fuselage.origin
        hydraulics.mass_properties.center_of_gravity            = .75*(wing.origin+wing.mass_properties.center_of_gravity) +.25*(propulsor.origin[0]+propulsor.mass_properties.center_of_gravity)       
    
    return 0
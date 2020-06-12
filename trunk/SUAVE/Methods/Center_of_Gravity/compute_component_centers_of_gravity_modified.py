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
import SUAVE.Components as C
from SUAVE.Components import Physical_Component
from SUAVE.Core import Units

# ----------------------------------------------------------------------
#  Computer Aircraft Center of Gravity
# ----------------------------------------------------------------------

## @ingroup Methods-Center_of_Gravity
def compute_component_centers_of_gravity_modified(vehicle, nose_load = 0.06, 
                                                  gear_pos_type = 'from_nose',
                                                  tipback = 15. * Units.deg,
                                                  rotation_angle = 14 * Units.deg,
                                                  rake_angle = 3. * Units.deg):
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
    
    cabin_origin_x = vehicle.fuselages.fuselage.cabin.origin[0][0]
    cabin_length   = vehicle.fuselages.fuselage.cabin.length    
    
    # Go through all wings
    for wing in vehicle.wings:
        
        if isinstance(wing,C.Wings.Main_Wing):
                wing.mass_properties.center_of_gravity[0][0] = .2*wing.chords.mean_aerodynamic +wing.aerodynamic_center[0]             
                
            
        elif isinstance(wing,C.Wings.Horizontal_Tail):
            chord_length_h_tail_35_percent_semi_span  = compute_chord_length_from_span_location(wing,.35*wing.spans.projected*.5)
            h_tail_35_percent_semi_span_offset        = np.tan(wing.sweeps.quarter_chord)*.35*.5*wing.spans.projected   
            wing.mass_properties.center_of_gravity[0][0] = .5*chord_length_h_tail_35_percent_semi_span + \
                                                                          h_tail_35_percent_semi_span_offset   
            tail_origin = wing.origin[0][0]
            tail_cg = wing.mass_properties.center_of_gravity[0][0]

        elif isinstance(wing,C.Wings.Vertical_Tail):
            chord_length_v_tail_35_percent_semi_span  = compute_chord_length_from_span_location(wing,.35*wing.spans.projected)
            v_tail_35_percent_semi_span_offset        = np.tan(wing.sweeps.quarter_chord)*.35*.5*wing.spans.projected
            wing.mass_properties.center_of_gravity[0][0] = .5*chord_length_v_tail_35_percent_semi_span + \
                                                                        v_tail_35_percent_semi_span_offset
            tail_origin = wing.origin[0][0]
            tail_cg = wing.mass_properties.center_of_gravity[0][0]
        else:
            span_location_mac = compute_span_location_from_chord_length(wing, wing.chords.mean_aerodynamic)
            mac_le_offset     = np.tan(wing.sweeps.leading_edge)*span_location_mac
            
            wing.mass_properties.center_of_gravity[0][0] = .5*wing.chords.mean_aerodynamic + mac_le_offset
            
            
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
        fuse.mass_properties.center_of_gravity[0][0]   = .6*fuse.lengths.total

    #---------------------------------------------------------------------------------
    # All other components
    #---------------------------------------------------------------------------------
                
    # unpack all components:
    avionics                                                = vehicle.systems.avionics
    furnishings                                             = vehicle.systems.furnishings
    apu                                                     = vehicle.systems.apu
    passenger_weights                                       = vehicle.systems.passengers
    air_conditioner                                         = vehicle.systems.air_conditioner
    optionals                                               = vehicle.systems.optionals  
    fuel                                                    = vehicle.systems.fuel 
    control_systems                                         = vehicle.systems.control_systems
    electrical_systems                                      = vehicle.systems.electrical_systems
    main_gear                                               = vehicle.landing_gear.main_landing_gear    
    nose_gear                                               = vehicle.landing_gear.nose_landing_gear 
    hydraulics                                              = vehicle.systems.hydraulics
        
    avionics.origin[0][0]                                      = cabin_origin_x - 10*Units.ft
    avionics.mass_properties.center_of_gravity[0][0]           = 0.0
    
    furnishings.origin[0][0]                                   = 0.51 * cabin_length + cabin_origin_x  
    furnishings.mass_properties.center_of_gravity[0][0]        = 0.0
    
    #assumption that it's at 90% of fuselage length (not from notes)
    apu.origin[0][0]                                           = cabin_origin_x + cabin_length + 10*Units.ft
    apu.mass_properties.center_of_gravity[0][0]                = 0.0
    
    passenger_weights.origin[0][0]                             = 0.51 * cabin_length + cabin_origin_x  
    passenger_weights.mass_properties.center_of_gravity[0][0]  = 0.0
    
    air_conditioner.origin[0][0]                               = cabin_origin_x
    air_conditioner.mass_properties.center_of_gravity[0][0]    = 0.0
    
    optionals.origin[0][0]                                     = 0.51 * cabin_length + cabin_origin_x  
    optionals.mass_properties.center_of_gravity[0][0]          = 0.0   
        
    fuel.origin[0][0]                                          = vehicle.wings.main_wing.origin[0][0] 
    fuel.mass_properties.center_of_gravity                     = vehicle.wings.main_wing.mass_properties.center_of_gravity
    
    control_systems.origin[0][0]                               = vehicle.wings.main_wing.origin[0][0] 
    control_systems.mass_properties.center_of_gravity[0][0]    = vehicle.wings.main_wing.mass_properties.center_of_gravity[0][0] + \
        .3*vehicle.wings.main_wing.chords.mean_aerodynamic
    
    
    electrical_systems.origin[0][0]                            = .75*(.5*cabin_length + cabin_origin_x) + propulsion_cg*.25
    electrical_systems.mass_properties.center_of_gravity[0][0] = 0.0
    
    hydraulics.origin[0][0]                                    = .75*(vehicle.wings.main_wing.origin[0][0] + vehicle.wings.main_wing.mass_properties.center_of_gravity[0][0]) + 0.25*(tail_origin+tail_cg)
    hydraulics.mass_properties.center_of_gravity[0][0]         = 0.0       
    
    # Now the landing gear
    
    # Nose gear
    nose_gear.origin[0][0]                                     = cabin_origin_x - 20 * Units.ft
    nose_gear.mass_properties.center_of_gravity[0][0]          = 0.0  
    
    def get_total_mass(vehicle):
        total = 0.0
        for key in vehicle.keys():
            item = vehicle[key]
            if isinstance(item,Physical_Component.Container):
                total += item.sum_mass()
        return total
    
    def get_CG(vehicle):
        total = np.array([[0.0,0.0,0.0]])
        for key in vehicle.keys():
            item = vehicle[key]
            if isinstance(item,Physical_Component.Container):
                total += item.total_moment()
                print(print(item.keys()),item.total_moment())
        mass = get_total_mass(vehicle)
        if mass ==0:
            mass = 1.
        CG = total/mass
        vehicle.mass_properties.center_of_gravity = CG  
        return CG
        
    
    # Main gear
    vehicle_moment_sans_main = get_CG(vehicle)[0][0]*get_total_mass(vehicle) # main gear moment is 0
    takeoff_weight      = vehicle.mass_properties.takeoff
    assert np.isclose(takeoff_weight,get_total_mass(vehicle))
    assert np.isclose(main_gear.origin[0][0],0)
    nose_gear_location  = nose_gear.origin[0][0]
    
    main_gear_location = (vehicle_moment_sans_main - nose_load*takeoff_weight*nose_gear_location)/((1-nose_load)*takeoff_weight-main_gear.mass_properties.mass)
    main_gear.origin[0][0]                                     = main_gear_location
    main_gear.mass_properties.center_of_gravity[0][0]          = 0.0
    
    # set proper CG with main gear location
    get_CG(vehicle)
    
    if gear_pos_type == 'from_angles':
        mm = main_gear.mass_properties.mass
        mn = nose_gear.mass_properties.mass
        mtot = get_total_mass(vehicle)
        mv = mtot - mm - mn
        f = nose_load
        phi = rotation_angle
        theta = np.pi/2-tipback
        L = vehicle.total_length
        
        xv = (get_CG(vehicle)[0][0]*get_total_mass(vehicle) - mm*main_gear_location - mn*nose_gear_location)/mv
        
        A = np.array([[1,-mm/mtot,-mn/mtot,0,0,0],
                      [1,-(1-f),-f,0,0,0],
                      [0,0,0,1,-np.tan(theta)*np.tan(phi)/(np.tan(theta) + np.tan(phi)),-np.tan(theta)*np.tan(phi)/(np.tan(theta) + np.tan(phi))],
                      [0,0,0,1,0,-np.tan(phi)],
                      [1,-1,0,0,1,0],
                      [0,1,0,0,0,1]])
        
        b = np.array([xv*mv/mtot,0,0,0,0,L])
        
        xcg, xm, xn, h, l1, l2 = np.linalg.solve(A, b)
        
        main_gear.origin[0][0] = xm
        nose_gear.origin[0][0] = xn        
        
        get_CG(vehicle)
    
    return
# empty.py
# 
# Created:  Jan 2014, A. Wendorff
# Modified: Feb 2016, M. Vegh

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import SUAVE
from SUAVE.Core import Units, Data
from fuselage import fuselage
from landing_gear import landing_gear
from payload import payload
from systems import systems
from tail_horizontal import tail_horizontal
from tail_vertical import tail_vertical
from wing_main import wing_main
from SUAVE.Methods.Weights.Correlations import Propulsion as Propulsion
import warnings

# ----------------------------------------------------------------------
#  Empty
# ----------------------------------------------------------------------

def empty(vehicle):
    """ output = SUAVE.Methods.Weights.Correlations.Tube_Wing.empty(engine,wing,aircraft,fuselage,horizontal,vertical)
        This is for a standard Tube and Wing aircraft configuration.        

        Inputs:
            engine - a data dictionary with the fields:                    
                thrust_sls - sea level static thrust of a single engine [Newtons]

            wing - a data dictionary with the fields:
                gross_area - wing gross area [meters**2]
                span - span of the wing [meters]
                taper - taper ratio of the wing [dimensionless]
                t_c - thickness-to-chord ratio of the wing [dimensionless]
                sweep - sweep angle of the wing [radians]
                mac - mean aerodynamic chord of the wing [meters]
                r_c - wing root chord [meters]

            aircraft - a data dictionary with the fields:                    
                Nult - ultimate load of the aircraft [dimensionless]
                Nlim - limit load factor at zero fuel weight of the aircraft [dimensionless]
                TOW - maximum takeoff weight of the aircraft [kilograms]
                zfw - maximum zero fuel weight of the aircraft [kilograms]
                num_eng - number of engines on the aircraft [dimensionless]
                num_pax - number of passengers on the aircraft [dimensionless]
                wt_cargo - weight of the bulk cargo being carried on the aircraft [kilograms]
                num_seats - number of seats installed on the aircraft [dimensionless]
                ctrl - specifies if the control system is "fully powered", "partially powered", or not powered [dimensionless]
                ac - determines type of instruments, electronics, and operating items based on types: 
                    "short-range", "medium-range", "long-range", "business", "cargo", "commuter", "sst" [dimensionless]
                w2h - tail length (distance from the airplane c.g. to the horizontal tail aerodynamic center) [meters]

            fuselage - a data dictionary with the fields:
                area - fuselage wetted area [meters**2]
                diff_p - Maximum fuselage pressure differential [Pascal]
                width - width of the fuselage [meters]
                height - height of the fuselage [meters]
                length - length of the fuselage [meters]                     

            horizontal
                area - area of the horizontal tail [meters**2]
                span - span of the horizontal tail [meters]
                sweep - sweep of the horizontal tail [radians]
                mac - mean aerodynamic chord of the horizontal tail [meters]
                t_c - thickness-to-chord ratio of the horizontal tail [dimensionless]
                exposed - exposed area ratio for the horizontal tail [dimensionless]

            vertical
                area - area of the vertical tail [meters**2]
                span - sweight = weight * Units.lbpan of the vertical [meters]
                t_c - thickness-to-chord ratio of the vertical tail [dimensionless]
                sweep - sweep angle of the vertical tail [radians]
                t_tail - factor to determine if aircraft has a t-tail, "yes" [dimensionless]

        Outputs:
            output - a data dictionary with fields:
                wt_payload - weight of the passengers plus baggage and paid cargo [kilograms]
                wt_pax - weight of all the passengers [kilogram]
                wt_bag - weight of all the baggage [kilogram]
                wt_fuel - weight of the fuel carried[kilogram]
                wt_empty - operating empty weight of the aircraft [kilograms]

        Assumptions:
            calculated aircraft weight from correlations created per component of historical aircraft
    """     

    # Unpack inputs
    fuel        = vehicle.fuel
    Nult        = vehicle.envelope.ultimate_load
    Nlim        = vehicle.envelope.limit_load
    TOW         = vehicle.mass_properties.max_takeoff
    wt_zf       = vehicle.mass_properties.max_zero_fuel
    num_pax     = vehicle.passengers
    wt_cargo    = vehicle.mass_properties.cargo
    num_seats   = vehicle.fuselages['fuselage'].number_coach_seats
    #ctrl_type  = vehicle.systems.control
    #ac_type    = vehicle.systems.accessories         
    q_c         = vehicle.design_dynamics_pressure
    mach_number = vehicle.design_mach_number
    
    propulsor_name = vehicle.propulsors.keys()[0] #obtain the key for the propulsor for assignment purposes
    propulsors     = vehicle.propulsors[propulsor_name]
    num_eng        = propulsors.number_of_engines
    
    
    
    
    if propulsor_name == 'turbofan':
        
                # thrust_sls should be sea level static thrust. Using design thrust results in wrong propulsor 
                # weight estimation. Engine sizing should return this value.
                # for now, using thrust_sls = design_thrust / 0.20, just for optimization evaluations
        thrust_sls                       = propulsors.sealevel_static_thrust
        wt_engine_jet                    = Propulsion.engine_jet(thrust_sls)
        wt_propulsion                    = Propulsion.integrated_propulsion(wt_engine_jet,num_eng)
        propulsors.mass_properties.mass  = wt_propulsion 
        
    elif propulsor_name == 'internal_combustion':
    
        rated_power                      = propulsors.rated_power/num_eng
        wt_engine_piston                 = Propulsion.engine_piston(rated_power)
        wt_propulsion                    = Propulsion.integrated_propulsion_general_aviation(wt_engine_piston,num_eng)
        propulsors.mass_properties.mass  = wt_propulsion 
      
    
    else: #propulsor used is not an IC Engine or Turbofan; assume mass_properties defined outside model
        wt_propulsion                    = propulsors.mass_properties.mass
        if wt_propulsion==0:
            warnings.warn("Propulsion mass= 0 ;e there is no Engine Weight being added to the Configuration", stacklevel=1)    
    #find fuel volume
    if not vehicle.has_key('fuel'): 
        warnings.warn("fuel mass= 0 ; fuel system volume is calculated incorrectly ", stacklevel=1)   
        N_tank     = 0 
        V_fuel     = 0.
        V_fuel_int = 0.
      
    else:
        m_fuel                      = fuel.mass_properties.mass
        landing_weight              = TOW-m_fuel  #just assume this for now
        N_tank                      = fuel.number_of_tanks
        V_fuel_int                  = fuel.internal_volume #fuel in internal (as opposed to external tanks)
        V_fuel                      = m_fuel/fuel.density  #total fuel
        fuel.mass_properties.volume = V_fuel 
    S_gross_w  = vehicle.reference_area
    #S_gross_w  = vehicle.wings['main_wing'].Areas.reference
    
    #main wing
    if not vehicle.wings.has_key('main_wing'):
        wt_wing = 0.0
        wing_c_r = 0.0
        warnings.warn("There is no Wing Weight being added to the Configuration", stacklevel=1)
        
    else:
        b          = vehicle.wings['main_wing'].spans.projected
        AR_w       = (b**2.)/S_gross_w
        taper_w    = vehicle.wings['main_wing'].taper
        t_c_w      = vehicle.wings['main_wing'].thickness_to_chord
        sweep_w    = vehicle.wings['main_wing'].sweep
        mac_w      = vehicle.wings['main_wing'].chords.mean_aerodynamic
        wing_c_r   = vehicle.wings['main_wing'].chords.root
        wt_wing    = wing_main(S_gross_w, m_fuel, AR_w, sweep_w, q_c, taper_w, t_c_w,Nult,TOW)
        vehicle.wings['main_wing'].mass_properties.mass = wt_wing        

    S_fus      = vehicle.fuselages['fuselage'].areas.wetted
    diff_p_fus = vehicle.fuselages['fuselage'].differential_pressure
    w_fus      = vehicle.fuselages['fuselage'].width
    h_fus      = vehicle.fuselages['fuselage'].heights.maximum
    l_fus      = vehicle.fuselages['fuselage'].lengths.structure
    V_fuse     = vehicle.fuselages['fuselage'].mass_properties.volume
    V_int      = vehicle.fuselages['fuselage'].internal_volume 
    
    if not vehicle.wings.has_key('horizontal_stabilizer'):
        wt_tail_horizontal = 0.0
        S_h = 0.0
        warnings.warn("There is no Horizontal Tail Weight being added to the Configuration", stacklevel=1)
        
    else:    
        S_h            = vehicle.wings['horizontal_stabilizer'].areas.reference
        b_h            = vehicle.wings['horizontal_stabilizer'].spans.projected
        AR_h           = (b_h**2.)/S_h
        taper_h        = vehicle.wings['horizontal_stabilizer'].spans.projected
        sweep_h        = vehicle.wings['horizontal_stabilizer'].sweep
        mac_h          = vehicle.wings['horizontal_stabilizer'].chords.mean_aerodynamic
        t_c_h          = vehicle.wings['horizontal_stabilizer'].thickness_to_chord
        h_tail_exposed = vehicle.wings['horizontal_stabilizer'].areas.exposed / vehicle.wings['horizontal_stabilizer'].areas.wetted
        l_w2h          = vehicle.wings['horizontal_stabilizer'].origin[0] + vehicle.wings['horizontal_stabilizer'].aerodynamic_center[0] - vehicle.wings['main_wing'].origin[0] - vehicle.wings['main_wing'].aerodynamic_center[0] #Need to check this is the length of the horizontal tail moment arm
        wt_tail_horizontal = tail_horizontal(S_h, AR_h, sweep_h, q_c, taper_h, t_c_h,Nult,TOW)                
        vehicle.wings['horizontal_stabilizer'].mass_properties.mass = wt_tail_horizontal        
    #vertical stabilizer
    if not vehicle.wings.has_key('vertical_stabilizer'):   
        output_3 = Data()
        output_3.wt_tail_vertical = 0.0
   
        S_v = 0.0
        warnings.warn("There is no Vertical Tail Weight being added to the Configuration", stacklevel=1)    
        
    else:     
        S_v        = vehicle.wings['vertical_stabilizer'].areas.reference
        b_v        = vehicle.wings['vertical_stabilizer'].spans.projected
        AR_v       = (b_v**2.)/S_v
        taper_v    = vehicle.wings['vertical_stabilizer'].taper
        t_c_v      = vehicle.wings['vertical_stabilizer'].thickness_to_chord
        sweep_v    = vehicle.wings['vertical_stabilizer'].sweep
        t_tail     = vehicle.wings['vertical_stabilizer'].t_tail  
        output_3   = tail_vertical(S_v, AR_v, sweep_v, q_c, taper_v, t_c_v, Nult,TOW,t_tail)
        vehicle.wings['vertical_stabilizer'].mass_properties.mass = output_3.wt_tail_vertical
    
    #landing gear
    if not vehicle.has_key('landing_gear'):
        warnings.warn('There is no Landing Gear weight being added to the vehicle', stacklevel=1)
        wt_landing_gear = Data()
        wt_landing_gear.main = 0.0
        wt_landing_gear.nose = 0.0
        
    else: 
        landing_gear_component = vehicle.landing_gear #landing gear previously defined
        strut_length_main      = landing_gear_component.main.strut_length
        strut_length_nose      = landing_gear_component.nose.strut_length
        wt_landing_gear        = landing_gear(landing_weight, Nult, strut_length_main, strut_length_nose)
        
        landing_gear_component.main.mass_properties.mass = wt_landing_gear.main
        landing_gear_component.nose.mass_properties.mass = wt_landing_gear.nose
    
    
    if not vehicle.has_key('avionics'):
        warnings.warn('There is no avionics weight being added to the vehicle; many weight correlations are dependant on this', stacklevel=1)
        avionics          = SUAVE.Components.Energy.Peripherals.Avionics()
        W_uav = 0.
        
    else:
        avionics = vehicle.avionics
        W_uav = avionics.mass_properties.uninstalled
    
    has_air_conditioner = vehicle.has_air_conditioner
    
    
    # Calculating Empty Weight of Aircraft
    wt_fuselage        = fuselage(S_fus, Nult, TOW, w_fus, h_fus, l_fus, l_w2h, q_c, V_fuse, diff_p_fus)
    output_2           = systems(W_uav,V_fuel, V_fuel_int, N_tank, num_eng, l_fus, b, TOW, Nult, num_seats, mach_number, has_air_conditioner)

    # Calculate the equipment empty weight of the aircraft
    
    wt_empty           = (wt_wing + wt_fuselage + wt_landing_gear.main+wt_landing_gear.nose + wt_propulsion + output_2.wt_systems + \
                          wt_tail_horizontal + output_3.wt_tail_vertical) 
    vehicle.fuselages['fuselage'].mass_properties.mass = wt_fuselage


    
    # packup outputs
    output                   = payload(TOW, wt_empty, num_pax,wt_cargo)
    output.wing              = wt_wing
    output.fuselage          = wt_fuselage
    output.propulsion        = wt_propulsion
    output.landing_gear      = Data()
    output.landing_gear_main = wt_landing_gear.main
    output.landing_gear_nose = wt_landing_gear.nose
    output.horizontal_tail   = wt_tail_horizontal
    output.vertical_tail     = output_3.wt_tail_vertical
    
    output.systems           = output_2.wt_systems       
    output.systems_breakdown = Data()
    output.systems_breakdown.control_systems   = output_2.wt_flt_ctrl    
    #output.systems_breakdown.apu               = output_2.wt_apu         
    output.systems_breakdown.hydraulics        = output_2.wt_hyd_pnu     
    output.systems_breakdown.avionics          = output_2.wt_avionics    
    output.systems_breakdown.electrical        = output_2.wt_elec        
    output.systems_breakdown.air_conditioner   = output_2.wt_ac          
    output.systems_breakdown.furnish           = output_2.wt_furnish    
    output.systems_breakdown.fuel_system       = output_2.wt_fuel_sys
    #define weights components

    
    
    control_systems   = SUAVE.Components.Physical_Component()
    electrical_systems= SUAVE.Components.Physical_Component()
    passengers        = SUAVE.Components.Physical_Component()
    furnishings       = SUAVE.Components.Physical_Component()
  
    apu               = SUAVE.Components.Physical_Component()
    hydraulics        = SUAVE.Components.Physical_Component()
   
    
    
    #assign output weights to objects
    control_systems.mass_properties.mass                             = output.systems_breakdown.control_systems
    electrical_systems.mass_properties.mass                          = output.systems_breakdown.electrical
    passengers.mass_properties.mass                                  = output.pax + output.bag
    furnishings.mass_properties.mass                                 = output.systems_breakdown.furnish
    avionics.mass_properties.mass                                    = output.systems_breakdown.avionics              
   
    #fuel.mass_properties.mass                                        = output.fuel
    hydraulics.mass_properties.mass                                  = output.systems_breakdown.hydraulics
    
    if has_air_conditioner:
        vehicle.air_conditioner.mass_properties.mass                 = output.systems_breakdown.air_conditioner
    #assign components to vehicle
    vehicle.control_systems                     = control_systems
    vehicle.electrical_systems                  = electrical_systems
    vehicle.avionics                            = avionics
    vehicle.furnishings                         = furnishings
    vehicle.passenger_weights                   = passengers 
   
    vehicle.apu                                 = apu
    vehicle.hydraulics                          = hydraulics
    vehicle.landing_gear                        = landing_gear_component

    #note; air conditioner optional, and weight is added to the air_conditioner object directly
    

    return output
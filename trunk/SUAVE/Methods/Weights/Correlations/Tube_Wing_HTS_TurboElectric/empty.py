## @ingroup Methods-Weights-Correlations-Tube_Wing_HTS_TurboElectric
# empty.py
# 
# Created:  Jan 2014, A. Wendorff
# Modified: Feb 2016, M. Vegh
#           Jul 2017, M. Clarke
#           Jun 2018, T. MacDonald
#           Mar 2020, K. Hamilton

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import SUAVE
from SUAVE.Core         import Units, Data
from SUAVE.Methods.Weights.Correlations.Tube_Wing.tube              import tube
from SUAVE.Methods.Weights.Correlations.Tube_Wing.systems           import systems
from SUAVE.Methods.Weights.Correlations.Tube_Wing.tail_horizontal   import tail_horizontal
from SUAVE.Methods.Weights.Correlations.Tube_Wing.tail_vertical     import tail_vertical
from SUAVE.Methods.Weights.Correlations.Common  import wing_main    as wing_main
from SUAVE.Methods.Weights.Correlations.Common  import landing_gear as landing_gear
from SUAVE.Methods.Weights.Correlations.Common  import payload      as payload
from SUAVE.Methods.Weights.Correlations         import Propulsion   as Propulsion

from SUAVE.Methods.Power.Turboelectric.Sizing                   import initialize_from_power
from SUAVE.Methods.Weights.Correlations.Propulsion.hts_motor    import hts_motor
from .SiC_Electronics import SiC_mass

import warnings

# ----------------------------------------------------------------------
#  Empty
# ----------------------------------------------------------------------

## @ingroup Methods-Weights-Correlations-Tube_Wing
def empty(vehicle,settings=None):
    """ This is for a standard Tube and Wing aircraft with a serial hybrid turboelectric drivetrain. I.e. the fuselage has 1 or more embedded turboshaft engines which generate electric power to be consumed by electrically driven ducted fans located remotely from the turboelectric generators.        

    Assumptions:
        calculated aircraft weight from correlations created per component of historical aircraft and other non-aircraft superconducting systems
    
    Source:
        N/A
        
    Inputs:
      engine - a data dictionary with the fields:                    
          thrust_sls - sea level static thrust of a single engine                [Newtons]

      wing - a data dictionary with the fields:
          gross_area - wing gross area                                           [meters**2]
          span - span of the wing                                                [meters]
          taper - taper ratio of the wing                                        [dimensionless]
          t_c - thickness-to-chord ratio of the wing                             [dimensionless]
          sweep - sweep angle of the wing                                        [radians]
          mac - mean aerodynamic chord of the wing                               [meters]
          r_c - wing root chord                                                  [meters]

      aircraft - a data dictionary with the fields:                    
          Nult - ultimate load of the aircraft                                   [dimensionless]
          Nlim - limit load factor at zero fuel weight of the aircraft           [dimensionless]
          TOW - maximum takeoff weight of the aircraft                           [kilograms]
          zfw - maximum zero fuel weight of the aircraft                         [kilograms]
          num_eng - number of engines on the aircraft                            [dimensionless]
          num_pax - number of passengers on the aircraft                         [dimensionless]
          wt_cargo - weight of the bulk cargo being carried on the aircraft      [kilograms]
          num_seats - number of seats installed on the aircraft                  [dimensionless]
          ctrl - specifies if the control system is "fully powered", "partially powered", or not powered [dimensionless]
          ac - determines type of instruments, electronics, and operating items based on types: 
              "short-range", "medium-range", "long-range", "business", "cargo", "commuter", "sst"        [dimensionless]
          w2h - tail length (distance from the airplane c.g. to the horizontal tail aerodynamic center)  [meters]

      fuselage - a data dictionary with the fields:
          area - fuselage wetted area                                            [meters**2]
          diff_p - Maximum fuselage pressure differential                        [Pascal]
          width - width of the fuselage                                          [meters]
          height - height of the fuselage                                        [meters]
          length - length of the fuselage                                        [meters]                     

      horizontal
          area - area of the horizontal tail                                     [meters**2]
          span - span of the horizontal tail                                     [meters]
          sweep - sweep of the horizontal tail                                   [radians]
          mac - mean aerodynamic chord of the horizontal tail                    [meters]
          t_c - thickness-to-chord ratio of the horizontal tail                  [dimensionless]
          exposed - exposed area ratio for the horizontal tail                   [dimensionless]

      vertical
          area - area of the vertical tail                                       [meters**2]
          span - sweight = weight * Units.lbpan of the vertical                  [meters]
          t_c - thickness-to-chord ratio of the vertical tail                    [dimensionless]
          sweep - sweep angle of the vertical tail                               [radians]
          t_tail - factor to determine if aircraft has a t-tail, "yes"           [dimensionless]
          
      settings.weight_reduction_factors.
          main_wing                                                              [dimensionless] (.1 is a 10% weight reduction)
          empennage                                                              [dimensionless] (.1 is a 10% weight reduction)
          fuselage                                                               [dimensionless] (.1 is a 10% weight reduction)
          

    Outputs:
        output - a data dictionary with fields:
            wt_payload - weight of the passengers plus baggage and paid cargo    [kilograms]
            wt_pax - weight of all the passengers                                [kilogram]
            wt_bag - weight of all the baggage                                   [kilogram]
            wt_fuel - weight of the fuel carried                                 [kilogram]
            wt_empty - operating empty weight of the aircraft                    [kilograms]
 
    Properties Used:
        N/A
    """     

    # Unpack inputs
    Nult       = vehicle.envelope.ultimate_load
    Nlim       = vehicle.envelope.limit_load
    TOW        = vehicle.mass_properties.max_takeoff
    wt_zf      = vehicle.mass_properties.max_zero_fuel
    num_pax    = vehicle.passengers
    wt_cargo   = vehicle.mass_properties.cargo
    num_seats  = vehicle.fuselages['fuselage'].number_coach_seats
    ctrl_type  = vehicle.systems.control
    ac_type    = vehicle.systems.accessories    
    
    if settings == None:
        wt_factors = Data()
        wt_factors.main_wing = 0.
        wt_factors.empennage = 0.
        wt_factors.fuselage  = 0.
    else:
        wt_factors = settings.weight_reduction_factors
     
    propulsor_name = list(vehicle.propulsors.keys())[0] # obtain the key for the propulsor for assignment purposes

    propulsors     = vehicle.propulsors[propulsor_name]
    num_eng        = propulsors.number_of_engines
    if propulsor_name=='turbofan' or propulsor_name=='Turbofan':
        # thrust_sls should be sea level static thrust. Using design thrust results in wrong propulsor weight estimation. Engine sizing should return this value.
        # for now, using thrust_sls = design_thrust / 0.20, just for optimization evaluations

        # This first part here is copied directly from the turbofan empty.py. It is assumed an electrofan with the same diameter as a turbojet will produce the same thrust. Likely the turbofan will produce slightly more as some thrust will be produced directly from the engine core, however the elctrofan will likley have a smaller core, resulting in more thrust directly from the fan.
        thrust_sls                      = propulsors.sealevel_static_thrust
        wt_engine_jet                   = Propulsion.engine_jet(thrust_sls)
        wt_factor                       = 1.0   # how many multiples of the dry engine weight is the operating engine.
        wt_turbojet                     = Propulsion.integrated_propulsion(wt_engine_jet,num_eng,wt_factor)

        # Ducted fan mass is typically 15% of a conventional turbofan engine, however the mass of the turbofans as estimated by SUAVE is proportionally higher for smaller engines, i.e. small engines have lower specific power, which may not be true for electrofans as turbine blade clearance may be the cause of lower performance in smaller turbojet engines. 15% of the turbojet mass is used for now.
        # Divided by the number of engines so the total can be added up later.
        wt_ductedfan        = wt_turbojet * 0.15 / num_eng

        # Size each HTS motor using the SUAVE corelation
        wt_motor            = hts_motor(propulsors.motor.rated_power)

        # Size the motor speed controller(s) (esc)
        wt_esc              = SiC_mass(propulsors.esc)

        # The HTS motor rotor does not have a mass as it is considered part of the motor mass.
        # However, the current supply leads do have mass.
        # It is assumed each motor has two leads.
        wt_leads            = propulsors.lead.mass * 2.0

        # Size the rotor current supply (ccs)
        wt_ccs              = SiC_mass(propulsors.ccs)

        # Size the rotor cryocooler. Each rotor has a seperate cryocooler as defined in serial_hts_turboelectric_sizing
        # This is skipped if the rotor(s) are cooled wholly by cryogen. The mass of the components required to store and deliver the cryogen are not considered.
        wt_cryocooler       = propulsors.cryocooler.mass

        # Sum the above propulsor components to give individual propulsor mass
        wt_propulsor        = wt_ductedfan + wt_motor + wt_esc + wt_leads + wt_ccs + wt_cryocooler
        # Multiply by the number of propulsors
        wt_all_propulsors   = wt_propulsor * num_eng

        # Retreive Turboelectric mass calculated in serial_hts_turboelectric_sizing. This value is all the turboelectric machines combined.
        wt_all_powersupplies    = propulsors.powersupply.mass_properties.mass

        # Total powertrain weight is the sum of the propulsors and the power suppliers
        wt_propulsion = wt_all_propulsors + wt_all_powersupplies
        propulsors.mass_properties.mass = wt_propulsion

        # pack up weight breakdown
        propulsors.motor.mass_properties.mass   = wt_motor * num_eng
        # propulsors.esc
        # propulsors.ccs
        # propulsors.cryocooler
        # propulsors.turboelectric


    else: #propulsor used is not a turbo_fan; assume mass_properties defined outside model
        wt_propulsion                   = propulsors.mass_properties.mass

        if wt_propulsion==0:
            warnings.warn("Propulsion mass= 0 ;e there is no Engine Weight being added to the Configuration", stacklevel=1)
            print("Propulsion mass= 0 ;e there is no Engine Weight being added to the Configuration")
    
    S_gross_w  = vehicle.reference_area
    if 'main_wing' not in vehicle.wings:
        wt_wing = 0.0
        wing_c_r = 0.0
        warnings.warn("There is no Wing Weight being added to the Configuration", stacklevel=1)
        
    else:
        b          = vehicle.wings['main_wing'].spans.projected
        lambda_w   = vehicle.wings['main_wing'].taper
        t_c_w      = vehicle.wings['main_wing'].thickness_to_chord
        sweep_w    = vehicle.wings['main_wing'].sweeps.quarter_chord
        mac_w      = vehicle.wings['main_wing'].chords.mean_aerodynamic
        wing_c_r   = vehicle.wings['main_wing'].chords.root
        wt_wing    = wing_main.wing_main(S_gross_w,b,lambda_w,t_c_w,sweep_w,Nult,TOW,wt_zf)
        wt_wing    = wt_wing*(1.-wt_factors.main_wing)
        vehicle.wings['main_wing'].mass_properties.mass = wt_wing        

    S_fus      = vehicle.fuselages['fuselage'].areas.wetted
    diff_p_fus = vehicle.fuselages['fuselage'].differential_pressure
    w_fus      = vehicle.fuselages['fuselage'].width
    h_fus      = vehicle.fuselages['fuselage'].heights.maximum
    l_fus      = vehicle.fuselages['fuselage'].lengths.total

    if 'horizontal_stabilizer' not in vehicle.wings:
        wt_tail_horizontal = 0.0
        S_h = 0.0
        warnings.warn("There is no Horizontal Tail Weight being added to the Configuration", stacklevel=1)
        
    else:    
        S_h            = vehicle.wings['horizontal_stabilizer'].areas.reference
        b_h            = vehicle.wings['horizontal_stabilizer'].spans.projected
        sweep_h        = vehicle.wings['horizontal_stabilizer'].sweeps.quarter_chord
        mac_h          = vehicle.wings['horizontal_stabilizer'].chords.mean_aerodynamic
        t_c_h          = vehicle.wings['horizontal_stabilizer'].thickness_to_chord
        h_tail_exposed = vehicle.wings['horizontal_stabilizer'].areas.exposed / vehicle.wings['horizontal_stabilizer'].areas.wetted
        l_w2h          = vehicle.wings['horizontal_stabilizer'].origin[0] + vehicle.wings['horizontal_stabilizer'].aerodynamic_center[0] - vehicle.wings['main_wing'].origin[0] - vehicle.wings['main_wing'].aerodynamic_center[0] #Need to check this is the length of the horizontal tail moment arm
        wt_tail_horizontal = tail_horizontal(b_h,sweep_h,Nult,S_h,TOW,mac_w,mac_h,l_w2h,t_c_h, h_tail_exposed)   
        wt_tail_horizontal = wt_tail_horizontal*(1.-wt_factors.empennage)
        vehicle.wings['horizontal_stabilizer'].mass_properties.mass = wt_tail_horizontal        

    if 'vertical_stabilizer' not in vehicle.wings:   
        output_3                  = Data()
        output_3.wt_tail_vertical = 0.0
        output_3.wt_rudder        = 0.0
        S_v                       = 0.0
        warnings.warn("There is no Vertical Tail Weight being added to the Configuration", stacklevel=1)    
        
    else:     
        S_v          = vehicle.wings['vertical_stabilizer'].areas.reference
        b_v          = vehicle.wings['vertical_stabilizer'].spans.projected
        t_c_v        = vehicle.wings['vertical_stabilizer'].thickness_to_chord
        sweep_v      = vehicle.wings['vertical_stabilizer'].sweeps.quarter_chord
        t_tail       = vehicle.wings['vertical_stabilizer'].t_tail  
        output_3     = tail_vertical(S_v,Nult,b_v,TOW,t_c_v,sweep_v,S_gross_w,t_tail)
        wt_vtail_tot = output_3.wt_tail_vertical + output_3.wt_rudder
        wt_vtail_tot = wt_vtail_tot*(1.-wt_factors.empennage)
        vehicle.wings['vertical_stabilizer'].mass_properties.mass = wt_vtail_tot
        
    # Calculating Empty Weight of Aircraft
    wt_landing_gear    = landing_gear.landing_gear(TOW)
    
    wt_fuselage        = tube(S_fus, diff_p_fus,w_fus,h_fus,l_fus,Nlim,wt_zf,wt_wing,wt_propulsion, wing_c_r) 
    wt_fuselage        = wt_fuselage*(1.-wt_factors.fuselage)
    output_2           = systems(num_seats, ctrl_type, S_h, S_v, S_gross_w, ac_type)  

    # Calculate the equipment empty weight of the aircraft
    wt_empty           = (wt_wing + wt_fuselage + wt_landing_gear + wt_propulsion + output_2.wt_systems + \
                          wt_tail_horizontal + wt_vtail_tot) 
    vehicle.fuselages['fuselage'].mass_properties.mass = wt_fuselage


    
    # packup outputs
    output                   = payload.payload(TOW, wt_empty, num_pax,wt_cargo)
    output.wing              = wt_wing
    output.fuselage          = wt_fuselage
    output.propulsion        = wt_propulsion
    output.landing_gear      = wt_landing_gear
    output.horizontal_tail   = wt_tail_horizontal
    output.vertical_tail     = output_3.wt_tail_vertical*(1.-wt_factors.empennage)
    output.rudder            = output_3.wt_rudder*(1.-wt_factors.empennage)
    output.systems           = output_2.wt_systems       
    output.systems_breakdown = Data()
    output.systems_breakdown.control_systems    = output_2.wt_flt_ctrl    
    output.systems_breakdown.apu                = output_2.wt_apu         
    output.systems_breakdown.hydraulics         = output_2.wt_hyd_pnu     
    output.systems_breakdown.instruments        = output_2.wt_instruments 
    output.systems_breakdown.avionics           = output_2.wt_avionics    
    output.systems_breakdown.optionals          = output_2.wt_opitems     
    output.systems_breakdown.electrical         = output_2.wt_elec        
    output.systems_breakdown.air_conditioner    = output_2.wt_ac          
    output.systems_breakdown.furnish            = output_2.wt_furnish

    # serial HTS powertrain components       
    output.propulsion_breakdown = Data()
    output.propulsion_breakdown.ductedfan       = wt_ductedfan
    output.propulsion_breakdown.motor           = wt_motor
    output.propulsion_breakdown.esc             = wt_esc
    output.propulsion_breakdown.leads           = wt_leads
    output.propulsion_breakdown.ccs             = wt_ccs
    output.propulsion_breakdown.cryocooler      = wt_cryocooler
    output.propulsion_breakdown.powersupply     = wt_all_powersupplies / propulsors.number_of_powersupplies
    
    #define weights components

    try: 
        landing_gear_component=vehicle.landing_gear #landing gear previously defined
    except AttributeError: # landing gear not defined
        landing_gear_component=SUAVE.Components.Landing_Gear.Landing_Gear()
        vehicle.landing_gear=landing_gear_component
    
    control_systems         = SUAVE.Components.Physical_Component()
    electrical_systems      = SUAVE.Components.Physical_Component()
    passengers              = SUAVE.Components.Physical_Component()
    furnishings             = SUAVE.Components.Physical_Component()
    air_conditioner         = SUAVE.Components.Physical_Component()
    fuel                    = SUAVE.Components.Physical_Component()
    apu                     = SUAVE.Components.Physical_Component()
    hydraulics              = SUAVE.Components.Physical_Component()
    optionals               = SUAVE.Components.Physical_Component()
    rudder                  = SUAVE.Components.Physical_Component()
    avionics                = SUAVE.Components.Energy.Peripherals.Avionics()
    
    
    #assign output weights to objects
    landing_gear_component.mass_properties.mass         = output.landing_gear
    control_systems.mass_properties.mass                = output.systems_breakdown.control_systems
    electrical_systems.mass_properties.mass             = output.systems_breakdown.electrical
    passengers.mass_properties.mass                     = output.pax + output.bag
    furnishings.mass_properties.mass                    = output.systems_breakdown.furnish
    avionics.mass_properties.mass                       = output.systems_breakdown.avionics \
        + output.systems_breakdown.instruments                  
    air_conditioner.mass_properties.mass                = output.systems_breakdown.air_conditioner
    fuel.mass_properties.mass                           = output.fuel
    apu.mass_properties.mass                            = output.systems_breakdown.apu
    hydraulics.mass_properties.mass                     = output.systems_breakdown.hydraulics
    optionals.mass_properties.mass                      = output.systems_breakdown.optionals
    rudder.mass_properties.mass                         = output.rudder
    
    
    #assign components to vehicle
    vehicle.control_systems                     = control_systems
    vehicle.electrical_systems                  = electrical_systems
    vehicle.avionics                            = avionics
    vehicle.furnishings                         = furnishings
    vehicle.passenger_weights                   = passengers 
    vehicle.air_conditioner                     = air_conditioner
    vehicle.fuel                                = fuel
    vehicle.apu                                 = apu
    vehicle.hydraulics                          = hydraulics
    vehicle.optionals                           = optionals
    vehicle.landing_gear                        = landing_gear_component
    vehicle.wings['vertical_stabilizer'].rudder = rudder

    return output
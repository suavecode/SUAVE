## @ingroup Methods-Weights-Correlations-Common
# weight_transport.py
#
# Created:  May 2020, W. Van Gijseghem
# Modified:

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import SUAVE
import numpy as np
from SUAVE.Core import Data
from SUAVE.Methods.Weights.Correlations import Propulsion as Propulsion
from SUAVE.Methods.Weights.Correlations.FLOPS.prop_system import total_prop_flops
from SUAVE.Methods.Weights.Correlations.FLOPS.systems import systems_FLOPS
from SUAVE.Methods.Weights.Correlations.FLOPS.operating_items import operating_items_FLOPS
from SUAVE.Methods.Weights.Correlations.FLOPS.wing import wing_weight_FLOPS
from SUAVE.Methods.Weights.Correlations.FLOPS.tail import tail_horizontal_FLOPS
from SUAVE.Methods.Weights.Correlations.FLOPS.tail import tail_vertical_FLOPS
from SUAVE.Methods.Weights.Correlations.FLOPS.fuselage import fuselage_weight_FLOPS
from SUAVE.Methods.Weights.Correlations.FLOPS.landing_gear import landing_gear_FLOPS
from SUAVE.Methods.Weights.Correlations.FLOPS.payload import payload_FLOPS

from SUAVE.Methods.Weights.Correlations.Common.systems import systems
from SUAVE.Methods.Weights.Correlations.Transport.tail_horizontal import tail_horizontal
from SUAVE.Methods.Weights.Correlations.Transport.tail_vertical import tail_vertical
from SUAVE.Methods.Weights.Correlations.Transport.operating_items import operating_items
from SUAVE.Methods.Weights.Correlations.Transport.tube import tube
from SUAVE.Methods.Weights.Correlations.Common import wing_main
from SUAVE.Methods.Weights.Correlations.Common import landing_gear as landing_gear_weight
from SUAVE.Methods.Weights.Correlations.Common import payload as payload_weight

from SUAVE.Methods.Weights.Correlations.Raymer import wing_main_raymer
from SUAVE.Methods.Weights.Correlations.Raymer.tail import tail_horizontal_Raymer, tail_vertical_Raymer
from SUAVE.Methods.Weights.Correlations.Raymer import fuselage_weight_Raymer
from SUAVE.Methods.Weights.Correlations.Raymer import landing_gear_Raymer
from SUAVE.Methods.Weights.Correlations.Raymer import systems_Raymer
from SUAVE.Methods.Weights.Correlations.Raymer import total_prop_Raymer

from SUAVE.Attributes.Solids.Aluminum import Aluminum

def empty_weight(vehicle, settings=None, method_type='New SUAVE'):
    """ Main function that estimates the zero-fuel weight of a transport aircraft:
        - MTOW = WZFW + FUEL
        - WZFW = WOE + WPAYLOAD
        - WOE = WE + WOPERATING_ITEMS
        - WE = WSTRCT + WPROP + WSYS
        Assumptions:

        Source:
            FLOPS method: The Flight Optimization System Weight Estimation Method
            SUAVE method: http://aerodesign.stanford.edu/aircraftdesign/AircraftDesign.html
            RAYMER method: Aircraft Design A Conceptual Approach
       Inputs:
            vehicle - data dictionary with vehicle properties               [dimensionless]
                -.networks: data dictionary with all the network elements and properties
                    -.total_weight: total weight of the propulsion system   [kg] 
                      (optional, calculated if not included)
                -.fuselages: data dictionary with the fuselage properties of the vehicle
                -.wings: data dictionary with all the wing properties of the vehicle, including horzinotal and vertical stabilizers
                -.wings['main_wing']: data dictionary with main wing properties
                    -.flap_ratio: flap surface area over wing surface area
                -.mass_properties: data dictionary with all the main mass properties of the vehicle including MTOW, ZFW, EW and OEW

            settings.weight_reduction_factors.
                    main_wing                                               [dimensionless] (.1 is a 10% weight reduction)
                    empennage                                               [dimensionless] (.1 is a 10% weight reduction)
                    fuselage                                                [dimensionless] (.1 is a 10% weight reduction)
            method_type - weight estimation method chosen, available:
                            - FLOPS Simple
                            - FLOPS Complex
                            - SUAVE
                            - New SUAVE
                            - Raymer
       Outputs:
            output - data dictionary with the weight breakdown of the vehicle
                        -.structures: structural weight
                            -.wing: wing weight
                            -.horizontal_tail: horizontal tail weight
                            -.vertical_tail: vertical tail weight
                            -.fuselage: fuselage weight
                            -.main_landing_gear: main landing gear weight
                            -.nose_landing_gear: nose landing gear weight
                            -.nacelle: nacelle weight
                            -.paint: paint weight
                            -.total: total strucural weight

                        -.propulsion_breakdown: propulsive system weight
                            -.engines: dry engine weight
                            -.thrust_reversers: thrust reversers weight
                            -.miscellaneous: miscellaneous items includes electrical system for engines and starter engine
                            -.fuel_system: fuel system weight
                            -.total: total propulsive system weight

                        -.systems_breakdown: system weight
                            -.control_systems: control system weight
                            -.apu: apu weight
                            -.electrical: electrical system weight
                            -.avionics: avionics weight
                            -.hydraulics: hydraulics and pneumatic system weight
                            -.furnish: furnishing weight
                            -.air_conditioner: air conditioner weight
                            -.instruments: instrumentation weight
                            -.anti_ice: anti ice system weight
                            -.total: total system weight

                        -.payload_breakdown: payload weight
                            -.passengers: passenger weight
                            -.bagage: baggage weight
                            -.cargo: cargo weight
                            -.total: total payload weight

                        -.operational_items: operational items weight
                            -.operating_items_less_crew: unusable fuel, engine oil, passenger service weight and cargo containers
                            -.flight_crew: flight crew weight
                            -.flight_attendants: flight attendants weight
                            -.total: total operating items weight

                        -.empty = structures.total + propulsion_breakdown.total + systems_breakdown.total
                        -.operating_empty = empty + operational_items.total
                        -.zero_fuel_weight = operating_empty + payload_breakdown.total
                        -.fuel = vehicle.mass_properties.max_takeoff - zero_fuel_weight


        Properties Used:
            N/A
    """
    
    Wings = SUAVE.Components.Wings
    Nets  = SUAVE.Components.Energy.Networks    
    
    if method_type == 'FLOPS Simple' or method_type == 'FLOPS Complex':
        if not hasattr(vehicle, 'design_mach_number'):
            raise ValueError("FLOPS requires a design mach number for sizing!")
        if not hasattr(vehicle, 'design_range'):
            raise ValueError("FLOPS requires a design range for sizing!")
        if not hasattr(vehicle, 'design_cruise_alt'):
            raise ValueError("FLOPS requires a cruise altitude for sizing!")
        if not hasattr(vehicle, 'flap_ratio'):
            if vehicle.systems.accessories == "sst":
                flap_ratio = 0.22
            else:
                flap_ratio = 0.33
            for wing in vehicle.wings:
                if isinstance(wing, Wings.Main_Wing):
                    wing.flap_ratio = flap_ratio

    # Set the factors
    if not hasattr(settings, 'weight_reduction_factors'):
        wt_factors = Data()
        wt_factors.main_wing    = 0.
        wt_factors.empennage    = 0.
        wt_factors.fuselage     = 0.
        wt_factors.structural   = 0.
        wt_factors.systems      = 0.
    else:
        wt_factors = settings.weight_reduction_factors
        if 'structural' in wt_factors and wt_factors.structural != 0.:
            print('Overriding individual structural weight factors')
            wt_factors.main_wing    = 0.
            wt_factors.empennage    = 0.
            wt_factors.fuselage     = 0.
            wt_factors.systems      = 0.
        else:
            wt_factors.structural   = 0.
            wt_factors.systems      = 0.

    # Prop weight (propulsion pod weight is calculated separately)
    wt_prop_total   = 0
    wt_prop_data    = None
    for prop in vehicle.networks:
        if isinstance(prop, Nets.Turbofan) or isinstance(prop, Nets.Turbojet_Super) or isinstance(prop,
                                                                                                  Nets.Propulsor_Surrogate):
            
            if not isinstance(prop.wing_mounted,list):
                prop.wing_mounted = [prop.wing_mounted] * int(prop.number_of_engines)
            else:
                if len(prop.wing_mounted) != int(prop.number_of_engines):
                    raise ValueError("Length of wing_mounted list does not match number of engines")            
            
            num_eng     = prop.number_of_engines
            thrust_sls  = prop.sealevel_static_thrust
            if 'total_weight' in prop.keys():
                wt_prop         = prop.total_weight
            elif method_type == 'FLOPS Simple' or method_type == 'FLOPS Complex':
                wt_prop_data    = total_prop_flops(vehicle, prop)
                wt_prop         = wt_prop_data.wt_prop
            elif method_type == 'Raymer':
                wt_prop_data    = total_prop_Raymer(vehicle, prop)
                wt_prop         = wt_prop_data.wt_prop

            else:
                wt_engine_jet = Propulsion.engine_jet(thrust_sls)
                wt_prop       = Propulsion.integrated_propulsion(wt_engine_jet, num_eng)

            if num_eng == 0.:
                wt_prop = 0.
            prop.mass_properties.mass = wt_prop
            wt_prop_total += wt_prop

    # Payload Weight
    if method_type == 'FLOPS Simple' or method_type == 'FLOPS Complex':
        payload = payload_FLOPS(vehicle)
    else:
        payload = payload_weight(vehicle)
        
    vehicle.payload.passengers = SUAVE.Components.Physical_Component()
    vehicle.payload.baggage    = SUAVE.Components.Physical_Component()
    vehicle.payload.cargo      = SUAVE.Components.Physical_Component()
    
    vehicle.payload.passengers.mass_properties.mass = payload.passengers
    vehicle.payload.baggage.mass_properties.mass    = payload.baggage
    vehicle.payload.cargo.mass_properties.mass      = payload.cargo
    
    # Operating Items Weight
    if method_type == 'FLOPS Simple' or method_type == 'FLOPS Complex':
        wt_oper = operating_items_FLOPS(vehicle)
    else:
        wt_oper = operating_items(vehicle)

    # System Weight
    if method_type == 'FLOPS Simple' or method_type == 'FLOPS Complex':
        wt_sys = systems_FLOPS(vehicle)
    elif method_type == 'Raymer':
        wt_sys = systems_Raymer(vehicle)
    else:
        wt_sys = systems(vehicle)
    for item in wt_sys.keys():
        wt_sys[item] *= (1. - wt_factors.systems)

    # Prop pod weight
    WPOD = 0.0
    if method_type == 'FLOPS Complex':
        network_name  = list(vehicle.networks.keys())[0]
        networks      = vehicle.networks[network_name]
        NENG            = networks.number_of_engines
        WTNFA = wt_prop_data.wt_eng + wt_prop_data.wt_thrust_reverser + wt_prop_data.wt_starter \
                + 0.25 * wt_prop_data.wt_engine_controls + 0.11 * wt_sys.wt_instruments + 0.13 * wt_sys.wt_elec \
                + 0.13 * wt_sys.wt_hyd_pnu + 0.25 * wt_prop_data.fuel_system
        WPOD = WTNFA / np.max([1, NENG]) + wt_prop_data.nacelle / np.max(
            [1.0, NENG + 1. / 2 * (NENG - 2 * np.floor(NENG / 2.))])

    # Wing Weight
    wt_main_wing        = 0.0
    wt_tail_horizontal  = 0.0
    wt_tail_vertical    = 0.0
    
    rho      = Aluminum().density
    sigma    = Aluminum().yield_tensile_strength      
    
    num_main_wings = 0
    for wing in vehicle.wings:
        if isinstance(wing, Wings.Main_Wing):
            num_main_wings += 1
    
    for wing in vehicle.wings:
        if isinstance(wing, Wings.Main_Wing):
            if method_type == 'SUAVE':
                wt_wing = wing_main(vehicle, wing, rho, sigma, computation_type='simple')
            elif method_type == 'New SUAVE':
                wt_wing = wing_main(vehicle, wing, rho, sigma,computation_type='segmented')
            elif method_type == 'FLOPS Simple' or method_type == 'FLOPS Complex':
                complexity = method_type.split()[1]
                wt_wing = wing_weight_FLOPS(vehicle, wing, WPOD, complexity, settings,
                                            num_main_wings)
            elif method_type == 'Raymer':
                wt_wing = wing_main_raymer(vehicle, wing)
            else:
                raise ValueError("This weight method is not yet implemented")
            # Apply weight factor
            wt_wing = wt_wing * (1. - wt_factors.main_wing) * (1. - wt_factors.structural)
            if np.isnan(wt_wing):
                wt_wing = 0.
            wing.mass_properties.mass = wt_wing
            wt_main_wing += wt_wing
        if isinstance(wing, Wings.Horizontal_Tail):
            if method_type == 'FLOPS Simple' or method_type == 'FLOPS Complex':
                wt_tail = tail_horizontal_FLOPS(vehicle, wing)
            elif method_type == 'Raymer':
                wt_tail = tail_horizontal_Raymer(vehicle, wing)
            else:
                wt_tail = tail_horizontal(vehicle, wing)
            if type(wt_tail) == np.ndarray:
                wt_tail = sum(wt_tail)
            # Apply weight factor
            wt_tail = wt_tail * (1. - wt_factors.empennage) * (1. - wt_factors.structural)
            # Pack and sum
            wing.mass_properties.mass = wt_tail
            wt_tail_horizontal += wt_tail
        if isinstance(wing, Wings.Vertical_Tail):
            if method_type == 'FLOPS Simple' or method_type == 'FLOPS Complex':
                wt_tail = tail_vertical_FLOPS(vehicle, wing)
            elif method_type == 'Raymer':
                wt_tail = tail_vertical_Raymer(vehicle, wing)
            else:
                wt_tail = tail_vertical(vehicle, wing)
            # Apply weight factor
            wt_tail = wt_tail * (1. - wt_factors.empennage) * (1. - wt_factors.structural)
            # Pack and sum
            wing.mass_properties.mass = wt_tail
            wt_tail_vertical += wt_tail

    # Fuselage Weight
    wt_fuse_total = 0
    for fuse in vehicle.fuselages:
        if method_type == 'FLOPS Simple' or method_type == 'FLOPS Complex':
            wt_fuse = fuselage_weight_FLOPS(vehicle)
        elif method_type == 'Raymer':
            wt_fuse = fuselage_weight_Raymer(vehicle, fuse, settings)
        else:
            wt_fuse = tube(vehicle, fuse, wt_main_wing, wt_prop_total)
        wt_fuse = wt_fuse * (1. - wt_factors.fuselage) * (1. - wt_factors.structural)
        fuse.mass_properties.mass = wt_fuse
        wt_fuse_total += wt_fuse

    # Landing Gear Weight
    if method_type == 'FLOPS Simple' or method_type == 'FLOPS Complex':
        landing_gear = landing_gear_FLOPS(vehicle)
    elif method_type == 'Raymer':
        landing_gear = landing_gear_Raymer(vehicle)
    else:
        landing_gear = landing_gear_weight(vehicle)

    # Distribute all weight in the output fields
    output                                  = Data()
    output.structures                       = Data()
    output.structures.wing                  = wt_main_wing
    output.structures.horizontal_tail       = wt_tail_horizontal
    output.structures.vertical_tail         = wt_tail_vertical
    output.structures.fuselage              = wt_fuse_total
    output.structures.main_landing_gear     = landing_gear.main
    output.structures.nose_landing_gear     = landing_gear.nose
    if wt_prop_data is None:
        output.structures.nacelle = 0
    else:
        output.structures.nacelle = wt_prop_data.nacelle
    if 'FLOPS' in method_type:
        print('Paint weight is currently ignored in FLOPS calculations.')
    output.structures.paint = 0  # TODO reconcile FLOPS paint calculations with Raymer and SUAVE baseline
    output.structures.total = output.structures.wing + output.structures.horizontal_tail + output.structures.vertical_tail \
                              + output.structures.fuselage + output.structures.main_landing_gear + output.structures.nose_landing_gear \
                              + output.structures.paint + output.structures.nacelle

    output.propulsion_breakdown = Data()
    if wt_prop_data is None:
        output.propulsion_breakdown.total               = wt_prop_total
        output.propulsion_breakdown.engines             = 0
        output.propulsion_breakdown.thrust_reversers    = 0
        output.propulsion_breakdown.miscellaneous       = 0
        output.propulsion_breakdown.fuel_system         = 0
    else:
        output.propulsion_breakdown.total               = wt_prop_total
        output.propulsion_breakdown.engines             = wt_prop_data.wt_eng
        output.propulsion_breakdown.thrust_reversers    = wt_prop_data.wt_thrust_reverser
        output.propulsion_breakdown.miscellaneous       = wt_prop_data.wt_engine_controls + wt_prop_data.wt_starter
        output.propulsion_breakdown.fuel_system         = wt_prop_data.fuel_system

    output.systems_breakdown                        = Data()
    output.systems_breakdown.control_systems        = wt_sys.wt_flight_control
    output.systems_breakdown.apu                    = wt_sys.wt_apu
    output.systems_breakdown.electrical             = wt_sys.wt_elec
    output.systems_breakdown.avionics               = wt_sys.wt_avionics
    output.systems_breakdown.hydraulics             = wt_sys.wt_hyd_pnu
    output.systems_breakdown.furnish                = wt_sys.wt_furnish
    output.systems_breakdown.air_conditioner        = wt_sys.wt_ac + wt_sys.wt_anti_ice # Anti-ice is sometimes included in ECS
    output.systems_breakdown.instruments            = wt_sys.wt_instruments
    output.systems_breakdown.total                  = output.systems_breakdown.control_systems + output.systems_breakdown.apu \
                                                    + output.systems_breakdown.electrical + output.systems_breakdown.avionics \
                                                    + output.systems_breakdown.hydraulics + output.systems_breakdown.furnish \
                                                    + output.systems_breakdown.air_conditioner + output.systems_breakdown.instruments

    output.payload_breakdown = Data()
    output.payload_breakdown = payload

    output.operational_items = Data()
    output.operational_items = wt_oper

    output.empty                = output.structures.total + output.propulsion_breakdown.total + output.systems_breakdown.total
    output.operating_empty      = output.empty + output.operational_items.total
    output.zero_fuel_weight     = output.operating_empty + output.payload_breakdown.total
    output.fuel                 = vehicle.mass_properties.max_takeoff - output.zero_fuel_weight
    output.max_takeoff          = vehicle.mass_properties.max_takeoff

    control_systems         = SUAVE.Components.Physical_Component()
    electrical_systems      = SUAVE.Components.Physical_Component()
    furnishings             = SUAVE.Components.Physical_Component()
    air_conditioner         = SUAVE.Components.Physical_Component()
    fuel                    = SUAVE.Components.Physical_Component()
    apu                     = SUAVE.Components.Physical_Component()
    hydraulics              = SUAVE.Components.Physical_Component()
    avionics                = SUAVE.Components.Energy.Peripherals.Avionics()
    optionals               = SUAVE.Components.Physical_Component()

    if not hasattr(vehicle.landing_gear, 'nose'):
        vehicle.landing_gear.nose       = SUAVE.Components.Landing_Gear.Nose_Landing_Gear()
    vehicle.landing_gear.nose.mass  = output.structures.nose_landing_gear
    if not hasattr(vehicle.landing_gear, 'main'):
        vehicle.landing_gear.main       = SUAVE.Components.Landing_Gear.Main_Landing_Gear()   
    vehicle.landing_gear.main.mass  = output.structures.main_landing_gear  

    control_systems.mass_properties.mass        = output.systems_breakdown.control_systems
    electrical_systems.mass_properties.mass     = output.systems_breakdown.electrical
    furnishings.mass_properties.mass            = output.systems_breakdown.furnish
    avionics.mass_properties.mass               = output.systems_breakdown.avionics \
                                                + output.systems_breakdown.instruments
    air_conditioner.mass_properties.mass        = output.systems_breakdown.air_conditioner
    fuel.mass_properties.mass                   = output.fuel
    apu.mass_properties.mass                    = output.systems_breakdown.apu
    hydraulics.mass_properties.mass             = output.systems_breakdown.hydraulics
    optionals.mass_properties.mass              = output.operational_items.operating_items_less_crew

    # assign components to vehicle
    vehicle.systems.control_systems         = control_systems
    vehicle.systems.electrical_systems      = electrical_systems
    vehicle.systems.avionics                = avionics
    vehicle.systems.furnishings             = furnishings
    vehicle.systems.air_conditioner         = air_conditioner
    vehicle.systems.fuel                    = fuel
    vehicle.systems.apu                     = apu
    vehicle.systems.hydraulics              = hydraulics
    vehicle.systems.optionals               = optionals   

    return output

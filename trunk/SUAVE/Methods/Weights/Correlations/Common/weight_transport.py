## @ingroup Methods-Weights-Correlations-Common
# weight_transport.pu
#
# Created:  May 2020, W. Van Gijseghem
# Modified:

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import SUAVE
import numpy as np
from SUAVE.Core import Data
import SUAVE.Components.Energy.Networks as Nets
from SUAVE.Methods.Weights.Correlations import Propulsion as Propulsion
from SUAVE.Methods.Weights.Correlations.FLOPS.prop_system import total_prop_flops
from SUAVE.Methods.Weights.Correlations.FLOPS.systems import systems_FLOPS
from SUAVE.Methods.Weights.Correlations.FLOPS.operating_items import operating_system_FLOPS
from SUAVE.Methods.Weights.Correlations.FLOPS.wing_weight import wing_weight_FLOPS
from SUAVE.Methods.Weights.Correlations.FLOPS.tail_weight import tail_horizontal_FLOPS
from SUAVE.Methods.Weights.Correlations.FLOPS.tail_weight import tail_vertical_FLOPS
from SUAVE.Methods.Weights.Correlations.FLOPS.fuselage import fuselage_weight_FLOPS
from SUAVE.Methods.Weights.Correlations.FLOPS.landing_gear import landing_gear_FLOPS
from SUAVE.Methods.Weights.Correlations.FLOPS.payload import payload_FLOPS

from SUAVE.Methods.Weights.Correlations.Tube_Wing.systems import systems
from SUAVE.Methods.Weights.Correlations.Tube_Wing.tail_horizontal import tail_horizontal
from SUAVE.Methods.Weights.Correlations.Tube_Wing.tail_vertical import tail_vertical
from SUAVE.Methods.Weights.Correlations.Tube_Wing.operating_items import operating_system
from SUAVE.Methods.Weights.Correlations.Tube_Wing.tube import tube
from SUAVE.Methods.Weights.Correlations.Common import wing_main
from SUAVE.Methods.Weights.Correlations.Common import wing_main_update
from SUAVE.Methods.Weights.Correlations.Common import landing_gear as landing_gear_weight
from SUAVE.Methods.Weights.Correlations.Common import payload as payload_weight

from SUAVE.Methods.Weights.Correlations.Raymer import wing_main_raymer
from SUAVE.Methods.Weights.Correlations.Raymer.tail_weight import tail_horizontal_Raymer, tail_vertical_Raymer
from SUAVE.Methods.Weights.Correlations.Raymer import fuselage_weight_Raymer
from SUAVE.Methods.Weights.Correlations.Raymer import landing_gear_Raymer
from SUAVE.Methods.Weights.Correlations.Raymer import systems_Raymer
from SUAVE.Methods.Weights.Correlations.Raymer import total_prop_Raymer
import SUAVE.Components.Wings as Wings


def empty_weight(vehicle, settings=None, conditions=None, method_type='SUAVE'):
    if method_type == 'FLOPS Simple' or method_type == 'FLOPS Complex':
        if not hasattr(vehicle, 'design_mach_number'):
            raise ValueError("FLOPS requires a design mach number for sizing!")
        if not hasattr(vehicle, 'design_range'):
            raise ValueError("FLOPS requires a design range for sizing!")
        if not hasattr(vehicle, 'design_cruise_alt'):
            raise ValueError("FLOPS requires a cruise altitude for sizing!")
        if not hasattr(vehicle, 'flap_ratio'):
            if vehicle.systems.accessories == "sst":
                vehicle.flap_ratio = 0.22
            else:
                vehicle.flap_ratio = 0.33

    # Set the factors
    if settings is None:
        wt_factors = Data()
        wt_factors.main_wing = 0.
        wt_factors.empennage = 0.
        wt_factors.fuselage = 0.
        wt_factors.structural = 0.
        wt_factors.systems = 0.
    else:
        wt_factors = settings.weight_reduction_factors
        if 'structural' in wt_factors and wt_factors.structural != 0.:
            print('Overriding individual structural weight factors')
            wt_factors.main_wing = 0.
            wt_factors.empennage = 0.
            wt_factors.fuselage = 0.
            wt_factors.systems = 0.
        else:
            wt_factors.structural = 0.
            wt_factors.systems = 0.

    # Prop weight
    wt_prop_total = 0
    wt_prop_data = None
    for prop in vehicle.propulsors:
        if isinstance(prop, Nets.Turbofan) or isinstance(prop, Nets.Turbojet_Super) or isinstance(prop,
                                                                                                  Nets.Propulsor_Surrogate):
            num_eng = prop.number_of_engines
            thrust_sls = prop.sealevel_static_thrust
            if 'total_weight' in prop.keys():
                wt_prop = prop.total_weight
            elif method_type == 'FLOPS Simple' or method_type == 'FLOPS Complex':
                wt_prop_data = total_prop_flops(vehicle, prop)
                wt_prop = wt_prop_data.wt_prop
            elif method_type == 'Raymer':
                wt_prop_data = total_prop_Raymer(vehicle, prop)
                wt_prop = wt_prop_data.wt_prop

            else:
                wt_engine_jet = Propulsion.engine_jet(thrust_sls)
                wt_prop = Propulsion.integrated_propulsion(wt_engine_jet, num_eng)

            if num_eng == 0.:
                wt_prop = 0.

            prop.mass_properties.mass = wt_prop

            wt_prop_total += wt_prop

    # Payload Weight
    if method_type == 'FLOPS Simple' or method_type == 'FLOPS Complex':
        payload = payload_FLOPS(vehicle)
    else:
        payload = payload_weight(vehicle)
    vehicle.payload = payload.total
    # Operating Items Weight
    if method_type == 'FLOPS Simple' or method_type == 'FLOPS Complex':
        wt_oper = operating_system_FLOPS(vehicle)
    else:
        wt_oper = operating_system(vehicle)

    # System Weight
    if method_type == 'FLOPS Simple' or method_type == 'FLOPS Complex':
        wt_sys = systems_FLOPS(vehicle)
    elif method_type == 'Raymer':
        wt_sys = systems_Raymer(vehicle)
    else:
        wt_sys = systems(vehicle)
    for item in wt_sys.keys():
        wt_sys[item] *= (1. - wt_factors.systems)

    WPOD = 0.0
    if method_type == 'FLOPS Complex':
        propulsor_name = list(vehicle.propulsors.keys())[0]
        propulsors = vehicle.propulsors[propulsor_name]
        NENG = propulsors.number_of_engines
        WTNFA = wt_prop_data.wt_eng + wt_prop_data.wt_thrust_reverser + wt_prop_data.wt_starter \
                + 0.25 * wt_prop_data.wt_engine_controls + 0.11 * wt_sys.wt_instruments + 0.13 * wt_sys.wt_elec \
                + 0.13 * wt_sys.wt_hyd_pnu + 0.25 * wt_prop_data.fuel_system
        WPOD = WTNFA / np.max([1, NENG]) + wt_prop_data.nacelle / np.max(
            [1.0, NENG + 1. / 2 * (NENG - 2 * np.floor(NENG / 2.))])

    # Wing Weight
    wt_main_wing = 0.0
    wt_tail_horizontal = 0.0
    wt_tail_vertical = 0.0
    for wing in vehicle.wings:
        if isinstance(wing, Wings.Main_Wing):
            if method_type == 'SUAVE':
                wt_wing = wing_main(vehicle, wing)
            elif method_type == 'New SUAVE':
                wt_wing = wing_main_update(vehicle, wing)
            elif method_type == 'FLOPS Simple' or method_type == 'FLOPS Complex':
                complexity = method_type.split()[1]
                wt_wing = wing_weight_FLOPS(vehicle, WPOD, complexity)
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
            wt_fuse = fuselage_weight_Raymer(vehicle, fuse)
        else:
            wt_fuse = tube(vehicle, fuse, wt_main_wing, wt_prop_total)
        wt_fuse = wt_fuse * (1. - wt_factors.fuselage) * (1. - wt_factors.structural)
        fuse.mass_properties.mass = wt_fuse
        wt_fuse_total += wt_fuse

    # Landing Gear Weigth
    if method_type == 'FLOPS Simple' or method_type == 'FLOPS Complex':
        landing_gear = landing_gear_FLOPS(vehicle)
    elif method_type == 'Raymer':
        landing_gear = landing_gear_Raymer(vehicle)
    else:
        landing_gear = landing_gear_weight(vehicle)

    output = Data()
    output.structures = Data()
    output.structures.wing = wt_main_wing
    output.structures.horizontal_tail = wt_tail_horizontal
    output.structures.vertical_tail = wt_tail_vertical
    output.structures.fuselage = wt_fuse_total
    output.structures.main_landing_gear = landing_gear.main
    output.structures.nose_landing_gear = landing_gear.nose
    if wt_prop_data is None:
        output.structures.nacelle = 0
    else:
        output.structures.nacelle = wt_prop_data.nacelle
    output.structures.paint = 0  # TODO change
    output.structures.total = output.structures.wing + output.structures.horizontal_tail + output.structures.vertical_tail \
                              + output.structures.fuselage + output.structures.main_landing_gear + output.structures.nose_landing_gear \
                              + output.structures.paint + output.structures.nacelle

    output.propulsion_breakdown = Data()
    if wt_prop_data is None:
        output.propulsion_breakdown.total = wt_prop_total
        output.propulsion_breakdown.engines = 0
        output.propulsion_breakdown.thrust_reversers = 0
        output.propulsion_breakdown.miscellaneous = 0
        output.propulsion_breakdown.fuel_system = 0
    else:
        output.propulsion_breakdown.total = wt_prop_total
        output.propulsion_breakdown.engines = wt_prop_data.wt_eng
        output.propulsion_breakdown.thrust_reversers = wt_prop_data.wt_thrust_reverser
        output.propulsion_breakdown.miscellaneous = wt_prop_data.wt_engine_controls + wt_prop_data.wt_starter
        output.propulsion_breakdown.fuel_system = wt_prop_data.fuel_system

    output.systems_breakdown = Data()
    output.systems_breakdown.control_systems = wt_sys.wt_flt_ctrl
    output.systems_breakdown.apu = wt_sys.wt_apu
    output.systems_breakdown.electrical = wt_sys.wt_elec
    output.systems_breakdown.avionics = wt_sys.wt_avionics
    output.systems_breakdown.hydraulics = wt_sys.wt_hyd_pnu
    output.systems_breakdown.furnish = wt_sys.wt_furnish
    output.systems_breakdown.air_conditioner = wt_sys.wt_ac
    output.systems_breakdown.instruments = wt_sys.wt_instruments
    output.systems_breakdown.anti_ice = wt_sys.wt_anti_ice
    output.systems_breakdown.total = output.systems_breakdown.control_systems + output.systems_breakdown.apu \
                                     + output.systems_breakdown.electrical + output.systems_breakdown.avionics \
                                     + output.systems_breakdown.hydraulics + output.systems_breakdown.furnish \
                                     + output.systems_breakdown.air_conditioner + output.systems_breakdown.instruments \
                                     + output.systems_breakdown.anti_ice

    output.payload_breakdown = Data()
    output.payload_breakdown = payload

    output.operational_items = Data()
    output.operational_items = wt_oper

    output.empty = output.structures.total + output.propulsion_breakdown.total + output.systems_breakdown.total
    output.operating_empty = output.empty + output.operational_items.total
    output.zero_fuel_weight = output.operating_empty + output.payload_breakdown.total
    output.fuel = vehicle.mass_properties.max_takeoff - output.zero_fuel_weight

    control_systems = SUAVE.Components.Physical_Component()
    electrical_systems = SUAVE.Components.Physical_Component()
    passengers = SUAVE.Components.Physical_Component()
    furnishings = SUAVE.Components.Physical_Component()
    air_conditioner = SUAVE.Components.Physical_Component()
    fuel = SUAVE.Components.Physical_Component()
    apu = SUAVE.Components.Physical_Component()
    hydraulics = SUAVE.Components.Physical_Component()
    avionics = SUAVE.Components.Energy.Peripherals.Avionics()

    # assign output weights to objects
    vehicle.landing_gear.mass_properties.mass = output.structures.main_landing_gear + output.structures.nose_landing_gear
    control_systems.mass_properties.mass = output.systems_breakdown.control_systems
    electrical_systems.mass_properties.mass = output.systems_breakdown.electrical
    passengers.mass_properties.mass = output.payload_breakdown.passengers + output.payload_breakdown.baggage
    furnishings.mass_properties.mass = output.systems_breakdown.furnish
    avionics.mass_properties.mass = output.systems_breakdown.avionics \
                                    + output.systems_breakdown.instruments
    air_conditioner.mass_properties.mass = output.systems_breakdown.air_conditioner
    fuel.mass_properties.mass = output.fuel
    apu.mass_properties.mass = output.systems_breakdown.apu
    hydraulics.mass_properties.mass = output.systems_breakdown.hydraulics

    # assign components to vehicle
    vehicle.control_systems = control_systems
    vehicle.electrical_systems = electrical_systems
    vehicle.avionics = avionics
    vehicle.furnishings = furnishings
    vehicle.passenger_weights = passengers
    vehicle.air_conditioner = air_conditioner
    vehicle.fuel = fuel
    vehicle.apu = apu
    vehicle.hydraulics = hydraulics

    return output

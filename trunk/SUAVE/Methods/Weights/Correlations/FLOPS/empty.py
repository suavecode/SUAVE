import SUAVE
from SUAVE.Core import Units, Data
from SUAVE.Methods.Weights.Correlations import Propulsion as Propulsion
import warnings
from .landing_gear import *
from .prop_system import *
from .fuselage import *
from .systems import *
from .tail_weight import *
from .wing_weight import *
from .operating_items import *
from .payload import *
def empty(vehicle, settings=None, conditions = None):

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
    propulsor_name = list(vehicle.propulsors.keys())[0]
    propulsors = vehicle.propulsors[propulsor_name]
    propulsor_name = list(vehicle.propulsors.keys())[0]
    NENG = propulsors.number_of_engines
    if propulsors.number_of_engines != len(propulsors.wing_mounted):
        raise ValueError("Make sure to indicate if all engines are wing mounted using wing_mounted = []")
    WUF, WOIL, WSRV, WCON = operating_system_FLOPS(vehicle)
    WSC, WAPU, WIN, WHYD, WELEC, WFURN, WAC, WAI, WSTUAB, WFLCRB, WAVONC = systems_FLOPS(vehicle)
    WPRO, WNAC, WFSYS, WEC, WSTART, WTHR, WENG = total_prop_flops(vehicle)
    WTNFA = NENG * WENG + WTHR + WSTART + 0.25 * WEC + 0.11 * WIN + 0.13 * WELEC + 0.13 * WHYD + 0.25 * WFSYS
    WPOD = WTNFA / np.max([1,NENG]) + WNAC / np.max([1.0,NENG + 1. / 2 * (NENG - 2 * np.floor(NENG / 2.))])

    if 'main_wing' not in vehicle.wings:
        WWING = 0.0
        warnings.warn("There is no Wing Weight being added to the Configuration", stacklevel=1)
    else:
        WWING = wing_weight_FLOPS(vehicle, WPOD, settings.complexity, conditions)
        vehicle.wings['main_wing'].mass_properties.mass = WWING

    if 'horizontal_stabilizer' not in vehicle.wings:
        WHT = 0.0
        warnings.warn("There is no Horizontal Tail Weight being added to the Configuration", stacklevel=1)
    else:
        WHT = tail_horizontal_FLOPS(vehicle)
        vehicle.wings['horizontal_stabilizer'].mass_properties.mass = WHT

    if 'vertical_stabilizer' not in vehicle.wings:
        WVT = 0.0
        warnings.warn("There is no Vertical Tail Weight being added to the Configuration", stacklevel=1)
    else:
        WVT = tail_vertical_FLOPS(vehicle)
        vehicle.wings['vertical_stabilizer'].mass_properties.mass = WVT

    WFUSE = fuselage_weight_FLOPS(vehicle)
    WLGM, WLGN = landing_gear_FLOPS(vehicle)
    try:
        landing_gear_component = vehicle.landing_gear  # landing gear previously defined
    except AttributeError:  # landing gear not defined
        landing_gear_component = SUAVE.Components.Landing_Gear.Landing_Gear()
        vehicle.landing_gear = landing_gear_component
    landing_gear_component.mass_properties.mass = (WLGM + WLGN)

    WNAC = nacelle_FLOPS(vehicle)
    WSTRCT = WWING + WHT + WVT + WFUSE + WLGM + WLGN + WNAC

    WSYS = WSC + WAPU + WIN + WHYD + WELEC + WAVONC + WFURN + WAC + WAI
    WWE = WSTRCT + WPRO + WSYS
    WOPIT = WFLCRB + WSTUAB + WUF + WOIL + WSRV + WCON
    WOWE = WWE + WOPIT

    WPAYLOAD, WPASS, WBAG = payload_FLOPS(vehicle)
    WZF = WOWE + WPAYLOAD

    output = Data()

    output.structures = Data()
    output.structures.total = WSTRCT
    output.structures.wing = WWING
    output.structures.horizontal_tail = WHT
    output.structures.vertical_tail = WVT
    output.structures.fuselage = WFUSE
    output.structures.main_landing_gear = WLGM
    output.structures.nose_landing_gear = WLGN
    output.structures.nacelle = WNAC
    output.structures.paint = 0 #  TODO change

    output.propulsion_breakdown = Data()
    output.propulsion_breakdown.total = WPRO
    output.propulsion_breakdown.engines = WENG
    output.propulsion_breakdown.thrust_reversers = WTHR
    output.propulsion_breakdown.miscellaneous = WEC + WSTART
    output.propulsion_breakdown.fuel_system = WFSYS

    output.systems_breakdown = Data()
    output.systems_breakdown.total = WSYS
    output.systems_breakdown.control_systems = WSC
    output.systems_breakdown.apu = WAPU
    output.systems_breakdown.electrical = WELEC
    output.systems_breakdown.avionics = WAVONC
    output.systems_breakdown.hydraulics = WHYD
    output.systems_breakdown.furnish = WFURN
    output.systems_breakdown.air_conditioner = WAC
    output.systems_breakdown.instruments = WIN
    output.systems_breakdown.optionals = 0 # TODO not sure what this includes

    output.payload_breakdown = Data()
    output.payload_breakdown.total = WPAYLOAD
    output.payload_breakdown.passengers = WPASS
    output.payload_breakdown.baggage = WBAG
    output.payload_breakdown.cargo = vehicle.mass_properties.cargo

    output.operational_items = Data()
    output.operational_items.total = WOPIT
    output.operational_items.flight_crew = WFLCRB
    output.operational_items.flight_attendants = WSTUAB
    output.operational_items.oil = WOIL
    output.operational_items.unusable_fuel = WUF
    output.operational_items.cargo_containers = WCON

    output.payload = WPAYLOAD
    output.pax = WPASS
    output.bag = WBAG
    output.empty = WWE
    output.wing = WWING
    output.fuselage = WFUSE
    output.propulsion = WPRO
    output.landing_gear = WLGM + WLGN
    output.horizontal_tail = WHT
    output.vertical_tail = WVT
    output.operating_empty = WOWE
    output.zero_fuel_weight = WZF


    output.fuel = vehicle.mass_properties.max_takeoff - output.zero_fuel_weight

    control_systems = SUAVE.Components.Physical_Component()
    electrical_systems = SUAVE.Components.Physical_Component()
    passengers = SUAVE.Components.Physical_Component()
    furnishings = SUAVE.Components.Physical_Component()
    air_conditioner = SUAVE.Components.Physical_Component()
    fuel = SUAVE.Components.Physical_Component()
    apu = SUAVE.Components.Physical_Component()
    hydraulics = SUAVE.Components.Physical_Component()
    optionals = SUAVE.Components.Physical_Component()
    rudder = SUAVE.Components.Physical_Component()
    avionics = SUAVE.Components.Energy.Peripherals.Avionics()

    # assign output weights to objects
    landing_gear_component.mass_properties.mass = output.landing_gear
    control_systems.mass_properties.mass = output.systems_breakdown.control_systems
    electrical_systems.mass_properties.mass = output.systems_breakdown.electrical
    passengers.mass_properties.mass = output.pax + output.bag
    furnishings.mass_properties.mass = output.systems_breakdown.furnish
    avionics.mass_properties.mass = output.systems_breakdown.avionics \
                                    + output.systems_breakdown.instruments
    air_conditioner.mass_properties.mass = output.systems_breakdown.air_conditioner
    fuel.mass_properties.mass = output.fuel
    apu.mass_properties.mass = output.systems_breakdown.apu
    hydraulics.mass_properties.mass = output.systems_breakdown.hydraulics
    optionals.mass_properties.mass = output.systems_breakdown.optionals
    rudder.mass_properties.mass = 0
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
    vehicle.optionals = optionals
    vehicle.landing_gear = landing_gear_component


    return output
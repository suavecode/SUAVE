## @ingroup Methods-Weights-Correlations-BWB
# empty.py
#
# Created:  Apr 2017, M. Clarke
# Modified: Jul 2017, M. Clarke
#           Apr 2020, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import SUAVE
from SUAVE.Core import Units, Data
from .cabin import cabin
from .aft_centerbody import aft_centerbody
from SUAVE.Methods.Weights.Correlations.Common.systems import systems
from SUAVE.Methods.Weights.Correlations.Common import wing_main as wing_main
from SUAVE.Methods.Weights.Correlations.Common import landing_gear as landing_gear_weight
from SUAVE.Methods.Weights.Correlations.Common import payload as payload_weight
from SUAVE.Methods.Weights.Correlations import Propulsion as Propulsion
from SUAVE.Methods.Weights.Correlations.Transport.operating_items import operating_items
from SUAVE.Attributes.Solids.Aluminum import Aluminum

import warnings


# ----------------------------------------------------------------------
#  Empty
# ----------------------------------------------------------------------

## @ingroup Methods-Weights-Correlations-BWB
def empty(vehicle):
    """ This is for a BWB aircraft configuration.

    Assumptions:
         Calculated aircraft weight from correlations created per component of historical aircraft
         The wings are made out of aluminum.
         A wing with the tag 'main_wing' exists.

    Source:
        N/A

    Inputs:
        engine - a data dictionary with the fields:
            thrust_sls - sea level static thrust of a single engine                                        [Newtons]

        wing - a data dictionary with the fields:
            gross_area - wing gross area                                                                   [meters**2]
            span - span of the wing                                                                        [meters]
            taper - taper ratio of the wing                                                                [dimensionless]
            t_c - thickness-to-chord ratio of the wing                                                     [dimensionless]
            sweep - sweep angle of the wing                                                                [radians]
            mac - mean aerodynamic chord of the wing                                                       [meters]
            r_c - wing root chord                                                                          [meters]

        aircraft - a data dictionary with the fields:
            Nult - ultimate load of the aircraft                                                           [dimensionless]
            Nlim - limit load factor at zero fuel weight of the aircraft                                   [dimensionless]
            TOW - maximum takeoff weight of the aircraft                                                   [kilograms]
            zfw - maximum zero fuel weight of the aircraft                                                 [kilograms]
            num_eng - number of engines on the aircraft                                                    [dimensionless]
            num_pax - number of passengers on the aircraft                                                 [dimensionless]
            wt_cargo - weight of the bulk cargo being carried on the aircraft                              [kilograms]
            num_seats - number of seats installed on the aircraft                                          [dimensionless]
            ctrl - specifies if the control system is "fully powered", "partially powered", or not powered [dimensionless]
            ac - determines type of instruments, electronics, and operating items based on types:
                "short-range", "medium-range", "long-range", "business", "cargo", "commuter", "sst"        [dimensionless]

         fuselage - a data dictionary with the fields:
            area - fuselage wetted area                                                                    [meters**2]
            diff_p - Maximum fuselage pressure differential                                                [Pascal]
            width - width of the fuselage                                                                  [meters]
            height - height of the fuselage                                                                [meters]
            length - length of the fuselage                                                                [meters]

    Outputs:
        output - a data dictionary with fields:
            wt_payload - weight of the passengers plus baggage and paid cargo                              [kilograms]
            wt_pax - weight of all the passengers                                                          [kilogram]
            wt_bag - weight of all the baggage                                                             [kilogram]
            wt_fuel - weight of the fuel carried                                                           [kilogram]
            wt_empty - operating empty weight of the aircraft                                              [kilograms]

    Properties Used:
    N/A
    """

    # Unpack inputs
    TOW         = vehicle.mass_properties.max_takeoff

    bwb_aft_centerbody_area     = vehicle.fuselages['fuselage_bwb'].aft_centerbody_area
    bwb_aft_centerbody_taper    = vehicle.fuselages['fuselage_bwb'].aft_centerbody_taper
    bwb_cabin_area              = vehicle.fuselages['fuselage_bwb'].cabin_area

    network_name = list(vehicle.networks.keys())[0]  # obtain the key f
    # for the network for assignment purposes

    networks = vehicle.networks[network_name]
    num_eng = networks.number_of_engines
    if network_name == 'turbofan' or network_name == 'Turbofan':
        # thrust_sls should be sea level static thrust. Using design thrust results in wrong network
        # weight estimation. Engine sizing should return this value.
        # for now, using thrust_sls = design_thrust / 0.20, just for optimization evaluations
        thrust_sls      = networks.sealevel_static_thrust
        wt_engine_jet   = Propulsion.engine_jet(thrust_sls)
        wt_propulsion   = Propulsion.integrated_propulsion(wt_engine_jet, num_eng)
        networks.mass_properties.mass = wt_propulsion

    else:  # network used is not a turbo_fan; assume mass_properties defined outside model
        wt_propulsion = networks.mass_properties.mass

        if wt_propulsion == 0:
            warnings.warn("Propulsion mass= 0; there is no Engine Weight being added to the Configuration",
                          stacklevel=1)

    if 'main_wing' not in vehicle.wings:
        wt_wing = 0.0
        warnings.warn("There is no Wing Weight being added to the Configuration", stacklevel=1)

    else:
        
        rho      = Aluminum().density
        sigma    = Aluminum().yield_tensile_strength           
        wt_wing  = wing_main(vehicle, vehicle.wings['main_wing'], rho, sigma, computation_type='simple')
        vehicle.wings['main_wing'].mass_properties.mass = wt_wing

        # Calculating Empty Weight of Aircraft
    landing_gear        = landing_gear_weight(vehicle)
    wt_cabin            = cabin(bwb_cabin_area, TOW)
    wt_aft_centerbody   = aft_centerbody(num_eng, bwb_aft_centerbody_area, bwb_aft_centerbody_taper, TOW)
    output_2            = systems(vehicle)
    # Calculate the equipment empty weight of the aircraft
    vehicle.fuselages['fuselage_bwb'].mass_properties.mass = wt_cabin

    # packup outputs
    payload = payload_weight(vehicle)
    
    vehicle.payload.passengers = SUAVE.Components.Physical_Component()
    vehicle.payload.baggage    = SUAVE.Components.Physical_Component()
    vehicle.payload.cargo      = SUAVE.Components.Physical_Component()
    
    vehicle.payload.passengers.mass_properties.mass = payload.passengers
    vehicle.payload.baggage.mass_properties.mass    = payload.baggage
    vehicle.payload.cargo.mass_properties.mass      = payload.cargo    
    
    wt_oper = operating_items(vehicle)
    # Distribute all weight in the output fields
    output = Data()
    output.structures                   = Data()
    output.structures.wing              = wt_wing
    output.structures.afterbody         = wt_aft_centerbody
    output.structures.fuselage          = wt_cabin
    output.structures.main_landing_gear = landing_gear.main
    output.structures.nose_landing_gear = landing_gear.nose
    output.structures.nacelle           = 0
    output.structures.paint             = 0  # TODO change
    output.structures.total   = output.structures.wing + output.structures.afterbody \
                                + output.structures.fuselage + output.structures.main_landing_gear + output.structures.nose_landing_gear \
                                + output.structures.paint + output.structures.nacelle

    output.propulsion_breakdown                     = Data()
    output.propulsion_breakdown.total               = wt_propulsion
    output.propulsion_breakdown.engines             = 0
    output.propulsion_breakdown.thrust_reversers    = 0
    output.propulsion_breakdown.miscellaneous       = 0
    output.propulsion_breakdown.fuel_system         = 0

    output.systems_breakdown                        = Data()
    output.systems_breakdown.control_systems        = output_2.wt_flight_control
    output.systems_breakdown.apu                    = output_2.wt_apu
    output.systems_breakdown.electrical             = output_2.wt_elec
    output.systems_breakdown.avionics               = output_2.wt_avionics
    output.systems_breakdown.hydraulics             = output_2.wt_hyd_pnu
    output.systems_breakdown.furnish                = output_2.wt_furnish
    output.systems_breakdown.air_conditioner        = output_2.wt_ac
    output.systems_breakdown.instruments            = output_2.wt_instruments
    output.systems_breakdown.anti_ice                = 0
    output.systems_breakdown.total = output.systems_breakdown.control_systems + output.systems_breakdown.apu \
                                     + output.systems_breakdown.electrical + output.systems_breakdown.avionics \
                                     + output.systems_breakdown.hydraulics + output.systems_breakdown.furnish \
                                     + output.systems_breakdown.air_conditioner + output.systems_breakdown.instruments \
                                     + output.systems_breakdown.anti_ice

    output.payload_breakdown = Data()
    output.payload_breakdown = payload

    output.operational_items = Data()
    output.operational_items = wt_oper

    output.empty            = output.structures.total + output.propulsion_breakdown.total + output.systems_breakdown.total
    output.operating_empty  = output.empty + output.operational_items.total
    output.zero_fuel_weight = output.operating_empty + output.payload_breakdown.total
    output.fuel             = vehicle.mass_properties.max_takeoff - output.zero_fuel_weight

    control_systems         = SUAVE.Components.Physical_Component()
    electrical_systems      = SUAVE.Components.Physical_Component()
    furnishings             = SUAVE.Components.Physical_Component()
    air_conditioner         = SUAVE.Components.Physical_Component()
    fuel                    = SUAVE.Components.Physical_Component()
    apu                     = SUAVE.Components.Physical_Component()
    hydraulics              = SUAVE.Components.Physical_Component()
    avionics                = SUAVE.Components.Energy.Peripherals.Avionics()
    optionals               = SUAVE.Components.Physical_Component()

    vehicle.landing_gear.nose       = SUAVE.Components.Landing_Gear.Main_Landing_Gear()
    vehicle.landing_gear.nose.mass  = output.structures.nose_landing_gear
    vehicle.landing_gear.main       = SUAVE.Components.Landing_Gear.Nose_Landing_Gear()   
    vehicle.landing_gear.main.mass  = output.structures.main_landing_gear  

    control_systems.mass_properties.mass    = output.systems_breakdown.control_systems
    electrical_systems.mass_properties.mass = output.systems_breakdown.electrical
    furnishings.mass_properties.mass        = output.systems_breakdown.furnish
    avionics.mass_properties.mass           = output.systems_breakdown.avionics \
                                            + output.systems_breakdown.instruments
    air_conditioner.mass_properties.mass    = output.systems_breakdown.air_conditioner
    fuel.mass_properties.mass               = output.fuel
    apu.mass_properties.mass                = output.systems_breakdown.apu
    hydraulics.mass_properties.mass         = output.systems_breakdown.hydraulics
    optionals.mass_properties.mass          = output.operational_items.operating_items_less_crew

    # assign components to vehicle
    vehicle.control_systems         = control_systems
    vehicle.electrical_systems      = electrical_systems
    vehicle.avionics                = avionics
    vehicle.furnishings             = furnishings
    vehicle.air_conditioner         = air_conditioner
    vehicle.fuel                    = fuel
    vehicle.apu                     = apu
    vehicle.hydraulics              = hydraulics
    vehicle.optionals               = optionals

    return output

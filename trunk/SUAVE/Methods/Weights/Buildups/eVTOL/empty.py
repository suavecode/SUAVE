## @ingroup Methods-Weights-Buildups-eVTOL
# empty.py
#
# Created:      Apr 2019, J. Smart
# Modified:     July 2021, R. Erhard
#               Dec 2022, J. Smart

#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------
import SUAVE
from SUAVE.Core import Units, Data

from SUAVE.Methods.Weights.Buildups.Common import *
from SUAVE.Components.Energy.Converters import Propeller, Lift_Rotor
from SUAVE.Components.Energy.Networks import Battery_Propeller
from SUAVE.Components.Energy.Networks import Lift_Cruise

import numpy as np

#-------------------------------------------------------------------------------
# Empty
#-------------------------------------------------------------------------------

## @ingroup Methods-Weights-Buildups-eVTOL

def empty(config,
          settings):

    """
    Calculates the empty vehicle mass for an eVTOL aircraft including seats,
    avionics, servomotors, ballistic recovery system, rotor and hub assembly,
    transmission, and landing gear.

    Sources:
    Project Vahana Conceptual Trade Study
    https://github.com/VahanaOpenSource


    Inputs:
        config:                     SUAVE Config Data Structure
        settings.method_settings.       [Data]
            contingency_factor          [Float] Secondary Weight Est.
            speed_of_sound              [Float] Design Point Speed of Sound
            max_tip_mach                [Float] Max Rotor Tip Mach Number
            disk_area_factor            [Float] k (ref. Johnson 2-6.2)
            safety_factor               [Float] Structural Factor of Safety
            max_thrust_to_weight_ratio  [Float] Design T/W Ratio
            max_g_load                  [Float] Design G Load
            motor_efficiency            [Float] Design Point Motor Efficiency

    Outputs:
        outputs:                    Data Dictionary of Component Masses [kg]

    Output data dictionary has the following bookkeeping  structure:

        Output
            Total.
                Empty.
                    Structural.
                        Fuselage
                        Nacelles
                        Wings
                        Landing Gear
                        Rotors
                        Hubs
                    Seats
                    Battery
                    Motors
                    Servo
                Systems.
                    Avionics
                    ECS               - Environmental Control System
                    BRS               - Ballistic Recovery System
                    Wiring            - Aircraft Electronic Wiring
                Payload

    """

    # Unpack Settings

    options = settings.method_settings

    contingency_factor            = options.contingency_factor
    speed_of_sound                = options.speed_of_sound
    max_tip_mach                  = options.max_tip_mach
    disk_area_factor              = options.disk_area_factor
    safety_factor                 = options.safety_factor
    max_thrust_to_weight_ratio    = options.max_thrust_to_weight_ratio
    max_g_load                    = options.max_g_load
    motor_efficiency              = options.motor_efficiency

    # Set up data structures for SUAVE weight methods
    output                   = Data()
    output.lift_rotors       = 0.0
    output.propellers        = 0.0
    output.lift_rotor_motors = 0.0
    output.propeller_motors  = 0.0
    output.battery           = 0.0
    output.servos            = 0.0
    output.hubs              = 0.0
    output.BRS               = 0.0

    output.payload = Data()
    output.payload.total = 0.0

    C = SUAVE.Components

    config.payload.passengers           = C.Physical_Component()
    config.payload.baggage              = C.Physical_Component()
    config.payload.cargo                = C.Physical_Component()
    control_systems                     = C.Physical_Component()
    electrical_systems                  = C.Physical_Component()
    furnishings                         = C.Physical_Component()
    air_conditioner                     = C.Physical_Component()
    fuel                                = C.Physical_Component()
    apu                                 = C.Physical_Component()
    hydraulics                          = C.Physical_Component()
    optionals                           = C.Physical_Component()
    avionics                            = C.Energy.Peripherals.Avionics()

    # assign components to vehicle
    config.systems.control_systems      = control_systems
    config.systems.electrical_systems   = electrical_systems
    config.systems.avionics             = avionics
    config.systems.furnishings          = furnishings
    config.systems.air_conditioner      = air_conditioner
    config.systems.fuel                 = fuel
    config.systems.apu                  = apu
    config.systems.hydraulics           = hydraulics
    config.systems.optionals            = optionals


    #---------------------------------------------------------------------------
    # Fixed Weights
    #---------------------------------------------------------------------------
    MTOW                = config.mass_properties.max_takeoff
    output.seats        = config.passengers * 15.   * Units.kg
    output.passengers   = config.passengers * 70.   * Units.kg
    output.avionics     = 15.                       * Units.kg
    output.landing_gear = MTOW * 0.02               * Units.kg
    output.ECS          = config.passengers * 7.    * Units.kg

    # Inputs and other constants
    tipMach        = max_tip_mach
    k              = disk_area_factor
    ToverW         = max_thrust_to_weight_ratio
    eta            = motor_efficiency
    rho_ref        = 1.225
    maxVTip        = speed_of_sound * tipMach         # Prop Tip Velocity
    maxLift        = MTOW * ToverW * 9.81             # Maximum Thrust
    AvgBladeCD     = 0.012                            # Average Blade CD

    # Select a length scale depending on what kind of vehicle this is
    length_scale = 1.
    nose_length  = 0.

    # Check if there is a fuselage

    if len(config.fuselages) == 0.:
        for w  in config.wings:
            if isinstance(w ,C.Wings.Main_Wing):
                b = w.chords.root
                if b>length_scale:
                    length_scale = b
                    nose_length  = 0.25*b
    else:
        for fuse in config.fuselages:
            nose   = fuse.lengths.nose
            length = fuse.lengths.total
            if length > length_scale:
                length_scale = length
                nose_length  = nose

    #---------------------------------------------------------------------------
    # Environmental Control System
    #---------------------------------------------------------------------------
    config.systems.air_conditioner.origin[0][0]          = 0.51 * length_scale
    config.systems.air_conditioner.mass_properties.mass  = output.ECS

    #---------------------------------------------------------------------------
    # Seats
    #---------------------------------------------------------------------------
    config.systems.furnishings.origin[0][0]          = 0.2 * length_scale
    config.systems.furnishings.mass_properties.mass  = output.seats

    #---------------------------------------------------------------------------
    # Network Weight
    #---------------------------------------------------------------------------
    for network in config.networks:

        #-----------------------------------------------------------------------
        # Battery Weight
        #-----------------------------------------------------------------------
        batt = network.battery
        batt.origin[0][0]                                = 0.51 * length_scale
        batt.mass_properties.center_of_gravity[0][0]     = 0.0

        output.battery += batt.mass_properties.mass * Units.kg

        #-----------------------------------------------------------------------
        # Payload Weight
        #-----------------------------------------------------------------------
        load = network.payload
        load.origin[0][0]                               = 0.51 * length_scale
        load.mass_properties.center_of_gravity[0][0]    = 0.0
        output.payload.total += load.mass_properties.mass * Units.kg

        #-----------------------------------------------------------------------
        # Avionics Weight
        #-----------------------------------------------------------------------
        avi = network.avionics
        avi.origin[0][0]                            = 0.4 * nose_length
        avi.mass_properties.center_of_gravity[0][0] = 0.0
        avi.mass_properties.mass                    = output.avionics

        #-----------------------------------------------------------------------
        # Servo, Hub and BRS Weights
        #-----------------------------------------------------------------------

        lift_rotor_hub_weight   = 4.   * Units.kg
        prop_hub_weight         = MTOW * 0.04  * Units.kg

        lift_rotor_BRS_weight   = 16.  * Units.kg

        #-----------------------------------------------------------------------
        # Rotor, Propeller, parameters for sizing
        #-----------------------------------------------------------------------
        if isinstance(network, Lift_Cruise):
            # Total number of rotors and propellers
            nLiftRotors   = network.number_of_lift_rotor_engines
            nThrustProps  = network.number_of_propeller_engines
            props         = network.propellers
            rots          = network.lift_rotors
            prop_motors   = network.propeller_motors
            rot_motors    = network.lift_rotor_motors

        elif isinstance(network, Battery_Propeller): 
            props         = network.propellers 
            prop_motors   = network.propeller_motors          
            nThrustProps  = 0  
            nLiftRotors   = 0    
            nProps        = 0 
            for rot_idx in range(len(props.keys())):               
                if type(props[list(props.keys())[rot_idx]]) == Propeller: 
                    props          = network.propellers
                    nThrustProps  +=1
    
                elif type(props[list(props.keys())[rot_idx]]) == Lift_Rotor:    
                    nLiftRotors   +=1  
                    
            if (nThrustProps == 0) and (nLiftRotors != 0):
                network.lift_rotors           = network.propellers
                rot_motors                    = network.propeller_motors  
                network.identical_lift_rotors = network.number_of_propeller_engines
        else:
            raise NotImplementedError("""eVTOL weight buildup only supports the Battery Propeller and Lift Cruise energy networks.\n
            Weight buildup will not return information on propulsion system.""", RuntimeWarning)

        
        nProps  = int(nLiftRotors + nThrustProps)  
        if nProps > 1:
            prop_BRS_weight     = 16.   * Units.kg
        else:
            prop_BRS_weight     = 0.   * Units.kg

        prop_servo_weight  = 0.0

        if nThrustProps > 0: 
            for idx, propeller in enumerate(network.propellers):
                proprotor    = propeller
                propmotor    = prop_motors[list(prop_motors.keys())[idx]]
                rTip_ref     = proprotor.tip_radius
                bladeSol_ref = proprotor.blade_solidity

                if proprotor.variable_pitch:
                    prop_servo_weight  = 5.2  * Units.kg

                # Compute and add propeller weights
                propeller_mass                 = prop(proprotor, maxLift/5.) * Units.kg
                output.propellers             += propeller_mass
                output.propeller_motors       += propmotor.mass_properties.mass
                proprotor.mass_properties.mass = propeller_mass + prop_hub_weight + prop_servo_weight

        lift_rotor_servo_weight = 0.0
        if nLiftRotors > 0: 
            for idx, lift_rotor in enumerate(network.lift_rotors):
                liftrotor    = lift_rotor
                liftmotor    = rot_motors[list(rot_motors.keys())[idx]]
                rTip_ref     = liftrotor.tip_radius
                bladeSol_ref = liftrotor.blade_solidity


                if liftrotor.variable_pitch:
                    lift_rotor_servo_weight = 0.65 * Units.kg

                # Compute and add lift_rotor weights
                lift_rotor_mass                = prop(liftrotor, maxLift / max(nLiftRotors - 1, 1))  * Units.kg
                output.lift_rotors            += lift_rotor_mass
                output.lift_rotor_motors      += liftmotor.mass_properties.mass
                liftrotor.mass_properties.mass = lift_rotor_mass + lift_rotor_hub_weight + lift_rotor_servo_weight

        # Add associated weights
        output.servos += (nLiftRotors * lift_rotor_servo_weight + nThrustProps * prop_servo_weight)
        output.hubs   += (nLiftRotors * lift_rotor_hub_weight + nThrustProps * prop_hub_weight)
        output.BRS    += (prop_BRS_weight + lift_rotor_BRS_weight)

        maxLiftPower   = 1.15*maxLift*(k*np.sqrt(maxLift/(2*rho_ref*np.pi*rTip_ref**2)) +
                                           bladeSol_ref*AvgBladeCD/8*maxVTip**3/(maxLift/(rho_ref*np.pi*rTip_ref**2)))
        # Tail Rotor
        if nLiftRotors == 1: # this assumes that the vehicle is an electric helicopter with a tail rotor
            
            maxLiftOmega   = maxVTip/rTip_ref
            maxLiftTorque  = maxLiftPower / maxLiftOmega

            tailrotor = next(iter(network.lift_rotors))
            output.tail_rotor   = prop(tailrotor, 1.5*maxLiftTorque/(1.25*rTip_ref))*0.2 * Units.kg
            output.lift_rotors += output.tail_rotor

    # sum motor weight
    output.motors = output.lift_rotor_motors + output.propeller_motors

    #-------------------------------------------------------------------------------
    # Wing and Motor Wiring Weight
    #-------------------------------------------------------------------------------
    total_wing_weight   = 0.0
    total_wiring_weight = 0.0
    output.wings        = Data()
    output.wiring       = Data()

    for w in config.wings:
        if w.symbolic:
            wing_weight = 0
        else:
            wing_weight            = wing(w, config, maxLift/5, safety_factor= safety_factor, max_g_load =  max_g_load )
            wing_tag               = w.tag
            output.wings[wing_tag] = wing_weight
            w.mass_properties.mass = wing_weight

        total_wing_weight    = total_wing_weight + wing_weight

        # wiring weight
        wiring_weight        = wiring(w, config, maxLiftPower/(eta*nProps)) * Units.kg
        total_wiring_weight  = total_wiring_weight + wiring_weight

    output.wiring            = total_wiring_weight
    output.total_wing_weight = total_wing_weight
    
    
    #-------------------------------------------------------------------------------
    # Control Systems
    #-------------------------------------------------------------------------------
    config.systems.control_systems.origin[0][0]          = 0.6 * length_scale
    config.systems.control_systems.mass_properties.mass  = output.servos
    
    #-------------------------------------------------------------------------------
    # Wiring
    #-------------------------------------------------------------------------------
    config.systems.electrical_systems.origin[0][0]          = 0.49 * length_scale
    config.systems.electrical_systems.mass_properties.mass  = output.wiring
    
    #-------------------------------------------------------------------------------
    # BRS (Optionals?)
    #-------------------------------------------------------------------------------
    config.systems.optionals.origin[0][0]          = 0.49 * length_scale
    config.systems.optionals.mass_properties.mass  = output.BRS
    
    
    #-------------------------------------------------------------------------------
    # Passengers are payload too
    #-------------------------------------------------------------------------------
    config.payload.passengers.origin[0][0]         = 0.2 * length_scale
    config.payload.passengers.mass_properties.mass = output.passengers  

    #-------------------------------------------------------------------------------
    # Landing Gear Weight
    #-------------------------------------------------------------------------------
    if not hasattr(config.landing_gear, 'nose'):
        config.landing_gear.nose       = SUAVE.Components.Landing_Gear.Nose_Landing_Gear()
    config.landing_gear.nose.mass_properties.mass      = 0.0
    if not hasattr(config.landing_gear, 'main'):
        config.landing_gear.main       = SUAVE.Components.Landing_Gear.Main_Landing_Gear()
    config.landing_gear.main.mass_properties.mass = output.landing_gear

    #-------------------------------------------------------------------------------
    # Fuselage Weight
    #-------------------------------------------------------------------------------
    output.fuselage = fuselage(config)

    fuse = config.fuselages.fuselage
    fuse.mass_properties.center_of_gravity[0][0] = .45 * fuse.lengths.total
    fuse.mass_properties.mass                    =  output.fuselage.total

    # -------------------------------------------------------------------------------
    # Nacelle Weight
    # -------------------------------------------------------------------------------

    total_nacelle_weight = 0. * Units.kg
    output.nacelles = Data()
    output.nacelles.total       = 0.0 * Units.kg
    output.nacelles.rotor_booms = 0.0 * Units.kg

    for nacelle in config.nacelles:
        if isinstance(nacelle, SUAVE.Components.Nacelles.Rotor_Boom):
            nacelle_weight = rotor_boom(nacelle, config).total * Units.kg
            output.nacelles.rotor_booms += nacelle_weight
        else:
            nacelle_weight = elliptical_shell(nacelle) * Units.kg

        nacelle.mass_properties.mass = nacelle_weight
        nacelle.mass_properties.center_of_gravity[0][0] = 0.45 * nacelle.length

        output.nacelles.total  += nacelle_weight

    #-------------------------------------------------------------------------------
    # Bookkeeping
    #-------------------------------------------------------------------------------
    output.motors       = Data()
    output.motors.total = 0.0 * Units.kg

    for key in [
        'lift_rotor_motors',
        'propeller_motors',
    ]:
        output.motors.total += output[key] * Units.kg
        output.motors[key] = output[key]
        del output[key]

    output.structural       = Data()
    output.structural.total = 0.0 * Units.kg

    for key in [
        'lift_rotors',
        'propellers',
        'hubs',
        'nacelles',
        'landing_gear',
    ]:
        output.structural.total += output[key] * Units.kg
        output.structural[key] = output[key]
        del output[key]

    output.structural.wings = output.wings
    output.structural.wings.total = total_wing_weight
    output.structural.total += total_wing_weight
    del output['wings']
    del output['total_wing_weight']

    output.structural.fuselage = output.fuselage
    output.structural.total += output.fuselage.total
    del output['fuselage']

    output.empty            = Data()
    output.empty.total      = 0.0 * Units.kg

    for key in [
        'seats',
        'avionics',
        'ECS',
        'servos',
        'wiring',
        'BRS',
    ]:
        output.empty.total += output[key] * Units.kg
        output.empty[key] = output[key]
        del output[key]

    for key in [
        'structural',
        'motors',
    ]:
        output.empty[key] = output[key]
        output.empty.total += output[key].total * Units.kg
        del output[key]

    output.empty.total *= contingency_factor

    output.empty.battery = output.battery
    output.empty.total += output.battery * Units.kg
    del output['battery']


    output.payload.passengers = output.passengers
    del output['passengers']

    output.total      = (output.empty.total +
                         output.payload.total)

    return output

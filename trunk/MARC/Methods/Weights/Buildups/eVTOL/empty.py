## @ingroup Methods-Weights-Buildups-eVTOL
# empty.py
#
# Created:    Apr, 2019, J. Smart
# Modified:   July, 2021, R. Erhard

#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------
import MARC
from MARC.Core import Units, Data

from MARC.Methods.Weights.Buildups.Common.fuselage import fuselage
from MARC.Methods.Weights.Buildups.Common.prop import prop
from MARC.Methods.Weights.Buildups.Common.wiring import wiring
from MARC.Methods.Weights.Buildups.Common.wing import wing
import MARC.Components.Energy.Converters.Propeller  as Propeller
import MARC.Components.Energy.Converters.Lift_Rotor as Lift_Rotor 
import MARC.Components.Energy.Converters.Prop_Rotor as Prop_Rotor
import MARC.Components.Energy.Converters.Rotor      as Rotor 
import MARC.Components.Energy.Networks.Battery_Electric_Rotor as  Battery_Electric_Rotor 

import numpy as np

#-------------------------------------------------------------------------------
# Empty
#-------------------------------------------------------------------------------

## @ingroup Methods-Weights-Buildups-eVTOL

def empty(config,
          settings,
          contingency_factor            = 1.1,
          speed_of_sound                = 340.294,
          max_tip_mach                  = 0.65,
          disk_area_factor              = 1.15,
          safety_factor                 = 1.5,
          max_thrust_to_weight_ratio    = 1.1,
          max_g_load                    = 3.8,
          motor_efficiency              = 0.85 * 0.98):

    """ Calculates the empty vehicle mass for an EVTOL-type aircraft including seats,
        avionics, servomotors, ballistic recovery system, rotor and hub assembly,
        transmission, and landing gear. Incorporates the results of the following
        common-use buildups:

            fuselage.py
            prop.py
            wing.py
            wiring.py

        Sources:
        Project Vahana Conceptual Trade Study
        https://github.com/VahanaOpenSource


        Inputs: 
            config:                     MARC Config Data Stucture
            contingency_factor          Factor capturing uncertainty in vehicle weight [Unitless]
            speed_of_sound:             Local Speed of Sound                           [m/s]
            max_tip_mach:               Allowable Tip Mach Number                      [Unitless]
            disk_area_factor:           Inverse of Disk Area Efficiency                [Unitless]
            max_thrust_to_weight_ratio: Allowable Thrust to Weight Ratio               [Unitless]
            safety_factor               Safety Factor in vehicle design                [Unitless]
            max_g_load                  Maximum g-forces load for certification        [UNitless]
            motor_efficiency:           Motor Efficiency                               [Unitless]

        Outputs: 
            outputs:                    Data Dictionary of Component Masses [kg]

        Output data dictionary has the following book-keeping hierarchical structure:

            Output
                Total.
                    Empty.
                        Structural.
                            Fuselage
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

    # Set up data structures for MARC weight methods
    weight                   = Data()  
    weight.battery           = 0.0
    weight.payload           = 0.0
    weight.servos            = 0.0
    weight.hubs              = 0.0
    weight.BRS               = 0.0  
    weight.motors            = 0.0
    weight.rotors            = 0.0
    weight.wiring            = 0.0
    weight.wings             = Data()
    weight.wings_total       = 0.0

    config.payload.passengers                      = MARC.Components.Physical_Component()
    config.payload.baggage                         = MARC.Components.Physical_Component()
    config.payload.cargo                           = MARC.Components.Physical_Component()
    control_systems                                = MARC.Components.Physical_Component()
    electrical_systems                             = MARC.Components.Physical_Component()
    furnishings                                    = MARC.Components.Physical_Component()
    air_conditioner                                = MARC.Components.Physical_Component()
    fuel                                           = MARC.Components.Physical_Component()
    apu                                            = MARC.Components.Physical_Component()
    hydraulics                                     = MARC.Components.Physical_Component()
    avionics                                       = MARC.Components.Energy.Peripherals.Avionics()
    optionals                                      = MARC.Components.Physical_Component()

    # assign components to vehicle
    config.systems.control_systems                 = control_systems
    config.systems.electrical_systems              = electrical_systems
    config.systems.avionics                        = avionics
    config.systems.furnishings                     = furnishings
    config.systems.air_conditioner                 = air_conditioner
    config.systems.fuel                            = fuel
    config.systems.apu                             = apu
    config.systems.hydraulics                      = hydraulics
    config.systems.optionals                       = optionals


    #-------------------------------------------------------------------------------
    # Fixed Weights
    #-------------------------------------------------------------------------------
    MTOW                = config.mass_properties.max_takeoff
    weight.seats        = config.passengers * 15.   * Units.kg
    weight.passengers   = config.passengers * 70.   * Units.kg
    weight.avionics     = 15.                       * Units.kg
    weight.landing_gear = MTOW * 0.02               * Units.kg
    weight.ECS          = config.passengers * 7.    * Units.kg

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
    C =  MARC.Components
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

    #-------------------------------------------------------------------------------
    # Environmental Control System
    #-------------------------------------------------------------------------------
    config.systems.air_conditioner.origin[0][0]          = 0.51 * length_scale
    config.systems.air_conditioner.mass_properties.mass  = weight.ECS

    #-------------------------------------------------------------------------------
    # Network Weight
    #-------------------------------------------------------------------------------
    for network in config.networks:

        #-------------------------------------------------------------------------------
        # Battery Weight
        #-------------------------------------------------------------------------------
        network.battery.origin[0][0]                                   = 0.51 * length_scale
        network.battery.mass_properties.center_of_gravity[0][0]        = 0.0
        weight.battery                                                += network.battery.mass_properties.mass * Units.kg

        #-------------------------------------------------------------------------------
        # Payload Weight
        #-------------------------------------------------------------------------------
        network.payload.origin[0][0]                                   = 0.51 * length_scale
        network.payload.mass_properties.center_of_gravity[0][0]        = 0.0
        weight.payload                                                += network.payload.mass_properties.mass * Units.kg

        #-------------------------------------------------------------------------------
        # Avionics Weight
        #-------------------------------------------------------------------------------
        network.avionics.origin[0][0]                                  = 0.4 * nose_length
        network.avionics.mass_properties.center_of_gravity[0][0]       = 0.0
        network.avionics.mass_properties.mass                          = weight.avionics


        #-------------------------------------------------------------------------------
        # Servo, Hub and BRS Weights
        #-------------------------------------------------------------------------------
        lift_rotor_hub_weight   = 4.   * Units.kg
        prop_hub_weight         = MTOW * 0.04  * Units.kg
        lift_rotor_BRS_weight   = 16.  * Units.kg

        #-------------------------------------------------------------------------------
        # Rotor, Propeller, parameters for sizing
        #------------------------------------------------------------------------------- 
        number_of_propellers    = 0.0  
        number_of_lift_rotors   = 0.0   
        total_number_of_rotors  = 0.0      
        lift_rotor_servo_weight = 0.0

        
        if not isinstance(network, Battery_Electric_Rotor):
            raise NotImplementedError("""eVTOL weight buildup only supports the Battery Electric Rotor energy network.\n
            Weight buildup will not return information on propulsion system.""",RuntimeWarning)
        
        

        # How many evaluations to do 
        rotor_group_indexes     = network.rotor_group_indexes
        motor_group_indexes     = network.motor_group_indexes
        motors                  = network.motors
        rotors                  = network.rotors
        n_rotors                = len(rotors)
        n_motors                = len(motors) 
        
        unique_rotor_groups,factors = np.unique(rotor_group_indexes, return_counts=True)
        unique_motor_groups,factors = np.unique(motor_group_indexes, return_counts=True) 
                
        # Count how many unknowns and residuals based on p)  
        if n_rotors!=len(rotor_group_indexes):
            assert('The number of rotor group indexes must be equal to the number of rotors')
        if n_motors!=len(motor_group_indexes):
            assert('The number of motor group indexes must be equal to the number of motors') 
        if len(rotor_group_indexes)!=len(motor_group_indexes):
            assert('The number of rotors is not the same as the number of motors')        
        i = 0
        for rotor,motor in zip(rotors,motors):    
            rotors           = network.rotors
            number_of_rotors = factors[rotor_group_indexes[i]]
            
            if len(rotors) != len(motors):
                assert("Number of rotors must be equal to the number of rotor motors") 
                  
            if type(rotor) == Propeller:
                ''' Propeller Weight '''  
                number_of_propellers     += 1   
                rTip_ref                  = rotor.tip_radius
                bladeSol_ref              = rotor.blade_solidity    
                if rotor.variable_pitch:
                    prop_servo_weight     = 5.2 * Units.kg  
                else:
                    prop_servo_weight     = 0
                propeller_mass            = prop(rotor, maxLift/5.) * Units.kg
                weight.rotors             += propeller_mass
                weight.motors             += motor.mass_properties.mass
                rotor.mass_properties.mass  =  propeller_mass + prop_hub_weight + prop_servo_weight
                weight.servos             += prop_servo_weight
                weight.hubs               += prop_hub_weight
                
            if (type(rotor) == Lift_Rotor or type(rotor) == Prop_Rotor) or type(rotor) == Rotor:
                ''' Lift Rotor, Prop-Rotor or Rotor Weight '''   
                number_of_lift_rotors   += 1  
                rTip_ref                = rotor.tip_radius
                bladeSol_ref            = rotor.blade_solidity      
                if rotor.variable_pitch:
                    lift_rotor_servo_weight = 0.65 * Units.kg 
                else:
                    prop_servo_weight     = 0 
                lift_rotor_mass             = prop(rotor, maxLift / max(number_of_rotors - 1, 1))  * Units.kg
                weight.rotors               += lift_rotor_mass
                weight.motors               += motor.mass_properties.mass
                rotor.mass_properties.mass    =  lift_rotor_mass + lift_rotor_hub_weight + lift_rotor_servo_weight
                weight.servos               += lift_rotor_servo_weight
                weight.hubs                 += lift_rotor_hub_weight
                
            i += 1
         
        total_number_of_rotors  = int(number_of_lift_rotors + number_of_propellers)  
        if total_number_of_rotors > 1:
            prop_BRS_weight     = 16.   * Units.kg
        else:
            prop_BRS_weight     = 0.   * Units.kg
 
        # Add associated weights  
        weight.BRS    += (prop_BRS_weight + lift_rotor_BRS_weight)  
        maxLiftPower   = 1.15*maxLift*(k*np.sqrt(maxLift/(2*rho_ref*np.pi*rTip_ref**2)) +
                                           bladeSol_ref*AvgBladeCD/8*maxVTip**3/(maxLift/(rho_ref*np.pi*rTip_ref**2)))
        # Tail Rotor
        if number_of_lift_rotors == 1: # this assumes that the vehicle is an electric helicopter with a tail rotor
            
            maxLiftOmega   = maxVTip/rTip_ref
            maxLiftTorque  = maxLiftPower / maxLiftOmega

            tailrotor = next(iter(network.lift_rotors))
            weight.tail_rotor  = prop(tailrotor, 1.5*maxLiftTorque/(1.25*rTip_ref))*0.2 * Units.kg
            weight.rotors     += weight.tail_rotor 

    #-------------------------------------------------------------------------------
    # Wing and Motor Wiring Weight
    #-------------------------------------------------------------------------------  
    for w in config.wings:
        if w.symbolic:
            wing_weight = 0
        else:
            wing_weight            = wing(w, config, maxLift/5, safety_factor= safety_factor, max_g_load =  max_g_load )
            wing_tag               = w.tag
            weight.wings[wing_tag] = wing_weight
            w.mass_properties.mass = wing_weight 
        weight.wings_total         += wing_weight

        # wiring weight
        weight.wiring  += wiring(w, config, maxLiftPower/(eta*total_number_of_rotors)) * Units.kg 

    #-------------------------------------------------------------------------------
    # Landing Gear Weight
    #-------------------------------------------------------------------------------
    if not hasattr(config.landing_gear, 'nose'):
        config.landing_gear.nose       = MARC.Components.Landing_Gear.Nose_Landing_Gear()
    config.landing_gear.nose.mass      = 0.0
    if not hasattr(config.landing_gear, 'main'):
        config.landing_gear.main       = MARC.Components.Landing_Gear.Main_Landing_Gear()
    config.landing_gear.main.mass      = weight.landing_gear

    #-------------------------------------------------------------------------------
    # Fuselage  Weight
    #-------------------------------------------------------------------------------
    weight.fuselage = fuselage(config) * Units.kg
    config.fuselages.fuselage.mass_properties.center_of_gravity[0][0] = .45*config.fuselages.fuselage.lengths.total
    config.fuselages.fuselage.mass_properties.mass                    =  weight.fuselage + weight.passengers + weight.seats +\
                                                                         weight.wiring + weight.BRS

    #-------------------------------------------------------------------------------
    # Pack Up Outputs
    #-------------------------------------------------------------------------------
    weight.structural = (weight.rotors + weight.hubs +
                                 weight.fuselage + weight.landing_gear +weight.wings_total)*Units.kg

    weight.empty      = (contingency_factor * (weight.structural + weight.seats + weight.avionics +weight.ECS +\
                        weight.motors + weight.servos + weight.wiring + weight.BRS) + weight.battery) *Units.kg

    weight.total      = weight.empty + weight.payload + weight.passengers

    return weight

## @ingroup Components-Energy-Networks
# Turboelectric_Ducted_Fan.py
#
# Created:  Nov 2019, K. Hamilton

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
import SUAVE

# package imports
import numpy as np
from SUAVE.Core import Data
from SUAVE.Components.Propulsors.Propulsor import Propulsor

# ----------------------------------------------------------------------
#  Network
# ----------------------------------------------------------------------

## @ingroup Components-Energy-Networks
class Turboelectric_Ducted_Fan(Propulsor):
    """ Simply connects a turboelectric power source to a ducted fan, with an assumed ducted fan motor efficiency
    
        Assumptions:
        None
        
        Source:
        None
    """
    
    def __defaults__(self):
        """ This sets the default values for the network to function.
    
            Assumptions:
            Your system always uses 90 amps...?
    
            Source:
            N/A
    
            Inputs:
            None
    
            Outputs:
            None
    
            Properties Used:
            N/A
        """         
        
        self.propulsor                  = None  # i.e. the ducted fan (not including the motor).
        self.motor                      = None  # the motor that drives the fan, which may be partial or fully HTS.
        self.powersupply                = None  # i.e. the turboelectric generator, the generator of which may be partial or fully HTS
        self.esc                        = None  # the electronics that supply the motor armature windings
        self.rotor                      = None  # the motor rotor (handled as a seperate component to the motor)
        self.lead                       = None  # the current leads that connect the rotor to the constant current supply
        self.ccs                        = None  # the electronics that supply the rotor field windings
        self.cryocooler                 = None  # cryocooler which cools the rotor using energy
        self.cryogenic_heat_exchanger   = None  # heat exchanger that cools the rotor using cryogen
        self.cryogen_proportion         = 1.0   # Proportion of cooling to be supplied by the cryogenic heat exchanger, rather than by the cryocooler

        # self.motor_efficiency   = .95
        self.nacelle_diameter       = 1.0
        self.engine_length          = 1.0
        self.bypass_ratio           = 0.0
        self.areas                  = Data()
        self.tag                    = 'Network'

        self.ambient_temp           = 300.0     # [K]  This is the temp of the outside of the rotor cryostat. This may match the ambient air tem, but does not need to.
    
    # manage process with a driver function
    def evaluate_thrust(self, conditions, state):
        """ Calculate thrust given the current state of the vehicle
    
            Assumptions:
            None
    
            Source:
            N/A
    
            Inputs:
            state [state()]
    
            Outputs:
            results.thrust_force_vector [newtons]
            results.vehicle_mass_rate   [kg/s]
    
            Properties Used:
            Defaulted values
        """         
        
        # unpack

        propulsor                   = self.propulsor                    # Electric ducted fan(s) not including the motor
        motor                       = self.motor                        # Motor(s) driving those fans
        powersupply                 = self.powersupply                  # Powersupply(s) providing electricity for the motor(s)
        esc                         = self.esc                          # Motor speed controller(s)
        rotor                       = self.rotor                        # Rotor(s) of the motor(s)
        lead                        = self.lead                         # Current supply leads supplying the rotor(s)
        ccs                         = self.ccs                          # Rotor constant current supply
        cryocooler                  = self.cryocooler                   # Rotor cryocoolers, powered by electricity
        heat_exchanger              = self.cryogenic_heat_exchanger     # Rotor cryocooling, powered by cryogen
        skin_temp                   = self.ambient_temp                 # Exterior temperature of the rotor
        cooling_share_cryogen       = self.cryogen_proportion           # Proportion of rotor cryocooling provided by cryogen
        cooling_share_cryocooler    = 1.0 - cooling_share_cryogen       # Proportion of rotor cryocooling provided by cryocooler

        amb_temp        = conditions.freestream.temperature
    
        conditions      = state.conditions
        numerics        = state.numerics

        # Solve the thrust using the other network (i.e. the ducted fan network)
        results = propulsor.evaluate_thrust(state)

        # Calculate the required electric power to be supplied to the ducted fan motor by dividing the shaft power required by the ducted fan by the efficiency of the ducted fan motor
        # Note here that the efficiency must not include the efficiency of the rotor and rotor supply components as these are handled separately below.
        # powersupply.inputs.power_in = propulsor.thrust.outputs.power/motor.motor_efficiency
        motor_power_in        = propulsor.thrust.outputs.power/motor.motor_efficiency

        # Calculate the power used by the power electronics. This does not include the power delivered by the power elctronics to the fan motor.
        esc_power             = motor_power_in/esc.efficiency - motor_power_in

        # Calculate the power that must be supplied to the rotor. This also calculates the cryo load at the rotor and stores this value as rotor.results.cryo_load
        rotor_power_in        = rotor.power(rotor.current, skin_temp)

        # Calculate the power loss in the rotor current supply leads. Two leads are required to complete the circuit. It is assumed the leads are operating at their design current or at zero current.
        lead_power            = 0
        if rotor.current != 0:
            lead_power        = 2*lead.minimum_Q

        # Calculate the power used by the rotor's current supply.
        ccs_power             = (lead_power+rotor_power_in)/ccs.efficiency - (lead_power+rotor_power_in)

        # # Calculate the power loss in the HTS Dynamo. This replaces the current supply leads.
        # lead_power          = 0
        # if rotor.current != 0:
            # lead_power      = dynamo.shaft_power(rotor_power_in)

        # # Calculate the power used by the HTS Dynamo powertrain, i.e. the esc, motor, and gearbox.
        # ccs_power           = dynamo.power_in

        # Retreive the cryogenic heat load from the rotor components (not including the leads).
        rotor_cryo_cryostat   = rotor.results.cryo_load

        # Retreive the cryogenic load due to the current supply leads
        lead_cryo_load        = 2*lead.unpowered_Q
        if rotor.current != 0:
            lead_cryo_load    = 2*lead.minimum_Q

        # # Retreive the cryogenic load due to the dynamo
        # lead_cryo_load        = 0.0
        # if rotor.current != 0:
        #     lead_cryo_load    = dynamo.cryo_load

        # Sum the two rotor cryogenic heat loads to give the total rotor cryogenic load.
        rotor_cryo_load = rotor_cryo_cryostat + lead_cryo_load

        # Calculate the power required from the cryocooler (if present)
        cryocooler_power = 0
        if cooling_share_cryocooler != 0:
            cryocooler_load = cooling_share_cryocooler * rotor_cryo_load
            cryocooler_power = cryocooler.power(cryocooler_load, cryocooler.cooler_type, rotor.temperature, amb_temp)[0]

        # Calculate the cryogen use required for cooling (if used)
        cryogen_mdot = 0
        if cooling_share_cryogen != 0:
            cryogen_load = cooling_share_cryogen * rotor_cryo_load
            cryogen_mdot = heat_exchanger.mdot(cryogen_load)

        # Sum all the power users to get the power required to be supplied by the powersupply, i.e. the turboelectric generator
        powersupply.inputs.power_in = motor_power_in + esc_power + rotor_power_in + lead_power + ccs_power + cryocooler_power

        # Calculate the fuel mass flow rate at the turboelectric power supply.
        fuel_mdot     = powersupply.energy_calc(conditions, numerics)

        # Sum the mass flow rates and store this total as vehicle_mass_rate so the vehicle mass change reflects both the fuel used and the cryogen used.
        results.vehicle_mass_rate   = fuel_mdot + cryogen_mdot

        # Pack up the mass flow rate components so they can be tracked.
        results.cryogen_mass_rate   = cryogen_mdot
        results.fuel_mass_rate      = fuel_mdot

        return results
            
    __call__ = evaluate_thrust

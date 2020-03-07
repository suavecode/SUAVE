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
        
        self.propulsor          = None      # i.e. the ducted fan (not including the motor).
        self.motor              = None      # the motor that drives the fan, which may be partial or fully HTS.
        self.powersupply        = None      # i.e. the turboelectric generator, the generator of which may be partial or fully HTS
        # self.motor_efficiency   = .95
        self.nacelle_diameter  = 1.0
        self.engine_length     = 1.0
        self.bypass_ratio      = 0.0
        self.areas             = Data()
        self.tag                = 'Network'
    
    # manage process with a driver function
    def evaluate_thrust(self,state):
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

        propulsor       = self.propulsor
        motor           = self.motor
        powersupply     = self.powersupply
    
        conditions      = state.conditions
        numerics        = state.numerics

        # Solve the thrust using the other network (i.e. the ducted fan network)
        results = propulsor.evaluate_thrust(state)

        # Calculate the required electric power to be supplied to the ducted fan motor by dividing the shaft power required by the ducted fan by the efficiency of the ducted fan motor
        # Note here that the efficiency must not include the efficiency of the rotor and rotor supply components as these are handled separately below.
        powersupply.inputs.power_in = propulsor.thrust.outputs.power/motor.motor_efficiency
        # motor_power_in        = propulsor.thrust.outputs.power/motor.motor_efficiency

        # # Calculate the power used by the power electronics. This does not include the power delivered by the power elctronics to the fan motor.
        # esc_power             = motor_power_in/esc.efficiency - motor_power_in

        # # Calculate the power that must be supplied to the rotor.
        # rotor_power_in        = rotor.power(rotor_current, ambient_temp)

        # # Calculate the power loss in the rotor current supply leads. Two leads are required to complete the circuit. This assumes optimum current or no current are the only possibilities.
        # lead_power            = 0
        # if rotor_current != 0:
        #     lead_power        = 2*lead.minimum_Q

        # # Calculate the power used by the rotor's current supply.
        # ccs_power             = (lead_power+rotor_power_in)/ccs.efficiency - (lead_power+rotor_power_in)

        # # Retreive the cryogenic heat load from the rotor components (not including the leads).
        # rotor_cryo_cryostat   = rotor.results.cryo_load

        # # Retreive the cryogenic load due to the current supply leads
        # lead_cryo_load        = 2*lead.unpowered_Q
        # if rotor_current != 0:
            # lead_cryo_load    = 2*lead.minimum_Q

        # # Sum the two rotor cryogenic heat loads to give the total rotor cryogenic load.
        # rotor_cryo_load = rotor_cryo_cryostat + lead_cryo_load

        # # Calculate the power required from the cryocooler (if present)
        # cryocooler_power = 0
        # if cooling_share_cryocooler != 0:
        #     cryocooler_load = cooling_share_cryocooler * rotor_cryo_load
        #     cryocooler_power = cryocooler(cryocooler_load, cooler_type, cryo_temp, amb_temp)[0]

        # # Calculate the cryogen use required for cooling (if used)
        # cryogen_mdot = 0
        # if cooling_share_cryogen != 0:
        #     cryogen_load = cooling_share_cryogen * rotor_cryo_load
        #     cryogen_mdot = cryogenic_heat_exchanger(cryogen_load, cryogen_type, cryogen_temp, cryo_temp)

        # # Sum all the power users to get the power required to be supplied by the powersupply, i.e. the turboelectric generator
        # powersupply.inputs.power_in = motor_power_in + esc_power + rotor_power_in + lead_power + ccs_power + cryocooler_power

        # # Calculate the fuel mass flow rate at the turboelectric power supply.
        # fuel_mdot     = powersupply.energy_calc(conditions, numerics)

        # # Sum the mass flow rates and store this total as vehicle_mass_rate so the vehicle mass change reflects both the fuel used and the cryogen used.
        # results.vehicle_mass_rate   = fuel_mdot + cryogen_mdot

        # # Pack up the mass flow rate components so they can be tracked.
        # results.cryogen_mass_rate   = cryogen_mdot
        # results.fuel_mass_rate      = fuel_mdot

        # return results

        # From here down can be deleted once the above is implemented properly

        # Calculate the fuel mass flow rate at the turboelectric power supply.
        # This assumes 100% of the electric power delivered by the turboelectric generator is delivered to the motor, i.e. there are no power electronics, avionics, transmission losses, cooling systems, or any other power users.
        mdot = powersupply.energy_calc(conditions, numerics)
        
        # Varying the mass in this way here prevents the causes of the difference in mass from being extracted later, for example if there are more than one fuel type.
        # Also if the change in mass is not wholly due to fuel this may cause difficulties in calculating SFC.
        # results.vehicle_cryogen_mass_rate    = mdot      # <- Possible fix.
        # Methods-Missions-Segments-Common.update_weights uses results.vehicle_mass_rate to update the vehicle mass, calling this change in mass "mdot_fuel". To prevent breaking this it is suggested the total mass change be stored in results.vehicle_mass_rate as is done elsewhere for mass varying batteries. To track the individual contributions to the mass difference new variables in "results" will need to be updated.
        # Modification of Methods-Missions-Segments-Common.update_weights is required as this is called during iteration, i.e. it either must be modified to consider other weights here, or updated multiple other places. A "cryogenic" flag in results would prevent other simulations failing due to a missing results.vehicle_fuel_mass_rate, or just add results.vehicle_cryogen_mass_rate to Analyses-Mission-Segments-Conditions.Aerodynamics initialised as zero and apply the difference regardless as it will usually be zero.
        results.vehicle_mass_rate       = mdot

        return results
            
    __call__ = evaluate_thrust

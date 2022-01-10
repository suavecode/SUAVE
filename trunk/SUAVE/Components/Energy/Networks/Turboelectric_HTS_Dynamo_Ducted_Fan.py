## @ingroup Components-Energy-Networks
# Turboelectric_HTS_Dynamo_Ducted_Fan.py
#
# Created:  Mar 2020, K. Hamilton

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
import SUAVE

# package imports
import numpy as np
from SUAVE.Core import Data
from SUAVE.Components.Energy.Networks.Network import Network

# ----------------------------------------------------------------------
#  Network
# ----------------------------------------------------------------------

## @ingroup Components-Energy-Networks
class Turboelectric_HTS_Dynamo_Ducted_Fan(Network):
    """ A serial hybrid powertrain with partially superconducting propulsion motors where the superconducting field coils are energised by a HTS Dynamo rather than the typical method of supplying current via resistive current leads.
    
        Assumptions:
        None
        
        Source:
        None
    """
    
    def __defaults__(self):
        """ This sets the default values for the network to function.
    
            Assumptions:
            N/A
    
            Source:
            N/A
    
            Inputs:
            None
    
            Outputs:
            None
    
            Properties Used:
            N/A
        """         

        self.ducted_fan                 = None  # i.e. the ducted fan (not including the motor).
        self.motor                      = None  # the motor that drives the fan, which may be partial or fully HTS.
        self.powersupply                = None  # i.e. the turboelectric generator, the generator of which may be partial or fully HTS
        self.esc                        = None  # the electronics that supply the motor armature windings
        self.rotor                      = None  # the motor rotor (handled as a seperate component to the motor)
        self.hts_dynamo                 = None  # HTS Dynamo used instead of the current leads. This component includes the dynamo motor
        self.dynamo_esc                 = None  # The dynamo motor speed controller
        self.cryocooler                 = None  # cryocooler which cools the rotor using energy
        self.heat_exchanger             = None  # heat exchanger that cools the rotor using cryogen
        self.cryogen_proportion         = 1.0   # Proportion of cooling to be supplied by the cryogenic heat exchanger, rather than by the cryocooler
        self.number_of_engines          = 1.0   # number of ducted_fans, also the number of propulsion motors.
        self.has_additional_fuel_type   = True

        self.engine_length              = 1.0
        self.bypass_ratio               = 0.0
        self.areas                      = Data()
        self.tag                        = 'Network'

        self.ambient_skin               = False  # flag to set whether the outer surface of the rotor is amnbient temperature or not.
        self.skin_temp                  = 300.0  # [K]  if self.ambient_skin is false, this is the temperature of the rotor skin. 
    
    # manage process with a driver function
    def evaluate_thrust(self, state):
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

        ducted_fan                  = self.ducted_fan               # Electric ducted fan(s) excluding motor
        motor                       = self.motor                    # Motor(s) driving those fans
        powersupply                 = self.powersupply              # Electricity producer(s)
        esc                         = self.esc                      # Motor speed controller(s)
        rotor                       = self.rotor                    # Rotor(s) of the motor(s)
        hts_dynamo                  = self.hts_dynamo               # HTS Dynamo supplying the rotor
        dynamo_esc                  = self.dynamo_esc               # HTS Dynamo speed controller
        cryocooler                  = self.cryocooler               # Rotor cryocoolers, powered by electricity
        heat_exchanger              = self.heat_exchanger           # Rotor cryocooling, powered by cryogen
        
        ambient_skin                = self.ambient_skin             # flag to indicate rotor skin temp
        rotor_surface_temp          = self.skin_temp                # Exterior temperature of the rotor
        cooling_share_cryogen       = self.cryogen_proportion       # Proportion of rotor cooling provided by cryogen
        cooling_share_cryocooler    = 1.0 - cooling_share_cryogen   # Proportion of rotor cooling provided by cryocooler
        number_of_engines           = self.number_of_engines        # number of propulsors and number of propulsion motors
        number_of_supplies          = self.powersupply.number_of_engines    # number of turboelectric generators
        cryogen_is_fuel             = self.heat_exchanger.cryogen_is_fuel   # Is the cryogen used as fuel.
    
        conditions      = state.conditions
        numerics        = state.numerics

        amb_temp        = conditions.freestream.temperature

        # Solve the thrust using the other network (i.e. the ducted fan network)
        results = ducted_fan.evaluate_thrust(state)


        # Calculate the required electric power to be supplied to the ducted fan motor by dividing the shaft power required by the ducted fan by the efficiency of the ducted fan motor
        # Note here that the efficiency must not include the efficiency of the rotor and rotor supply components as these are handled separately below.
        # powersupply.inputs.power_in = propulsor.thrust.outputs.power/motor.motor_efficiency
        motor_power_in        = ducted_fan.thrust.outputs.power/motor.motor_efficiency

        # Calculate the power used by the power electronics. This does not include the power delivered by the power elctronics to the fan motor.
        esc_power             = motor_power_in/esc.efficiency - motor_power_in

        # Set the rotor skin temp. Either it's ambient, or it's the temperature set in the rotor.
        skin_temp = amb_temp * 1

        if ambient_skin == False:
            skin_temp[:]    = rotor_surface_temp 

        # If the rotor current is to be varied depending on the motor power here is the place to do it. For now the rotor current is set as constant.
        #rotor_current       = np.full_like(motor_power_in, 800)
        rotor_current       = np.linspace(200, 1400, num = len(motor_power_in))
        rotor_current       = rotor_current.reshape(len(motor_power_in),1)
  
        # Calculate the power that must be supplied to the rotor. This also calculates the cryo load per rotor and stores this value as rotor.outputs.cryo_load
        single_rotor_power  = rotor.power(rotor_current, skin_temp)
        rotor_power_in      = single_rotor_power * ducted_fan.number_of_engines

        # --------  Current Supply Dynamo --------------------

        # Calculate the power loss in the HTS Dynamo.
        dynamo_powers           = hts_dynamo.shaft_power(rotor.temperature, rotor_current, single_rotor_power)
        dynamo_shaft_power      = dynamo_powers[0]

        # Calculate the power used by the HTS Dynamo powertrain, i.e. the esc, motor, and gearbox.
        dynamo_esc_power        = dynamo_esc.power_in(hts_dynamo, dynamo_shaft_power, rotor_current)

        # Retreive the cryogenic load due to the dynamo
        dynamo_cryo_load        = dynamo_powers[1]

        # Rename dynamo power components as lead power components. As the dynamo replaces the current supply leads the power required is stored as lead_power to minimise code changes elsewhere.
        lead_power          = dynamo_shaft_power
        lead_cryo_load      = dynamo_cryo_load
        ccs_power           = dynamo_esc_power

        # -------- End of Rotor Current Supply --------------------------

        # Multiply the power (electrical and cryogenic) required by the rotor components by the number of rotors, i.e. the number of propulsion motors
        all_leads_power             = number_of_engines * lead_power    
        all_leads_cryo              = number_of_engines * lead_cryo_load
        all_ccs_power               = number_of_engines * ccs_power     

        # Retreive the cryogenic heat load from the rotor components (not including the leads).
        rotor_cryo_cryostat         = rotor.outputs.cryo_load * number_of_engines

        # Sum the two rotor cryogenic heat loads to give the total rotor cryogenic load.
        rotor_cryo_load             = rotor_cryo_cryostat + all_leads_cryo

        # Calculate the power required from the cryocoolers (if present)
        cryocooler_power = 0.0

        if cooling_share_cryocooler != 0.0:
            cryocooler_load         = cooling_share_cryocooler * rotor_cryo_load
            cryocooler_power        = cryocooler.energy_calc(cryocooler_load, rotor.temperature, amb_temp)

        # Calculate the cryogen use required for cooling (if used)
        cryogen_mdot = 0.0
        if cooling_share_cryogen != 0.0:
            cryogen_load            = cooling_share_cryogen * rotor_cryo_load
            cryogen_mdot            = heat_exchanger.energy_calc(cryogen_load, conditions)

        # Sum all the power users to get the power required to be supplied by each powersupply, i.e. the turboelectric generators
        powersupply.inputs.power_in = (motor_power_in + esc_power + rotor_power_in + all_leads_power + all_ccs_power + cryocooler_power) / number_of_supplies

        # Calculate the fuel mass flow rate at the turboelectric power supply.
        fuel_mdot                   = number_of_supplies * powersupply.energy_calc(conditions, numerics)

        # Sum the mass flow rates and store this total as vehicle_mass_rate so the vehicle mass change reflects both the fuel used and the cryogen used, unless the cryogen is fuel.
        results.vehicle_mass_rate   = fuel_mdot + (cryogen_mdot * (1.0-cryogen_is_fuel))

        # Pack up the mass flow rate components so they can be tracked.
        results.vehicle_additional_fuel_rate   = cryogen_mdot
        results.vehicle_fuel_rate              = fuel_mdot   

        return results
            
    __call__ = evaluate_thrust
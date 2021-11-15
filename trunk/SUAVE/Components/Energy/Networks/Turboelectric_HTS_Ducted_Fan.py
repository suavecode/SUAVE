## @ingroup Components-Energy-Networks
# Turboelectric_HTS_Ducted_Fan.py
#
# Created:  Mar 2020, K. Hamilton
# Modified: Nov 2021, S. Claridge

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
import SUAVE

# package imports
import numpy as np
from SUAVE.Core import Data

from SUAVE.Methods.Cooling.Leads.copper_lead import Q_offdesign

from SUAVE.Components.Energy.Networks.Network import Network

# ----------------------------------------------------------------------
#  Network
# ----------------------------------------------------------------------

## @ingroup Components-Energy-Networks
class Turboelectric_HTS_Ducted_Fan(Network):
    """ A serial hybrid powertrain with partially superconducting propulsion motors where the superconducting field coils are energised by resistive current leads.
    
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
   
        self.leads                      = 2.0   # number of cryogenic leads supplying the rotor(s). Typically twice the number of rotors.
        self.number_of_engines          = 1.0   # number of ducted_fans, also the number of propulsion motors.

        self.nacelle_diameter           = 1.0
        self.engine_length              = 1.0
        self.bypass_ratio               = 0.0
        self.areas                      = Data()
        self.tag                        = 'Turboelectric_HTS_Ducted_Fan'

        self.ambient_skin               = 0         # flag to set whether the outer surface of the rotor is amnbient temperature or not.
        self.skin_temp                  = 300.0     # [K]  if self.ambient_skin is false, this is the temperature of the rotor skin. 
    
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
        lead                        = self.lead                     # Current leads supplying the rotor(s)
        ccs                         = self.ccs                      # Rotor constant current supply
        cryocooler                  = self.cryocooler               # Rotor cryocoolers, powered by electricity
        
        
        
        ambient_skin                = self.ambient_skin             # flag to indicate rotor skin temp
        rotor_surface_temp          = self.skin_temp                # Exterior temperature of the rotors
        leads                       = self.leads                    # number of rotor leads, typically twice the number of rotors
        number_of_engines           = self.number_of_engines        # number of propulsors and number of propulsion motors
        number_of_supplies          = self.powersupply.number_of_engines    # number of turboelectric generators

       
    
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
        skin_temp           = np.empty_like(amb_temp)
        skin_temp[:]        = amb_temp
        if ambient_skin == 0:
            skin_temp[:]    = rotor_surface_temp 

        # If the rotor current is to be varied depending on the motor power here is the place to do it. For now the rotor current is set as constant.
        rotor_current       = np.full_like(motor_power_in, rotor.current)

        # Calculate the power that must be supplied to the rotor. This also calculates the cryo load per rotor and stores this value as rotor.outputs.cryo_load
        single_rotor_power  = rotor.power(rotor_current, skin_temp)
        rotor_power_in      = single_rotor_power * ducted_fan.number_of_engines


        # -------- Rotor Current Supply ---------------------------------


        # Calculate the power loss in the rotor current supply leads.
        # The cryogenic loading due to the leads is also calculated here.
        lead_power            = np.zeros_like(rotor_current)
        lead_cryo_load        = np.full_like(rotor_current, lead.unpowered_Q)
        # Iterate through each element in the rotor_current array to build the cryo_load and lead_power array
        for index, r_current in np.ndenumerate(rotor_current):
            if r_current != 0.0:
                lead_powers             = Q_offdesign(lead, r_current)
                lead_power[index]       = lead_powers[1]
                lead_cryo_load[index]   = lead_powers[0]
                
        # Multiply the lead powers by the number of leads, this is typically twice the number of motors
        lead_power          = lead_power * leads
        lead_cryo_load      = lead_cryo_load * leads

        # Calculate the power used by the rotor's current supply.
        ccs_power             = (lead_power+rotor_power_in)/ccs.efficiency - (lead_power+rotor_power_in)


        # Multiply the power (electrical and cryogenic) required by the rotor components by the number of rotors, i.e. the number of propulsion motors
        all_leads_power             = number_of_engines * lead_power    
        all_leads_cryo              = number_of_engines * lead_cryo_load
        all_ccs_power               = number_of_engines * ccs_power     
        

        # Retreive the cryogenic heat load from the rotor components (not including the leads).
        rotor_cryo_cryostat         = rotor.outputs.cryo_load * number_of_engines

        # Sum the two rotor cryogenic heat loads to give the total rotor cryogenic load.
        rotor_cryo_load             = rotor_cryo_cryostat + all_leads_cryo
        

        # Calculate the power required from the cryocoolers
        
        cryocooler_load         = rotor_cryo_load
        cryocooler_power        = cryocooler.energy_calc(cryocooler_load, rotor.temperature, amb_temp)

        # Sum all the power users to get the power required to be supplied by each powersupply, i.e. the turboelectric generators
        powersupply.inputs.power_in = (motor_power_in + esc_power + rotor_power_in + all_leads_power + all_ccs_power + cryocooler_power) / number_of_supplies

        # Calculate the fuel mass flow rate at the turboelectric power supply.
        fuel_mdot                   = number_of_supplies * powersupply.energy_calc(conditions, numerics)

        # Sum the mass flow rates and store this total as vehicle_mass_rate 
        results.vehicle_mass_rate   = fuel_mdot 

        # Pack up the mass flow rate components so they can be tracked.
        results.vehicle_fuel_rate      = fuel_mdot   

        return results
            
    __call__ = evaluate_thrust

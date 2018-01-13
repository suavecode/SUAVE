## @ingroup Components-Energy-Networks
# Ramjet.py
#
# Created:  Jun 2017, P. Goncalves

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
import SUAVE

from SUAVE.Core import Data, Units
from SUAVE.Components.Propulsors.Propulsor import Propulsor

# ----------------------------------------------------------------------
#  Ramjet Network
# ----------------------------------------------------------------------

## @ingroup Components-Energy-Networks
class Ramjet(Propulsor):
    """ This is a ramjet for supersonic flight.

        Assumptions:
        None

        Source:
        Most of the componentes come from this book:
        https://web.stanford.edu/~cantwell/AA283_Course_Material/AA283_Course_Notes/
    """

    def __defaults__(self):
        """ This sets the default values for the network to function.

        Assumptions:
        None

        Source:
        N/A

        Inputs:
        None

        Outputs:
        None

        Properties Used:
        N/A
        """

        #setting the default values
        self.tag = 'Ramjet'
        self.number_of_engines = 1.0
        self.nacelle_diameter  = 1.0
        self.engine_length     = 1.0

    _component_root_map = None


    def evaluate_thrust(self,state):
        """ Calculate thrust given the current state of the vehicle

        	Assumptions:
        	None

        	Source:
        	N/A

        	Inputs:
        	state [state()]

        	Outputs:
        	results.thrust_force_vector                   [newtons]
        	results.vehicle_mass_rate                     [kg/s]
        	conditions.propulsion.acoustic_outputs:
        	    core:
        		exit_static_temperature                  [K]
        		exit_static_pressure                     [K]
        		exit_stagnation_temperature              [K]
        		exit_stagnation_pressure                 [Pa]
        		exit_velocity                            [m/s]
        	    fan:
        		exit_static_temperature                  [K]
        		exit_static_pressure                     [K]
        		exit_stagnation_temperature              [K]
        		exit_stagnation_pressure                 [Pa]
        		exit_velocity                            [m/s]

        	Properties Used:
        	Defaulted values
            """

        #Unpack
        conditions = state.conditions

        ram                       = self.ram
        inlet_nozzle              = self.inlet_nozzle
        combustor                 = self.combustor
        core_nozzle               = self.core_nozzle
        thrust                    = self.thrust
        number_of_engines         = self.number_of_engines

        #Creating the network by manually linking the different components

        #set the working fluid to determine the fluid properties
        ram.inputs.working_fluid                               = self.working_fluid

        #Flow through the ram , this computes the necessary flow quantities and stores it into conditions
        ram(conditions)

        #link inlet nozzle to ram
        inlet_nozzle.inputs.stagnation_temperature             = ram.outputs.stagnation_temperature
        inlet_nozzle.inputs.stagnation_pressure                = ram.outputs.stagnation_pressure

        #Flow through the inlet nozzle
        inlet_nozzle(conditions)

        #link the combustor to the inlet nozzle
        combustor.inputs.stagnation_temperature                = inlet_nozzle.outputs.stagnation_temperature
        combustor.inputs.stagnation_pressure                   = inlet_nozzle.outputs.stagnation_pressure
        combustor.inputs.mach_number                           = inlet_nozzle.outputs.mach_number

        #flow through the high pressor comprresor
        combustor.compute_rayleigh(conditions)


        #link the core nozzle to the low pressure turbine
        core_nozzle.inputs.stagnation_temperature              = combustor.outputs.stagnation_temperature
        core_nozzle.inputs.stagnation_pressure                 = combustor.outputs.stagnation_pressure

        #flow through the core nozzle
        core_nozzle.compute_limited_geometry(conditions)

        # compute the thrust using the thrust component
        #link the thrust component to the core nozzle
        thrust.inputs.core_exit_velocity                       = core_nozzle.outputs.velocity
        thrust.inputs.core_area_ratio                          = core_nozzle.outputs.area_ratio
        thrust.inputs.core_nozzle                              = core_nozzle.outputs

        #link the thrust component to the combustor
        thrust.inputs.fuel_to_air_ratio                        = combustor.outputs.fuel_to_air_ratio

        #link the thrust component to the low pressure compressor
        thrust.inputs.stag_temp_lpt_exit                       = core_nozzle.outputs.stagnation_temperature
        thrust.inputs.stag_press_lpt_exit                      = core_nozzle.outputs.stagnation_pressure
        thrust.inputs.number_of_engines                        = number_of_engines
        thrust.inputs.flow_through_core                        =  1.0 #scaled constant to turn on core thrust computation
        thrust.inputs.flow_through_fan                         =  0.0 #scaled constant to turn on fan thrust computation

        #compute the thrust
        thrust(conditions)

        #getting the network outputs from the thrust outputs
        F            = thrust.outputs.thrust*[1,0,0]
        mdot         = thrust.outputs.fuel_flow_rate
        Isp          = thrust.outputs.specific_impulse
        output_power = thrust.outputs.power
        F_vec        = conditions.ones_row(3) * 0.0
        F_vec[:,0]   = F[:,0]
        F            = F_vec

        results = Data()
        results.thrust_force_vector = F
        results.vehicle_mass_rate   = mdot

        return results

    def size(self,state):

        """ Size the ramjet

            Assumptions:
            None

            Source:
            N/A

            Inputs:
            State [state()]

            Outputs:
            None

            Properties Used:
            N/A
        """

        #Unpack components
        conditions = state.conditions
        ram                       = self.ram
        inlet_nozzle              = self.inlet_nozzle
        combustor                 = self.combustor
        core_nozzle               = self.core_nozzle
        thrust                    = self.thrust

        #Creating the network by manually linking the different components
        #set the working fluid to determine the fluid properties
        ram.inputs.working_fluid                               = self.working_fluid

        #Flow through the ram , this computes the necessary flow quantities and stores it into conditions
        ram(conditions)

        #link inlet nozzle to ram
        inlet_nozzle.inputs.stagnation_temperature             = ram.outputs.stagnation_temperature
        inlet_nozzle.inputs.stagnation_pressure                = ram.outputs.stagnation_pressure

        #Flow through the inlet nozzle
        inlet_nozzle(conditions)


        #link the combustor to the high pressure compressor
        combustor.inputs.stagnation_temperature                = inlet_nozzle.outputs.stagnation_temperature
        combustor.inputs.stagnation_pressure                   = inlet_nozzle.outputs.stagnation_pressure
        combustor.inputs.mach_number                           = inlet_nozzle.outputs.mach_number

        #flow through the high pressure compressor
        combustor.compute_rayleigh(conditions)

        #link the core nozzle to the low pressure turbine
        core_nozzle.inputs.stagnation_temperature              = combustor.outputs.stagnation_temperature
        core_nozzle.inputs.stagnation_pressure                 = combustor.outputs.stagnation_pressure

        #flow through the core nozzle
        core_nozzle(conditions)

        # compute the thrust using the thrust component
        #link the thrust component to the core nozzle
        thrust.inputs.core_exit_velocity                       = core_nozzle.outputs.velocity
        thrust.inputs.core_area_ratio                          = core_nozzle.outputs.area_ratio
        thrust.inputs.core_nozzle                              = core_nozzle.outputs

        #link the thrust component to the combustor
        thrust.inputs.fuel_to_air_ratio                        = combustor.outputs.fuel_to_air_ratio

        #link the thrust component to the low pressure compressor
        thrust.inputs.stag_temp_lpt_exit                       = core_nozzle.outputs.stagnation_temperature
        thrust.inputs.stag_press_lpt_exit                      = core_nozzle.outputs.stagnation_pressure

        #compute the thrust
        thrust.size(conditions)

    __call__ = evaluate_thrust

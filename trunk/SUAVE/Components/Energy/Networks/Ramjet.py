## @ingroup Components-Energy-Networks
# Ramjet.py
#
# Created:  Jun 2017, P. Goncalves
# Modified: Jan 2018, W. Maier

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
		results.thrust_force_vector                      [newtons]
		results.vehicle_mass_rate                        [kg/s]
		results.specific_impulse                         [s]
		conditions.propulsion.acoustic_outputs:
		    core:
			exit_static_temperature                  [K]
			exit_static_pressure                     [K]
			exit_stagnation_temperature              [K]
			exit_stagnation_pressure                 [Pa]
			exit_velocity                            [m/s]

		Properties Used:
		Defaulted values
	    """

        # unpack
        conditions                = state.conditions
        ram                       = self.ram
        inlet_nozzle              = self.inlet_nozzle
        combustor                 = self.combustor
        core_nozzle               = self.core_nozzle
        thrust                    = self.thrust
        number_of_engines         = self.number_of_engines

        # creating the network by manually linking the different components

        # set the working fluid to determine the fluid properties
        ram.inputs.working_fluid                               = self.working_fluid

        # flow through the ram
        ram(conditions)

        # link inlet nozzle to ram
        inlet_nozzle.inputs          = ram.outputs
	
        # flow through the inlet nozzle
        inlet_nozzle(conditions)

        # link the combustor to the inlet nozzle
        combustor.inputs             = inlet_nozzle.outputs

        # flow through the combustor
        combustor.compute_rayleigh(conditions)

        #link the core nozzle to the combustor
        core_nozzle.inputs           = combustor.outputs

        # flow through the core nozzle
        core_nozzle.compute_limited_geometry(conditions)

        # compute the thrust using the thrust component
        
        # link the thrust component to the core nozzle
        thrust.inputs.core_nozzle                              = core_nozzle.outputs
	thrust.inputs.total_temperature_reference              = core_nozzle.outputs.stagnation_temperature
	thrust.inputs.total_pressure_reference                 = core_nozzle.outputs.stagnation_pressure
	
        # link the thrust component to the combustor
        thrust.inputs.fuel_to_air_ratio                        = combustor.outputs.fuel_to_air_ratio

        # link the thrust component 
        thrust.inputs.number_of_engines                        = number_of_engines
        thrust.inputs.flow_through_core                        =  1.0 #scaled constant to turn on core thrust computation
        thrust.inputs.flow_through_fan                         =  0.0 #scaled constant to turn on fan thrust computation

        # compute the thrust
        thrust(conditions)

        # getting the network outputs from the thrust outputs
        F            = thrust.outputs.thrust*[1,0,0]
        mdot         = thrust.outputs.fuel_flow_rate
        Isp          = thrust.outputs.specific_impulse
        output_power = thrust.outputs.power
        F_vec        = conditions.ones_row(3) * 0.0
        F_vec[:,0]   = F[:,0]
        F            = F_vec

        results                     = Data()
        results.thrust_force_vector = F
        results.vehicle_mass_rate   = mdot
	results.specific_impulse    = Isp
	
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

        # Unpack components
        conditions                = state.conditions
        ram                       = self.ram
        inlet_nozzle              = self.inlet_nozzle
        combustor                 = self.combustor
        core_nozzle               = self.core_nozzle
        thrust                    = self.thrust

        # Creating the network by manually linking the different components
        
        # set the working fluid to determine the fluid properties
        ram.inputs.working_fluid = self.working_fluid

        # flow through the ram
        ram(conditions)

        # link inlet nozzle to ram
        inlet_nozzle.inputs = ram.outputs

        # flow through the inlet nozzle
        inlet_nozzle(conditions)

        # link the combustor to the high pressure compressor
        combustor.inputs = inlet_nozzle.outputs.mach_number

        # flow through the high pressure compressor
        combustor.compute_rayleigh(conditions)

        # link the core nozzle to the low pressure turbine
        core_nozzle.inputs = combustor.outputs.stagnation_pressure

        # flow through the core nozzle
        core_nozzle(conditions)

        # compute the thrust using the thrust component
        
        # link the thrust component to the core nozzle
        thrust.inputs.stagnation_temperature                   = core_nozzle.outputs.stagnation_temperature
        thrust.inputs.stagnation_pressure                      = core_nozzle.outputs.stagnation_pressure
	thrust.inputs.core_nozzle                              = core_nozzle.outputs

        # link the thrust component to the combustor
        thrust.inputs.fuel_to_air_ratio                        = combustor.outputs.fuel_to_air_ratio

        # compute the thrust
        thrust.size(conditions)

    __call__ = evaluate_thrust

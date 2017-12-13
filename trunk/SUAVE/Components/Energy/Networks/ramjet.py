## @ingroup Components-Energy-Networks
#Ramjet.py
#
# Created:  Dec 2017, W. Maier,
# Modified:

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
#  Ramjet Network
# ----------------------------------------------------------------------

## @ingroup Components-Energy-Networks
class Ramjet(Propulsor):
    """ This is a ramjet supersonic/hypersonic flight.

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

        #areas needed for drag; not in there yet
        self.areas             = Data()
        self.areas.wetted      = 0.0
        self.areas.maximum     = 0.0
        self.areas.exit        = 0.0
        self.areas.inflow      = 0.0
    _component_root_map = None

    # linking the different network components
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
            conditions.propulsion.acoustic_outputs:
                core:
                    exit_static_temperature
                    exit_static_pressure
                    exit_stagnation_temperature
                    exit_stagnation_pressure
                    exit_velocity
                fan:
                    exit_static_temperature
                    exit_static_pressure
                    exit_stagnation_temperature
                    exit_stagnation_pressure
                    exit_velocity

            Properties Used:
            Defaulted values
        """

        #Unpack
        conditions = state.conditions

        #Creating the network by manually linking the different components
        ram                       = self.ram
        inlet_nozzle              = self.inlet_nozzle
        combustor                 = self.combustor
        core_nozzle               = self.core_nozzle
        thrust                    = self.thrust
        number_of_engines         = self.number_of_engines

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
        combustor.inputs.stagnation_temperature                = inlet_nozzle.stagnation_temperature
        combustor.inputs.stagnation_pressure                   = inlet_nozzle.stagnation_pressure
        combustor.inputs.mach_number                           = inlet_nozzle.outputs.mach_number

        #flow through the combustor
        combustor.compute_RayleighFlow(conditions)

        #link the core nozzle to the low pressure turbine combustor
        core_nozzle.inputs.stagnation_temperature              = low_pressure_turbine.outputs.stagnation_temperature
        core_nozzle.inputs.stagnation_pressure                 = low_pressure_turbine.outputs.stagnation_pressure

        #flow through the core nozzle
        core_nozzle(conditions)

        # compute the thrust using the thrust component
        # link the thrust component to the core nozzle
        thrust.inputs.core_exit_velocity                       = core_nozzle.outputs.velocity
        thrust.inputs.core_area_ratio                          = core_nozzle.outputs.area_ratio
        thrust.inputs.core_nozzle                              = core_nozzle.outputs

        #link the thrust component to the combustor
        thrust.inputs.fuel_to_air_ratio                        = combustor.outputs.fuel_to_air_ratio

        #link the thrust component to the inlet nozzle
        thrust.inputs.stag_press_comb                          = inlet_nozzle.outputs.stagnation_temperature
        thrust.inputs.stag_tempe_comb                          = inlet_nozzle.outputs.stagnation_pressure
        thrust.inputs.number_of_engines                        = number_of_engines
        thrust.inputs.flow_through_core                        = 1
        thrust.inputs.bypass_ratio                             = 0

        #compute the thrust
        thrust(conditions)

        #getting the network outputs from the thrust outputs
        F            = thrust.outputs.thrust*[1,0,0]
        mdot         = thrust.outputs.fuel_flow_rate
        output_power = thrust.outputs.output_power

        F_vec        = conditions.ones_row(3) * 0.0
        F_vec[:,0]   = F[:,0]
        F            = F_vec

        # Add ISP HERE
        results = Data()
        results.thrust_force_vector = F
        results.vehicle_mass_rate   = mdot

        # store data
        results_conditions = Data
        conditions.propulsion.acoustic_outputs.core = results_conditions(


        return results

    def size(self,state):
        """ Size the Ramjet

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
        thrust     = self.thrust
        thrust.size(conditions)

    def engine_out(self,state):
        """ Lose an engine

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

        temp_throttle = np.zeros(len(state.conditions.propulsion.throttle))

        for i in xrange(0,len(state.conditions.propulsion.throttle)):
            temp_throttle[i] = state.conditions.propulsion.throttle[i]
            state.conditions.propulsion.throttle[i] = 1.0

        results = self.evaluate_thrust(state)

        for i in xrange(0,len(state.conditions.propulsion.throttle)):
            state.conditions.propulsion.throttle[i] = temp_throttle[i]

        results.thrust_force_vector = results.thrust_force_vector/self.number_of_engines*(self.number_of_engines-1)
        results.vehicle_mass_rate   = results.vehicle_mass_rate/self.number_of_engines*(self.number_of_engines-1)

        return results

    __call__ = evaluate_thrust

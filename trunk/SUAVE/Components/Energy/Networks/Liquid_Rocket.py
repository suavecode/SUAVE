## @ingroup Components-Energy-Networks
# Liquid_Rocket.py
#
# Created:  Feb 2018, W. Maier
# Modified: Apr 2021, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Data
from .Network import Network

# ----------------------------------------------------------------------
#  Liquid Rocket Network
# ----------------------------------------------------------------------
## @ingroup Components-Energy-Networks
class Liquid_Rocket(Network):
    """ This sets up the equations for a liquid rocket.

        Assumptions:
        Quasi 1-D flow
        Adiabatic Nozzles
        Throat Always Choked

        Source:
        Chapter 7
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
        self.tag = 'Liquid_Rocket'
        self.number_of_engines = None
        self.engine_length     = None
        self.nacelle_diameter  = None

        # For Drag calculations 
        self.areas             = Data()
        self.areas.wetted      = None       
        self.internal          = True

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
        	conditions.noise.sources.liquid_rocket:
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
        combustor                 = self.combustor
        core_nozzle               = self.core_nozzle
        thrust                    = self.thrust
        number_of_engines         = self.number_of_engines

        # creating the network by manually linking the different components

        # flow through the combustor
        combustor.compute(conditions)

        # link the core nozzle to the low pressure turbine
        core_nozzle.inputs        = combustor.outputs

        # flow through the core nozzle
        core_nozzle.compute(conditions)

        # compute the thrust using the thrust component

        # link the thrust component to the core nozzle
        thrust.inputs                                          = core_nozzle.outputs

        # link the thrust component 
        thrust.inputs.number_of_engines                        = number_of_engines

        # compute the thrust
        thrust(conditions)

        # getting the network outputs from the thrust outputs
        F            = thrust.outputs.thrust*[1,0,0]
        mdot         = thrust.outputs.vehicle_mass_rate
        Isp          = thrust.outputs.specific_impulse
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
        combustor                 = self.combustor
        core_nozzle               = self.core_nozzle
        thrust                    = self.thrust

        #--Creating the network by manually linking the different components--

        # flow through the high pressure compressor
        combustor.compute(conditions)

        # link the core nozzle to the low pressure turbine
        core_nozzle.inputs = combustor.outputs

        # flow through the core nozzle
        core_nozzle(conditions)

        # link the thrust component to the core nozzle
        thrust.inputs      = core_nozzle.outputs

        # compute the thrust
        thrust.size(conditions)

    __call__ = evaluate_thrust
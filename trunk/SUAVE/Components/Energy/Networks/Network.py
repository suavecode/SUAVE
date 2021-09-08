## @ingroup Components-Energy-Networks
# Propulsor.py
# 
# Created:  
# Modified: Feb 2016, T. MacDonald
#           May 2020, E. Botero
#           Jul 2021, E. Botero


# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Components import Physical_Component
from SUAVE.Core import Data

# ----------------------------------------------------------------------
#  Network
# ----------------------------------------------------------------------

## @ingroup Components-Energy-Networks
class Network(Physical_Component):

    """ SUAVE.Components.Energy.Networks.Network()
    
        The Top Level Network Class
            
            Assumptions:
            None
            
            Source:
            N/As
    
    """

    def __defaults__(self):
        
        """ This sets the default attributes for the network.
        
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
        self.tag = 'network'
        self.generative_design_max_per_vehicle = 1
        self.non_dimensional_origin = [[0.0,0.0,0.0]]
        self.number_of_engines = 1.0 
        self.engine_length     = 1.0
        self.wing_mounted      = True
        
        self.areas             = Data()
        self.areas.wetted      = 0.0
        self.areas.maximum     = 0.0
        self.areas.exit        = 0.0
        self.areas.inflow      = 0.0
        
## @ingroup Components-Energy-Networks
class Container(Physical_Component.Container):
    """ SUAVE.Components.Energy.Networks.Network.Container()
        
        The Network Container Class
    
            Assumptions:
            None
            
            Source:
            N/A
    
    """
    def get_children(self):
        """ Returns the components that can go inside
        
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
        import SUAVE.Components.Energy.Networks as Nw
                
        return [Nw.Turbofan,Nw.Turbojet_Super]

                

    
    def evaluate_thrust(self,state):
        """ This is used to evaluate the thrust produced by the network.
        
                Assumptions:
                Network has "evaluate_thrust" method
                If multiple networks are attached their masses will be summed
                
                Source:
                N/A
                
                Inputs:
                State variables
                
                Outputs:
                Results of the "evaluate_thrust" method
                
                Properties Used:
                N/A
        """
        
        ones_row = state.ones_row
        
        results = Data()
        results.thrust_force_vector = 0.*ones_row(3)
        results.vehicle_mass_rate   = 0.*ones_row(1)

        for net in self.values():
            results_p = net.evaluate_thrust(state) 
            
            for key in results.keys():
                results[key] += results_p[key]

        return results

# ----------------------------------------------------------------------
#  Handle Linking
# ----------------------------------------------------------------------

Network.Container = Container

## @ingroup Components-Propulsors
# Propulsor.py
# 
# Created:  
# Modified: Feb 2016, T. MacDonald
#           May 2020, E. Botero


# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Components import Physical_Component
from SUAVE.Core import Data

# ----------------------------------------------------------------------
#  Propulsor
# ----------------------------------------------------------------------

## @ingroup Components-Propulsors
class Propulsor(Physical_Component):

    """ SUAVE.Components.Propulsor()
    
        The Top Level Propulsor Class
            
            Assumptions:
            None
            
            Source:
            N/As
    
    """

    def __defaults__(self):
        
        """ This sets the default attributes for the propulsor.
        
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
        self.tag = 'Propulsor'
        self.generative_design_max_per_vehicle = 1
        self.non_dimensional_origin = [[0.0,0.0,0.0]]
        self.number_of_engines = 1.0
        self.nacelle_diameter  = 1.0
        self.engine_length     = 1.0
        self.wing_mounted      = True
        
        self.areas             = Data()
        self.areas.wetted      = 0.0
        self.areas.maximum     = 0.0
        self.areas.exit        = 0.0
        self.areas.inflow      = 0.0
        
## @ingroup Components-Propulsors
class Container(Physical_Component.Container):
    """ SUAVE.Components.Propulsor.Container()
        
        The Propulsor Container Class
    
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
        
        #return [Nw.Battery_Propeller,Nw.Battery_Ducted_Fan,Nw.Lift_Forward_Propulsor,Nw.Ramjet,Nw.Solar, \
                #Nw.Turbofan,Nw.Turbojet_Super]
                
        return [Nw.Turbofan,Nw.Turbojet_Super]

                

    
    def evaluate_thrust(self,state):
        """ This is used to evaluate the thrust produced by the propulsor.
        
                Assumptions:
                Propulsor has "evaluate_thrust" method
                If multiple propulsors are attached their masses will be summed
                
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

        for propulsor in self.values():
            results_p = propulsor.evaluate_thrust(state) 
            
            for key in results.keys():
                results[key] += results_p[key]
            
            
            
        return results

# ----------------------------------------------------------------------
#  Handle Linking
# ----------------------------------------------------------------------

Propulsor.Container = Container

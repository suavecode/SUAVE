# Propulsor.py
# 
# Created:  
# Modified: Feb 2016, T. MacDonald

""" Propulsor.py: parent class for propulsion systems """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Components import Physical_Component

# ----------------------------------------------------------------------
#  Propulsor
# ----------------------------------------------------------------------

class Propulsor(Physical_Component):

    """ A component that makes go-ification """

    def __defaults__(self):
        self.tag = 'Propulsor'
        
class Container(Physical_Component.Container):
    """ Contains many SUAVE.Components.Propulsor()
    
    """
    
    def evaluate_thrust(self,state):

        for propulsor in self.values():
            results = propulsor.evaluate_thrust(state) 
            
        return results

# ----------------------------------------------------------------------
#  Handle Linking
# ----------------------------------------------------------------------

Propulsor.Container = Container
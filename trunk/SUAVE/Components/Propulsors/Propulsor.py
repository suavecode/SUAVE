""" Propulsor.py: parent class for propulsion systems """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
from SUAVE.Core import Data, Data_Exception, Data_Warning
from SUAVE.Components import Component, Physical_Component, Lofted_Body
from SUAVE.Attributes.Gases import Air

# ----------------------------------------------------------------------
#  Propulsor
# ----------------------------------------------------------------------

class Propulsor(Physical_Component):

    """ A component that makes go-ification """

    def __defaults__(self):
        self.tag = 'Propulsor'


class Container(Physical_Component.Container):
    """ Contains many SUAVE.Components.Propulsor()
    
        Search Methods
            import SUAVE.Components.Propulsors
            example: find_instances(Propulsors.Motor)    > return all Motors
            example: find_instances(Propulsors.Turbojet) > return all Turbojets
    """
    
    def evaluate_thrust(self,state):

        for propulsor in self.values():
            results = propulsor.evaluate_thrust(state) 
            
        return results

# ----------------------------------------------------------------------
#  Handle Linking
# ----------------------------------------------------------------------

Propulsor.Container = Container
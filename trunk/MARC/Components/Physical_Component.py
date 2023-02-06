## @ingroup Components
# Physical_Component.py
# 
# Created:  
# Modified: Feb 2016, T. MacDonald
#           May 2020, E. Botero


# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from .Component import Component
from .Mass_Properties import Mass_Properties

import numpy as np

# ----------------------------------------------------------------------
#  Physical Component
# ----------------------------------------------------------------------

## @ingroup Components
class Physical_Component(Component):
    """ A component that has a Mass_Properties Data
        
        Assumptions:
        None
        
        Source:
        None
    """
    def __defaults__(self):
        """This sets the default values.
    
            Assumptions:
            None
    
            Source:
            N/A
    
            Inputs:
            None
    
            Outputs:
            None
    
            Properties Used:
            None
        """         
        self.tag = 'Component'
        self.mass_properties = Mass_Properties()
        self.origin = np.array([[0.0,0.0,0.0]])
        self.symmetric = False

## @ingroup Components    
class Container(Component.Container):
    """ A container of physical components
        
        Assumptions:
        None
        
        Source:
        None
    """ 
    def sum_mass(self):
        """ will recursively search the data tree and sum
            any Comp.Mass_Properties.mass, and return the total sum
            
            Assumptions:
            None
    
            Source:
            N/A
    
            Inputs:
            None
    
            Outputs:
            mass  [kg]
    
            Properties Used:
            None
        """   
        total = 0.0
        for key,Comp in self.items():
            if isinstance(Comp,Physical_Component.Container):
                total += Comp.sum_mass() # recursive!
            elif isinstance(Comp,Physical_Component):
                total += Comp.mass_properties.mass
                
        return total
    
    def total_moment(self):
        """ will recursively search the data tree and sum
            any Comp.Mass_Properties.mass, and return the total sum of moments
            
            Assumptions:
            None
    
            Source:
            N/A
    
            Inputs:
            None
    
            Outputs:
            total moment [kg*m]
    
            Properties Used:
            None
        """   
        total = np.array([[0.0,0.0,0.0]])
        for key,Comp in self.items():
            if isinstance(Comp,Physical_Component.Container):
                total += Comp.total_moment() # recursive!
            elif isinstance(Comp,Physical_Component):
                total += Comp.mass_properties.mass*(np.sum(np.array(Comp.origin),axis=0)/len(Comp.origin)+Comp.mass_properties.center_of_gravity)

        return total
    
    
# ------------------------------------------------------------
#  Handle Linking
# ------------------------------------------------------------

Physical_Component.Container = Container
## @ingroup Components
# Physical_Component.py
# 
# Created:  
# Modified: Feb 2016, T. MacDonald

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from .Component import Component
from .Mass_Properties import Mass_Properties


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
        self.origin  = [[0.0,0.0,0.0]]
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
            None
    
            Properties Used:
            None
        """   
        total = 0.0
        for key,Comp in self.items():
            if isinstance(Comp,PhysicalComponentContainer):
                total += Comp.sum_mass() # recursive!
            elif isinstance(Comp,Physical_Component):
                total += Comp.mass_properties.mass
                
        return total
    
    
# ------------------------------------------------------------
#  Handle Linking
# ------------------------------------------------------------

Physical_Component.Container = Container
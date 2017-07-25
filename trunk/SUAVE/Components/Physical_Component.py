# Physical_Component.py
# 
# Created:  
# Modified: Feb 2016, T. MacDonald

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from Component import Component
from Mass_Properties import Mass_Properties


# ----------------------------------------------------------------------
#  Physical Component
# ----------------------------------------------------------------------

class Physical_Component(Component):
    """ SUAVE.Components.Physical_Component()
        a component that has a Mass_Properties Data
    """
    def __defaults__(self):
        self.tag = 'Component'
        self.mass_properties = Mass_Properties()
        self.origin  = [[0.0,0.0,0.0]]
        self.symmetric = False
    
class Container(Component.Container):
    """ SUAVE.Components.Physical_Component.Container()
        a container of physical components
        
        Methods:
            sum_mass(): will recursively search the data tree and sum
                        any Comp.Mass_Properties.mass, and return the total sum
    """    
    def sum_mass(self):
        """ an example of how to recursivly sum the mass of 
            a tree of physical components
        """
        total = 0.0
        for key,Comp in self.iteritems():
            if isinstance(Comp,PhysicalComponentContainer):
                total += Comp.sum_mass() # recursive!
            elif isinstance(Comp,Physical_Component):
                total += Comp.mass_properties.mass
                
        return total
    
    
# ------------------------------------------------------------
#  Handle Linking
# ------------------------------------------------------------

Physical_Component.Container = Container
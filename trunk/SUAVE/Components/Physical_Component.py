

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Structure import Data, Data_Exception, Data_Warning
from Component import Component
from Mass_Props import Mass_Props


# ----------------------------------------------------------------------
#  Physical Component
# ----------------------------------------------------------------------

class Physical_Component(Component):
    """ SUAVE.Components.Physical_Component()
        a component that has a Mass_Props Data
    """
    def __defaults__(self):
        self.tag = 'Component'
        self.Mass_Props = Mass_Props()
        self.position  = [0.0,0.0,0.0]
        self.symmetric = False
        self.joints    = None
        self.leafs     = None
    
class Container(Component.Container):
    """ SUAVE.Components.Physical_Component.Container()
        a container of physical components
        
        Methods:
            sum_mass(): will recursively search the data tree and sum
                        any Comp.Mass_Props.mass, and return the total sum
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
                total += Comp.Mass_Props.mass
        return total
    
    
# ------------------------------------------------------------
#  Handle Linking
# ------------------------------------------------------------

Physical_Component.Container = Container


# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Structure import Data, Data_Exception, Data_Warning


# ----------------------------------------------------------------------
#  Mass Properties
# ----------------------------------------------------------------------

class Mass_Properties(Data):
    """ SUAVE.Components.Mass_Properties()
        mass properties for a physical component
    """
    def __defaults__(self):
        
        self.mass   = 0.0
        self.volume = 0.0
        self.center_of_gravity = [0.0,0.0,0.0]
        
        self.moments_of_inertia = Data()
        self.moments_of_inertia.center = [0.0,0.0,0.0]
        self.moments_of_inertia.tensor   = [[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]]
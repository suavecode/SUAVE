

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Structure import Data, Data_Exception, Data_Warning


# ----------------------------------------------------------------------
#  Mass Properties
# ----------------------------------------------------------------------

class Mass_Properties(Data):
    """ SUAVE.Components.Mass_Props()
        mass properties for a physical component
    """
    def __defaults__(self):
        
        self.mass   = 0.0
        self.volume = 0.0
        self.center_of_gravity = [0.0,0.0,0.0]
        
        self.Moments_Of_Inertia = Data()
        self.Moments_Of_Inertia.center = [0.0,0.0,0.0]
        self.Moments_Of_Inertia.tensor   = [[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]]
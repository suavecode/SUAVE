

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Structure import Data, Data_Exception, Data_Warning


# ----------------------------------------------------------------------
#  Mass Properties
# ----------------------------------------------------------------------

class Mass_Props(Data):
    """ SUAVE.Components.Mass_Props()
        mass properties for a physical component
    """
    def __defaults__(self):
        self.mass   = 0.0
        self.volume = 0.0
        self.pos_cg = [0.0,0.0,0.0]
        self.I_cg   = [[0.0,0.0,0.0],
                       [0.0,0.0,0.0],
                       [0.0,0.0,0.0]]
        
    

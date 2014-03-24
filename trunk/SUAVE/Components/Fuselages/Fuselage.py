

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Structure import Data, Data_Exception, Data_Warning
from SUAVE.Components import Component, Physical_Component, Lofted_Body


# ------------------------------------------------------------
#  Fuselage
# ------------------------------------------------------------

class Fuselage(Lofted_Body):
    def __defaults__(self):
        self.tag = 'Fuselage'
        self.length      = 0.0
        self.aero_center = [0.0,0.0,0.0]
        self.Sections    = Lofted_Body.Section.Container()
        self.Segments    = Lofted_Body.Segment.Container()
        self.length_cabin = 0.0
            
        
class Container(Physical_Component.Container):
    pass


# ------------------------------------------------------------
#  Handle Linking
# ------------------------------------------------------------

Fuselage.Container = Container

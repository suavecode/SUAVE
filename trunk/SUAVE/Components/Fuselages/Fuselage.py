

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
        self.aerodynamic_center = [0.0,0.0,0.0]
        self.Sections    = Lofted_Body.Section.Container()
        self.Segments    = Lofted_Body.Segment.Container()
        
        self.number_coach_seats = 0.0
        self.seats_abreast = 0.0
        self.seat_pitch = 1.0
        
        self.Areas = Data()
        self.Areas.front_projected = 0.0
        self.Areas.side_projected = 0.0
        self.Areas.wetted = 0.0
        
        self.effective_diameter = 0.0
        self.width = 0.0
        
        self.Heights = Data()
        self.Heights.maximum = 0.0
        self.Heights.at_quarter_length = 0.0
        self.Heights.at_three_quarters_length = 0.0
        self.Heights.at_wing_root_quarter_chord = 0.0
        
        self.Lengths = Data()
        self.Lengths.nose = 0.0
        self.Lengths.tail = 0.0
        self.Lengths.total = 0.0
        self.Lengths.cabin = 0.0
        self.Lengths.fore_space = 0.0
        self.Lengths.aft_space = 0.0
            
        self.Fineness = Data()
        self.Fineness.nose = 0.0
        self.Fineness.tail = 0.0
        
        self.differential_pressure = 0.0
            
        
class Container(Physical_Component.Container):
    pass


# ------------------------------------------------------------
#  Handle Linking
# ------------------------------------------------------------

Fuselage.Container = Container

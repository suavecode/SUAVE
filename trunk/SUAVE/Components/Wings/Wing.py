

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Structure import Data, Data_Exception, Data_Warning
from SUAVE.Components import Component, Physical_Component, Lofted_Body, Mass_Properties
from Airfoils import Airfoil

# ------------------------------------------------------------
#   Wing
# ------------------------------------------------------------

class Wing(Lofted_Body):
    def __defaults__(self):
        self.tag = 'Wing'
        self.Mass_Properties = Mass_Properties()
        self.position  = [0.0,0.0,0.0]
        self.symmetric = False

        self.sweep        = 0.0
        self.taper        = 0.0
        self.dihedral     = 0.0
        self.aspect_ratio = 0.0
        self.thickness_to_chord = 0.0
        self.span_efficiency = 0.9
        self.aerodynamic_center = [0.0,0.0,0.0]
        
        self.Spans = Data()
        self.Spans.projected = 0.0
        
        self.Areas = Data()
        self.Areas.reference = 0.0
        self.Areas.exposed = 0.0
        self.Areas.affected = 0.0
        self.Areas.wetted = 0.0
        
        self.Chords = Data()
        self.Chords.mean_aerodynamic = 0.0
        self.Chords.mean_geometric = 0.0
        self.Chords.root = 0.0
        self.Chords.tip = 0.0
        
        self.Twists = Data()
        self.Twists.root = 0.0
        self.Twists.tip = 0.0
        
        self.Flaps = Data()
        self.Flaps.chord = 0.0
        self.Flaps.angle = 0.0
        self.Flaps.span_start = 0.0
        self.Flaps.span_end = 0.0
        self.Flaps.type = None
        
        self.Slats = Data()
        self.Slats.chord = 0.0
        self.Slats.angle = 0.0
        self.Slats.span_start = 0.0
        self.Slats.span_end = 0.0
        self.Slats.type = None        
        
        self.high_lift     = False
        self.high_mach     = False
        self.vortex_lift   = False
        self.transistion_x=0.0     # Normalized 0 to 1
        
    def append_segment(self,segment):
        """ adds a segment to the wing """

        # assert database type
        if not isinstance(segment,Data):
            raise Component_Exception, 'input component must be of type Data()'

        # store data
        self.Segments.append(segment)
       
        return


class Container(Component.Container):
    pass


# ------------------------------------------------------------
#  Wing Sections
# ------------------------------------------------------------

class Section(Lofted_Body.Section):
    pass

class Airfoil(Airfoil):
    pass

class SectionContainer(Lofted_Body.Section.Container):
    pass

# ------------------------------------------------------------
#  Wing Segments
# ------------------------------------------------------------

class Segment(Lofted_Body.Segment):
    def __defaults__(self):
        self.tag = 'Section'
        self.AR        = 0.0
        self.TR        = 0.0
        self.area      = 0.0
        self.span      = 0.0
        self.TC        = 0.0
        self.RC        = 0.0
        self.sweep     = 0.0
        self.sweep_loc = 0.0
        self.twist     = 0.0
        self.twist_loc = 0.0


class SegmentContainer(Lofted_Body.Segment.Container):
    pass

# ------------------------------------------------------------
#  Handle Linking
# ------------------------------------------------------------
Wing.Container = Container
Section.Container  = SectionContainer #propogates to Airfoil
Segment.Container  = SegmentContainer
Wing.Section = Section
Wing.Airfoil = Airfoil
Wing.Segment = Segment




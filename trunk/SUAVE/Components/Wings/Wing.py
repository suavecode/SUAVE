

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Structure import Data, Data_Exception, Data_Warning
from SUAVE.Components import Component, Physical_Component, Lofted_Body
from Airfoils import Airfoil

# ------------------------------------------------------------
#   Wing
# ------------------------------------------------------------

class Wing(Lofted_Body):
    def __defaults__(self):
        self.tag = 'Wing'
        self.symmetric      = True
        self.sweep          = 0.0
        self.taper          = 0.0
        self.dihedral       = 0.0
        self.span           = 0.0
        self.aspect_ratio   = 0.0
        self.aero_center    = [0.0,0.0,0.0]
        self.total_area     = 0.0
        self.total_span     = 0.0
        self.avg_chord      = 0.0
        self.mac_chord      = 0.0
        self.ref            = Data()
        self.ref.A          = 0.0
        self.ref.span       = 0.0
        self.ref.chord      = 0.0
        self.t_c            = 0.0
        self.totals         = Data()
        self.totals.area    = 0.0
        self.totals.span    = 0.0
        self.totals.proj_span = 0.0
        self.Sections       = SectionContainer()
        self.Segments       = SegmentContainer()
        self.highlift       = False
        self.flaps_chord    = 0.0
        self.flaps_angle    = 0.0
        self.slats_angle    = 0.0
        self.hl             = 0
        self.flap_type      = 'none'
        self.S_affected     = 0.
        self.vertical       = False
        self.transition_x_u = 0.0
        self.transition_x_l = 0.0
          
        
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




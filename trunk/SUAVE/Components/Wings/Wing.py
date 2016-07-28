# Wing.py
# 
# Created:  
# Modified: Feb 2016, T. MacDonald

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Data
from SUAVE.Components import Component, Lofted_Body, Mass_Properties
from Airfoils import Airfoil

# ------------------------------------------------------------
#   Wing
# ------------------------------------------------------------

class Wing(Lofted_Body):
    def __defaults__(self):

        self.tag             = 'wing'
        self.mass_properties = Mass_Properties()
        self.position        = [0.0,0.0,0.0]

        self.symmetric                 = True
        self.vertical                  = False
        self.t_tail                    = False
        self.taper                     = 0.0
        self.dihedral                  = 0.0
        self.aspect_ratio              = 0.0
        self.thickness_to_chord        = 0.0
        self.span_efficiency           = 0.9
        self.aerodynamic_center        = [0.0,0.0,0.0]
        self.exposed_root_chord_offset = 0.0

        self.spans = Data()
        self.spans.projected = 0.0
        

        self.areas = Data()
        self.areas.reference = 0.0
        self.areas.exposed   = 0.0
        self.areas.affected  = 0.0
        self.areas.wetted    = 0.0

        self.chords = Data()
        self.chords.mean_aerodynamic = 0.0
        self.chords.mean_geometric   = 0.0
        self.chords.root             = 0.0
        self.chords.tip              = 0.0
        
        self.sweeps               = Data()
        self.sweeps.quarter_chord = 0.0
        self.sweeps.leading_edge  = 0.0
        self.sweeps.half_chord    = 0.0        

        self.twists = Data()
        self.twists.root = 0.0
        self.twists.tip  = 0.0

        self.control_surfaces = Data()

        self.flaps = Data()
        self.flaps.chord      = 0.0
        self.flaps.angle      = 0.0
        self.flaps.span_start = 0.0
        self.flaps.span_end   = 0.0
        self.flaps.type       = None
        self.flaps.area       = 0.0

        self.slats = Data()
        self.slats.chord      = 0.0
        self.slats.angle      = 0.0
        self.slats.span_start = 0.0
        self.slats.span_end   = 0.0
        self.slats.type       = None

        self.high_lift     = False
        self.high_mach     = False
        self.vortex_lift   = False

        self.transition_x_upper = 0.0
        self.transition_x_lower = 0.0


    def append_segment(self,segment):
        """ adds a segment to the wing """

        # assert database type
        if not isinstance(segment,Data):
            raise Exception, 'input component must be of type Data()'

        # store data
        self.Segments.append(segment)

        return


    def append_control_surface(self,control_surface):
        """ adds a component to vehicle """

        # assert database type
        if not isinstance(control_surface,Data):
            raise Exception, 'input control surface must be of type Data()'

        # store data
        self.control_surfaces.append(control_surface)

        return


class Container(Component.Container):
    pass


# ------------------------------------------------------------
#  Wing Sections
# ------------------------------------------------------------

class Section(Lofted_Body.Section):
    def __defaults__(self):
        self.tag                   = 'section'
        self.twist                 = 0.0
        self.chord                 = 0.0
        self.origin                = [0.0,0.0,0.0]
        self.transformation_matrix = [[1,0,0],[0,1,0],[0,0,1]]
        self.airfoil = None


class SectionContainer(Lofted_Body.Section.Container):
    pass


# ------------------------------------------------------------
#  Wing Segments
# ------------------------------------------------------------

class Segment(Lofted_Body.Segment):
    def __defaults__(self):
        self.tag = 'segment'
        self.aspect_ratio = 0.0
        self.taper_ratio  = 0.0
        self.area         = 0.0
        self.span         = 0.0
        self.tip_chord    = 0.0
        self.root_chord   = 0.0
        self.sweep        = 0.0
        self.sweep_loc    = 0.0
        self.twist        = 0.0
        self.twist_loc    = 0.0


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





# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np

from SUAVE.Structure import Data, Data_Exception, Data_Warning
from SUAVE.Components import Component, Physical_Component, Lofted_Body, Mass_Properties
from Airfoils import Airfoil

# ------------------------------------------------------------
#   Wing
# ------------------------------------------------------------

class Wing(Lofted_Body):
    def __defaults__(self):

        self.tag = 'Wing'
        self.mass_properties = Mass_Properties()
        self.position  = [0.0,0.0,0.0]

        self.symmetric = True
        self.vertical  = False
        self.t_tail    = False
        self.sweep        = 0.0
        self.taper        = 0.0
        self.dihedral     = 0.0
        self.aspect_ratio = 0.0
        self.thickness_to_chord = 0.0
        self.span_efficiency = 0.9
        self.aerodynamic_center = [0.0,0.0,0.0]
        self.exposed_root_chord_offset = 0.0

        self.spans = Data()
        self.spans.projected = 0.0

        self.areas = Data()
        self.areas.reference = 0.0
        self.areas.exposed = 0.0
        self.areas.affected = 0.0
        self.areas.wetted = 0.0

        self.chords = Data()
        self.chords.mean_aerodynamic = 0.0
        self.chords.mean_geometric = 0.0
        self.chords.root = 0.0
        self.chords.tip = 0.0

        self.twists = Data()
        self.twists.root = 0.0
        self.twists.tip = 0.0

        self.control_surfaces = Data()

        self.flaps = Data()
        self.flaps.chord = 0.0
        self.flaps.angle = 0.0
        self.flaps.span_start = 0.0
        self.flaps.span_end = 0.0
        self.flaps.type = None

        self.slats = Data()
        self.slats.chord = 0.0
        self.slats.angle = 0.0
        self.slats.span_start = 0.0
        self.slats.span_end = 0.0
        self.slats.type = None

        self.high_lift     = False
        self.high_mach     = False
        self.vortex_lift   = False

        self.transition_x_upper = 0.0
        self.transition_x_lower = 0.0


    def append_segment(self,segment):
        """ adds a segment to the wing """

        # assert database type
        if not isinstance(segment,Data):
            raise Component_Exception, 'input component must be of type Data()'

        # store data
        self.Segments.append(segment)

        return


    def append_control_surface(self,control_surface):
        """ adds a component to vehicle """

        # assert database type
        if not isinstance(control_surface,Data):
            raise Component_Exception, 'input control surface must be of type Data()'

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
        self.tag    = 'Section'
        self.twist  = 0.0
        self.chord  = 0.0
        self.origin = [0.0,0.0,0.0]
        self.transformation_matrix = [[1,0,0],[0,1,0],[0,0,1]]
        self.airfoil = None
    	

class SectionContainer(Lofted_Body.Section.Container):
    pass


# ------------------------------------------------------------
#  Wing Segments
# ------------------------------------------------------------

class Segment(Lofted_Body.Segment):
    def __defaults__(self):
        self.tag = 'Segment'
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
#  Control Surfaces
# ------------------------------------------------------------

class Control_Surface(Lofted_Body):
	def __defaults__(self):
		self.tag    = 'Control Surface'
		self.span   = 0.0
		self.span_fraction = 0.0
		self.deflection_symmetry = 1.0
		self.origin = [0.0,0.0,0.0]
		self.transformation_matrix = [[1,0,0],[0,1,0],[0,0,1]]
		
		self.sections = Data()
		
		
	# Are the reference points for span/chord fractions and twists correct??
	def populate(self,span_fractions,chord_fractions,relative_twists,wing):
		"""
		Creates Control_Surface_Sections defining a control surface such that:
			-There are as many Control_Surface_Sections as the length of
			span_fractions
			-The i-th Control_Surface_Section has the local chord fraction
			given in chord_fractions[i], the twist given in twists[i], and is
			at the spanwise position, span_fractions[i]
			-The control surface origin is defined based on the geometry of the
			wing (i.e., dimensional projected span, LE sweep, root chord, taper)
			
		Preconditions:
			-span_fractions has as many sections as will be used to define the
			control surface (at least two)
			-relative_twists has the same length as span_fractions and
			indicates the twist in the local chord frame of reference - twist
			is zero if the control surface chord line is parallel to the local
			chord line in the undeflected state.
			-span_fractions and chord_fractions have the same length and
			indicate the y- and x-coordinates of the section leading edge as
			fractions of wingspan and local chord, repsectively
			
		Postconditions:
			-Control_Surface.sections contains len(span_fractions)
			Control_Surface_Sections with size and position parameters filled
			
		Assumes a trailing-edge control surface
		"""
		
		if len(span_fractions) < 2:
			raise ValueError('Two or more sections required for control surface definition')
			
		sw   = wing.sweep
		di   = wing.dihedral
		span = wing.spans.projected
		c_r  = wing.chords.root
		tpr  = wing.taper
		orig = wing.origin
		
		inboard = Control_Surface_Section()
		inboard.tag = 'Inboard_Section'
		inboard.origins.span_fraction  = span_fractions[0]
		inboard.origins.chord_fraction = 1. - chord_fractions[0]
		local_chord = c_r * (1 + 2. * span_fractions[0] * (tpr - 1))
		inboard.origins.dimensional[0] = orig[0] + span*span_fractions[0]*np.tan(sw) + local_chord*inboard.origins.chord_fraction
		inboard.origins.dimensional[1] = orig[1] + span*span_fractions[0]
		inboard.origins.dimensional[2] = orig[2] + span*span_fractions[0]*np.tan(di)
		inboard.chord_fraction = chord_fractions[0]
		inboard.twist = relative_twists[0]
		self.append_section(inboard)
		
		outboard = Control_Surface_Section()
		outboard.tag = 'Outboard_Section'
		outboard.origins.span_fraction  = span_fractions[-1]
		outboard.origins.chord_fraction = 1. - chord_fractions[-1]
		local_chord = c_r * (1 + 2. * span_fractions[-1] * (tpr - 1))
		inboard.origins.dimensional[0] = orig[0] + span*span_fractions[-1]*np.tan(sw) + local_chord*inboard.origins.chord_fraction
		inboard.origins.dimensional[1] = orig[1] + span*span_fractions[-1]
		inboard.origins.dimensional[2] = orig[2] + span*span_fractions[-1]*np.tan(di)
		outboard.chord_fraction = chord_fractions[-1]
		outboard.twist = relative_twists[-1]
		self.append_section(outboard)
		
		self.span_fraction = abs(span_fractions[-1] - span_fractions[0])
		
		if len(span_fractions) > 2:
			i = 1
			while i+1 < len(span_fractions):
				section = Control_Surface_Section()
				section.tag = ('Control_Section{}'.format(i))
				section.origins.span_fraction  = span_fractions[i]
				section.origins.chord_fraction = 1. - chord_fractions[i]
				local_chord = c_r * (1 + 2. * span_fractions[i] * (tpr - 1))
				inboard.origins.dimensional[0] = orig[0] + span*span_fractions[i]*np.tan(sw) + local_chord*inboard.origins.chord_fraction
				inboard.origins.dimensional[1] = orig[1] + span*span_fractions[i]
				inboard.origins.dimensional[2] = orig[2] + span*span_fractions[i]*np.tan(di)
				section.chord_fraction = chord_fractions[i]
				section.twist = relative_twists[i]
				self.append_section(section)
				i += 1
				
		return self
					

	def append_section(self,section):
		"""adds a component to vehicle """

		# assert database type
		if not isinstance(section,Data):
			raise Component_Exception, 'input control surface section must be of type Data()'
		
		# store data
		self.sections.append(section)

		return
		
		

class Control_Surface_Section(Lofted_Body.Section):
    def __defaults__(self):
        self.tag    = 'Control Section'
        self.chord  = 0.0
        self.chord_fraction = 0.0
        self.twist  = 0.0 # Offset / deflection in neutral position

        self.origins = Data()
        self.origins.dimensional    = [0.0,0.0,0.0]
        self.origins.span_fraction  = 0.0
        self.origins.chord_fraction = 0.0

        self.points = Data()
        self.points.leading_edge =  0.0
        self.points.trailing_edge = 0.0


# ------------------------------------------------------------
#  Handle Linking
# ------------------------------------------------------------
Wing.Container = Container
Section.Container  = SectionContainer #propogates to Airfoil
Segment.Container  = SegmentContainer
Wing.Section = Section
Wing.Airfoil = Airfoil
Wing.Segment = Segment




# Wing.py
# 
# Created:  Oct 2014, T. Momose
# Modified: Jan 2016, E. Botero
# Modified: Arp 2017, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Data

# ------------------------------------------------------------
#   Wing
# ------------------------------------------------------------

class Wing(Data):
	def __defaults__(self):

		self.tag = 'wing'
		self.symmetric = True
		self.vertical  = False
		self.origin    = [0.,0.,0.]

		self.sweep        = 0.0
		self.dihedral     = 0.0

		self.sections = Data()
		self.configuration = Data()
		self.control_surfaces = Data()

		self.configuration.nspanwise = 10
		self.configuration.nchordwise = 5
		self.configuration.sspace = 1.0
		self.configuration.cspace = 1.0


	def append_section(self,section):
		""" adds a segment to the wing """

		# assert database type
		if not isinstance(section,Data):
			raise Exception, 'input component must be of type Data()'

		# store data
		self.sections.append(section)
		return


# ------------------------------------------------------------
#  AVL Wing Sections
# ------------------------------------------------------------

class Section(Data):
	def __defaults__(self):
		self.tag    = 'section'
		self.origin = [0.0,0.0,0.0]
		self.chord  = 0.0
		self.twist  = 0.0
		self.airfoil_coord_file = None
		self.control_surfaces = Data()
		
				
	def append_control_surface(self,control):
		""" adds a control_surface to the wing section """

		# assert database type
		if not isinstance(control,Data):
			raise Exception, 'input component must be of type Data()'

		# store data
		self.control_surfaces.append(control)
		return


# ------------------------------------------------------------
#  AVL Control Surface
# ------------------------------------------------------------

class Control_Surface(Data):
	def __defaults__(self):
		self.tag            = 'control_surface'
		self.gain           = 0.0
		self.x_hinge        = 0.0
		self.hinge_vector   = '0. 0. 0.'
		self.sign_duplicate = '+1'	# sign_duplicate: 1.0 or -1.0 - the sign of
						# the duplicate control on the mirror wing.
						# Use 1.0 for a mirrored control surface,
						# like an elevator. Use -1.0 for an aileron.


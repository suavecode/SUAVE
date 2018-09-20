## @ingroup Methods-Aerodynamics-AVL-Data
#Wing.py
# 
# Created:  Oct 2014, T. Momose
# Modified: Jan 2016, E. Botero
#           Jul 2017, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Data

# ------------------------------------------------------------
#   Wing
# ------------------------------------------------------------

## @ingroup Methods-Aerodynamics-AVL-Data
class Wing(Data):
	""" A class that defines parameters of the AVL aircraft wing

	Assumptions:
	    None
    
	Source:
	    None
    
	Inputs:
	    None
    
	Outputs:
	    None
    
	Properties Used:
	    N/A
	"""    
	
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
			raise Exception('input component must be of type Data()')

		# store data
		self.sections.append(section)
		return


# ------------------------------------------------------------
#  AVL Wing Sections
# ------------------------------------------------------------

class Section(Data):
	""" A class that defines the sections of the aircraft wing in AVL.
	Each section can be thought of as a trapezoid

	Assumptions:
	    None
    
	Source:
	    None
    
	Inputs:
	    None
    
	Outputs:
	    None
    
	Properties Used:
	    N/A
	"""   	
	def __defaults__(self):
		""" Sets the defaunts of the aircraft wing geometry 
	
		Assumptions:
		    None
	
		Source:
		    None
	
		Inputs:
		    None
	
		Outputs:
		    None
	
		Properties Used:
		    N/A
		"""   		
		self.tag    = 'section'
		self.origin = [0.0,0.0,0.0]
		self.chord  = 0.0
		self.twist  = 0.0
		self.airfoil_coord_file = None
		self.control_surfaces = Data()
		
				
	def append_control_surface(self,control):
		""" Adds a control_surface to the wing section in AVL
	
		Assumptions:
		    None
	
		Source:
		    None
	
		Inputs:
		    None
	
		Outputs:
		    None
	
		Properties Used:
		    N/A
		"""   		

		# assert database type
		if not isinstance(control,Data):
			raise Exception('input component must be of type Data()')

		# store data
		self.control_surfaces.append(control)
		return


# ------------------------------------------------------------
#  AVL Control Surface
# ------------------------------------------------------------

class Control_Surface(Data):
	""" A class that defines the control surface geometry and deflection
	on the aircraft wing in AVL

	Assumptions:
	    None
    
	Source:
	    None
    
	Inputs:
	    None
    
	Outputs:
	    None
    
	Properties Used:
	    N/A
	"""   	
	def __defaults__(self):
		""" Sets the defaults of the control surface on the aircraft wing
		in AVL
	
		Assumptions:
		    None
	
		Source:
		    None
	
		Inputs:
		    None
	
		Outputs:
		    None
	
		Properties Used:
		    N/A
		"""   		
		self.tag            = 'control_surface'
		self.gain           = 0.0
		self.x_hinge        = 0.0
		self.hinge_vector   = '0. 0. 0.'
		self.sign_duplicate = '+1'	# sign_duplicate: 1.0 or -1.0 - the sign of
						# the duplicate control on the mirror wing.
						# Use 1.0 for a mirrored control surface,
						# like an elevator. Use -1.0 for an aileron.


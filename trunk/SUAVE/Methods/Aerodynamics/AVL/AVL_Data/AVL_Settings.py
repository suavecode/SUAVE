# Tim Momose, December 2014

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import SUAVE
from SUAVE.Structure import Data
from AVL_Cases import AVL_Cases

# ------------------------------------------------------------
#   Configuration
# ------------------------------------------------------------

class AVL_Settings(Data):

	def __defaults__(self):
		
		self.run_cases = AVL_Cases()
		self.filenames = Data()
		self.flow_symmetry = Data()
		self.discretization = Data()
		
		self.num_control_surfaces = 0
		
		self.discretization.defaults = Data()
		self.discretization.surfaces = Data()
		self.discretization.defaults.wing = AVL_Discretization_Settings()
		self.discretization.defaults.fuselage = AVL_Discretization_Settings()
		self.discretization.defaults.fuselage.spanwise_spacing = 3
		self.discretization.defaults.fuselage.spanwise_spacing_scheme = 'equal'
		
		self.filenames.avl_bin_name    = 'avl' # to call avl from command line. If avl is not on the system path, include absolute path to the avl binary
		self.filenames.run_folder      = SUAVE.__path__[0] + '/temporary_files/'
		self.filenames.features        = 'aircraft.avl'
		self.filenames.cases           = 'avl_cases.case'
		self.filenames.input_deck      = 'avl_commands.run'
		self.filenames.output_template = 'avl_results{}.txt'
		self.filenames.log_filename    = 'avl_log.txt'
		#self.filenames.results         = []
		
		#------------------------------------------
		# 1: Symmetry about the plane
		# -1: Antizymmetry (Cp constant on plane)
		# 0: Symmetry not guaranteed
		#------------------------------------------
		self.flow_symmetry.xz_plane = 0	# Symmetry across the xz-plane, y=0
		self.flow_symmetry.xy_parallel = 0 # Symmetry across the z=z_symmetry_plane plane
		self.flow_symmetry.z_symmetry_plane = 0.0
		


# ------------------------------------------------------------
#  AVL Case
# ------------------------------------------------------------

class AVL_Discretization_Settings(Data):
	def __defaults__(self):
		"""
		SPACING SCHEMES:
			- 'cosine' : ||  |    |      |      |    |  || (bunched at both ends)
			- 'sine'   : || |  |   |    |    |     |     | (bunched at start)
			- 'equal'  : |   |   |   |   |   |   |   |   | (equally spaced)
			- '-sine'  : |     |     |    |    |   |  | || (bunched at end)
		"""
		
		self.chordwise_elements = 5
		self.chordwise_spacing_scheme = 'equal'
		self.spanwise_elements  = 5
		self.spanwise_spacing_scheme  = 'cosine'

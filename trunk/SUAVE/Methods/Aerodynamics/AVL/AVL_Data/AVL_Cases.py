# Tim Momose, October 2014

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Structure import Data, Data_Exception, Data_Warning


# ------------------------------------------------------------
#   Configuration
# ------------------------------------------------------------

class AVL_Cases(Data):

	def __defaults__(self):
		
		self.num_cases = 0
		self.cases = Data()


	def append_case(self,case):
		""" adds a case to the set of run cases """

		# assert database type
		if not isinstance(case,Data):
			raise Component_Exception, 'input component must be of type Data()'

		# store data with the appropriate case index
		# AVL uses indices starting from 1, not 0!
		self.num_cases += 1
		case.index = self.num_cases
		self.cases.append(case)

		return

# ------------------------------------------------------------
#  AVL Case
# ------------------------------------------------------------

class AVL_Run_Case(Data):
	def __defaults__(self):
		self.index  = 0		# Will be overwritten when appended to an AVL_Cases structure
		self.tag    = 'Case'
		
		self.conditions = Data()
		self.conditions.mach  = 0.0
		self.conditions.v_inf = 0.0
		self.conditions.rho   = 1.225
		self.conditions.gravitational_acc = 9.81
		
		self.angles = Data()
		self.angles.alpha = 0.0
		self.angles.beta  = 0.0
		
		self.control_deflections = Data()


	def append_control_deflection(self,control_tag,deflection):
		""" adds a control deflection case """
		control_deflection = Data()
		control_deflection.tag        = control_tag
		control_deflection.deflection = deflection
		self.control_deflection.append(control_deflection)
		
		return


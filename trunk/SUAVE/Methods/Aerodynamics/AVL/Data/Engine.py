# Body.py
# 
# Created:  Mar 2017, M.Clarke
# 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Data #, Data_Exception, Data_Warning
from Wing        import Section

# ------------------------------------------------------------
#   Engine
# ------------------------------------------------------------

class Engine(Data):
	"""	A data class defining the parameters of an engine
	"""

	def __defaults__(self):
		
		self.tag = 'engine'
		self.symmetric = True
		self.origin    = [0.,0.,0.]

		self.number_of_engines = 0.0
		self.bypass_ratio  = 0.0
		self.engine_length = 0.0
		self.nacelle_diameter  = 0.0

		self.configuration.nspanwise = 6
		self.configuration.nchordwise = 12
		self.configuration.sspace = 0
		self.configuration.cspace = 1.0


	def append_section(self,section):
		""" adds a section to engine """

		# assert database type
		if not isinstance(section,Data):
			raise Component_Exception, 'input component must be of type Data()'

		# store data
		self.sections.append(section)
		return


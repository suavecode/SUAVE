# Tim Momose, October 2014

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Structure import Data, Data_Exception, Data_Warning

from AVL_Wing import AVL_Wing
from AVL_Body import AVL_Body


# ------------------------------------------------------------
#   Aircraft
# ------------------------------------------------------------

class AVL_Aircraft(Data):
	
	def __defaults__(self):
		
		self.tag = 'Aircraft'
		self.wings = Data()
		self.bodies = Data()
	
	def append_wing(self,wing):
		# assert database type
		if not isinstance(wing,AVL_Wing):
			raise Component_Exception, 'input component must be of type AVL_Wing()'

		# store data
		self.wings.append(wing)
		return


	def append_body(self,body):
		# assert database type
		if not isinstance(body,AVL_Body):
			raise Component_Exception, 'input component must be of type AVL_Body()'

		# store data
		self.bodies.append(body)
		return


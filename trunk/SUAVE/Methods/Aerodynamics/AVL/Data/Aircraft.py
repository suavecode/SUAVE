# Tim Momose, October 2014

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Data, Data_Exception, Data_Warning

from Wing import Wing
from Body import Body


# ------------------------------------------------------------
#   Aircraft
# ------------------------------------------------------------

class Aircraft(Data):
	
	def __defaults__(self):
		
		self.tag = 'aircraft'
		self.wings = Data()
		self.bodies = Data()
	
	def append_wing(self,wing):
		# assert database type
		if not isinstance(wing,Wing):
			raise Component_Exception, 'input component must be of type AVL.Data.Wing()'

		# store data
		self.wings.append(wing)
		return


	def append_body(self,body):
		# assert database type
		if not isinstance(body,Body):
			raise Component_Exception, 'input component must be of type AVL.Data.Body()'

		# store data
		self.bodies.append(body)
		return


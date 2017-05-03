# Aircraft.py
# 
# Created:  Oct 2014, T. Momose
# Modified: Jan 2016, E. Botero
# Modified: Arp 2017, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Data

from Wing import Wing
from Body import Body
#from Engine import Engine

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
			raise Exception, 'input component must be of type AVL.Data.Wing()'

		# store data
		self.wings.append(wing)
		return


	def append_body(self,body):
		# assert database type
		if not isinstance(body,Body):
			raise Exception, 'input component must be of type AVL.Data.Body()'

		# store data
		self.bodies.append(body)
		return


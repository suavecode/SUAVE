# Inputs.py
# 
# Created:  Oct 2014, T. Momose
# Modified: Jan 2016, E. Botero
#           Jul 2017, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Data

# ------------------------------------------------------------
#   Configuration
# ------------------------------------------------------------

class Inputs(Data):

	def __defaults__(self):
		
		self.configuration  = Data()
		self.aircraft       = Data()
		self.cases          = Data()
		self.avl_bin_path   ='avl'
		
		filenames           = Data()
		filenames.geometry  = 'aircraft.avl'
		filenames.results   = []
		filenames.cases     = 'aircraft.cases'
		filenames.deck      = 'commands.run'
		filenames.reference_path = SUAVE.__path__[0] + '/temporary_files/'
		# Currently dumping input files in the SUAVE main
		# directory. This should be fixed once AVL has a place in
		# the SUAVE file structure
		
		self.input_files = filenames
## @ingroup Methods-Aerodynamics-AVL-Data
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

## @ingroup Methods-Aerodynamics-AVL-Data
class Inputs(Data):
	""" A data class defining filenames for the AVL executable

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
		""" Defines the data structure  and defaults of aircraft configuration and cases 
	
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
		self.input_files = filenames
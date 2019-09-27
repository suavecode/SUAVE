## @ingroup Methods-Aerodynamics-AVL-Data
# Body.py
# 
# Created:  Oct 2014, T. Momose
# Modified: Jan 2016, E. Botero
#           Jul 2017, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Data
from .Wing        import Section

# ------------------------------------------------------------
#   Body
# ------------------------------------------------------------

## @ingroup Methods-Aerodynamics-AVL-Data
class Body(Data):
	""" A data class defining the parameters of a fuselage or other body modeled
	by side and planform projections arranged in a plus (+) shape (when viewed from
	the front).
		    
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
		""" Defines the data structure and defaults for aircraft body geometry 
	
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
		self.tag                      = 'body'
		self.symmetric                = True
		self.origin                   = [0.,0.,0.]

		self.lengths                  = Data()
		self.lengths.total            = 0.0
		self.lengths.nose             = 0.0
		self.lengths.tail             = 0.0

		self.widths                   = Data()
		self.widths.maximum           = 0.0
		self.heights                  = Data()
		self.heights.maximum          = 0.0

		self.sections                 = Data()
		self.sections.vertical        = Data()
		self.sections.horizontal      = Data()
		self.configuration            = Data()
        
		self.configuration.nspanwise  = 10
		self.configuration.nchordwise = 5
		self.configuration.sspace     = 1.0
		self.configuration.cspace     = 1.0


	def append_section(self,section,orientation='Horizontal'):
		""" Adds a section to the body vertical or horizontal segment 
	
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
		if not isinstance(section,Data):
			raise Exception('input component must be of type Data()')

		# store data
		if orientation.lower() == 'horizontal':
			self.sections.horizontal.append(section)
		elif orientation.lower() == 'vertical':
			self.sections.vertical.append(section)
		else:
			raise KeyError('No key, "{}". Use "Horizontal" or "Vertical".'.format(orientation))
		return
	
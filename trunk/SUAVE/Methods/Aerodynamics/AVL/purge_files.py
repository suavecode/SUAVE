## @ingroup Methods-Aerodynamics-AVL
#purge_files.py
# 
# Created:  Oct 2014, T. Momose
# Modified: Jan 2016, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import os

## @ingroup Methods-Aerodynamics-AVL
def purge_files(filenames_array,directory=''):
	""" Purges folder folder of conflicting files
    
	Assumptions:
            None
     
	Source:
	    Drela, M. and Youngren, H., AVL, http://web.mit.edu/drela/Public/web/avl
    
	Inputs:
	    None
    
	Outputs:
            None
    
	Properties Used:
	    N/A
	"""    	
	for f in filenames_array:
		try:
			os.remove(os.path.abspath(os.path.join(directory,f)))
		except OSError:
			pass
			#print 'File {} was not found. Skipping purge.'.format(f)
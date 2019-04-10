## @ingroup Methods-Aerodynamics-AVL
#purge_directory.py
# 
# Created:  Oct 2014, T. Momose
# Modified: Jan 2016, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import os
from .purge_files import purge_files

## @ingroup Methods-Aerodynamics-AVL
def purge_directory(path,purge_subdirectories=False):
	""" Deletes the contents of a directory, then the directory itself.
	If purge_subdirectories is True, subdirectories of the directory will also
	be purged, recursively. In this case, the directory specified in the 'path'
	argument will always be removed at the end of this function. 
	However, if purge_subdirectories is False, files in 'path' will be deleted,
	but subdirectories and their contents will be left untouched. In this case, 
	the directory ('path' argument) will only be removed if it contains no
	subdirectories.


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
	
	contents = os.listdir(path)
	subdir   = []
	print(contents)
	
	for item in contents:
		print(os.path.abspath(os.path.join(path,item)))
		if os.path.isdir(os.path.abspath(os.path.join(path,item))):
			subdir.append(item)
			contents.remove(item)
	
	print(subdir, contents)
	purge_files(contents,path)
	
	if purge_subdirectories:
		for sub in subdir:
			purge_directory(os.path.join(path,sub),True)
	
	contents = os.listdir(path)
	if contents:
		print("Directory contains subdirectories which were not specified for deletion. Directory not fully deleted.")
	else:
		os.rmdir(path)
		
	return
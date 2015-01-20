# Tim Momose, December 2014

import os

def purge_files(filenames_array,directory=''):
	
	for f in filenames_array:
		try:
			os.remove(os.path.abspath(os.path.join(directory,f)))
		except OSError:
			print 'File {} was not found. Skipping purge.'.format(f)
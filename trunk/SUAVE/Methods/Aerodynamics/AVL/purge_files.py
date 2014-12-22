# Tim Momose, December 2014

import os

def purge_files(filenames_array,directory=''):
	
	if directory == '':
		pass
	elif not directory[-1] == '/':
		directory = directory + '/'
	
	for f in filenames_array:
		try:
			os.remove(directory+f)
		except OSError:
			print 'File {} was not found. Skipping purge.'.format(f)
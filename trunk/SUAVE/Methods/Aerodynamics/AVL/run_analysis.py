# Tim Momose, October 2014

import os

def run_analysis(avl_inputs):
	
	avl_bin_path      = avl_inputs.avl_bin_path
	files_path        = avl_inputs.input_files.reference_path
	geometry_filename = avl_inputs.input_files.geometry
	deck_filename     = avl_inputs.input_files.deck
	
	geometry_path = os.path.abspath(os.path.join(files_path,geometry_filename))
	deck_path     = os.path.abspath(os.path.join(files_path,deck_filename))
	
	command = build_avl_command(geometry_path,deck_path,avl_bin_path)
	run_command(command)



def build_avl_command(geometry_path,deck_path,avl_bin_path):
	""" builds a command to run an avl analysis on the specified geometry,
		according to the commands in the input deck. 
		filenames are referenced to AVL_Callable.settings.filenames.run_folder
	"""
	
	command_skeleton = '{0} {1}<{2}' # {avl_path} {geometry}<{input_deck}
	command = command_skeleton.format(avl_bin_path,geometry_path,deck_path)
	
	return command


def run_command(command):
	
	import sys
	import time
	import SUAVE.Plugins.VyPy.tools.redirect as redirect

	with redirect.output('avl_log.txt','stderr.txt'):
		ctime = time.ctime() # Current date and time stamp
		sys.stdout.write("Log File of System stdout from AVL Run \n{}\n\n".format(ctime))
		sys.stderr.write("Log File of System stderr from AVL Run \n{}\n\n".format(ctime))
		exit_status = os.system(command)
	
	return exit_status


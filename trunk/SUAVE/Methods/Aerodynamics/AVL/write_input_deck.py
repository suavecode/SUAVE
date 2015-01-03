# Tim Momose, October 2014

#from SUAVE.Structure import Data, Data_Exception, Data_Warning
from purge_files import purge_files


def write_input_deck(avl_object):

	base_input = \
'''LOAD {0}
CASE {1}
OPER
'''
	# unpack
	files_path        = avl_object.settings.filenames.run_folder
	geometry_filename = avl_object.settings.filenames.geometry
	cases_filename    = avl_object.settings.filenames.cases
	deck_filename     = avl_object.settings.filenames.input_deck
	kases             = avl_object.settings.run_cases
	
	deck_path     = files_path + deck_filename
	cases_path    = files_path + cases_filename
	geometry_path = files_path + geometry_filename
	
	input_deck = open(deck_path,'w')
	results_files = []
	
	try:
		input_deck.write(base_input.format(geometry_path,cases_path))
		for case in kases.cases:
			case_command,res_file = make_case_command(files_path,case)
			results_files.append(res_file)
			input_deck.write(case_command)
		avl_object.settings.filenames.results = results_files
		input_deck.write('\n\nQUIT\n')
	finally:
		input_deck.close()
	
	return avl_object


def make_case_command(directory,case):
	
	base_case_command = \
'''{0}
x
{1}
{2}{3}
'''
	
	index = case.index
	case_tag = case.tag
	res_type = 'st' # Eventually make this variable, or multiple, depending on user's desired outputs
	results_file = 'results_{}.txt'.format(case_tag)
	purge_files([directory+results_file])
	case_command = base_case_command.format(index,res_type,directory,results_file)
	
	return case_command,results_file

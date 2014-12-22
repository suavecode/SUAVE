# Tim Momose, October 2014

#from SUAVE.Structure import Data, Data_Exception, Data_Warning
from purge_files import purge_files


def write_input_deck(avl_inputs):

	base_input = \
'''LOAD {0}
CASE {1}
OPER
'''
	# unpack
	files_path        = avl_inputs.input_files.reference_path
	geometry_filename = avl_inputs.input_files.geometry
	cases_filename    = avl_inputs.input_files.cases
	deck_filename     = avl_inputs.input_files.deck
	kases             = avl_inputs.cases
	
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
		avl_inputs.input_files.results = results_files
		input_deck.write('\n\nQUIT\n')
	finally:
		input_deck.close()
	
	return avl_inputs



def make_case_command(directory,case):
	
	base_case_command = \
'''{0}
x
{1}
{2}results{0}.txt
'''
	
	index = case.index
	res_type = 'st' # Eventually make this variable, or multiple, depending on user's desired outputs
	results_file = 'results{}.txt'.format(index)
	purge_files([directory+results_file])
	#if index == 1:
	#	results_command = 'st\nresults.txt\n'
	#else:
	#	results_command = 'st\nresults.txt\na\n'
	case_command = base_case_command.format(index,res_type,directory)
	
	return case_command,results_file

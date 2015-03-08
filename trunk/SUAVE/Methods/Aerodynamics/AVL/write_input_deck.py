# Tim Momose, October 2014

#from SUAVE.Core import Data, Data_Exception, Data_Warning
from purge_files import purge_files


def write_input_deck(avl_object):

    open_runs = \
'''CASE {}
'''
    base_input = \
'''OPER
'''
    # unpack
    files_path    = avl_object.settings.filenames.run_folder
    batch         = avl_object.analysis_temps.current_batch_file
    deck_filename = avl_object.settings.filenames.input_deck

    # purge old versions and write the new input deck
    purge_files([deck_filename])
    with open(deck_filename,'w') as input_deck:
        input_deck.write(open_runs.format(batch))
        input_deck.write(base_input)
        for case in avl_object.analysis_temps.current_cases:
            case_command = make_case_command(avl_object,case)
            input_deck.write(case_command)
        input_deck.write('\n\nQUIT\n')

    return


def make_case_command(avl_object,case):

    base_case_command = \
'''{0}
x
{1}
{2}
'''
    directory = avl_object.settings.filenames.run_folder
    index = case.index
    case_tag = case.tag
    res_type = 'st' # This needs to change to multiple ouputs if you want to add the ability to read other types of results
    results_file = case.result_filename
    purge_files([results_file],directory)
    case_command = base_case_command.format(index,res_type,results_file)

    return case_command
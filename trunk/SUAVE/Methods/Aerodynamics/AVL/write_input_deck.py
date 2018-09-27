## @ingroup Methods-Aerodynamics-AVL
# write_input_deck.py
# 
# Created:  Oct 2014, T. Momose
# Modified: Jan 2016, E. Botero
#           Apr 2017, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from .purge_files import purge_files

## @ingroup Methods-Aerodynamics-AVL
def write_input_deck(avl_object):
    """ This fucntions writes the execution steps used in the AVL call

    Assumptions:
        None
        
    Source:
        Drela, M. and Youngren, H., AVL, http://web.mit.edu/drela/Public/web/avl

    Inputs:
        avl_object

    Outputs:
        None
 
    Properties Used:
        N/A
    """    
    
    open_runs = \
'''CASE {}
'''
    base_input = \
'''OPER
'''
    # unpack
    batch         = avl_object.current_status.batch_file
    deck_filename = avl_object.current_status.deck_file 

    # purge old versions and write the new input deck
    avl_regression_flag = avl_object.regression_flag
    if not avl_regression_flag:
        purge_files([deck_filename]) 
        with open(deck_filename,'w') as input_deck:
            input_deck.write(open_runs.format(batch))
            input_deck.write(base_input)
            for case_name in avl_object.current_status.cases:
                
                # write and store aerodynamic and static stability result files 
                case = avl_object.current_status.cases[case_name]
                case_command = make_case_command(avl_object,case)
                input_deck.write(case_command)
                
            #-----------------------------------------------------------------
            #          SUAVE-AVL dynamic stability analysis under development
            #          
            # write store dynamics stability result files 
            # em_case_command = make_eigen_mode_case_command(avl_object,case)
            # input_deck.write(em_case_command)
            #
            #-----------------------------------------------------------------
             
            input_deck.write('\n\nQUIT\n')

    return


def make_case_command(avl_object,case):
    """ Makes commands for case execution in AVK

    Assumptions:
        None
        
    Source:
        None

    Inputs:
        case.index
        case.tag
        case.result_filename

    Outputs:
        case_command
 
    Properties Used:
        N/A
    """  
    base_case_command = \
'''{0}
x
{1}
{2}
'''
    index = case.index
    case_tag = case.tag
    res_type = 'st' # This needs to change to multiple ouputs if you want to add the ability to read other types of results
    results_file = case.result_filename
    purge_files([results_file])
    case_command = base_case_command.format(index,res_type,results_file)

    return case_command


#def make_eigen_mode_case_command(avl_object,case):

    #em_base_case_command = \
#'''
#MODE
#n
#w
#{0}
#'''
    #em_results_file = case.eigen_result_filename
    #purge_files([em_results_file])
    #em_case_command = em_base_case_command.format(em_results_file)

    #return em_case_command


     
     
     
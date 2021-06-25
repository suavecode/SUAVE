## @ingroup Methods-Aerodynamics-AVL
# write_input_deck.py
# 
# Created:  Oct 2014, T. Momose
# Modified: Jan 2016, E. Botero
#           Apr 2017, M. Clarke
#           Aug 2019, M. Clarke
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from .purge_files import purge_files

## @ingroup Methods-Aerodynamics-AVL
def write_input_deck(avl_object,trim_aircraft):
    """ This function writes the execution steps used in the AVL call
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
    mass_file_input = \
'''MASS {0}
mset
0
PLOP
G
'''   
    open_runs = \
'''CASE {}
'''
    base_input = \
'''OPER
'''
    # unpack
    batch         = avl_object.current_status.batch_file
    deck_filename = avl_object.current_status.deck_file 
    mass_filename = avl_object.settings.filenames.mass_file

    # purge old versions and write the new input deck
    purge_files([deck_filename]) 
    with open(deck_filename,'w') as input_deck:
        input_deck.write(mass_file_input.format(mass_filename))
        input_deck.write(open_runs.format(batch))
        input_deck.write(base_input)
        for case in avl_object.current_status.cases:
            # write and store aerodynamic and static stability result files 
            case_command = make_case_command(avl_object,case,trim_aircraft)
            input_deck.write(case_command)

        input_deck.write('\nQUIT\n')

    return


def make_case_command(avl_object,case,trim_aircraft):
    """ Makes commands for case execution in AVL
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
    # This is a template (place holder) for the input deck. Think of it as the actually keys
    # you will type if you were to manually run an analysis
    base_case_command = \
'''{0}{1}
x
{2}
{3}
{4}
{5}
{6}
{7}
{8}
{9}
''' 
    # if trim analysis is specified, this function writes the trim commands 
    if trim_aircraft:
        trim_command = make_trim_text_command(case)
    else:
        trim_command = ''
    
    index          = case.index
    case_tag       = case.tag
    
    # AVL executable commands which correlate to particular result types 
    aero_command_1 = 'st' # stability axis derivatives   
    aero_command_2 = 'fn' # surface forces 
    aero_command_3 = 'fs' # strip forces 
    aero_command_4 = 'sb' # body axis derivatives 
                   
    # create aliases for filenames for future handling
    aero_file_1    = case.aero_result_filename_1 
    aero_file_2    = case.aero_result_filename_2 
    aero_file_3    = case.aero_result_filename_3 
    aero_file_4    = case.aero_result_filename_4
    
    # purge files 
    if not avl_object.keep_files:
        purge_files([aero_file_1])
        purge_files([aero_file_2])
        purge_files([aero_file_3])      
    
    # write input deck for avl executable 
    case_command = base_case_command.format(index,trim_command,aero_command_1 , aero_file_1 ,aero_command_2  \
                                            , aero_file_2 , aero_command_3 , aero_file_3, aero_command_4 , aero_file_4) 
        
    return case_command

def make_trim_text_command(case):
    """ Writes the trim command currently for a specified AoA or flight CL condition
    Assumptions:
        None
        
    Source:
        None
    Inputs:
        case
    Outputs:
        trim_command
 
    Properties Used:
        N/A
    """      
    
    base_trim_command = \
'''
c1
{0}
{1}
''' 
    CL_val   = case.conditions.aerodynamics.flight_CL
    velocity = case.conditions.freestream.velocity
    G_force  = case.conditions.freestream.gravitational_acceleration
    # if Angle of Attack command is specified, write A 
    if case.conditions.aerodynamics.flight_CL is None:
        condition = 'A'
        val       = case.conditions.aerodynamics.angle_of_attack
    else: # if Flight Lift Coefficient command is specified, write C
        condition = 'C'
        val       = case.conditions.aerodynamics.flight_CL 
        
    # write trim commands into template 
    trim_command = base_trim_command.format(condition,val)
    
    return trim_command

def control_surface_deflection_command(case,aircraft): 
    """Writes the control surface command template
    Assumptions:
        None
        
    Source:
        None
    Inputs:
        avl_object
        case
    Outputs:
        em_case_command
 
    Properties Used:
        N/A
    """     
    cs_template = \
'''
D{0}
D{1}
{2}'''
    cs_idx = 1 
    cs_commands = ''
    for wing in aircraft.wings:
        for ctrl_surf in wing.control_surfaces:
            cs_command = cs_template.format(cs_idx,cs_idx,wing.control_surfaces[ctrl_surf].deflection)
            cs_commands = cs_commands + cs_command
            cs_idx += 1
    return cs_commands 
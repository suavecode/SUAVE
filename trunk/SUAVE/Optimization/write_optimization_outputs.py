## @ingroup Optimization
# write_optimization_outputs.py
#
# Created:  May 2016, M. Vegh
# Modified:


# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from .helper_functions import get_values, scale_obj_values, scale_const_values

# ----------------------------------------------------------------------
#  write_optimization_outputs
# ----------------------------------------------------------------------


## @ingroup Optimization
def write_optimization_outputs(nexus, filename):
    """ Writes the optimization outputs to a file

    Assumptions:
    N/A

    Source:
    N/A

    Inputs:
    nexus            [nexus()]
    filename         [str]

    Outputs:
    N/A

    Properties Used:
    N/A
    """       
 
    #unpack optimization problem values
    objective          = nexus.optimization_problem.objective
    aliases            = nexus.optimization_problem.aliases
    constraints        = nexus.optimization_problem.constraints
    
    #inputs
    unscaled_inputs    = nexus.optimization_problem.inputs[:,1] #use optimization problem inputs here
    input_scaling      = nexus.optimization_problem.inputs[:,3]
    scaled_inputs      = unscaled_inputs/input_scaling
    
    #objective
    objective_value    = get_values(nexus,objective,aliases)
    scaled_objective   = scale_obj_values(objective , objective_value)
    
    #constraints
    constraint_values  = get_values(nexus,constraints,aliases) 
    scaled_constraints = scale_const_values(constraints,constraint_values)
    
    problem_inputs  = []
    problem_constraints = []
    for value in scaled_inputs:
        problem_inputs.append(value)  #writing to file is easier when you use list
    for value in scaled_constraints:
        problem_constraints.append(value)
    
    
    file=open(filename, 'a')
    file.write('iteration = ')
    file.write(str(nexus.total_number_of_iterations))
    file.write(' , ')
    file.write('objective = ')
    file.write(str(scaled_objective[0]))
    file.write(', inputs = ')
    file.write(str(problem_inputs))
    file.write(', constraints = ')
    file.write(str(problem_constraints))
    
    file.write('\n') 
    file.close()
    
    return
def write_optimization_outputs(nexus):
 
    #unpack optimization problem values
    objective          = nexus.optimization_problem.objective
    aliases            = nexus.optimization_problem.aliases
    constraints        = nexus.optimization_problem.constraints
    
    #inputs
    unscaled_inputs    = nexus.optimization_problem.inputs[:,1] #use optimization problem inputs here
    input_scaling      = nexus.optimization_problem.inputs[:,3]
    scaled_inputs      = unscaled_inputs/input_scaling
    
    #objective
    objective_value    = SUAVE.Optimization.helper_functions.get_values(nexus,objective,aliases)
    scaled_objective   = SUAVE.Optimization.helper_functions.scale_obj_values(objective , objective_value)
    
    #constraints
    constraint_values  = SUAVE.Optimization.helper_functions.get_values(nexus,constraints,aliases) 
    scaled_constraints = SUAVE.Optimization.helper_functions.scale_const_values(constraints,constraint_values)
    
    
    
    range              = nexus.target_range/Units.nautical_miles
    
    problem_inputs  = []
    problem_constraints = []
    for value in scaled_inputs:
        problem_inputs.append(value)  #writing to file is easier when you use list
    for value in scaled_constraints:
        problem_constraints.append(value)
    
    filename = 'optimizer_sizing '+str(int(range))+'_nautical_mile_range.txt'
    
    file=open(filename, 'ab')
    file.write('iteration = ')
    file.write(str(nexus.total_number_of_iterations))
    file.write(' , ')
    file.write('objective = ')
    file.write(str(scaled_objective))
    file.write(', inputs = ')
    file.write(str(problem_inputs))
    file.write(', constraints = ')
    file.write(str(problem_constraints))
    
    file.write('\n') 
    file.close()
    
    return nexus
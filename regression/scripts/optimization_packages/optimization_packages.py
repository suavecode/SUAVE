# optimization_packages.py
# Created: Sep. 2019, M. Clarke
#          Mar 2020, M. Clarke
#          Jul 2020, M. Clarke

# ----------------------------------------------------------------------        
#   Imports
# ----------------------------------------------------------------------    
import SUAVE
from SUAVE.Core import Units, Data
import numpy as np
import vehicle_opt_pack
import procedure_opt_pack 
from SUAVE.Optimization import Nexus, carpet_plot 
import SUAVE.Optimization.Package_Setups.scipy_setup as scipy_setup
import SUAVE.Optimization.write_optimization_outputs as write_optimization_outputs
import os , sys
# ----------------------------------------------------------------------        
#   Run the whole thing
# ----------------------------------------------------------------------  
def main():
    seed = np.random.seed(1)  
    
    # ------------------------------------------------------------------
    #   SLSQP
    # ------------------------------------------------------------------    
    solver_name = 'SLSQP'
    problem     = setup(solver_name)
    problem.optimization_problem.constraints = np.array([
        [ 'x1' , '>', -10., 1., 1*Units.less],
        [ 'x1' , '=',   0., 1., 1*Units.less],
        [ 'x2' , '>',   1., 1., 1*Units.less],
        [ 'x2' , '<',   2., 1., 1*Units.less],
    ],dtype=object)        
    print('\n\n Checking basic additive with one active constraint...') 
    # suppress iteration printout 
    Nexus.translate(problem)
    sys.stdout = open(os.devnull,'w')   
    outputs = scipy_setup.SciPy_Solve(problem, solver='SLSQP' , sense_step = 1.4901161193847656e-08, pop_size =  10 , prob_seed = seed )  
    # end suppression of interation printout
    sys.stdout = sys.__stdout__  
    print(outputs)  
    obj = scipy_setup.SciPy_Problem(problem,outputs)[0]
    x1 = outputs[0]
    x2 = outputs[1] 
     
    #   Check Results 
    assert( np.isclose(obj,  1, atol=1e-6) )
    assert( np.isclose(x1 ,  0, atol=1e-2) )
    assert( np.isclose(x2 ,  1, atol=1e-2))    
    
    # test writing output function 
    filename = 'optimization_output.txt'
    write_optimization_outputs(problem,filename)  
    os.remove('optimization_output.txt')
    
    # ------------------------------------------------------------------
    #   Differential Evolution 
    # ------------------------------------------------------------------  
    print('\n\n Checking differential evolution algorithm')
    solver_name = 'differential_evolution' 
    problem     = setup(solver_name)
    problem.optimization_problem.constraints = np.array([
        [ 'x1' , '>', -10., 1., 1*Units.less],
        [ 'x1' , '=',   0., 1., 1*Units.less],
        [ 'x2' , '>',   1., 1., 1*Units.less],
        [ 'x2' , '<',   2., 1., 1*Units.less],
    ],dtype=object)          
    # suppress iteration printout 
    sys.stdout = open(os.devnull,'w')      
    outputs = scipy_setup.SciPy_Solve(problem, solver='differential_evolution' , sense_step = 1.4901161193847656e-08, pop_size =  10 , prob_seed = seed )  
    # end suppression of interation printout
    sys.stdout = sys.__stdout__  
    print(outputs)   
    obj = outputs.fun
    x1 = outputs.x[0]
    x2 = outputs.x[1] 
     
    #   Check Results 
    assert( np.isclose(obj,  1, atol=1e-5) )
    assert( np.isclose(x1 ,  0, atol=1e-4) )
    assert( np.isclose(x2 ,  1, atol=1e-4) )     
 
 
    # ------------------------------------------------------------------
    #   Particle Swarm Optimization
    # ------------------------------------------------------------------     
    solver_name = 'particle_swarm_optimization'
    problem     = setup(solver_name)        
    problem.optimization_problem.constraints = np.array([
        [ 'x1' , '>', -10., 1., 1*Units.less],
        [ 'x2' , '>',   1., 1., 1*Units.less],
        [ 'x2' , '<',   2., 1., 1*Units.less],
    ],dtype=object)     
    print('\n\n Checking particle swarm optimization algorithm')
    # suppress iteration printout 
    sys.stdout = open(os.devnull,'w')      
    outputs = scipy_setup.SciPy_Solve(problem, solver='particle_swarm_optimization' , sense_step = 1.4901161193847656e-08, pop_size =  100 , prob_seed = seed )  
    # end suppression of interation printout
    sys.stdout = sys.__stdout__  
    print(outputs)   
    obj = outputs[1][0]
    x1 = outputs[0][0]
    x2 = outputs[0][1]
     
    #   Check Results 
    assert( np.isclose(obj,  1, atol=1e-2) )
    assert( np.isclose(x1 ,  0, atol=1e-1) )
    assert( np.isclose(x2 ,  1, atol=1e-1) )     
    
     
    return

# ----------------------------------------------------------------------        
#   Inputs, Objective, & Constraints
# ----------------------------------------------------------------------  

def setup(solver_name):

    nexus = Nexus()
    problem = Data()
    nexus.optimization_problem = problem
    nexus.solver_name = solver_name
    # -------------------------------------------------------------------
    # Inputs
    # -------------------------------------------------------------------

    problem.inputs = np.array([
    #   [ tag   , initial,(   lb   ,   ub   )     , scaling , units ]
        [ 'x1'  ,  1.  ,    -2.   ,   2.     ,   1.   , 1*Units.less],
        [ 'x2'  ,  1.  ,    -2.   ,   2.     ,   1.   , 1*Units.less],
    ],dtype=object)
    
    # -------------------------------------------------------------------
    # Objective
    # -------------------------------------------------------------------

    # throw an error if the user isn't specific about wildcards
    # [ tag, scaling, units ]
    problem.objective = np.array([
        ['y',1.,1*Units.less]
    ],dtype=object)
    
    # -------------------------------------------------------------------
    # Constraints
    # -------------------------------------------------------------------
    
    # [ tag, sense, edge, scaling, units ]
    problem.constraints = np.array([
        [ 'x1' , '>', -10., 1., 1*Units.less],
        [ 'x2' , '>', -50., 1., 1*Units.less],
    ])
    
    # -------------------------------------------------------------------
    #  Aliases
    # -------------------------------------------------------------------
    
    # [ 'alias' , ['data.path1.name','data.path2.name'] ]

    # don't set wing_area for initial configuration so that values can be used later
    problem.aliases = [
        [ 'x1'                        ,    'vehicle_configurations.base.x1'       ],
        [ 'x2'                        ,    'vehicle_configurations.base.x2'       ],
        [ 'y'                         ,    'obj'                            ],
    ]    
    
    # -------------------------------------------------------------------
    #  Vehicles
    # -------------------------------------------------------------------
    nexus.vehicle_configurations = vehicle_opt_pack.setup()
    
    
    # -------------------------------------------------------------------
    #  Analyses
    # -------------------------------------------------------------------
    nexus.analyses = None
    
    
    # -------------------------------------------------------------------
    #  Missions
    # -------------------------------------------------------------------
    nexus.missions = None
    
    
    # -------------------------------------------------------------------
    #  Procedure
    # -------------------------------------------------------------------    
    nexus.procedure = procedure_opt_pack.setup()
    
    # -------------------------------------------------------------------
    #  Summary
    # -------------------------------------------------------------------    
    nexus.summary = Data()    
    nexus.total_number_of_iterations = 0
    return nexus

if __name__ == '__main__':
    main()
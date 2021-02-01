# Optimize.py
# Created:  Feb 2016, M. Vegh
# Modified: Nov 2016, T. MacDonald
#           Oct 2019, T. MacDonald

# ----------------------------------------------------------------------        
#   Imports
# ----------------------------------------------------------------------    

import SUAVE
from SUAVE.Core import Units, Data
import numpy as np
import vehicle_mf
import procedure_mf
import matplotlib.pyplot as plt
from SUAVE.Optimization import Nexus, carpet_plot
from SUAVE.Optimization.Package_Setups.additive_setup import Additive_Solver
import SUAVE.Optimization.Package_Setups.TRMM.Trust_Region_Optimization as tro
from SUAVE.Optimization.Package_Setups.TRMM.Trust_Region import Trust_Region
import os

# ----------------------------------------------------------------------        
#   Run the whole thing
# ----------------------------------------------------------------------  
def main():
    np.random.seed(0)
    
    problem = setup()
    tol = 1e-8
    
    def set_add_solver():
        solver = Additive_Solver()
        solver.local_optimizer = 'SLSQP'
        solver.global_optimizer = 'SHGO'
        return solver
    
    ################### Basic Additive ##################################################
    
    # ------------------------------------------------------------------
    #   Inactive constraints
    # ------------------------------------------------------------------     
    
    solver = set_add_solver()
    
    problem.optimization_problem.constraints = np.array([
        [ 'x1' , '>', -10., 1., 1*Units.less],
        [ 'x2' , '>', -50., 1., 1*Units.less],
    ],dtype=object)    
    
    print('Checking basic additive with no active constraints...')
    outputs = solver.Additive_Solve(problem,max_iterations=10,num_samples=20,tolerance=1e-8,print_output=False)
    print(outputs)   
    obj,x1,x2 = get_results(outputs)
    
    # ------------------------------------------------------------------
    #   Check Results
    # ------------------------------------------------------------------    

    assert( np.isclose(obj,  0, atol=1e-6) )
    assert( np.isclose(x1 ,-.1, atol=1e-2) )
    assert( np.isclose(x2 ,  0, atol=1e-2) )      
    
    # ------------------------------------------------------------------
    #   Active constraint
    # ------------------------------------------------------------------     
    
    solver = set_add_solver()
    
    problem.optimization_problem.constraints = np.array([
        [ 'x1' , '>', -10., 1., 1*Units.less],
        [ 'x2' , '>',   1., 1., 1*Units.less],
    ],dtype=object)    
    
    print('Checking basic additive with one active constraint...')
    outputs = solver.Additive_Solve(problem,max_iterations=1000,num_samples=20,tolerance=1e-8,print_output=False)
    print(outputs)   
    obj,x1,x2 = get_results(outputs)
    
    # ------------------------------------------------------------------
    #   Check Results
    # ------------------------------------------------------------------    

    assert( np.isclose(obj,  1, atol=1e-6) )
    assert( np.isclose(x1 ,-.1, atol=1e-2) )
    assert( np.isclose(x2 ,  1, atol=1e-2) )     
    
    # ------------------------------------------------------------------
    #   Other active constraints
    # ------------------------------------------------------------------     
    
    solver = set_add_solver()
    
    problem.optimization_problem.constraints = np.array([
        [ 'x1' , '=',   2., 1., 1*Units.less],
        [ 'x2' , '<',  -1., 1., 1*Units.less],
    ],dtype=object)    
    
    print('Checking basic additive with two active constraints...')
    outputs = solver.Additive_Solve(problem,max_iterations=1000,num_samples=20,tolerance=1e-8,print_output=False)
    print(outputs)   
    obj,x1,x2 = get_results(outputs)
    
    # ------------------------------------------------------------------
    #   Check Results
    # ------------------------------------------------------------------    

    assert( np.isclose(obj,5.41, atol=1e-6) )
    assert( np.isclose(x1 ,   2, atol=1e-2) )
    assert( np.isclose(x2 ,  -1, atol=1e-2) )  
    
    ################# Additive MEI ##################################################
    
    # ------------------------------------------------------------------
    #   Inactive constraints
    # ------------------------------------------------------------------     
    
    solver = set_add_solver()
    
    problem.optimization_problem.constraints = np.array([
        [ 'x1' , '>', -10., 1., 1*Units.less],
        [ 'x2' , '>', -50., 1., 1*Units.less],
    ],dtype=object)    
    
    print('Checking MEI additive with no active constraint...')
    outputs = solver.Additive_Solve(problem,max_iterations=10,num_samples=20,tolerance=tol,print_output=False,opt_type='MEI')
    print(outputs)   
    obj,x1,x2 = get_results(outputs)
    
    # ------------------------------------------------------------------
    #   Check Results
    # ------------------------------------------------------------------    

    assert( np.isclose(obj,  0, atol=1e-6) )
    assert( np.isclose(x1 ,-.1, atol=1e-2) )
    assert( np.isclose(x2 ,  0, atol=1e-2) )      
    
    # ------------------------------------------------------------------
    #   Active constraint
    # ------------------------------------------------------------------     
    
    solver = set_add_solver()
    
    problem.optimization_problem.constraints = np.array([
        [ 'x1' , '>', -10., 1., 1*Units.less],
        [ 'x2' , '>',   1., 1., 1*Units.less],
    ],dtype=object)    
    
    print('Checking MEI additive with one active constraint...')
    outputs = solver.Additive_Solve(problem,max_iterations=10,num_samples=20,tolerance=tol,print_output=False,opt_type='MEI')
    print(outputs)   
    obj,x1,x2 = get_results(outputs)
    
    # ------------------------------------------------------------------
    #   Check Results
    # ------------------------------------------------------------------    

    assert( np.isclose(obj,  1, atol=1e-4) ) # optimizer does not reach exactly optimum here
    assert( np.isclose(x1 ,-.1, atol=1e-2) )
    assert( np.isclose(x2 ,  1, atol=1e-2) )     
    
    #------------------------------------------------------------------
    #   Other active constraints
    #------------------------------------------------------------------     
    
    solver = set_add_solver()
    
    problem.optimization_problem.constraints = np.array([
        [ 'x1' , '=',   2., 1., 1*Units.less],
        [ 'x2' , '<',  -1., 1., 1*Units.less],
    ],dtype=object)    
    
    print('Checking MEI additive with two active constraints...')
    outputs = solver.Additive_Solve(problem,max_iterations=10,num_samples=20,tolerance=tol,print_output=False,opt_type='MEI')
    print(outputs)   
    obj,x1,x2 = get_results(outputs)
    
    # ------------------------------------------------------------------
    #   Check Results
    # ------------------------------------------------------------------    

    assert( np.isclose(obj,5.41, atol=1e-6) )
    assert( np.isclose(x1 ,   2, atol=1e-6) )
    assert( np.isclose(x2 ,  -1, atol=1e-6) )     
    
    ################# TRMM ##################################################
    
    tr_optimizer = 'SLSQP'
    
    # ------------------------------------------------------------------
    #   Inactive constraints
    # ------------------------------------------------------------------     
    
    problem.optimization_problem.constraints = np.array([
        [ 'x1' , '>', -10., 1., 1*Units.less],
        [ 'x2' , '>', -50., 1., 1*Units.less],
    ],dtype=object)    
    
    tr = Trust_Region()
    problem.trust_region = tr
    TRM_opt = tro.Trust_Region_Optimization()
    TRM_opt.trust_region_max_iterations           = 20
    TRM_opt.optimizer  = tr_optimizer    
    print('Checking TRMM with no active constraints...')
    outputs = TRM_opt.optimize(problem,print_output=False)
    print(outputs)   
    obj,x1,x2 = get_results(outputs)
    
    # ------------------------------------------------------------------
    #   Check Results
    # ------------------------------------------------------------------    

    assert( np.isclose(obj,  0, atol=1e-6) )
    assert( np.isclose(x1 ,-.1, atol=1e-2) )
    assert( np.isclose(x2 ,  0, atol=1e-2) )       
    
    # ------------------------------------------------------------------
    #   Active constraint
    # ------------------------------------------------------------------     
    
    problem.optimization_problem.constraints = np.array([
        [ 'x1' , '>', -10., 1., 1*Units.less],
        [ 'x2' , '>',   1., 1., 1*Units.less],
    ],dtype=object)   
    
    tr = Trust_Region()
    problem.trust_region = tr
    TRM_opt = tro.Trust_Region_Optimization()
    TRM_opt.trust_region_max_iterations           = 20
    TRM_opt.optimizer  = tr_optimizer    
    print('Checking TRMM with one active constraint...')
    outputs = TRM_opt.optimize(problem,print_output=False)
    print(outputs)   
    obj,x1,x2 = get_results(outputs)
    
    # ------------------------------------------------------------------
    #   Check Results
    # ------------------------------------------------------------------    

    assert( np.isclose(obj,  1, atol=1e-6) )
    assert( np.isclose(x1 ,-.1, atol=1e-2) )
    assert( np.isclose(x2 ,  1, atol=1e-2) )  
    
    # ------------------------------------------------------------------
    #   Other constraints
    # ------------------------------------------------------------------     
    
    problem.optimization_problem.constraints = np.array([
        [ 'x1' , '=',   2., 1., 1*Units.less],
        [ 'x2' , '<',  -1., 1., 1*Units.less],
    ],dtype=object)     
    
    tr = Trust_Region()
    problem.trust_region = tr
    TRM_opt = tro.Trust_Region_Optimization()
    TRM_opt.trust_region_max_iterations           = 20
    TRM_opt.optimizer  = tr_optimizer    
    print('Checking TRMM with active constraints...')
    outputs = TRM_opt.optimize(problem,print_output=False)
    print(outputs)   
    obj,x1,x2 = get_results(outputs)
    

    # removes files from folder after regression is completed 
    os.remove("add_hist.txt")  
    os.remove("TRM_hist.txt") 
    
    # ------------------------------------------------------------------
    #   Check Results
    # ------------------------------------------------------------------    

    assert( np.isclose(obj,5.41, atol=1e-6) )
    assert( np.isclose(x1 ,   2, atol=1e-2) )
    assert( np.isclose(x2 ,  -1, atol=1e-2) )      
     
    return

# ----------------------------------------------------------------------        
#   Inputs, Objective, & Constraints
# ----------------------------------------------------------------------  

def setup():

    nexus = Nexus()
    problem = Data()
    nexus.optimization_problem = problem

    # -------------------------------------------------------------------
    # Inputs
    # -------------------------------------------------------------------

    #   [ tag                            , initial, (lb,ub)             , scaling , units ]
    problem.inputs = np.array([
        [ 'x1'  ,  1.  , (   -2.   ,   2.   )  ,   1.   , 1*Units.less],
        [ 'x2'  ,  1.  , (   -2.   ,   2.   )  ,   1.   , 1*Units.less],
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
    ],dtype=object)
    
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
    nexus.vehicle_configurations = vehicle_mf.setup()
    
    
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
    nexus.procedure = procedure_mf.setup()
    
    # -------------------------------------------------------------------
    #  Summary
    # -------------------------------------------------------------------    
    nexus.summary = Data()    
    nexus.total_number_of_iterations = 0
    return nexus

def get_results(outputs):
    obj = outputs[0]   
    x1  = outputs[1][0] 
    x2  = outputs[1][1] 
    return obj, x1, x2

if __name__ == '__main__':
    main()
    
    

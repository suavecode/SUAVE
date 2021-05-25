## @ingroup Optimization
#  line_plot.py
#
# Created:  Oct 2017, M. Vegh 
# Modified: Nov 2017, M. Vegh
#           May 2021, E. Botero 

# ----------------------------------------------------------------------
#  Imports
# -------------------------------------------
 
from SUAVE.Core import Data
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
#  line_plot
# ----------------------------------------------------------------------


def line_plot(problem, number_of_points,  plot_obj=1, plot_const=1, sweep_index=0): 
    """
    Takes in an optimization problem and runs a line plot of the first  variable of sweep index
    sweep_index. i.e. sweep_index=0 means you want to sweep the first variable, sweep_index = 4 is the 5th variable)
    
        Assumptions:
        N/A
    
        Source:
        N/A
    
        Inputs:
        problem            [Nexus Class]
        number_of_points   [int]
        plot_obj           [int]
        plot_const         [int]
        sweep_index        [int]

        
        Outputs:
        Beautiful plots!
            Outputs:
                inputs     [array]
                objective  [array]
                constraint [array]
    
        Properties Used:
        N/A
    """         
    
    
    

    idx0            = sweep_index # local name

    opt_prob        = problem.optimization_problem
    base_inputs     = opt_prob.inputs
    names           = base_inputs[:,0] # Names
    bndl            = base_inputs[:,2] # Bounds
    bndu            = base_inputs[:,3] # Bounds
    base_objective  = opt_prob.objective
    obj_name        = base_objective[0][0] #objective function name (used for scaling)
    obj_scaling     = base_objective[0][1]
    base_constraints= opt_prob.constraints
    constraint_names= base_constraints[:,0]
   
    #define inputs, output, and constraints for sweep
    inputs          = np.zeros([2,number_of_points])
    obj             = np.zeros([number_of_points])
    constraint_num  = np.shape(base_constraints)[0] # of constraints
    constraint_val  = np.zeros([constraint_num,number_of_points])
    
    
    #create inputs matrix
    inputs[0,:] = np.linspace(bndl[idx0], bndu[idx0], number_of_points)
 

    #inputs defined; now run sweep
    for i in range(0, number_of_points):
        opt_prob.inputs[:,1][idx0]= inputs[0,i]
   
        obj[i]             = problem.objective()*obj_scaling
        constraint_val[:,i]= problem.all_constraints().tolist()
  
    if plot_obj==1:
        plt.figure(0)
        plt.plot(inputs[0,:], obj, lw = 2)
        plt.xlabel(names[idx0])
        plt.ylabel(obj_name)
        

    if plot_const==1:
        for i in range(0, constraint_num):
            plt.figure(i+1)
            plt.plot(inputs[0,:], constraint_val[i,:], lw = 2)
            plt.xlabel(names[idx0])
            plt.ylabel(constraint_names[i])

       
    plt.show(block=True)      
       
        
    #pack outputs
    outputs= Data()
    outputs.inputs         = inputs
    outputs.objective      = obj
    outputs.constraint_val =constraint_val
    return outputs
    
    
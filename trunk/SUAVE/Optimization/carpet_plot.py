## @ingroup Optimization
# carpet_plot.py
#
# Created:  Feb 2016, M. Vegh 
# Modified: Feb 2017, M. Vegh
#           May 2021, E. Botero 

# ----------------------------------------------------------------------
#  Imports
# -------------------------------------------
 
from SUAVE.Core import Data
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
#  carpet_plot
# ----------------------------------------------------------------------

## @ingroup Optimization
def carpet_plot(problem, number_of_points,  plot_obj=1, plot_const=0, sweep_index_0=0, sweep_index_1=1): 
    """ Takes in an optimization problem and runs a carpet plot of the first 2 variables
        sweep_index_0, sweep_index_1 is index of variables you want to run carpet plot (i.e. sweep_index_0=0 means you want to sweep first variable, sweep_index_0 = 4 is the 5th variable)
    
        Assumptions:
        N/A
    
        Source:
        N/A
    
        Inputs:
        problem            [Nexus Class]
        number_of_points   [int]
        plot_obj           [int]
        plot_const         [int]
        sweep_index_0      [int]
        sweep_index_1      [int]
        
        Outputs:
        Beautiful Beautiful Plots!
            Outputs:
                inputs     [array]
                objective  [array]
                constraint [array]
    
        Properties Used:
        N/A
    """         

    #unpack
    idx0            = sweep_index_0 # local name
    idx1            = sweep_index_1
    opt_prob        = problem.optimization_problem
    base_inputs     = opt_prob.inputs
    names           = base_inputs[:,0] # Names
    bndl            = base_inputs[:,2] # Bounds
    bndu            = base_inputs[:,3] # Bounds
    scl             = base_inputs[:,4] # Scaling
    base_objective  = opt_prob.objective
    obj_name        = base_objective[0][0] #objective function name (used for scaling)
    obj_scaling     = base_objective[0][1]
    base_constraints= opt_prob.constraints
    constraint_names= base_constraints[:,0]
    constraint_scale= base_constraints[:,3]
   
    #define inputs, output, and constraints for sweep
    inputs          = np.zeros([2,number_of_points])
    obj             = np.zeros([number_of_points,number_of_points])
    constraint_num  = np.shape(base_constraints)[0] # of constraints
    constraint_val  = np.zeros([constraint_num,number_of_points,number_of_points])
    
    
    #create inputs matrix
    inputs[0,:] = np.linspace(bndl[idx0], bndu[idx0], number_of_points)
    inputs[1,:] = np.linspace(bndl[idx1], bndu[idx1], number_of_points)

    
    #inputs defined; now run sweep
    for i in range(0, number_of_points):
        for j in range(0,number_of_points):
            #problem.optimization_problem.inputs=base_inputs  #overwrite any previous modification
            opt_prob.inputs[:,1][idx0]= inputs[0,i]
            opt_prob.inputs[:,1][idx1]= inputs[1,j]

            obj[j,i]             = problem.objective()*obj_scaling
            constraint_val[:,j,i]= problem.all_constraints().tolist()
  
    if plot_obj==1:
        plt.figure(0)
        CS = plt.contourf(inputs[0,:],inputs[1,:], obj, linewidths=2)
        cbar = plt.colorbar(CS)
        cbar.ax.set_ylabel(obj_name)
        plt.xlabel(names[idx0])
        plt.ylabel(names[idx1])

       
    if plot_const==1:
        
        for i in range(0, constraint_num): #constraint_num):
            plt.figure(i+1)
            CS_const=plt.contour(inputs[0,:],inputs[1,:], constraint_val[i,:,:])
            cbar = plt.colorbar(CS_const)
            cbar.ax.set_ylabel(constraint_names[i])
            plt.xlabel(names[idx0])
            plt.ylabel(names[idx1])
    plt.show(block=True)      
       

    #pack outputs
    outputs= Data()
    outputs.inputs         = inputs
    outputs.objective      = obj
    outputs.constraint_val = constraint_val
    
    return outputs
    
    
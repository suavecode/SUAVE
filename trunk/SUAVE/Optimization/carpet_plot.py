# carpet_plot.py
#
# Created : Feb 2016, M. Vegh 
from SUAVE.Core import Data
import numpy as np
import matplotlib.pyplot as plt
def carpet_plot(problem, number_of_points, plot_obj=1, plot_const=0): 
    #SUAVE.Optimization.carpet_plot(problem, ):
    #takes in an optimization problem and runs a carpet plot of the first 2 variables

    #unpack
    opt_prob        = problem.optimization_problem
    idx0            = 0   #index of variable location
    idx1            = 1
    base_inputs     = opt_prob.inputs
    names           = base_inputs[:,0] # Names
    bnd             = base_inputs[:,2] # Bounds
    scl             = base_inputs[:,3] # Scaling
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
    inputs[0,:] = np.linspace(bnd[idx0][0], bnd[idx0][1], number_of_points)
    inputs[1,:] = np.linspace(bnd[idx1][0], bnd[idx1][1], number_of_points)

    
    #inputs defined; now run sweep
    for i in range(0, number_of_points):
        for j in range(0,number_of_points):
            #problem.optimization_problem.inputs=base_inputs  #overwrite any previous modification
            opt_prob.inputs[:,1][idx0]= inputs[0,i]
            opt_prob.inputs[:,1][idx1]= inputs[1,j]
   
            obj[i,j]             = problem.objective()*obj_scaling
            constraint_val[:,i,j]= problem.all_constraints().tolist()
  
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
    plt.show()      
       
        
    #pack outputs
    outputs= Data()
    outputs.inputs         = inputs
    outputs.objective      = obj
    outputs.constraint_val =constraint_val
    return outputs
    
    
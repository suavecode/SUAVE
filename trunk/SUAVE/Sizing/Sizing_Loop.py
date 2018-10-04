## @ingroup Sizing
#Sizing_Loop.py
#Created:  Jun 2016, M. Vegh
#Modified: May 2018, M. Vegh

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Data
from SUAVE.Surrogate.svr_surrogate_functions import check_svr_accuracy
import scipy.interpolate as interpolate

import sklearn.svm as svm
import sklearn.ensemble as ensemble
import sklearn.gaussian_process as gaussian_process
from sklearn.gaussian_process.kernels import RationalQuadratic 
import sklearn.linear_model as linear_model
import sklearn.neighbors as neighbors
from .write_sizing_outputs import write_sizing_outputs
from .read_sizing_inputs import read_sizing_inputs
from .write_sizing_residuals import write_sizing_residuals


import numpy as np
import scipy as sp
import time


## @ingroup Sizing
class Sizing_Loop(Data):
    def __defaults__(self):
        """
        Data class that solves a fixed point iteration problem to size the aircraft. Includes
        a variety of methods to solve the subproblem, including successive substitution, newton-raphson,
        broyden's method, as well as a damped newton method. Also includes machine learning algorithms
        from scikit-learn to aid in finding a good initial guess for your sizing parameters.
        """
   
        #parameters common to all methods
        self.tolerance             = None
        self.initial_step          = None  #'Default', 'Table', 'SVR', 'GradientBoosting', ExtraTrees', 'RandomForest', 'Bagging', 'GPR', 'RANSAC', 'Neighbors'  
        self.update_method         = None  #'successive_substitution', 'newton-raphson', 'broyden'
        self.default_y             = None  #default inputs in case the guess is very far from 
        self.default_scaling       = None  #scaling value to make sizing parameters ~1
        self.maximum_iterations    = None  #cutoff point for sizing loop to close
        self.output_filename       = None  #stores optimization parameters and closed sizing parameters
        self.sizing_evaluation     = None  #defined in the Procedure script
        self.write_threshhold      = 3     #number of iterations before it writes, regardless of how close it is to currently written values (i.e. this step is hard to converge)
        self.max_y                 = None  #vector of highest allowable y values 
        self.min_y                 = None  #vector of lowest allowable y values
        self.hard_max_bound        = False #set to true if you want the solver to backtrack if it reaches this bound, otherwise, just don't allow to start higher than this value
        self.hard_min_bound        = True  #set to true if you want the solver to backtrack if it reaches this bound, otherwise, just don't allow to start lower than this value
        self.write_threshhold      = 3     #number of iterations before it writes,
        self.write_residuals       = False  #set to True to write the residuals at every iteration
        self.residual_filename     = 'y_err_values.txt'
        
        #parameters that may only apply to certain methods
        self.iteration_options     = Data()
        self.iteration_options.newton_raphson_tolerance          = 5E-2             #threshhold of convergence when you start using newton raphson
        self.iteration_options.max_newton_raphson_tolerance      = 2E-3             #threshhold at which newton raphson is no longer used (to prevent overshoot and extra iterations)
        self.iteration_options.h                                 = 1E-6             #finite difference step for Newton iteration
        self.iteration_options.initialize_jacobian               = 'newton-raphson' #how Jacobian is initialized for broyden; newton-raphson by default
        self.iteration_options.jacobian                          = np.array([np.nan])
        self.iteration_options.write_jacobian                    = True             #set to True if you want to write the Jacobian at each iteration (to keep track of every iteration
        self.iteration_options.max_initial_step                  = 1.               #maximum distance at which interpolation is allowed
        self.iteration_options.min_fix_point_iterations          = 2                #minimum number of iterations to perform fixed-point iteration before starting newton-raphson
        self.iteration_options.min_surrogate_step                = .011             #minimum distance at which SVR is used (if closer, table lookup is used)
        self.iteration_options.min_write_step                    = .011             #minimum distance at which sizing data are written
        self.iteration_options.min_surrogate_length              = 4                #minimum number data points needed before SVR is used
        self.iteration_options.number_of_surrogate_calls         = 0
        self.iteration_options.newton_raphson_damping_threshhold = 5E-5
        self.iteration_options.n_neighbors                       = 5
        self.iteration_options.err_save                          = 0.
        
        #backtracking 
        backtracking                         = Data()
        backtracking.backtracking_flag       = True     #True means you do backtracking when err isn't decreased
        backtracking.threshhold              = 1.      # factor times the msq at which to terminate backtracking
        backtracking.max_steps               = 5
        backtracking.multiplier              = .5
        self.iteration_options.backtracking  = backtracking
        
    def evaluate(self, nexus):
        
        if nexus.optimization_problem != None: #make it so you can run sizing without an optimization problem
            unscaled_inputs = nexus.optimization_problem.inputs[:,1] #use optimization problem inputs here
            input_scaling   = nexus.optimization_problem.inputs[:,3]
            scaled_inputs   = unscaled_inputs/input_scaling
            problem_inputs  = []
            for value in scaled_inputs:
                problem_inputs.append(value)  #writing to file is easier when you use list
            
            nexus.problem_inputs = problem_inputs
            opt_flag = 1 #tells if you're running an optimization case or not-used in writing outputs
        else:
            opt_flag = 0
  
        #unpack inputs
        tol               = self.tolerance #percentage difference in mass and energy between iterations
        h                 = self.iteration_options.h 
        y                 = self.default_y
        max_iter          = self.maximum_iterations
        scaling           = self.default_scaling
        sizing_evaluation = self.sizing_evaluation
        iteration_options = self.iteration_options
        err               = [1000] #initialize error
        
        #initialize
        converged = 0     #marker to tell if it's converged
        i         = 0  #function evals
        
        #determine the initial step
        min_norm = 1000.
        if self.initial_step != 'Default':
            data_inputs, data_outputs, read_success = read_sizing_inputs(self, scaled_inputs)
            
            if read_success:
                min_norm, i_min_dist = find_min_norm(scaled_inputs, data_inputs)
                
                if min_norm<iteration_options.max_initial_step: #make sure data is close to current guess
                    if self.initial_step == 'Table' or min_norm<iteration_options.min_surrogate_step or len(data_outputs[:,0])< iteration_options.min_surrogate_length:
                        regr    = neighbors.KNeighborsRegressor( n_neighbors = 1)
                      
                    else:
                        print('running surrogate method')
                        if self.initial_step == 'SVR':
                            #for SVR, can optimize parameters C and eps for closest point
                            print('optimizing svr parameters')
                            x = [2.,-1.] #initial guess for 10**C, 10**eps
                        
                            out = sp.optimize.minimize(check_svr_accuracy, x, method='Nelder-Mead', args=(data_inputs, data_outputs, imin_dist))
                            t2=time.time()
                            c_out = 10**out.x[0]
                            eps_out = 10**out.x[1]
                            if c_out > 1E10:
                                c_out = 1E10
                            if eps_out<1E-8:
                                eps_out = 1E-8
            
                            regr        = svm.SVR(C=c_out,  epsilon = eps_out)
                            
                        elif self.initial_step == 'GradientBoosting':
                            regr        = ensemble.GradientBoostingRegressor()
                            
                        elif self.initial_step == 'ExtraTrees':
                            regr        = ensemble.ExtraTreesRegressor()
                        
                        elif self.initial_step == 'RandomForest':
                            regr        = ensemble.RandomForestRegressor()
                        
                        elif self.initial_step == 'Bagging':
                            regr        = ensemble.BaggingRegressor()
                            
                        elif self.initial_step == 'GPR':
                            gp_kernel_RQ = RationalQuadratic(length_scale=1.0, alpha=1.0)
                            regr        = gaussian_process.GaussianProcessRegressor(kernel=gp_kernel_RQ,normalize_y=True)
                            
                        elif self.initial_step == 'RANSAC':
                            regr        = linear_model.RANSACRegressor()
                        
                        elif self.initial_step == 'Neighbors':
                            n_neighbors = min(iteration_options.n_neighbors, len(data_outputs))
                            if iteration_options.neighbors_weighted_distance  == True:
                                regr    = neighbors.KNeighborsRegressor( n_neighbors = n_neighbors ,weights = 'distance')
                            
                            else:  
                                regr    = neighbors.KNeighborsRegressor( n_neighbors = n_neighbors)
                        
                        #now run the fits/guesses  
                    
                        iteration_options.number_of_surrogate_calls += 1
                    y = []    
                    input_for_regr = scaled_inputs.reshape(1,-1)
                    for j in range(len(data_outputs[0,:])):
                        y_surrogate = regr.fit(data_inputs, data_outputs[:,j])
                        y.append(y_surrogate.predict(input_for_regr)[0])    
                        if y[j] > self.max_y[j] or y[j]< self.min_y[j]: 
                            print('sizing variable range violated, val = ', y[j], ' j = ', j)
                            n_neighbors = min(iteration_options.n_neighbors, len(data_outputs))
                            regr_backup = neighbors.KNeighborsRegressor( n_neighbors = n_neighbors)
                            y = []
                            for j in range(len(data_outputs[0,:])):
                                y_surrogate = regr_backup.fit(data_inputs, data_outputs[:,j])
                                y.append(y_surrogate.predict(input_for_regr)[0])
                            break
                    y = np.array(y)
                   
        # initialize previous sizing values
        y_save   = 1*y  #save values to detect oscillation
        y_save2  = 3*y
        norm_dy2 = 1   #used to determine if it's oscillating; if so, do a successive_substitution iteration
        nr_start = 0 #flag to switch between methods; if you do nr too early, sizing diverges
        
        #now start running the sizing loop
        while np.max(np.abs(err))>tol:
            #save the previous iterations for backtracking
            iteration_options.err_save2 = 1.*np.array(iteration_options.err_save)
            iteration_options.err_save  = err
        
            if self.update_method == 'successive_substitution':
                err,y, i   = self.successive_substitution_update(y,err, sizing_evaluation, nexus, scaling, i, iteration_options)
                
            elif self.update_method == 'newton-raphson':
                if i==0:
                    nr_start=0  
                if np.max(np.abs(err))> self.iteration_options.newton_raphson_tolerance or np.max(np.abs(err))<self.iteration_options.max_newton_raphson_tolerance or i<self.iteration_options.min_fix_point_iterations:
                    err,y, i = self.successive_substitution_update(y,err, sizing_evaluation, nexus, scaling, i, iteration_options)

                else:          
                    if nr_start==0:
                        err,y, i   = self.newton_raphson_update(y_save2, err, sizing_evaluation, nexus, scaling, i, iteration_options)
                        nr_start   = 1
                    else:
                        err,y, i   = self.newton_raphson_update(y, err, sizing_evaluation, nexus, scaling, i, iteration_options)
                        nr_start   = 1
            
            elif self.update_method == 'broyden':
                
                if (np.max(np.abs(err))> self.iteration_options.newton_raphson_tolerance or np.max(np.abs(err))<self.iteration_options.max_newton_raphson_tolerance or i<self.iteration_options.min_fix_point_iterations) and nr_start ==0:
                    if i>1:  #obtain this value so you can get an ok value initialization from the Jacobian w/o finite differincing
                        err_save   = iteration_options.err_save
                    err,y, i   = self.successive_substitution_update(y,err, sizing_evaluation, nexus, scaling, i, iteration_options)
                    nr_start   = 0 #in case broyden update diverges
                    
                else:
                    
                    if nr_start==0:
                        if self.iteration_options.initialize_jacobian == 'newton-raphson':
                            err,y, i   = self.newton_raphson_update(y_save2, err, sizing_evaluation, nexus, scaling, i, iteration_options)
                        
                        
                        else:
                            #from http://www.jnmas.org/jnmas2-5.pdf
                            D                             = np.diag((y-y_save2)/(err-self.iteration_options.err_save))
                            self.iteration_options.y_save = y_save
                            self.iteration_options.Jinv   = D
                        
                            err,y, i   = self.broyden_update(y, err, sizing_evaluation, nexus, scaling, i, iteration_options)
                     
                        nr_start = 1
                        
                    else:
                        err,y, i   = self.broyden_update(y, err, sizing_evaluation, nexus, scaling, i, iteration_options)
                      
            y        = self.stay_inbounds(y_save, y)           
            dy       = y-y_save
            dy2      = y-y_save2
            norm_dy  = np.linalg.norm(dy)
            norm_dy2 = np.linalg.norm(dy2)
            if self.iteration_options.backtracking.backtracking_flag == True:
                err_save           = iteration_options.err_save
                backtracking       = iteration_options.backtracking
                back_thresh        = backtracking.threshhold
                i_back             = 0
                min_err_back       = 1000.
                y_back_list        = [y]
                err_back_list      = [err]
                norm_err_back_list = [np.linalg.norm(err)]
                
                while np.linalg.norm(err)>back_thresh*np.linalg.norm(err_save) and i_back<backtracking.max_steps  : #while?
                    print('backtracking')
                    print('err, err_save = ', np.linalg.norm(err), np.linalg.norm(err_save))
                    p                 = y-y_save
                    backtrack_y       = y_save+p*(backtracking.multiplier**(i_back+1))
                    err,y_back, i     = self.successive_substitution_update(backtrack_y, err, sizing_evaluation, nexus, scaling, i, iteration_options)
                    
                    y_back_list.append(backtrack_y)
                    err_back_list.append(err)
                    norm_err_back_list.append(np.linalg.norm(err))
                    min_err_back = min(np.linalg.norm(err_back_list), min_err_back)
                    i_back+=1
                
                i_min_back = np.argmin(norm_err_back_list)
                y          = y_back_list[i_min_back]
                err        = err_back_list[i_min_back]
        
            #keep track of previous iterations, as they're used to transition between methods + for saving results
            y_save2 = 1.*y_save
            y_save  = 1. *y  
    
            print('err = ', err)
            
            if self.write_residuals:  #write residuals at every iteration
                write_sizing_residuals(self, y_save, scaled_inputs, err)
        
            if i>max_iter: #
                print("###########sizing loop did not converge##########")
                break
        
        if i<max_iter and not np.isnan(err).any() and opt_flag == 1:  #write converged values to file
            converged = 1
            #check how close inputs are to what we already have        
            if converged and (min_norm>self.iteration_options.min_write_step or i>self.write_threshhold): #now output to file, writing when it's either not a FD step, or it takes a long time to converge
            #make sure they're in right format      
            #use y_save2, as it makes derivatives consistent
                write_sizing_outputs(self, y_save2, problem_inputs)
                
        nexus.total_number_of_iterations += i
        nexus.number_of_iterations = i #function calls
        results=nexus.results
        
        print('number of function calls=', i)
        print('number of iterations total=', nexus.total_number_of_iterations)

        nexus.sizing_loop.converged    = converged
        nexus.sizing_loop.norm_error   = np.linalg.norm(err)
        nexus.sizing_loop.max_error    = max(err)
        nexus.sizing_loop.output_error = err  #save in case you want to write this
        nexus.sizing_variables         = y_save2

        return nexus
        
    def successive_substitution_update(self,y, err, sizing_evaluation, nexus, scaling, iter, iteration_options):
        """
        Uses a successive substitution update to try to zero the residual
        """
        
        err_out, y_out = sizing_evaluation(y, nexus, scaling)
        iter += 1
        return err_out, y_out, iter
    
    def newton_raphson_update(self,y, err, sizing_evaluation, nexus, scaling, iter, iteration_options):
        """
        Finite differences the problem to calculate the Jacobian, then
        tries to use that to zero the residual
        """
        
        h = iteration_options.h
        print('###begin Finite Differencing###')
        J, iter = Finite_Difference_Gradient(y,err, sizing_evaluation, nexus, scaling, iter, h)
        try:
            
            Jinv     = np.linalg.inv(J)  
            p        = -np.dot(Jinv,err)
            y_update = y + p
            y_update = self.stay_inbounds(y, y_update)  #make sure bounds aren't exceeded

            err_out, y_out = sizing_evaluation(y_update, nexus, scaling)
            iter           += 1 
            
            #save these values in case of Broyden update
            iteration_options.Jinv     = Jinv 
            iteration_options.y_save   = y
           

            
        except np.linalg.LinAlgError:
            print('singular Jacobian detected, use successive_substitution')
            err_out, y_update, iter = self.successive_substitution_update(y, err, sizing_evaluation, nexus, scaling, iter, iteration_options)
        
       
        return err_out, y_update, iter
        
    def broyden_update(self,y, err, sizing_evaluation, nexus, scaling, iter, iteration_options):
        """
        uses an approximation to update the Jacobian without
        the use of finite differencing
        """
        y_save      = iteration_options.y_save
        err_save    = iteration_options.err_save2 
        dy          = y - y_save
        df          = err - err_save
        Jinv        = iteration_options.Jinv

        update_step = ((dy - Jinv*df)/np.linalg.norm(df))* df
        Jinv_out    = Jinv + update_step
    
        p                      = -np.dot(Jinv_out,err)
        y_update               = y + p
        err_out, y_out         = sizing_evaluation(y_update, nexus, scaling)
        #pack outputs
        iteration_options.Jinv     = Jinv_out
        iteration_options.y_save   = y
        iter                       = iter+1
        
        return err_out, y_update, iter
        
    def check_bounds(self, y):
        """
        checks if the corresponding y value violates the min or max bounds of the sizing loops:
        returns the violated bound if it does, along with a flag indicating this
        
        Inputs:
        y              [array]
        
        Outputs:
        y_out          [array]
        bound_violated [int]
        
        
        """
        y_out          = 1.*y #create copy
        bound_violated = 0
        for j in range(len(y)):  #handle variable bounds to prevent going to weird areas (such as negative mass)
            if self.hard_min_bound:
                if y[j]<self.min_y[j]:
                    y_out[j]       = self.min_y[j]*1.
                    bound_violated = 1
            if self.hard_max_bound:
                if y[j]>self.max_y[j]:
                    y_out[j]       = self.max_y[j]*1.
                    bound_violated = 1
        return y_out, bound_violated
    
    def stay_inbounds(self, y, y_update):
        """
        Checks if the corresponding y_update violates the bounds if so, performs a backtracking linesearch until
        y_update is within the preset bounds
        
        Inputs:
        y        [array]
        y_update [array]
        
        Outputs:
        y_update  [array]
        
        """
        sizing_evaluation     = self.sizing_evaluation
        scaling               = self.default_scaling
        p                     = y_update-y #search step

        y_out, bound_violated = self.check_bounds(y_update)

        backtrack_step        = self.iteration_options.backtracking.multiplier
        bounds_violated       = 1 #counter to determine how many bounds are violated
        while bound_violated:
            print('bound violated, backtracking')
            print('y_update, y_out = ',  y_update, y_out)
            bound_violated = 0
            for j in range(len(y_out)):
                if not np.isclose(y_out[j], y_update[j]) or np.isnan(y_update).any():
                    y_update        = y+p*backtrack_step
                    bounds_violated = bounds_violated+1
                    backtrack_step  = backtrack_step*.5
                    break

            y_out, bound_violated = self.check_bounds(y_update)
        return y_update
        

    __call__ = evaluate
    
## @ingroup Sizing    
def Finite_Difference_Gradient(x,f , my_function, inputs, scaling, iter, h):
    """
    Uses a first-order finite difference step to calculate the Jacobian
    
    Inputs:
    x               [array]
    f               [array]
    my_function     function that returns the residual f and sizing variable y-y_save
    inputs          ordered dict that is unpacked within my_function
    scaling         [array]
    iter            [int]
    h               [float]
    
    Outputs:
    J               [array,array]
    iter            [int]
    
    """

    J = np.nan*np.ones([len(x), len(x)])
    for i in range(len(x)):
        xu        = 1.*x;
        xu[i]     = x[i]+h *x[i]  #use FD step of H*x
        fu, y_out = my_function(xu, inputs,scaling)
        J[:,i]    = (fu-f)/(xu[i]-x[i])
        iter=iter+1
        
    return J, iter

def find_min_norm(scaled_inputs, data_inputs):
    """
    Finds the minimum and location of the L2 norm of two sets of data
    
    Inputs:
    scaled_inputs    [array]
    data_outputs     [array]
    
    Outputs:
    min_norm         [float]
    imin_dist        [int]
    """
    min_norm  = 1E9
    diff      = np.subtract(scaled_inputs, data_inputs) #check how close inputs are to tabulated values  
  
    imin_dist = -1 
    for k in range(len(diff[:,-1])):
        row      = diff[k,:]
        row_norm = np.linalg.norm(row)
        if row_norm < min_norm:
            min_norm  = row_norm
            imin_dist = k*1 
    
    return min_norm, imin_dist

        
        
        
    
#Sizing_Loop.py
#Created: Jun 2016, M. Vegh

from SUAVE.Core import Data
import numpy as np
import scipy.interpolate as interpolate
import sklearn.svm as svm
from write_sizing_outputs import write_sizing_outputs
from read_sizing_inputs import read_sizing_inputs

class Sizing_Loop(Data):
    def __defaults__(self):
        #parameters common to all methods
        self.tolerance             = None
        self.initial_step          = None  #'Default', 'Table', 'SVR'
        self.update_method         = None  #'fixed_point', 'newton-raphson'
        self.default_y             = None
        self.max_y                 = None  #maximum values for y to take (can be useful when large jumps happen)
        self.min_y                 = None  #minimum values for y to take
        self.default_scaling       = None  #scaling value to make sizing parameters ~1
        self.maximum_iterations    = None
        self.output_filename       = None
        self.function_evaluation   = None  #defined in the Procedure script
        self.write_threshhold      = 9     #number of iterations before it writes, regardless of how close it is to currently written values
        
        #parameters that may only apply to certain methods
        self.iteration_options     = Data()
        self.iteration_options.newton_raphson_tolerance     = 5E-2   #threshhold of convergence when you start using newton raphson
        self.iteration_options.max_newton_raphson_tolerance = 2E-4   #threshhold at which newton raphson is no longer used (to prevent overshoot and extra iterations)
        self.iteration_options.h                            = 1E-6   #finite difference step for Newton iteration
        self.iteration_options.max_initial_step             = 1.     #maximum distance at which interpolation is allowed
        self.iteration_options.min_fix_point_iterations     = 2      #minimum number of iterations to perform fixed-point iteration before starting newton-raphson
        self.iteration_options.min_svr_step                 = .011   #minimum distance at which SVR is used (if closer, table lookup is used)
        self.iteration_options.min_svr_length               = 4      #minimum number data points needed before SVR is used
        
    def evaluate(self, nexus):
        unscaled_inputs = nexus.optimization_problem.inputs[:,1] #use optimization problem inputs here
        input_scaling   = nexus.optimization_problem.inputs[:,3]
        scaled_inputs   = unscaled_inputs/input_scaling
        
        
        problem_inputs=[]
        for value in scaled_inputs:
            problem_inputs.append(value)  #writing to file is easier when you use list
        
        nexus.problem_inputs = problem_inputs
        print 'problem inputs=', problem_inputs
        #data_inputs, data_outputs, read_success = read_sizing_inputs(problem_inputs)
        
        
        #unpack inputs
        
        tol               = self.tolerance #percentage difference in mass and energy between iterations
        h                 = self.iteration_options.h 
        y                 = self.default_y
        max_iter          = self.maximum_iterations
        scaling           = self.default_scaling
        function_eval     = self.function_evaluation
        iteration_options = self.iteration_options
        err               = [1000] #initialize error
        
        #initialize
        converged         = 0     #marker to tell if it's converged
        j=0  #major iterations
        i=0  #function evals
        
        
        #determine the initial step
        min_norm =1000.
        if self.initial_step == 'Table' or self.initial_step == 'SVR':
            data_inputs, data_outputs, read_success = read_sizing_inputs(self, scaled_inputs)
            if read_success:
                #check how close inputs are to tabulated values             
                diff=np.subtract(scaled_inputs, data_inputs)#/input_scaling

                for row in diff:
                    row_norm = np.linalg.norm(row)
                    min_norm = min(min_norm, row_norm)
                    
                if min_norm<self.iteration_options.max_initial_step and len(data_inputs[0,:])>self.iteration_options.min_svr_length:  #make sure data is close to current guess
                
                    if self.initial_step == 'Table':
                        interp = interpolate.griddata(data_inputs, data_outputs, scaled_inputs, method = 'nearest') 
                        y = interp[0]
                
                    elif self.initial_step == 'SVR':
                        if min_norm>=self.iteration_options.min_svr_step and len(data_outputs[:,0]) > self.iteration_options.min_svr_length:
                            print 'running surrogate'
                            y = []
                            for j in range (len(data_outputs[0,:])):
                                clf         = svm.SVR(C=1E3,  epsilon = .01)
                                y_surrogate = clf.fit(data_inputs, data_outputs[:,j])
                                y.append(y_surrogate.predict(scaled_inputs)[0])
                            y = np.array(y)
                        else:
                            print 'running table'
                            interp = interpolate.griddata(data_inputs, data_outputs, scaled_inputs, method = 'nearest') 
                            y = interp[0]
        y_save  = 2*y  #save values to detect oscillation
        y_save2 = 3*y
        norm_dy2 = 1   #used to determine if it's oscillating; if so, do a fixed point iteration
        err = 1.
        
        #handle input data
        
        
        #now start running the sizing loop
        while np.max(np.abs(err))>tol:        
            if self.update_method == 'fixed_point':
                err,y, i   = self.fixed_point_update(y,err, function_eval, nexus, scaling, i, iteration_options)
                
            elif self.update_method == 'newton-raphson':
                if i==0:
                    nr_start=0
                if np.max(np.abs(err))> self.iteration_options.newton_raphson_tolerance or np.max(np.abs(err))<self.iteration_options.max_newton_raphson_tolerance or i<self.iteration_options.min_fix_point_iterations:
                    err,y, i   = self.fixed_point_update(y,err, function_eval, nexus, scaling, i, iteration_options)
                

                else:
                    if nr_start==0:
                        err,y, i   = self.newton_raphson_update(y_save2, err, function_eval, nexus, scaling, i, iteration_options)
                        nr_start=1
                    else:
                        err,y, i   = self.newton_raphson_update(y, err, function_eval, nexus, scaling, i, iteration_options)
                
           
           
       
            dy  = y-y_save
            dy2 = y-y_save2
            norm_dy  = np.linalg.norm(dy)
            norm_dy2 = np.linalg.norm(dy2)
            print 'norm(dy)=', norm_dy
            print 'norm(dy2)=', norm_dy2
    
        
            
            #save the previous input values
            y_save2 = 1.*y_save
            y_save = 1. *y  
            
            
            print 'err=', err
            
            j+=1
            if i>max_iter:
                
                print "###########maximum number of iterations exceeded##########"
                break
    
        if i<max_iter and not np.isnan(err).any():  #write converged values to file
            converged = 1
            #check how close inputs are to what we already have        
            print 'min_norm out=', min_norm
            
            if converged and (min_norm>self.iteration_options.min_svr_step or i>self.write_threshhold): #now output to file, writing when it's either not a FD step, or it takes a long time to converge
            #make sure they're in right format      
            #use y_save2, as it makes derivatives more consistent
                write_sizing_outputs(self, y_save2, problem_inputs)
                

        nexus.total_number_of_iterations += i
        nexus.number_of_iterations = i #function calls
        
        #nexus.mass_guess=mass
        results=nexus.results
        
    
        print 'number of function calls=', i
        print 'number of iterations total=', nexus.total_number_of_iterations

    
        nexus.converged = converged
        nexus.sizing_variables = y_save2
    
        
        return nexus
    def fixed_point_update(self,y, err, function_eval, nexus, scaling, iter, iteration_options):
        err, y_out = function_eval(y, nexus, scaling)
        iter += 1
        print 'y_out=', y_out
        print 'err_out=', err
        
        return err, y_out, iter
    
    def newton_raphson_update(self,y, err, function_eval, nexus, scaling, iter, iteration_options):
        h = iteration_options.h
        print '###begin Finite Differencing###'
        J, iter = Finite_Difference_Gradient(y,err, function_eval, nexus, scaling, iter, h)
        try:
            Jinv =np.linalg.inv(J)  
            p = -np.dot(Jinv,err)
            print 'Jinv=', Jinv
            print 'p=', p
            y_update = y + p
      
            
            '''
            for i in range(len(y_update)):  #handle variable bounds
                if y_update[i]<self.min_y[i]:
                    y_update[i] = self.min_y[i]*1.
                elif y_update[i]>self.max_y[i]:
                    y_update[i] = self.max_y[i]*1.
            '''
            err, y_out = function_eval(y_update, nexus, scaling)
            iter += 1 
            print 'err_out=', err
        except np.linalg.LinAlgError("Singular matrix"):
            print 'singular Jacobian detected, use fixed point'
            err, y_update, iter = fixed_point_update(self,y, err, function_eval, nexus, scaling, iter, iteration_options)
        
       
        return err, y_update, iter
    __call__ = evaluate
    

    


    
def Finite_Difference_Gradient(x,f , my_function, inputs, scaling, iter, h):
    #use forward difference
    #print 'type(x)= ', type(x)

    print 'h=', h
    #h=scaling[0]*.0001
    J=np.nan*np.ones([len(x), len(x)])
    for i in range(len(x)):
        xu=1.*x;
        xu[i]=x[i]+h *x[i]  #use FD step of H*x
        fu, y_out = my_function(xu, inputs,scaling)
        print 'fu=', fu
        print 'fbase=', f
        J[:,i] = (fu-f)/(xu[i]-x[i])
        iter=iter+1
        



    return J, iter



    
def interpolate_consistency_variables(self, nexus, data_inputs, data_outputs):
    unscaled_inputs= nexus.optimization_problem.inputs[:,1]
    input_scaling  = nexus.optimization_problem.inputs[:,3]
    opt_inputs     = unscaled_inputs/input_scaling
    diff=np.subtract(opt_inputs, data_inputs)#/input_scaling
    in_norm=1000.
    for row in diff:
        row_norm = np.linalg.norm(row)
        min_norm = min(min_norm, row_norm)    
        
    if len(data_inputs[:,0])>4 and min_norm>.011:
        print 'running surrogate'
        x = []
        for j in range(len(data_outputs[0,:])):
            clf = svm.SVR(C=1E4)
            y_surrogate = clf.fit(data_inputs, data_outputs[:,j])
            y.append(y_surrogate.predict(opt_inputs)[0])
   
        y = np.array(y)

    else:
        print 'running table'
        interp = interpolate.griddata(data_inputs , data_outputs, opt_inputs, method='nearest')
        x = interp[0] #use correct data size
 
    
    
    
    
        
        
        
        
    
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
        self.initial_step          = None  #None, 'Table', 'SVR'
        self.update_method         = None  #'fixed_point', 'newton-raphson'
        self.default_y             = None
        self.default_scaling       = None  #scaling value to make sizing parameters ~1
        self.maximum_iterations    = None
        self.output_filename       = None
        self.function_evaluation   = None  #defined in the Procedure script
        self.write_threshhold      = 9     #number of iterations before it writes, regardless of how close it is to currently written values
        
        #parameters that may only apply to certain methods
        self.iteration_options     = Data()
        self.iteration_options.newton_raphson_tolerance = 5E-2   #threshhold of convergence when you start using newton raphson
        self.iteration_options.h                        = 1E-6   #finite difference step for Newton iteration
        self.iteration_options.max_initial_step         = 1.     #maximum distance at which interpolation is allowed
        self.iteration_options.min_fix_point_iterations = 2      #minimum number of iterations to perform fixed-point iteration before starting newton-raphson
        self.iteration_options.min_svr_step             = .011   #minimum distance at which SVR is used (if closer, table lookup is used)
        
        
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
        
        if self.initial_step == 'Table' or self.initial_step == 'SVR':
            data_inputs, data_outputs, read_success = read_sizing_inputs(self, scaled_inputs)
            if read_success:
                #check how close inputs are to tabulated values             
                diff=np.subtract(scaled_inputs, data_inputs)#/input_scaling
                
                min_norm=1000.
                for row in diff:
                    row_norm = np.linalg.norm(row)
                    min_norm = min(min_norm, row_norm)
                    
                if min_norm<self.iteration_options.max_initial_step:  #make sure data is close to 
                
                    if self.initial_step == 'Table':
                        interp = interpolate.griddata(data_inputs, data_outputs, scaled_inputs, method = 'nearest') 
                        y = interp[0]
                
                    elif self.initial_step == 'SVR':
                        if min_norm>=self.iteration_options.min_svr_step:
                            print 'running surrogate'
                            y = []
                            for j in range (len(data_outputs[0,:])):
                                clf = svm.SVR(C=1E4)
                                y_surrogate = clf.fit(data_inputs, data_outputs[:,j])
                                y.append(y_surrogate.predict(scaled_inputs_inputs)[0])
                            y = np.array(y)
                        else:
                            print 'running table'
                            interp = interpolate.griddata(data_inputs, data_outputs, scaled_inputs, method = 'nearest') 
                            y = interp[0]
        y_save  = 2*y  #save values to detect oscillation
        y_save2 = 3*y
        norm_dy2 = 1   #used to determine if it's oscillating; if so, do a fixed point iteration
        err = 1.
        min_norm =1000.
        #handle input data
        
        
        #now start running the sizing loop
        while np.max(np.abs(err))>tol:        
            if self.update_method == 'fixed_point':
                err,y, i   = fixed_point_update(y,err, function_eval, nexus, scaling, i, iteration_options)
            
            elif self.update_method == 'newton-raphson':
                if np.max(np.abs(err))> self.iteration_options.newton_raphson_tolerance or i<self.iteration_options.min_fix_point_iterations:
                    err,y, i   = fixed_point_update(y, err,function_eval, nexus, scaling, i, iteration_options)
                
                else:
                    err,y, i   = newton_raphson_update(y, err, function_eval, nexus, scaling, i, iteration_options)
                
           
       
            dy  = y-y_save
            dy2 = y-y_save2
            norm_dy  = np.linalg.norm(dy)
            norm_dy2 = np.linalg.norm(dy2)
            print 'norm(dy)=', norm_dy
            print 'norm(dy2)=', norm_dy2
    
        
            
            #save the input values, in the hopes of detecting oscillation
            y_save2 = 1.*y_save
            y_save = 1. *y  
            
            
            print 'err=', err
            
            j+=1
            if i>max_iter:
                
                print "###########maximum number of iterations exceeded##########"
                break
    
        if i<max_iter and not np.isnan(err.any()):  #write converged values to file
            converged = 1
            #check how close inputs are to what we already have        
            print 'min_norm out=', min_norm
            
            
            if min_norm>self.iteration_options.min_svr_step or i>9: #now output to file, writing when it's either not a FD step, or it takes a long time to converge
            #make sure they're in right format 
            
                write_sizing_outputs(self, y_save, problem_inputs)
                
        elif np.linalg.norm(err)>1E-2:  #give a nan only when it's diverging, or looks like it won't converge
            results.segments[-1].conditions.weights.total_mass[-1,0] = float('nan')
        
        nexus.total_number_of_iterations += i
        nexus.number_of_iterations = i #function calls
        
        #nexus.mass_guess=mass
        results=nexus.results
        
    
        print 'number of function calls=', i
        print 'number of iterations total=', nexus.total_number_of_iterations

    
        nexus.converged = converged
        nexus.sizing_variables = y
    
        
        return nexus
    __call__ = evaluate
    
def fixed_point_update(y, err, function_eval, nexus, scaling, iter, iteration_options):
    err, y_out = function_eval(y, nexus, scaling)
    iter += 1
    return err, y_out, iter
    
def newton_raphson_update(y, err, function_eval, nexus, scaling, iter, iteration_options):
    h = iteration_options.h
    J, iter = Finite_Difference_Gradient(y,err, function_eval, nexus, scaling, iter, h)
    Jinv =np.linalg.inv(J)
    p = -np.dot(Jinv,err)
    y_update = y + p
    err, y_out = function_eval(y_update, nexus, scaling)
    iter += 1 
    return err, y_update, iter
    


    
def Finite_Difference_Gradient(x,f , my_function, inputs, scaling, iter, h):
    #use forward difference
    #print 'type(x)= ', type(x)

    
    #h=scaling[0]*.0001
    J=np.nan*np.ones([len(x), len(x)])
    for i in range(len(x)):
        xu=1.*x;
        xu[i]=x[i]+h *x[i]  #use FD step of H*x
        fu, y_out = my_function(xu, inputs,scaling)
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
 
    
    
    
    
        
        
        
        
    
#Sizing_Loop.py
#Created: Jun 2016, M. Vegh

from SUAVE.Core import Data
import numpy as np
import scipy.interpolate as interpolate
import sklearn.svm as svm

class Sizing_Loop(Data):
    def __defaults__(self):
        self.tolerance           = None
        self.update_method       = None
        self.default_y           = None
        self.default_scaling     = None  #scaling value to make sizing parameters ~1
        self.maximum_iterations  = None
        self.output_filename     = None
        self.function_evaluation = None  #defined in the Procedure script
        self.write_threshhold    = 9     #number of iterations before it writes, regardless of how close it is to currently written values
        self.iteration_options   = Data()
        self.iteration_options.h = 1E-6  #finite difference step for Newton iteration
    
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
        
        #use interpolated data if you can
      
        #if read_success:
       
      
        y_save  = 2*y  #some value to prevent
        y_save2 = 3*y
        norm_dy2 = 1   #used to determine if it's oscillating; if so, do a fixed point iteration
        err = 1.
        #handle input data
        
        
        
        while np.any(abs(err))>tol:
            if self.update_method == 'fixed_point':
                y, err, i   = fixed_point_update(y, function_eval, nexus, scaling, iteration_options)
            
            #do newton-raphson if within the specified tolerance to speed up convergence
        
       
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
            
            
            if min_norm>.011 or i>9: #now output to file, writing when it's either not a FD step, or it takes a long time to converge
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
    
def fixed_point_update(y, function_eval, nexus, scaling, iteration_options):
    err, y_out = function_eval(y, nexus, scaling)
    i = i+1
    return err, y_out
    
def newton_update(y, function_eval, nexus, scaling, iteration_options):
    i = iteration_options.i
    h = iteration_options.h
    y_update, Jinv_out, i =  Finite_Difference_Gradient(y,  function_eval, inputs, scaling, iter, h)
    iteration_options.i +=1
    err, y_out = function_eval(y_out, nexus, scaling)
    return err, y_update
    
def newton_raphson_iter(y, my_function, nexus, scaling, iter, h=1E-6):
    #f is a vector of size m data points
    #x is a matrix of input variables
    #assumes you have enough data points to finite difference Hessian and gradient
    alpha0=1
    print '############Begin Finite Difference############'

    f, J, iter =Finite_Difference_Gradient(y, my_function, nexus, scaling, iter, h)
    print '############End Finite Difference############'
    
    
    print 'J=', J
    print 'y=', x
    
    #x_out = x - f/g
    #f_out = my_function(x_out, inputs, scaling)
    Jinv   = np.linalg.inv(J)
    p      = -np.dot(Jinv,f)
    
    
    y_out  = y + p
    #f_out  = my_function(x_out, inputs, scaling)
    #iter  += 1
    
    #f_out,x_out, iter = line_search(x, f, J, alpha0, p, my_function, inputs, scaling, iter)
    print 'y_out=', y_out


    
    return  y_out, Jinv, iter
    

    
def Finite_Difference_Gradient(x,  my_function, inputs, scaling, iter, h):
    #use forward difference
    #print 'type(x)= ', type(x)
    f = my_function(x, inputs, scaling)
    iter = iter+1
    
    #h=scaling[0]*.0001
    J=np.nan*np.ones([len(x), len(x)])
    for i in range(len(x)):
        xu=1.*x;
        xu[i]=x[i]+h *x[i]  #use FD step of H*x
        fu = my_function(xu, inputs,scaling)
        J[:,i] = (fu-f)/(xu[i]-x[i])
        iter=iter+1
        



    return f, J, iter



    
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
 
    
    
    
    
        
        
        
        
    
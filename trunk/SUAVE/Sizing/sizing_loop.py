#Sizing_Loop.py
#Created: Jun 2016, M. Vegh

from SUAVE.Core import Data

class Sizing_Loop(Data):
    def __defaults__(self):
    self.tolerance          = None
    self.update_method      = None
    self.default_y          = None
    self.default_scaling    = None
    self.maximum_iterations = None
    
    def evaluate(nexus):
        vehicle    = nexus.vehicle_configurations.base
        configs    = nexus.vehicle_configurations
        analyses   = nexus.analyses
        mission    = nexus.missions.base
        
        
        unscaled_inputs = nexus.optimization_problem.inputs[:,1] #use optimization problem inputs here
        input_scaling   = nexus.optimization_problem.inputs[:,3]
        scaled_inputs   = unscaled_inputs/input_scaling
        
        
        problem_inputs=[]
        for value in scaled_inputs:
            problem_inputs.append(value)  #writing to file is easier when you use list
        
        nexus.problem_inputs = problem_inputs
        print 'problem inputs=', problem_inputs
        
    
    
        #converge numbers
        converged         = 0     #marker to tell if it's converged
        tol               = self.tolerance #percentage difference in mass and energy between iterations
        #h                 = 1E-6
        #nr_tol            = 5E-2 #threshold to start newton raphson
        #osc_tol           = 1E-5 #tolerance to not use Newton Raphson when it's oscillating
        #fixed_point_force = 0 #flag to force fixed point iteration in case of oscillation
        err               = 1000 #initialize error
        dm                = 10000. 
        dE                = 10000.
        dE_primary        = 1.
        dE_auxiliary      = 1.
        dE_total          = 1.
        max_iter          = self.maximum_iterations
        
        if self.update_method = 'fixed_point':
            update_inputs = fixed_point_update
    
        #nexus.tol         = tol
        '''
        primary_battery   = configs.base.energy_network['primary_battery']
        auxiliary_battery = configs.base.energy_network['auxiliary_battery']
        ducted_fan        = configs.base.propulsors['ducted_fan']
        
        #make it so all configs handle the exact same battery object
        configs.cruise.energy_network['primary_battery']   =primary_battery 
        configs.takeoff.energy_network['primary_battery']  =primary_battery
        configs.landing.energy_network['primary_battery']  =primary_battery
        configs.cruise.energy_network['auxiliary_battery'] =auxiliary_battery
        configs.takeoff.energy_network['auxiliary_battery']=auxiliary_battery
        configs.landing.energy_network['auxiliary_battery']=auxiliary_battery
        
        nexus.mass_guess=[]
        '''
        j=0  #major iterations
        i=0  #function evals
        
        #use interpolated data if you can
        iteration_options   = Data()
        iteration_options.i = i
        iteration_options.h = h
        y = self.default_y
        scaling = self.default_scaling
        x = x/scaling
        y_save  = 2*y  #some value to prevent
        y_save2 = 3*y
        norm_dy2 = 1   #used to determine if it's oscillating; if so, do a fixed point iteration
        err = 1.
        #handle input data
        try:
            unscaled_inputs= nexus.optimization_problem.inputs[:,1]
            input_scaling  = nexus.optimization_problem.inputs[:,3]
            opt_inputs     = unscaled_inputs/input_scaling
            
            file_in    = open('x_values_%(range)d_nautical_mile_range.txt' %{'range':nexus.target_range/Units.nautical_miles})
            
            #read data from previous iterations
            data=file_in.readlines()
            file_in.close()
            data=format_input_data(data) #format data so we can work with it
            
            
            data_inputs = data[:, 0:len(opt_inputs)]  #values from optimization problem
            data_outputs= data[:,len(opt_inputs):16]  #variables we iterate on in sizing loop
    
            
            diff=np.subtract(opt_inputs, data_inputs)#/input_scaling
        
            min_norm=1000.
            for row in diff:
                row_norm = np.linalg.norm(row)
                min_norm = min(min_norm, row_norm)
            print 'min_norm=', min_norm
            x_out = interpolate_consistency_variables(opt_inputs, data_inputs, data_outputs, min_norm)
            #see how close data is to current optimization variables
            print 'x_out=', x_out
        
            x = x_out
            '''
            if min_norm<1:  #give a threshold to hopefully prevent divergence
                x = x_out
            else:
                print 'tabulated values not close enough; use default iteration variables'
            '''
        except:
            print 'no data to read; use default iteration variables'
            min_norm=1.
    
            
    
            
        while np.any(abs(err))>tol:
            if self.update_method = 'fixed_point':
                y, err, i   = fixed_point_update(y, function_eval, nexus, scaling, iteration_options)
            
            #do newton-raphson if within the specified tolerance to speed up convergence
        
            soc_aux            = auxiliary_battery.current_energy[-1]/auxiliary_battery.max_energy
            
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
            
                file=open('x_values_%(range)d_nautical_mile_range.txt' %{'range':nexus.target_range/Units.nautical_miles}, 'ab')
            
                file.write(str(problem_inputs))
                file.write(' ')
            
                file.write(str(x_save.tolist()))
                file.write('\n') 
                file.close()
        elif np.linalg.norm(err)>1E-2:  #give a nan only when it's diverging, or looks like it won't converge
            results.segments[-1].conditions.weights.total_mass[-1,0] = float('nan')
        
        nexus.total_number_of_iterations += i
        nexus.number_of_iterations = i #function calls
        
        #nexus.mass_guess=mass
        results=nexus.results
        
    
        print 'number of function calls=', i
        print 'number of iterations total=', nexus.total_number_of_iterations
    
        
        
        nexus.results=results
        nexus.vehicle_configurations=configs
        #now find operating costs
        nexus=evaluate_cost(nexus)
        nexus=compute_volumes(nexus)
        nexus.converged = converged
        nexus.sizing_variables = y
    
        
        return nexus
        
    def fixed_point_update(y, function_eval, nexus, scaling, iteration_options):
        err, y_out = function_eval(y, nexus_scaling):
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
        
    
    __call__ = evaluate
    
    
        
        
        
        
    
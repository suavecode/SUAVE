import numpy as np
import copy
import SUAVE

from SUAVE.Core import Units, Data
from SUAVE.Optimization import helper_functions as help_fun

class Greedy_Optimization(Data):
        
    def __defaults__(self):
        
        self.tag                                = 'Greedy_Opt'
        self.gradients                          = 'FD'
        self.trust_region_max_iterations        = 30
        self.optimizer_max_iterations           = 30
        #self.max_optimizer_function_evaluations = 1000
        
        self.soft_convergence_tolerance         = 1E-6
        self.soft_convergence_limit             = 3
        self.hard_convergence_tolerance         = 1E-6
        self.optimizer_convergence_tolerance    = 1E-6  #used in SNOPT
        self.optimizer_constraint_tolerance     = 1E-6  #used in SNOPT only
        self.difference_interval                = 1E-6  #used in evaluating high fidelity case
        self.optimizer_function_precision       = 1E-12 #used in SNOPT only
        self.function_precision                 = 1E-12
        self.optimizer_verify_level             = 0
        self.fidelity_levels                    = 2  
        self.evaluation_order                   = [1,2]
        self.mutual_setup_step                  = 0
        self.function_dependency                = 0
        self.number_cores                       = 1        
        self.iteration_index                    = 0
        self.trust_region_center_index          = 0
        self.trust_region_center                = None
        self.shared_data_index                  = 0 
        self.truth_history                      = dict() # history for truth function evaluations
        self.surrogate_history                  = dict() # history for evaluation of surrogate models (all fidelity levels)
        self.trust_region_history               = None
        self.number_truth_evals                 = dict()
        self.number_duplicate_truth_evals       = 0
        self.number_surrogate_evals             = 0
        self.user_data                          = Data()
        self.root_directory                     = None
        self.objective_history                  = []
        self.constraint_history                 = []
        self.relative_difference_history        = []
        self.derivative_flag                    = False #tells you whether to evaluate derivatves
        
    def optimize(self,problem):
        inp = problem.optimization_problem.inputs
        obj = problem.optimization_problem.objective
        con = problem.optimization_problem.constraints 
  
        # Set inputs
        nam = inp[:,0] # Names
        ini = inp[:,1] # Initials
        bnd = inp[:,2] # Bounds
        scl = inp[:,3] # Scale
        typ = inp[:,4] # Type
    
        # Pull out the constraints and scale them
        bnd_constraints = help_fun.scale_const_bnds(con)
        scaled_constraints = help_fun.scale_const_values(con,bnd_constraints)

        x   = ini/scl        
        # need to make this into a vector of some sort that can be added later
        lbd  = []#np.zeros(np.shape(bnd[:][1]))
        ubd  = []#np.zeros(np.shape(bnd[:][1]))
        edge = []#np.zeros(np.shape(bnd[:][1]))
        name = []#[None]*len(bnd[:][1])
        up_edge  = []
        low_edge = []
        
        
        #bnd[1000]
        for ii in xrange(0,len(inp)):
            lbd.append(bnd[ii][0]/scl[ii])
            ubd.append(bnd[ii][1]/scl[ii])

        for ii in xrange(0,len(con)):
            name.append(con[ii][0])
            edge.append(scaled_constraints[ii])
            if con[ii][1]=='<':
                up_edge.append(edge[ii])
                low_edge.append(-np.inf)
            elif con[ii][1]=='>':
                up_edge.append(np.inf)
                low_edge.append(edge[ii])
                
            elif con[ii][1]=='=':
                up_edge.append(edge[ii])
                low_edge.append(edge[ii])
            
        lbd = np.array(lbd)
        ubd = np.array(ubd)
        edge = np.array(edge)
        up_edge  = np.array(up_edge)         
        low_edge = np.array(low_edge)    
        
        # ---------------------------
        # Trust region specific code
        # ---------------------------
        
        iterations = 0
        max_iterations = self.trust_region_max_iterations

        x = np.array(x,dtype='float')

        x_initial = x*1.
        g_violation = 0 # pre subproblem constraint violation
        g_violation2_hi = 0 # post subproblem constraint violation (high fidelity)
        g_violation2_lo = 0 # post subproblem constraint violation (low fidelity)
     
        xOpt = np.zeros(np.shape(x))
        fOpt = None
        gOpt = np.zeros(np.shape(scaled_constraints))
        
        while iterations < max_iterations:
            iterations += 1
            self.increment_iteration()


            
            
            
            f    = [None]*self.fidelity_levels
            df   = [None]*self.fidelity_levels
            g    = [None]*self.fidelity_levels
            dg   = [None]*self.fidelity_levels            
            
            for level in self.evaluation_order:
                problem.fidelity_level = level
                der_flag = self.derivative_flag
                res = self.evaluate_model(problem,x,scaled_constraints, der_flag = der_flag )
                if der_flag == True:
                    f[level-1]  = res[0]    # objective value
                    df[level-1] = res[1]    # objective derivate vector
                    g[level-1]  = res[2]    # constraints vector
                    dg[level-1] = res[3]    # constraints jacobian
                else:
                    f[level-1]  = res[0]
                    g[level-1]  = res[1]
            if iterations == 0:
                self.objective_history.append(f[0])
                self.constraint_history.append(g[0])

            # Setup SNOPT 
            #opt_wrap = lambda x:self.evaluate_corrected_model(problem,x,corrections=corrections,tr=tr)
            import pyOpt
            opt_prob = pyOpt.Optimization('SUAVE',self.evaluate_corrected_model)
            
            for ii in xrange(len(obj)):
                opt_prob.addObj('f',f[-1]) 
            for ii in xrange(0,len(inp)):
                vartype = 'c'

                opt_prob.addVar(nam[ii],vartype,lower=lbd[ii],upper=ubd[ii],value=x[ii])    
            for ii in xrange(0,len(con)):
                if con[ii][1]=='<':
                    opt_prob.addCon(name[ii], type='i', upper=edge[ii])
                    
                elif con[ii][1]=='>':
                    opt_prob.addCon(name[ii], type='i', lower=edge[ii],upper=np.inf)
               
                    
                elif con[ii][1]=='=':
                    opt_prob.addCon(name[ii], type='e', equal=edge[ii])      
                    
               
            opt = pyOpt.pySNOPT.SNOPT()
            #opt.max_iterations = 15
            #opt.max_function_evaluations = 300
            #opt.setOption('Major iterations limit',opt.max_iterations)
            
            #CD_step = (sense_step**2.)**(1./3.)  #based on SNOPT Manual Recommendations
            #opt.setOption('Function precision', sense_step**2.)
            #opt.setOption('Difference interval', sense_step)
            #opt.setOption('Central difference interval', CD_step)         
           
            opt.setOption('Major iterations limit'     , self.optimizer_max_iterations)
            opt.setOption('Major optimality tolerance' , self.optimizer_convergence_tolerance)
            opt.setOption('Major feasibility tolerance', self.optimizer_constraint_tolerance)
            opt.setOption('Function precision'         , self.optimizer_function_precision)
            opt.setOption('Verify level'               , self.optimizer_verify_level) 

            problem.fidelity_level = 1
            if self.gradients == 'FD':
                outputs = opt(opt_prob, sens_type='FD',problem=problem)#, sens_step = sense_step)  
            else:
                outputs = opt(opt_prob, sens_type=self.problem_derivative_model , problem=problem)
                #outputs = opt(opt_prob, sens_type=problem.finite_difference,corrections=corrections,tr=tr)
            
            fOpt_lo = outputs[0]
            xOpt_lo = outputs[1]
            gOpt_lo = np.zeros([1,len(con)])[0]
            
          
           
            for ii in xrange(len(con)):
                gOpt_lo[ii] = opt_prob._solutions[0]._constraints[ii].value
       
            g_violation_opt_lo = self.calculate_constraint_violation(gOpt,low_edge,up_edge)
            
            success_indicator = outputs[2]['value'][0]
            # hard convergence check
            if (success_indicator==1 and np.sum(np.isclose(xOpt_lo,xOpt,rtol=1e-14,atol=1e-12))==len(x)):
                print 'Hard convergence reached'
                return outputs
            print 'fOpt_lo = ', fOpt_lo
            print 'xOpt_lo = ', xOpt_lo
            print 'gOpt_lo = ', gOpt_lo
            #x = 1.*xOpt_lo #restart problem from previous optimum
                       
            # Evaluate high-fidelity at optimum (including derivatives)
            problem.fidelity_level = np.max(self.fidelity_levels)
            if der_flag == True:
                fOpt_hi, gOpt_hi, dfOpt_hi, dgOpt_hi = self.evaluate_model(problem,xOpt_lo,scaled_constraints,der_flag=der_flag)
            else:
                fOpt_hi, gOpt_hi = self.evaluate_model(problem,xOpt_lo,scaled_constraints,der_flag=der_flag)
            
            
            
            
            self.objective_history.append(fOpt_hi)
            self.constraint_history.append(gOpt_hi)
            
            g_violation_opt_hi = self.calculate_constraint_violation(gOpt_hi,low_edge,up_edge)

            problem.fidelity_level = 2
 
            # Soft convergence test
            if( np.abs(fOpt_hi) <= self.function_precision and np.abs(f[-1]) <= self.function_precision ):
                relative_diff = 0
                
            elif( np.abs(fOpt_hi) <= self.function_precision):
                relative_diff = (fOpt_hi - f[-1])/f[-1]
            else:
                relative_diff = (fOpt_hi - f[-1])/fOpt_hi
                
            self.relative_difference_history.append(relative_diff)
            diff_hist = self.relative_difference_history
            
            ind1 = max(0,iterations-1-self.soft_convergence_limit)
            ind2 = len(diff_hist) - 1
            converged = 1
            while ind2 >= ind1:
                if( np.abs(diff_hist[ind1]) > self.soft_convergence_tolerance ):
                    converged = 0
                    break
                ind1 += 1
            if( converged and len(self.relative_difference_history) >= self.soft_convergence_limit):
                print 'Soft convergence reached'
                return outputs     
            
            # Acceptance Test
            accepted = 0
            if( fOpt_hi < f[-1] ):
                print 'update accepted since objective value is lower\n'
                accepted = 1
        
            else:
                print 'Update rejected (filter)\n'        
          
            
            
            '''
            # Terminate if solution is infeasible, no change is detected, and trust region does not expand
            if( success_indicator == 13 and tr_action < 3 and \
                np.sum(np.isclose(xOpt,x,rtol=1e-15,atol=1e-14)) == len(x) ):
                print 'Solution infeasible, no improvement can be made'
                return -1       
            '''
            
            print iterations
            x_diff = xOpt_lo-xOpt
            print 'x_diff = ', x_diff
            print 'xOpt_lo = ', xOpt_lo
            print 'xOpt = ', xOpt
            xOpt = xOpt_lo*1.
            if np.linalg.norm(x_diff)< self.soft_convergence_tolerance: 
                print 'soft convergence reached'
                return (fOpt,xOpt_lo)
           
           
            #update
            
           
            aa = 0
            
        print 'Max iteration limit reached'
        return (fOpt,xOpt)
            
        
    def evaluate_model(self,problem,x,cons,der_flag=True):
        #duplicate_flag, obj, gradient = self.check_for_duplicate_evals(x)
        duplicate_flag = False
        f  = np.array(0.)
        g  = np.zeros(np.shape(cons))
        df = np.zeros(np.shape(x))
        dg = np.zeros([np.size(cons),np.size(x)])
        
        
        f  = problem.objective(x)
        g  = problem.all_constraints(x)
        if der_flag == False:
            return f,g
        df, dg = problem.evaluate_derivatives(x)
        #problem.evaluate_d
        
        '''
        if problem.fidelity_level == 2:
            #let SUAVE handle it, including gradients
            problem.finite_difference_step = self.difference_interval    
            df, dg, flag = problem.finite_difference(x)
        
        elif problem.fidelity_level == 1:

        # build derivatives
        
            fd_step = self.difference_interval
            
            for ii in xrange(len(x)):
                x_fd = x*1.
                x_fd[ii] = x_fd[ii] + fd_step
                obj = problem.objective(x_fd)
                grad_cons = problem.all_constraints(x_fd)
    
                df[ii] = (obj - f)/fd_step
    
                for jj in xrange(len(cons)):
                    
                    dg[jj,ii] = (grad_cons[jj] - g[jj])/fd_step   
        
                         
        '''
        return (f,df,g,dg)
    def check_for_duplicate_evals(self, problem, x):
        
        return None
    def evaluate_corrected_model(self,x,problem=None):
        #duplicate_flag, obj, gradient = self.check_for_duplicate_evals(x)
        duplicate_flag = False
        if duplicate_flag == False:
            obj   = problem.objective(x)
            const = problem.all_constraints(x).tolist()
            #const = []
            fail  = np.array(np.isnan(obj) or np.isnan(np.array(const).any())).astype(int)
            

            
            obj   = obj 

        
            print 'Inputs'
            print x
            print 'Obj'
            print obj
            print 'Con'
            print const
            
        return obj,const,fail
        
  
    def problem_derivative_model(self,x, f= None, g = None,problem=None): #f and g not necessary here
     

        duplicate_flag = False
        if duplicate_flag == False:
            df, dg = problem.evaluate_derivatives(x)
            df = df.tolist()
            dg = dg.tolist()
  
            return df, dg, 0
          
        
            
        
    def calculate_constraint_violation(self,gval,lb,ub):
        gdiff = []
  
        for i in range(len(gval)):
            if len(lb) > 0:
                if( gval[i] < lb[i] ):
                    gdiff.append(lb[i] - gval[i])
            if len(ub) > 0:    
                if( gval[i] > ub[i] ):
                    gdiff.append(gval[i] - ub[i])
    
        return np.linalg.norm(gdiff) # 2-norm of violation  
    
    def calculate_correction(self,f,df,g,dg,tr):
        print 'f = ', f
        print 'df = ', df
        print 'g = ', g
        print 'dg = ', dg
        nr = 1 + len(g[0])
        nc = len(df[0])
        
        A = np.empty((nr,nc))
        b = np.empty(nr)
            
        # objective correction
        #handle data typing
        A[0,:] = np.array(df[1]) - np.array(df[0])
        b[0] = np.array(f[1]) - np.array(f[0])
            
        # constraint corrections
        A[1:,:] = np.array(dg[1]) - np.array(dg[0])
        b[1:] = np.array(g[1]) - np.array(g[0])
            
        corr = (A,b)
        print 'corr = ', corr

        return corr        
        
    def add_user_data(self,value):
        raise NotImplementedError
    
    def assign_to_history(self,tag,x,val):
        self.truth_history[(tag,x)] = val
        try:
            self.number_truth_evals[tag] += 1
        except KeyError:
            self.number_truth_evals[tag] = 1
            
    def assign_to_trust_region_history(self):
        raise NotImplementedError
    
    def assign_to_surrogate_history(self):
        raise NotImplementedError
        
    def get_shared_data_index(self):
        return self.shared_data_index
    
    def get_trust_region_center(self):
        return self.trust_region_center
    
    def get_trust_region_center_index(self):
        return self.trust_region_center_index
    
    def get_user_data(self,key):
        return self.user_data[key]
    
    def increment_iteration(self):
        self.iteration_index += 1
        
    def increment_shared_data_index(self):
        self.shared_data_index += 1
        
    def increment_trust_region_center_index(self):
        self.trust_region_center_index += 1
        
    # do not allow revert user data update
    
    def set_trust_region_center(center):
        # center should be a numpy array
        self.trust_region_center = center
    
    def setup_history():
        raise NotImplementedError
    
    def setup_surrogate_history():
        raise NotImplementedError
import numpy as np
import copy
import SUAVE
import Trust_Region
import pyOpt
from SUAVE.Core import Units, Data
from SUAVE.Optimization import helper_functions as help_fun

class Trust_Region_Optimization(Data):
        
    def __defaults__(self):
        
        self.tag                                = 'TR_Opt'
        self.trust_region_max_iterations        = 30
        self.optimizer_max_iterations           = 30
        #self.max_optimizer_function_evaluations = 1000
        
        self.soft_convergence_tolerance         = 1E-6
        self.hard_convergence_tolerance         = 1E-6
        self.optimizer_convergence_tolerance    = 1E-6  #used in SNOPT
        self.optimizer_constraint_tolerance     = 1E-6  #used in SNOPT only
        self.difference_interval                = 1E-6  #used in evaluating high fidelity case
        self.optimizer_function_precision       = 1E-12 #used in SNOPT only
        self.trust_region_function_precision    = 1E-12
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
        
    def optimize(self,problem):
        inp = problem.optimization_problem.inputs
        obj = problem.optimization_problem.objective
        con = problem.optimization_problem.constraints 
        tr = problem.trust_region
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
        #tr = Trust_Region.Trust_Region()
        #tr.initialize()
        x = np.array(x,dtype='float')
        tr.set_center(x)
        tr_size = tr.size

        trc = x # trust region center
        x_initial = x*1.
        g_violation = 0 # pre subproblem constraint violation
        g_violation2_hi = 0 # post subproblem constraint violation (high fidelity)
        g_violation2_lo = 0 # post subproblem constraint violation (low fidelity)
        tr_index = self.trust_region_center_index
        
        while iterations < max_iterations:
            iterations += 1
            self.increment_iteration()
            #self.assign_to_trust_region_history(iterations,tr_index,trc,tr_size)
            tr.set_center(x)
            
            xOpt = np.zeros(np.shape(x))
            fOpt = None
            gOpt = np.zeros(np.shape(scaled_constraints))
            f    = [None]*self.fidelity_levels
            df   = [None]*self.fidelity_levels
            g    = [None]*self.fidelity_levels
            dg   = [None]*self.fidelity_levels            
            
            for level in self.evaluation_order:
                problem.fidelity_level = level
                res = self.evaluate_model(problem,x,scaled_constraints)
                f[level-1]  = res[0]    # objective value
                df[level-1] = res[1]    # objective derivate vector
                g[level-1]  = res[2]    # constraints vector
                dg[level-1] = res[3]    # constraints jacobian
                
            if iterations == 0:
                self.objective_history.append(f[0])
                self.constraint_history.append(g[0])
                
            # Calculate correction
            corrections = self.calculate_correction(f,df,g,dg,tr)
            
            # Calculate constraint violations
            g_violation_hi_center = self.calculate_constraint_violation(g[-1],low_edge,up_edge)
            
            # Subproblem
            tr_size = tr.size
            tr.lower_bound = np.max(np.vstack([lbd,x-tr_size]),axis=0)
            tr.upper_bound = np.min(np.vstack([ubd,x+tr_size]),axis=0)

            # Setup SNOPT 
            #opt_wrap = lambda x:self.evaluate_corrected_model(problem,x,corrections=corrections,tr=tr)

            opt_prob = pyOpt.Optimization('SUAVE',self.evaluate_corrected_model, corrections=corrections,tr=tr)
            
            for ii in xrange(len(obj)):
                opt_prob.addObj('f',f[-1]) 
            for ii in xrange(0,len(inp)):
                vartype = 'c'
                opt_prob.addVar(nam[ii],vartype,lower=tr.lower_bound[ii],upper=tr.upper_bound[ii],value=x[ii])    
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
           
            outputs = opt(opt_prob, sens_type='FD',problem=problem,corrections=corrections,tr=tr)#, sens_step = sense_step)  
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
           
            
            # Evaluate high-fidelity at optimum
            problem.fidelity_level = np.max(self.fidelity_levels)
            fOpt_hi, gOpt_hi = self.evaluate_model(problem,xOpt_lo,scaled_constraints,der_flag=False)
            
            self.objective_history.append(fOpt_hi)
            self.constraint_history.append(gOpt_hi)
            
            g_violation_opt_hi = self.calculate_constraint_violation(gOpt_hi,low_edge,up_edge)
            
            # Calculate ratio
            offset = 0.
            problem.fidelity_level = 2
            high_fidelity_center  = tr.evaluate_function(f[-1],g_violation_hi_center)
            high_fidelity_optimum = tr.evaluate_function(fOpt_hi,g_violation_opt_hi)
            low_fidelity_center   = tr.evaluate_function(f[-1],g_violation_hi_center)
            low_fidelity_optimum  = tr.evaluate_function(fOpt_lo,g_violation_opt_lo)
            if ( np.abs(low_fidelity_center-low_fidelity_optimum) < self.trust_region_function_precision):
                rho = 1.
            else:
                rho = (high_fidelity_center-high_fidelity_optimum)/(low_fidelity_center-low_fidelity_optimum)
            
            # Soft convergence test
            if( np.abs(fOpt_hi) <= self.trust_region_function_precision and np.abs(f[-1]) <= self.trust_region_function_precision ):
                relative_diff = 0
            elif( np.abs(fOpt_hi) <= self.trust_region_function_precision):
                relative_diff = (fOpt_hi - f[-1])/f[-1]
            else:
                relative_diff = (fOpt_hi - f[-1])/fOpt_hi
            self.relative_difference_history.append(relative_diff)
            diff_hist = self.relative_difference_history
            
            ind1 = max(0,iterations-1-tr.soft_convergence_limit)
            ind2 = len(diff_hist) - 1
            converged = 1
            while ind2 >= ind1:
                if( np.abs(diff_hist[ind1]) > tr.soft_convergence_tolerance ):
                    converged = 0
                    break
                ind1 += 1
            if( converged and len(self.relative_difference_history) >= tr.soft_convergence_limit):
                print 'Soft convergence reached'
                return outputs     
            
            # Acceptance Test
            accepted = 0
            if( fOpt_hi < f[-1] ):
                print 'Trust region update accepted since objective value is lower\n'
                accepted = 1
            elif( g_violation_opt_hi < g_violation_hi_center ):
                print 'Trust region update accepted since nonlinear constraint violation is lower\n'
                accepted = 1
            else:
                print 'Trust region update rejected (filter)\n'        
            
            # Update Trust Region Size
            print tr
            tr_size_previous = tr.size
            tr_action = 0 # 1: shrink, 2: no change, 3: expand
            if( not accepted ): # shrink trust region
                tr.size = tr.size*tr.contraction_factor
                tr_action = 1
                print 'Trust region shrunk from %f to %f\n\n' % (tr_size_previous,tr.size)        
            elif( rho < 0. ): # bad fit, shrink trust region
                tr.size = tr.size*tr.contraction_factor
                tr_action = 1
                print 'Trust region shrunk from %f to %f\n\n' % (tr_size_previous,tr.size)
            elif( rho <= tr.contract_threshold ): # okay fit, shrink trust region
                tr.size = tr.size*tr.contraction_factor
                tr_action = 1
                print 'Trust region shrunk from %f to %f\n\n' % (tr_size_previous,tr.size)
            elif( rho <= tr.expand_threshold ): # pretty good fit, retain trust region
                tr_action = 2
                print 'Trust region size remains the same at %f\n\n' % tr.size
            elif( rho <= 1.25 ): # excellent fit, expand trust region
                tr.size = tr.size*tr.expansion_factor
                tr_action = 3
                print 'Trust region expanded from %f to %f\n\n' % (tr_size_previous,tr.size)
            else: # rho > 1.25, okay-bad fit, but good for us, retain trust region
                tr_action = 2
                print 'Trust region size remains the same at %f\n\n' % tr.size  
                
            # Terminate if trust region too small
            if( tr.size < tr.minimum_size ):
                print 'Trust region too small'
                return -1
            
            # Terminate if solution is infeasible, no change is detected, and trust region does not expand
            if( success_indicator == 13 and tr_action < 3 and \
                np.sum(np.isclose(xOpt,x,rtol=1e-15,atol=1e-14)) == len(x) ):
                print 'Solution infeasible, no improvement can be made'
                return -1       
            
            # Update Trust Region Center
            if accepted == 1:
                x = xOpt_lo
                t = self.increment_trust_region_center_index()
                trc = xOpt_lo
            else:
                aa = 0
                pass
            
            print iterations
            print x
            print fOpt_hi
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
     
                         
        
        return (f,df,g,dg)
    def check_for_duplicate_evals(self, problem, x):
        
        return None
    def evaluate_corrected_model(self,x,problem=None,corrections=None,tr=None):
        #duplicate_flag, obj, gradient = self.check_for_duplicate_evals(x)
        duplicate_flag = False
        if duplicate_flag == False:
            obj   = problem.objective(x)
            const = problem.all_constraints(x).tolist()
            #const = []
            fail  = np.array(np.isnan(obj) or np.isnan(np.array(const).any())).astype(int)
            
            A, b = corrections
            x0   = tr.center
            
            obj   = obj + np.dot(A[0,:],(x-x0))+b[0]
            const = const + np.matmul(A[1:,:],(x-x0))+b[1:]
            const = const.tolist()
        
            print 'Inputs'
            print x
            print 'Obj'
            print obj
            print 'Con'
            print const
            
        return obj,const,fail
        
        
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
        nr = 1 + g[0].size
        nc = df[0].size
            
        A = np.empty((nr,nc))
        b = np.empty(nr)
            
        # objective correction
        A[0,:] = df[1] - df[0]
        b[0] = f[1] - f[0]
            
        # constraint corrections
        A[1:,:] = dg[1] - dg[0]
        b[1:] = g[1] - g[0]
            
        corr = (A,b)
        
        
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
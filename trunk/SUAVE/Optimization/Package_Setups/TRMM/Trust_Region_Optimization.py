## @ingroup Optimization-Package_Setups-TRMM
# Trust_Region_Optimization.py
#
# Created:  Apr 2017, T. MacDonald
# Modified: Jun 2017, T. MacDonald
#           Oct 2019, T. MacDonald

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
import SUAVE
try:
    import pyOpt
except:
    pass
from SUAVE.Core import Data
from SUAVE.Optimization import helper_functions as help_fun
import os
import sys
from scipy.optimize import minimize

# ----------------------------------------------------------------------
#  Trust Region Optimization Class
# ----------------------------------------------------------------------
## @ingroup Optimization-Package_Setups-TRMM
class Trust_Region_Optimization(Data):
    """A trust region optimization
    
    Assumptions:
    Only SNOPT is implemented
    
    Source:
    None
    """      
        
    def __defaults__(self):
        """This sets the default values.

        Assumptions:
        None

        Source:
        N/A

        Inputs:
        None

        Outputs:
        None

        Properties Used:
        None
        """          
        
        self.tag                                = 'TR_Opt'
        self.trust_region_max_iterations        = 30
        self.optimizer_max_iterations           = 30
        self.convergence_tolerance              = 1E-6
        self.optimizer_convergence_tolerance    = 1E-6  # used in SNOPT
        self.optimizer_constraint_tolerance     = 1E-6  # used in SNOPT
        self.difference_interval                = 1E-6 
        self.optimizer_function_precision       = 1E-12 # used in SNOPT
        self.trust_region_function_precision    = 1E-12
        self.optimizer_verify_level             = 0
        self.fidelity_levels                    = 2     # only two are currently supported
        self.evaluation_order                   = [1,2] # currently this order is necessary for proper functionality   
        self.optimizer                          = 'SLSQP'
        
    def optimize(self,problem,print_output=False):
        """Optimizes the problem

        Assumptions:
        Currently only works with SNOPT

        Source:
        "A trust-region framework for managing the use of approximation models in optimization," Alexandrov et. al., 1998
        Details do not follow exactly.

        Inputs:
        problem.                 <Nexus class> (also passed into other functions)
          optimization_problem. 
            inputs               Numpy array matching standard SUAVE optimization setup
            objective            Numpy array matching standard SUAVE optimization setup
            constraints          Numpy array matching standard SUAVE optimization setup
          fidelity_level         [-]
        print_output             <boolean> Determines if output is printed during the optimization run

        Outputs:
        (fOpt_corr,xOpt_corr,str):
            fOpt_corr            <float>
            xOpt_corr            <numpy array>
            str                  Varies depending on the result of the optimization
            

        Properties Used:
        self.
          trust_region_max_iterations         [-]    
          fidelity_levels                     [-]
          evaluation_order                    List of the fidelity level order
          evaluate_model(..)
          calculate_correction(..)
          calculate_constraint_violation(..)
          optimizer                           <string> Determines what optimizer is used
          evaluate_corrected_model(..)
          optimizer_max_iterations            [-]
          optimizer_convergence_tolerance     [-]
          optimizer_constraint_tolerance      [-]
          optimizer_function_precision        [-]
          optimizer_verify_level              Int determining if SNOPT will verify that the minimum is level
          accuracy_ratio(..)
          update_tr_size(..)
          convergance_tolerance               [-]
        """          
        if print_output == False:
            devnull = open(os.devnull,'w')
            sys.stdout = devnull
            
        # History writing
        f_out = open('TRM_hist.txt','w')
        import datetime
        f_out.write(str(datetime.datetime.now())+'\n')       
        
        inp = problem.optimization_problem.inputs
        obj = problem.optimization_problem.objective
        con = problem.optimization_problem.constraints 
        tr  = problem.trust_region
        
        # Set inputs
        nam = inp[:,0] # names
        ini = inp[:,1] # initials
        bnd = inp[:,2] # x bounds
        scl = inp[:,3] # x scale
        typ = inp[:,4] # type
    
        (x,scaled_constraints,x_low_bound,x_up_bound,con_up_edge,con_low_edge,name) = self.scale_vals(inp, con, ini, bnd, scl)
        
        # ---------------------------
        # Trust region specific code
        # ---------------------------
        
        iterations = 0
        max_iterations = self.trust_region_max_iterations
        x = np.array(x,dtype='float')
        tr.center = x
        tr_center = x # trust region center
        x_initial = x*1.      
        
        while iterations < max_iterations:
            iterations += 1
            
            # History writing
            f_out.write('Iteration ----- ' + str(iterations) + '\n')
            f_out.write('x_center: ' + str(x.tolist()) + '\n')
            f_out.write('tr size  : ' + str(tr.size) + '\n')   
            
            f    = [None]*self.fidelity_levels
            df   = [None]*self.fidelity_levels
            g    = [None]*self.fidelity_levels
            dg   = [None]*self.fidelity_levels            
            
            for level in self.evaluation_order:
                problem.fidelity_level = level
                res = self.evaluate_model(problem,x)
                f[level-1]  = res[0]    # objective value
                df[level-1] = res[1]    # objective derivate vector
                g[level-1]  = res[2]    # constraints vector
                dg[level-1] = res[3]    # constraints jacobian
                # History writing
                f_out.write('Level    : ' + str(level) + '\n')
                f_out.write('f        : ' + str(res[0][0]) + '\n')
                f_out.write('df       : ' + str(res[1].tolist()) + '\n')
            # assumes high fidelity is last
            f_center = f[-1][0]
                
            # Calculate correction
            corrections = self.calculate_correction(f,df,g,dg,tr)
            
            # Calculate constraint violation
            g_violation_hi_center = self.calculate_constraint_violation(g[-1],con_low_edge,con_up_edge)
            
            # Subproblem
            tr_size = tr.size
            tr.lower_bound = np.max(np.vstack([x_low_bound,x-tr_size]),axis=0)
            tr.upper_bound = np.min(np.vstack([x_up_bound,x+tr_size]),axis=0)      
            
            # Set to base fidelity level for optimizing the corrected model
            problem.fidelity_level = 1
            
            if self.optimizer == 'SNOPT':
                opt_prob = pyOpt.Optimization('SUAVE',self.evaluate_corrected_model, corrections=corrections,tr=tr)
                
                for ii in range(len(obj)):
                    opt_prob.addObj('f',f_center) 
                for ii in range(0,len(inp)):
                    vartype = 'c'
                    opt_prob.addVar(nam[ii],vartype,lower=tr.lower_bound[ii],upper=tr.upper_bound[ii],value=x[ii])    
                for ii in range(0,len(con)):
                    if con[ii][1]=='<':
                        opt_prob.addCon(name[ii], type='i', upper=con_up_edge[ii])  
                    elif con[ii][1]=='>':
                        opt_prob.addCon(name[ii], type='i', lower=con_low_edge[ii],upper=np.inf)
                    elif con[ii][1]=='=':
                        opt_prob.addCon(name[ii], type='e', equal=con_up_edge[ii])      
                        
                   
                opt = pyOpt.pySNOPT.SNOPT()       
                
                opt.setOption('Major iterations limit'     , self.optimizer_max_iterations)
                opt.setOption('Major optimality tolerance' , self.optimizer_convergence_tolerance)
                opt.setOption('Major feasibility tolerance', self.optimizer_constraint_tolerance)
                opt.setOption('Function precision'         , self.optimizer_function_precision)
                opt.setOption('Verify level'               , self.optimizer_verify_level)           
                
                outputs = opt(opt_prob, sens_type='FD',problem=problem,corrections=corrections,tr=tr)
                
                # output value of 13 indicates that the optimizer could not find an optimum
                if outputs[2]['value'][0] == 13:
                    feasible_flag = False
                else:
                    feasible_flag = True
                fOpt_corr = outputs[0][0]
                xOpt_corr = outputs[1]
                gOpt_corr = np.zeros([1,len(con)])[0]  
                for ii in range(len(con)):
                    gOpt_corr[ii] = opt_prob._solutions[0]._constraints[ii].value  
                    

            elif self.optimizer == 'SLSQP':
                
                bounds = []
                for lb, ub in zip(tr.lower_bound, tr.upper_bound):
                    bounds.append((lb,ub))
                
                constraints = self.initialize_SLSQP_constraints(con,problem,corrections,tr)
                # need corrections, tr
    
                res = minimize(self.evaluate_corrected_model, x, constraints=constraints, \
                               args=(problem,corrections,tr), bounds=bounds)
                
                fOpt_corr = res['fun']
                xOpt_corr = res['x']
                gOpt_corr = problem.all_constraints(xOpt_corr)
                
                if res['success']:
                    feasible_flag = True
                else:
                    feasible_flag = False
                
            else:
                raise ValueError('Selected optimizer not implemented')
            success_flag = feasible_flag            
        
            f_out.write('fopt = ' + str(fOpt_corr)+'\n')
            f_out.write('xopt = ' + str(xOpt_corr)+'\n')
            f_out.write('gopt = ' + str(gOpt_corr)+'\n')
            
            
            # Constraint minization ------------------------------------------------------------------------
            if feasible_flag == False:
                print('Infeasible within trust region, attempting to minimize constraint')
                
                if self.optimizer == 'SNOPT':
                    opt_prob = pyOpt.Optimization('SUAVE',self.evaluate_constraints, corrections=corrections,tr=tr,
                                                  lb=con_low_edge,ub=con_up_edge)
                    for ii in range(len(obj)):
                        opt_prob.addObj('constraint violation',0.) 
                    for ii in range(0,len(inp)):
                        vartype = 'c'
                        opt_prob.addVar(nam[ii],vartype,lower=tr.lower_bound[ii],upper=tr.upper_bound[ii],value=x[ii])           
                    opt = pyOpt.pySNOPT.SNOPT()            
                    opt.setOption('Major iterations limit'     , self.optimizer_max_iterations)
                    opt.setOption('Major optimality tolerance' , self.optimizer_convergence_tolerance)
                    opt.setOption('Major feasibility tolerance', self.optimizer_constraint_tolerance)
                    opt.setOption('Function precision'         , self.optimizer_function_precision)
                    opt.setOption('Verify level'               , self.optimizer_verify_level)                 
                   
                    con_outputs = opt(opt_prob, sens_type='FD',problem=problem,corrections=corrections,tr=tr,
                                      lb=con_low_edge,ub=con_up_edge)
                    xOpt_corr = con_outputs[1]
                    new_outputs = self.evaluate_corrected_model(x, problem=problem,corrections=corrections,tr=tr)
        
                    fOpt_corr = new_outputs[0][0]
                    gOpt_corr = np.zeros([1,len(con)])[0]   
                    for ii in range(len(con)):
                        gOpt_corr[ii] = new_outputs[1][ii]
                elif self.optimizer == 'SLSQP':
                    bounds = []
                    for lb, ub in zip(con_low_edge, con_up_edge):
                        bounds.append((lb,ub)) 
                        
                    res = minimize(self.evaluate_constraints, x, \
                                   args=(problem,corrections,tr,con_low_edge,con_up_edge), bounds=bounds, method='slsqp')                    
                        
                    xOpt_corr = res['x']
                    new_outputs = self.evaluate_corrected_model(x, problem=problem,corrections=corrections,tr=tr,
                                                                return_cons=True)
                    
                    fOpt_corr = new_outputs[0][0]
                    gOpt_corr = np.zeros([1,len(con)])[0]   
                    for ii in range(len(con)):
                        gOpt_corr[ii] = new_outputs[1][ii]               
                        
                else:
                    raise ValueError('Selected optimizer not implemented')
                
                # Constraint minization end ------------------------------------------------------------------------
                

            print('fOpt_corr = ', fOpt_corr)
            print('xOpt_corr = ', xOpt_corr)
            print('gOpt_corr = ', gOpt_corr)
            
            # Evaluate high-fidelity at optimum
            problem.fidelity_level = np.max(self.fidelity_levels)
            fOpt_hi, gOpt_hi = self.evaluate_model(problem,xOpt_corr,der_flag=False)
            fOpt_hi = fOpt_hi[0]
        
            g_violation_opt_corr = self.calculate_constraint_violation(gOpt_corr,con_low_edge,con_up_edge)
            g_violation_opt_hi = self.calculate_constraint_violation(gOpt_hi,con_low_edge,con_up_edge)
            
            # Calculate ratio
            rho = self.accuracy_ratio(f_center,fOpt_hi, fOpt_corr, g_violation_hi_center, g_violation_opt_hi, 
                                      g_violation_opt_corr,tr)  
            
            # Acceptance Test
            accepted = 0
            if( fOpt_hi < f_center ):
                print('Trust region update accepted since objective value is lower\n')
                accepted = 1
            elif( g_violation_opt_hi < g_violation_hi_center ):
                print('Trust region update accepted since nonlinear constraint violation is lower\n')
                accepted = 1
            else:
                print('Trust region update rejected (filter)\n')        
            
            # Update Trust Region Size
            print(tr)
            tr_action = self.update_tr_size(rho,tr,accepted)  
                
            # Terminate if trust region too small
            if( tr.size < tr.minimum_size ):
                print('Trust region too small')
                f_out.write('Trust region too small')
                f_out.close()
                if print_output == False:
                    sys.stdout = sys.__stdout__                  
                return (fOpt_corr,xOpt_corr,'Trust region too small')
            
            # Terminate if solution is infeasible, no change is detected, and trust region does not expand
            if( success_flag == False and tr_action < 3 and\
                np.sum(np.isclose(xOpt_corr,x,rtol=1e-15,atol=1e-14)) == len(x) ):
                print('Solution infeasible, no improvement can be made')
                f_out.write('Solution infeasible, no improvement can be made')
                f_out.close()
                if print_output == False:
                    sys.stdout = sys.__stdout__                  
                return (fOpt_corr,xOpt_corr,'Solution infeasible')      
            
            # History writing
            f_out.write('x opt    : ' + str(xOpt_corr.tolist()) + '\n')
            f_out.write('low obj  : ' + str(fOpt_corr)          + '\n')
            f_out.write('hi  obj  : ' + str(fOpt_hi)            + '\n')
            
            # Convergence check
            if (accepted==1 and (np.abs(f_center-fOpt_hi) < self.convergence_tolerance)):
                print('Hard convergence reached')
                f_out.write('Hard convergence reached')
                f_out.close()
                if print_output == False:
                    sys.stdout = sys.__stdout__                  
                return (fOpt_corr,xOpt_corr,'convergence reached')            
            
            # Update trust region center
            if accepted == 1:
                x = xOpt_corr*1.
                tr.center = x*1.             
            
            print('Iteration number: ' + str(iterations))
            print('x value: ' + str(x.tolist()))
            print('Objective value: ' + str(fOpt_hi))
        
        f_out.write('Max iteration limit reached')
        f_out.close()
        print('Max iteration limit reached')
        if print_output == False:
            sys.stdout = sys.__stdout__          
        return (fOpt_corr,xOpt_corr,'Max iteration limit reached')
            
        
    def evaluate_model(self,problem,x,der_flag=True):
        """Evaluates the SUAVE nexus problem. This is often a mission evaluation.

        Assumptions:
        None

        Source:
        N/A

        Inputs:
        problem.                 <Nexus class>
          objective(..)
          all_constraints(..)
          finite difference(..)
        x                        <numpy array>
        der_flag                 <boolean>  Determines if finite differencing is done

        Outputs:
        f  - function value      <float>
        df - derivative of f     <numpy array>
        g  - constraint value    <numpy array> (only returned if der_flag is True)
        dg - jacobian of g       <numpy array> (only returned if der_flag is True)


        Properties Used:
        self.difference_interval [-]
        """              
        f  = problem.objective(x)
        g  = problem.all_constraints(x)
        
        if der_flag == False:
            return f,g
        
        # build derivatives
        fd_step = self.difference_interval
        df, dg  = problem.finite_difference(x,diff_interval=fd_step)
        
        return (f,df,g,dg)


    def evaluate_corrected_model(self,x,problem=None,corrections=None,tr=None,return_cons=False):
        """Evaluates the SUAVE nexus problem and applies corrections to the results.
        
        Assumptions:
        None

        Source:
        N/A

        Inputs:
        problem.                 <Nexus class>
          objective(..)
          all_constraints(..)
        corrections              <tuple> Contains correction factors
        tr.center                <array>

        Outputs:
        obj                      function objective
        cons                     list of contraint values
        fail                     indicates if the evaluation was successful

        Properties Used:
        None
        """              
        obj   = problem.objective(x)
        const = problem.all_constraints(x).tolist()
        fail  = np.array(np.isnan(obj.tolist()) or np.isnan(np.array(const).any())).astype(int)
        
        A, b = corrections
        x0   = tr.center
        
        obj   = obj + np.dot(A[0,:],(x-x0))+b[0]
        const = const + np.matmul(A[1:,:],(x-x0))+b[1:]
        const = const.tolist()
    
        print('Inputs')
        print(x)
        print('Obj')
        print(obj)
        print('Con')
        print(const)
            
        if self.optimizer == 'SNOPT' or return_cons:
            return obj,const,fail
        elif self.optimizer == 'SLSQP':
            return obj
        else:
            raise NotImplemented
    
    
    def evaluate_constraints(self,x,problem=None,corrections=None,tr=None,lb=None,ub=None):
        """Evaluates the SUAVE nexus problem provides an objective value based on constraint violation.
        Correction factors are applied to the evaluation results.
        
        Assumptions:
        None

        Source:
        N/A

        Inputs:
        problem.                 <Nexus class>
          objective(..)
          all_constraints(..)
        corrections              <tuple> Contains correction factors
        tr.center                <numpy array>
        lb                       <numpy array> lower bounds on the constraints
        up                       <numpy array> upper bounds on the constraints

        Outputs:
        obj_cons                 objective based on constraint violation
        cons                     list of contraint values
        fail                     indicates if the evaluation was successful

        Properties Used:
        self.calculate_constraint_violation(..)
        """            
        obj      = problem.objective(x) # evaluate the problem
        const    = problem.all_constraints(x).tolist()
        fail     = np.array(np.isnan(obj.tolist()) or np.isnan(np.array(const).any())).astype(int)
        
        A, b = corrections
        x0   = tr.center
        
        const = const + np.matmul(A[1:,:],(x-x0))+b[1:]
        const = const.tolist()
        
        # get the objective that matters here
        obj_cons = self.calculate_constraint_violation(const,lb,ub)
        const    = None
        
        print('Inputs')
        print(x)
        print('Cons violation')
        print(obj_cons)         
        if self.optimizer == 'SNOPT':
            return obj_cons,const,fail  
        else:
            return obj_cons
        
        
    def calculate_constraint_violation(self,gval,lb,ub):
        """Calculates the constraint violation using a 2-norm of the violated constraint values.
        
        Assumptions:
        None

        Source:
        N/A

        Inputs:
        gval                     <numpy array> constraint values
        lb                       <numpy array> lower bounds on the constraints
        up                       <numpy array> upper bounds on the constraints

        Outputs:
        constraint violation     [-]

        Properties Used:
        None
        """                  
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
        """Calculates additive correction factors.
        
        Assumptions:
        None

        Source:
        N/A

        Inputs:
        f  - function value      <float>
        df - derivative of f     <numpy array>
        g  - constraint value    <numpy array> (only returned if der_flag is True)
        dg - jacobian of g       <numpy array> (only returned if der_flag is True)

        Outputs:
        corr                     <tuple> correction factors

        Properties Used:
        None
        """             
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
    
    
    def scale_vals(self,inp,con,ini,bnd,scl):
        """Scales inputs, constraints, and their bounds.
        
        Assumptions:
        None

        Source:
        N/A

        Inputs:
        (all SUAVE format specific numpy arrays)
        inp                 Design variables
        con                 Constraint limits
        ini                 Initial values
        bnd                 Variable bounds
        scl                 Scaling factors

        Outputs:
        x                   <numpy array> Scaled design variables
        scaled_constraints  <numpy array>
        x_low_bound         <numpy array>
        x_up_bound          <numpy array>
        con_up_edge         <numpy array>
        con_low_edge        <numpy array>
        name                <list of strings> List of variable names

        Properties Used:
        None
        """    
        
        # Pull out the constraints and scale them
        bnd_constraints = help_fun.scale_const_bnds(con)
        scaled_constraints = help_fun.scale_const_values(con,bnd_constraints)

        x            = ini/scl        
        x_low_bound  = []
        x_up_bound   = []
        edge         = []
        name         = []
        con_up_edge  = []
        con_low_edge = []
        
        for ii in range(0,len(inp)):
            x_low_bound.append(bnd[ii][0]/scl[ii])
            x_up_bound.append(bnd[ii][1]/scl[ii])

        for ii in range(0,len(con)):
            name.append(con[ii][0])
            edge.append(scaled_constraints[ii])
            if con[ii][1]=='<':
                con_up_edge.append(edge[ii])
                con_low_edge.append(-np.inf)
            elif con[ii][1]=='>':
                con_up_edge.append(np.inf)
                con_low_edge.append(edge[ii])
            elif con[ii][1]=='=':
                con_up_edge.append(edge[ii])
                con_low_edge.append(edge[ii])
            
        x_low_bound  = np.array(x_low_bound)
        x_up_bound   = np.array(x_up_bound)
        con_up_edge  = np.array(con_up_edge)         
        con_low_edge = np.array(con_low_edge)        
        
        return (x,scaled_constraints,x_low_bound,x_up_bound,con_up_edge,con_low_edge,name)
    
    
    def accuracy_ratio(self,f_center,f_hi,f_corr,g_viol_center,g_viol_hi,g_viol_corr,tr):
        """Compute the trust region accuracy ratio.
        
        Assumptions:
        None

        Source:
        N/A

        Inputs:
        f_center                   Objective value at the center of the trust region
        f_hi                       High-fidelity objective value at the expected optimum
        f_corr                     Corrected low-fidelity objective value at the expected optimum
        g_viol_center              Constraint violation at the center of the trust region
        g_viol_hi                  High-fidelity constraint violation at the expected optimum
        g_viol_corr                Corrected low-fidelity constraint violation at the expected optimum
        tr.evaluation_function(..)

        Outputs:
        rho                        [-] accuracy ratio

        Properties Used:
        self.trust_region_function_precision [-]
        """          
        # center value does not change since the corrected function already matches
        high_fidelity_center  = tr.evaluate_function(f_center,g_viol_center)
        high_fidelity_optimum = tr.evaluate_function(f_hi,g_viol_hi)
        low_fidelity_center   = tr.evaluate_function(f_center,g_viol_center)
        low_fidelity_optimum  = tr.evaluate_function(f_corr,g_viol_corr)
        if ( np.abs(low_fidelity_center-low_fidelity_optimum) < self.trust_region_function_precision):
            rho = 1.
        else:
            rho = (high_fidelity_center-high_fidelity_optimum)/(low_fidelity_center-low_fidelity_optimum) 
            
        return rho
    
    
    def update_tr_size(self,rho,tr,accepted):
        """Updates the trust region size based on the accuracy ratio and if it has been accepted.
        
        Assumptions:
        None

        Source:
        N/A

        Inputs:
        rho                  [-] accuracy ratio
        tr. 
          size               [-]
          contraction_factor [-]
          contract_threshold [-]
          expand_threshold   [-]
          expansion_factor   [-]

        Outputs:
        tr_action            [-] number indicating the type of action done by the trust region

        Properties Used:
        None
        """     
        
        tr_size_previous = tr.size
        tr_action = 0 # 1: shrink, 2: no change, 3: expand
        if( not accepted ): # shrink trust region
            tr.size = tr.size*tr.contraction_factor
            tr_action = 1
            print('Trust region shrunk from %f to %f\n\n' % (tr_size_previous,tr.size))        
        elif( rho < 0. ): # bad fit, shrink trust region
            tr.size = tr.size*tr.contraction_factor
            tr_action = 1
            print('Trust region shrunk from %f to %f\n\n' % (tr_size_previous,tr.size))
        elif( rho <= tr.contract_threshold ): # okay fit, shrink trust region
            tr.size = tr.size*tr.contraction_factor
            tr_action = 1
            print('Trust region shrunk from %f to %f\n\n' % (tr_size_previous,tr.size))
        elif( rho <= tr.expand_threshold ): # pretty good fit, retain trust region
            tr_action = 2
            print('Trust region size remains the same at %f\n\n' % tr.size)
        elif( rho <= 1.25 ): # excellent fit, expand trust region
            tr.size = tr.size*tr.expansion_factor
            tr_action = 3
            print('Trust region expanded from %f to %f\n\n' % (tr_size_previous,tr.size))
        else: # rho > 1.25, okay-bad fit, but good for us, retain trust region
            tr_action = 2
            print('Trust region size remains the same at %f\n\n' % tr.size)        
            
        return tr_action
    
    def initialize_SLSQP_constraints(self,con,problem,corrections,tr):
        # Initialize variables according to SLSQP requirements
        slsqp_con_list = []
        for i,c in enumerate(con):
            c_dict = {}
            if c[1] == '<' or c[1] == '>':
                c_dict['type'] = 'ineq'
            elif c[1] == '=':
                c_dict['type'] = 'eq'
            else:
                raise ValueError('Constraint specification not recognized.')
            c_dict['fun'] = self.unpack_constraints_slsqp
            if c[1] == '<':
                c_dict['args'] = [i,-1,c[2],problem,corrections,tr]
            else:
                c_dict['args'] = [i,1,c[2],problem,corrections,tr]
            slsqp_con_list.append(c_dict)
        
        return slsqp_con_list
    
    def unpack_constraints_slsqp(self,x,con_ind,sign,edge,problem,corrections,tr):
    
        A, b = corrections
        x0   = tr.center
        
        const = problem.all_constraints(x).tolist()
        
        const = const + np.matmul(A[1:,:],(x-x0))+b[1:]
        const = const.tolist()   
        
        con = (const[con_ind]-edge)*sign
        
        return con
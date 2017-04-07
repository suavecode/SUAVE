"""
Setup and run trust region optimization problem.

Rick Fenrich 2/16/17
"""

import time
import datetime
import copy
import shutil
import os
import sys
import multiprocessing
import subprocess

import numpy as np

from pyOpt import Optimization
from pyOpt import SNOPT

import market as M
import user_setup
#from userFcns import linearInequalityConstraints

def setup_shared_model_data(x,y,log_file,flow):  #setupSharedModelData
    
    # Increment global variables
    s = M.incrementSharedDataIndex()
    
    # Evaluate user setup function
    if( flow.function_evals_in_unique_directory == 1):
    
        dirname = 'SHARED_' + str(s);
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        else:
            print 'directory %s already exists' % dirname

        os.chdir(dirname)
        
        # Link necessary files
        for f in flow.setup_link_files:
            if( os.path.exists(f) ):
                pass
            else:
                os.link(os.path.join('..',f),f)
                
        # Link (potentially) necessary files
        for f in flow.link_files:
            if( os.path.exists(f) ):
                pass
            else:
                os.link(os.path.join('..',f),f)                

        user_setup.setup(x,y,flow.n_cores)
        
        os.chdir('..')
        
        log = open(log_file,'a')
        log.write('Mutual setup completed in directory %s\n\n' % dirname)
        log.write('User data:\n')
        for key in M.USER_DATA:
            log.write('%s: ' % key)
            log.write('{}\n'.format(M.USER_DATA[key]))
        log.write('\n')
        log.close()
        
    else:
    
        user_setup.setup(x,y,flow.n_cores)
        
        log = open(log_file,'a')
        log.write('Mutual setup complete\n\n')
        log.close()        
    
    return 0
    
def evaluate_model(xval,y,log_file,flow,opt,level,ret, my_function):
    if( ret == 'all' ):
        f, g, df, dg = user_setup.function(xval,y,flow,opt,level,ret, my_function)
    elif( ret == 'val' ):
        f, g = user_setup.function(xval,y,flow,opt,level,ret, my_function)[0:2]          
    elif( ret == 'der' ):
        df, dg = user_setup.function(xval,y,flow,opt,level,ret, my_function)[2:]
    else:
        raise ValueError("evalModel only returns 'all', 'val', or 'der'")
        
    if( ret == 'all' ):
        log = open(log_file,'a')
        log.write('Level %i model value and derivative evaluation complete\n\n' % level)
        log.close()
        return f, g, df, dg
    elif( ret == 'val' ):
        log = open(log_file,'a')
        log.write('Level %i model value evaluation complete\n\n' % level)
        log.close()
        return f, g
    elif( ret == 'der' ):
        log = open(log_file,'a')
        log.write('Level %i model derivative evaluation complete\n\n' % level)
        log.close()
        return df, dg
        
        
def evaluate_corrected_model(xval,y=None,corr=None,flow=None,opt=None,tr=None,level=None,ret='all', my_function = None,**kwargs):

    fail = 0
    
    if( len(corr) == 2 ): # assume additive correction
    
        A, b = corr
        x0 = tr.center[0:xval.size]

        if( ret == 'all' ):
        
            f, g, df, dg = user_setup.function(xval,y,flow,opt,level,ret,my_function)
            f = f + np.dot(A[0,:],(xval-x0)) + b[0]
            g = g + np.matmul(A[1:,:],(xval-x0)) + b[1:]
            df = df + A[0,:]
            dg = dg + A[1:,:]
            
            return f, g, df, dg, fail
        
        elif( ret == 'val' ):
        
            f, g = user_setup.function(xval,y,flow,opt,level,ret, my_function)[0:2]
            f = f + np.dot(A[0,:],(xval-x0)) + b[0]
            g = g + np.dot(A[1:,:],(xval-x0)) + b[1:]    #g + np.matmul(A[1:,:],(xval-x0)) + b[1:]       
            
            return f, g, fail     
                    
        elif( ret == 'der' ):
        
            df, dg = user_setup.function(xval,y,flow,opt,level,ret)[2:]
            df = df + A[0,:]
            dg = dg + A[1:,:]
            
            return df, dg, fail
                        
        else:
            raise ValueError("evalCorrectedModel only returns 'all', 'val', or 'der'")
            
    else: # assume no correction
    
        raise NotImplementedError('No correction is applied?')
    
        if( ret == 'all' ):
        
            f, g, df, dg = user_setup.function(xval,y,flow,opt,level,ret)            
            return f, g, df, dg, fail
        
        elif( ret == 'val' ):
        
            f, g = user_setup.function(xval,y,flow,opt,level,ret)[0:2]           
            return f, g, fail     
                    
        elif( ret == 'der' ):
        
            df, dg = user_setup.function(xval,y,flow,opt,level,ret)[2:]
            return df, dg, fail
                        
        else:
            raise ValueError("evalCorrectedModel only returns 'all', 'val', or 'der'")
    
        
        
def calculate_correction(f,g,df,dg,tr):

    nf = len(f) # number of fidelity levels
    nr = 1 + g[0].size
    nc = df[0].size
        
    if( nf == 1 ):
    
        A = np.zeros((nr,nc))
        b = np.zeros(nr)
        return (A,b)
        
    elif( nf > 2 ):
        raise NotImplementedError('Correction calculations currently only implemented for 2 fidelity levels')

    if( tr.correction_type == 'additive' ):
        
        A = np.empty((nr,nc))
        b = np.empty(nr)
        
        # objective correction
        A[0,:] = df[1] - df[0]
        b[0] = f[1] - f[0]
        
        # constraint corrections
        A[1:,:] = dg[1] - dg[0]
        b[1:] = g[1] - g[0]
        
        corr = (A,b)
    
    else:
        
        raise NotImplementedError('Only additive correction implemented')
    
    return corr
    
    
def calculate_constraint_violation(gval,opt):
    
    # For nonlinear inequality constraints only

    lb = opt.constraint_lower_bounds
    ub = opt.constraint_upper_bounds

    gdiff = []

    for i in range(len(gval)):
        if len(lb) > 0:
            if( gval[i] < lb[i] ):
                gdiff.append(lb[i] - gval[i])
        if len(ub) > 0:    
            if( gval[i] > ub[i] ):
                gdiff.append(gval[i] - ub[i])

    return np.linalg.norm(gdiff) # 2-norm of violation
        

def run(x,y,log_file_rel,tr,opt,flow,mi,me,ai, my_function):

    # Convert log filename to absolute path
    log_file = os.path.join(os.getcwd(),log_file_rel)

    # =========================================================================
    # Write Initial Settings to File
    # =========================================================================
    
    log = open(log_file,'w')
    log.write('%s\n\n' % str(datetime.datetime.now()))
    log.write('PROBLEM STATEMENT\n\n')
    log.write('1 objective function\n')
    log.write('%i nonlinear constraints\n' % (mi+me))
    log.write('%i linear constraints\n' % ai)
    #log.write('%i nonlinear inequality constraints\n' % mi)
    #log.write('%i nonlinear equality constraints\n' % me)
    #log.write('%i linear inequality constraints\n' % ai)
    log.write('\n')
    log.write('%i design variables\n' % x.n)
    log.write(x.__print__())
    log.write('\n')
    log.flush()
    log.write('%i parameters\n' % y.n)
    log.write(y.__print__())
    log.write('\n')
    log.flush()
    log.write('TRUST REGION SETTINGS\n\n')
    log.write('Initial size:          %f\n' % tr.initial_size)
    log.write('Minimum size:          %e\n' % tr.minimum_size)
    log.write('Contraction threshold: %f\n' % tr.contract_threshold)
    log.write('Expansion threshold:   %f\n' % tr.expand_threshold)
    log.write('Contraction factor:    %f\n' % tr.contraction_factor)
    log.write('Expansion factor:      %f\n' % tr.expansion_factor)
    log.write('Approx. subproblem:    %s\n' % tr.approx_subproblem)
    log.write('Merit function:        %s\n' % tr.merit_function)
    log.write('Acceptance test:       %s\n' % tr.acceptance_test)
    if( tr.correction_order == 0 ):
        log.write('Correction:            %s 0th order\n' % tr.correction_type)
    elif( tr.correction_order == 1 ):
        log.write('Correction:            %s 1st order\n' % tr.correction_type)
    elif( tr.correction_order == 2 ):
        log.write('Correction:            %s 2nd order\n' % tr.correction_type)
    log.write('Max iterations:        %i\n' % tr.max_iterations)
    log.write('Soft converg. limit:   %i\n' % tr.soft_convergence_limit)
    log.write('Convergence tolerance: %e\n' % tr.convergence_tolerance)
    log.write('Constraint tolerance:  %e\n' % tr.constraint_tolerance)
    log.write('\n')
    log.flush()
    log.write('OPTIMIZATION SETTINGS\n\n')
    log.write('Algorithm:             %s\n' % opt.name)
    if( opt.name == 'SNOPT' ):
        log.write('Verify level:             %i\n' % opt.verify_level)
    log.write('Max iterations:        %i\n' % opt.max_iterations)
    log.write('Max function evals:    %i\n' % opt.max_function_evaluations)
    log.write('Convergence tolerance: %e\n' % opt.convergence_tolerance)
    log.write('Constraint tolerance:  %e\n' % opt.constraint_tolerance)
    log.write('Gradient type:         %s\n' % opt.gradients)
    log.write('Finite difference abs. step: %e\n' % opt.difference_interval)
    if( mi + me > 0 ):
        log.write('Nonlinear constraints:\n')
        log.write('    Lower bounds: {}\n'.format(', '.join(str(e) for e in opt.constraint_lower_bounds)))
        log.write('    Upper bounds: {}\n'.format(', '.join(str(e) for e in opt.constraint_upper_bounds))) 
    log.write('\n')
    log.flush()
    log.write('PROGRAM FLOW SETTINGS\n\n')
    log.write('# of fidelity levels:  %s\n' % flow.fidelity_levels)
    log.write('Evaluation order:      level {}\n'.format(" then ".join(str(e) for e in flow.evaluation_order)))
    log.write('Mutual function setup step: %s\n' % ("yes" if flow.mutual_setup_step == 1 else "no"))
    log.write('Number of cores:       %i\n' % flow.n_cores)
    log.write('Function evals in unique directory: %s\n' % ("yes" if flow.function_evals_in_unique_directory == 1 else "no"))
    log.write('Duplicate evaluation checking:\n')
    for i in range(flow.fidelity_levels):
        log.write('  Level %i depends on %s\n' % (i+1,flow.function_dependency[i]))
    log.write('Gradient method:\n')
    for i in range(flow.fidelity_levels):
        log.write('  Level %i: %s\n' % (i+1,flow.gradient_evaluation[i]))
    log.write('Function evaluation files to link: {}\n'.format(', '.join(e for e in flow.link_files)))
    log.write('Function setup files to link: {}\n'.format(', '.join(e for e in flow.setup_link_files)))
    log.write('\n')
    log.write('INITIALIZATION\n\n')
    log.close()

    # =========================================================================
    # Initialize All Variables
    # =========================================================================
    
    # Global variables
    M.init()
    M.setup_history(flow.history_tags) # history for truth functions
    M.setup_surrogate_history(range(1,flow.fidelity_levels+1)) # history for each surrogate function
    M.setup_trust_region_history() # history for trust region
    
    # Local variables
    t = M.increment_trust_region_center_index() # trust region center index
    trc = M.set_trust_region_center(np.hstack((x.value,y.value))) # center of trust region    
    k = M.K # counter for number of iterations
    xi = copy.copy(x.value)
    
    fRelDiff = [] # relative difference in objective func for soft convergence criteria
    fHistory = [] # record optimal objective value at end of every iteration
    gHistory = [] # record optimal constraint values at end of every iteration
    terminateMessage = 'max iterations reached'
    
    Aviol = 0 # linear constraint violation pre-subproblem solution
    Aviol2 = 0 # linear constraint violation post-subproblem solution
    gviol = 0 # nonlinear constraint violation pre-subproblem solution
    gviol2_hi = 0 # nonlinear constraint violation post-subproblem solution (high-fi)
    gviol2_lo = 0 # nonlinear constraint violation post-subproblem solution (low-fi)
    
    # =========================================================================
    # Do initial mutual setup step for functions if necessary
    # =========================================================================
    
    if( flow.mutual_setup_step ):
        log = open(log_file,'a')
        log.write('Performing initial mutual setup step for model functions\n\n')
        log.close()
            
        setupSharedModelData(x,y,log_file,flow)
    
    # =========================================================================
    # Begin Trust Region Iterations
    # =========================================================================

    startTime = time.time()
    while k < tr.max_iterations:
    
        k += 1
        M.increment_iteration()
        M.assign_to_trust_region_history(k,t,trc,tr.size)
        tr.set_center(trc)
        
        # Create new directory if necessary 
        if( flow.function_evals_in_unique_directory ):
            dirname = 'ITER_' + str(k)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            else:
                print 'directory %s already exists' % dirname

            os.chdir(dirname)
        
        # Set necessary local variables
        xOpt = np.zeros(x.n) # values of design variables at optimum
        fOpt = None
        gOpt = np.zeros(mi+me) # values of nonlinear constraints at optimum
        
        sys.stdout.write('\nTRMM ITERATION %i\n\n' % k)
        
        log = open(log_file,'a')
        log.write('ITERATION %i\n\n' % k)
        log.write('Center of trust region:\n')
        log.write('{}\n'.format(" ".join(str(e) for e in trc)))
        log.write('Trust region size: %f\n\n' % tr.size)
        log.close()

        # =========================================================================
        # Evaluate function values and derivatives @ center of trust region
        # =========================================================================
        log = open(log_file,'a')
        log.write('Beginning function value & derivative evaluations at trust region center\n\n')
        log.close()
        f = [None]*flow.fidelity_levels
        g = [None]*flow.fidelity_levels
        df = [None]*flow.fidelity_levels
        dg = [None]*flow.fidelity_levels
        for level in flow.evaluation_order: # evaluate model fidelities in order
            results = evaluate_model(x.value,y,log_file,flow,opt,level,'all', my_function)
            f[level-1] = results[0]
            g[level-1] = results[1]
            df[level-1] = results[2]
            dg[level-1] = results[3]
        log = open(log_file,'a')
        log.write('Evaluation history to date:\n\n')
        for key in M.EVAL:
            log.write('%i total evals of truth %s function\n' % (M.EVAL[key],key))
        for key in M.SURR_EVAL:
            log.write('%i total evals of surrogate level %i\n' % (M.SURR_EVAL[key],key))
        log.write('\n')         
        log.write('Function values and derivatives at trust region center:\n\n')
        for i in range(len(f)):
            log.write('Obj value, level %i: %0.16f\n' % (i+1,f[i]))
        for i in range(len(f)):
            log.write('Obj grad,  level %i: ' % (i+1))
            log.write('{}\n'.format(' '.join(str(e) for e in df[i])))
        log.write('\n')
        if( mi + me > 0 ):
            for i in range(len(g)):
                log.write('Con value, level %i: ' % (i+1))
                log.write('{}\n'.format(' '.join(str(e) for e in g[i])))
            for i in range(len(dg)):
                log.write('Con grad,  level %i:\n' % (i+1))
                for j in range(len(g[0])):
                    log.write('{}\n'.format(dg[i][j,:])) 
            log.write('\n') 
        log.close()
        
        # Record initial values to general history
        if( k == 1 ):
            fHistory.append(f[-1])
            gHistory.append(g[-1])
            
        # =========================================================================
        # Calculate correction for all responses
        # =========================================================================         
        log = open(log_file,'a')
        log.write('Calculating correction for all responses\n\n')
        log.close()
        
        corr = calculate_correction(f,g,df,dg,tr)
        
        log = open(log_file,'a')
        if( tr.correction_type == 'additive' ):
            log.write('Additive gradient correction matrix:\n')
            for i in range(1+len(g[0])):
                log.write('{}\n'.format(corr[0][i,:]))
            log.write('Additive value correction vector:\n')
            log.write('{}\n\n'.format(corr[1]))
        else:
            log.write('Correction output not enabled for %s correction\n\n' % tr.correction_type)
        log.close()

        # =========================================================================
        # Calculate constraint violation for all fidelities
        # =========================================================================  
        '''
        if( ai > 0 ):
            LHS, Alinear, b_upper = linearInequalityConstraints(x.value)
            b_lower = -np.inf*np.ones(b_upper.size)
            Aviol = np.linalg.norm([e for e in LHS if e > 0])
            log = open(log_file,'a')
            log.write('Linear constraint violation at trust region center: %f\n\n' % Aviol)
            if( Aviol > 0 ):
                log.write(' Ax - b = {}\n\n'.format(LHS))            
            log.close()
        '''
        if( mi + me > 0 ):
            print 'f=', f
            print 'g= ', g
            gviol = calculate_constraint_violation(g[-1],opt)
            log = open(log_file,'a')
            log.write('Nonlinear constraint violation at trust region center: %f\n\n' % gviol)
            log.close()
                 
        # =========================================================================
        # Setup Subproblem
        # =========================================================================
        log = open(log_file,'a')
        log.write('Setting up subproblem\n\n')
        log.close()

        # Set trust region bounds
        xTrLower = np.max(np.vstack((x.lower_bound,x.value-tr.size)),axis=0)
        xTrUpper = np.min(np.vstack((x.upper_bound,x.value+tr.size)),axis=0)
        
        opt_prob = Optimization('TRMM Subproblem',evaluate_corrected_model)
        for i in range(x.n):
            variableName = 'x' + str(i)
            opt_prob.addVar(variableName,'c',lower=xTrLower[i],upper=xTrUpper[i],value=x.value[i])
        opt_prob.addObj('f',value=f[-1])
#        for i in range(mi+1,r_lo.size-1):
#            constraintName = 'ge' + str(i)
#            opt_prob.addCon(constraintName,'e',equal=0.,value=r_lo[i])
        
        for i in range(mi):
            constraintName = 'gi' + str(i)
            print 'g=', g
            print g[-1][i]
            opt_prob.addCon(constraintName,'i',lower=opt.constraint_lower_bounds[i], \
                            upper=opt.constraint_upper_bounds[i],value=g[-1][i])
        if( ai > 0 ):            
            opt_prob.addLinCon('a1','i',matrix=Alinear,upper=b_upper,lower=b_lower)
            
        print opt_prob   
            
        # =========================================================================
        # Solve Subproblem
        # =========================================================================
        log = open(log_file,'a')
        log.write('Trust region:\n')    
        log.write('lower bounds:   {}\n'.format(" ".join(str(e) for e in xTrLower)))
        log.write('center:         {}\n'.format(" ".join(str(e) for e in trc)))
        log.write('upper bounds:   {}\n'.format(" ".join(str(e) for e in xTrUpper)))
        log.write('\n')
        log.write('Optimizing using %s...\n\n' % opt.name)
        log.close()          

        if( opt.name == 'SNOPT' ):
        
            # Setup SNOPT
            snopt = SNOPT()
            if( hasattr(opt,'max_iterations') ):
                snopt.setOption('Major iterations limit',opt.max_iterations)
            if( hasattr(opt,'convergence_tolerance') ):
                snopt.setOption('Major optimality tolerance',opt.convergence_tolerance)
            if( hasattr(opt,'constraint_tolerance') ):
                snopt.setOption('Major feasibility tolerance',opt.constraint_tolerance)
            if( hasattr(opt,'function_precision') ):
                snopt.setOption('Function precision',opt.function_precision)
            if( hasattr(opt,'verify_level') ):
                snopt.setOption('Verify level',opt.verify_level)
            snopt.setOption('Difference interval',opt.difference_interval)
            
            # Solve problem
            if( opt.gradients == 'FD' ): # finite difference gradients via pyOpt
                snopt(opt_prob, sens_type='FD', my_function=my_function, y=y, corr=corr, flow=flow, \
                      opt=opt, tr=tr, level=1, ret='val')
            elif( opt.gradients == 'user' ):
                print 'WARNING: Specifying user for opt.gradients requires modified pySNOPT.py file in pyOpt'
                snopt(opt_prob, sens_type='user', y=y, corr=corr, flow=flow, \
                      opt=opt, tr=tr, level=1)
            else:
                raise NotImplementedError('Only FD or user are currently accepted for pyOpt gradient calculations')

            # Extract optimization results
            # optInformValues:
            # 1: successful termination
            # 13: infeasible (w.r.t. nonlinear constraints)
            optInformValue = opt_prob._solutions[0].opt_inform['value']
            optInformMessage = opt_prob._solutions[0].opt_inform['text']
            fOpt = opt_prob._solutions[0]._objectives[0].value
            for i in range(x.n):
                xOpt[i] = opt_prob._solutions[0]._variables[i].value
            for i in range(mi+me):
                gOpt[i] = opt_prob._solutions[0]._constraints[i].value

        else:
            raise NotImplementedError('Only SNOPT (using pyOpt) is currently implemented')        
        
        # Write optimization results to log file
        log = open(log_file,'a')
        log.write('Optimization terminated with inform code %i: %s\n' % (optInformValue,optInformMessage))
        log.write('Design Variables: {}\n\n'.format(" ".join(str(e) for e in xOpt)))
        log.write('Low-fidelity response evaluated at optimum:\n')
        log.write('Objective:     %0.16f\n' % fOpt)
        for i in range(mi+me):
            log.write('Const. %i:     %0.16f\n' % (i,gOpt[i]))       
        log.write('\n')
        log.close()
        
        # Check linear constraint violation
        '''
        if( ai > 0 ):
            LHS2 = linearInequalityConstraints(xOpt)[0]
            Aviol2 = np.linalg.norm([e for e in LHS2 if e > 0])
            log = open(log_file,'a')
            log.write('Linear constraint violation at optimum: %f\n\n' % Aviol2)
            log.close()   
        '''    
        # Check nonlinear constraint violation for low-fidelity model
        if( mi + me > 0 ):
            gviol2_lo = calculate_constraint_violation(gOpt,opt)     
            log = open(log_file,'a')
            log.write('Nonlinear constraint violation at optimum (lo-fi): %f\n\n' % gviol2_lo)
            log.close()  
                    
        # Rename optimization output file
        if( opt.name == 'SNOPT' ):
            os.rename('SNOPT_summary.out','SNOPT_summary_' + str(k) + '.out')
            os.rename('SNOPT_print.out','SNOPT_print_' + str(k) + '.out')
        else:
            raise NotImplementedError('Renaming optimization files not implemented for %s algorithm' % opt.name)
            
        # =========================================================================
        # Leave New Directory if Necessary
        # =========================================================================
        if( flow.function_evals_in_unique_directory ):
            os.chdir('..')
            
        # =========================================================================
        # Check Hard Convergence
        # =========================================================================
        # If optimizer terminates successfully at initial value of design variables
        # then we have convergence
        if( optInformValue == 1 and \
            np.sum(np.isclose(xOpt.astype(np.float64),x.value.astype(np.float64),rtol=1e-14,atol=1e-12)) == x.value.size ):
            
            log = open(log_file,'a')
            log.write('Optimization converged - hard convergence limit reached\n\n')  
            log.flush()
            log.close()
            fHistory.append(fOpt)
            gHistory.append(gOpt)
            fRelDiff.append(0.0)
            terminateMessage = 'hard convergence limit reached'
            break
            
        # =========================================================================
        # Obtain Trust Region Ratio
        # =========================================================================
        
        # Perform setup to evaluate high-fidelity if necessary
        if( flow.mutual_setup_step ):
            log = open(log_file,'a')
            log.write('Performing mutual setup step for model functions\n\n')
            log.close()
            
            x2 = copy.copy(x)
            x2.value = xOpt
            setupSharedModelData(x2,y,log_file,flow)
                        
        # Evaluate highest-fidelity function at optimum
        # Create new directory if necessary 
        if( flow.function_evals_in_unique_directory ):
            dirname = 'ITER_' + str(k+1)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            else:
                print 'directory %s already exists' % dirname

            os.chdir(dirname) 
                   
        fOpt_hi, gOpt_hi = evaluate_model(xOpt,y,log_file,flow,opt,np.max(flow.fidelity_levels),'val', my_function)
        
        if( flow.function_evals_in_unique_directory ):
            os.chdir('..') 
        
        # Append optimal value to history
        fHistory.append(fOpt_hi)
        gHistory.append(gOpt_hi)
                    
        # Calculate constraint violation for high-fidelity model
        if( mi + me > 0 ):
            gviol2_hi = calculate_constraint_violation(gOpt_hi,opt)
            log = open(log_file,'a')
            log.write('Nonlinear constraint violation at optimum (hi-fi): %f\n\n' % gviol2_hi)
            log.close()  
                    
        # Calculate ratio
        offset = 0.
        hiCenter = tr.evaluate_function(f[-1],gviol,k,offset)
        hiOptim = tr.evaluate_function(fOpt_hi,gviol2_hi,k,offset)
        loCenter = tr.evaluate_function(f[-1],gviol,k,offset) # assume correction
        loOptim = tr.evaluate_function(fOpt,gviol2_lo,k,offset)       
        if( np.abs(loCenter - loOptim) < 1e-12 ):
            rho = 1.
        else:
            rho = (hiCenter - hiOptim)/(loCenter - loOptim)
        
        log = open(log_file,'a')
        log.write('High-fidelity response evaluated at optimum:\n')
        log.write('Objective:     %0.16f\n' % fOpt_hi)
        for i in range(mi+me):
            log.write('Const. %i:     %0.16f\n' % (i,gOpt_hi[i]))       
        log.write('\n')
        log.flush()
        log.write('Ratio: %f\n\n' % rho)
        log.close()
        
        # =========================================================================
        # Soft Convergence Test
        # =========================================================================
        # Calculate % difference in objective function
        if( np.abs(fOpt_hi) <= 1e-12 and np.abs(f[-1]) <= 1e-12 ):
            relDiff = 0
        elif( np.abs(fOpt_hi) <= 1e-12 ):
            relDiff = (fOpt_hi - f[-1])/f[-1]
        else:
            relDiff = (fOpt_hi - f[-1])/fOpt_hi
        fRelDiff.append(relDiff)
                
        log = open(log_file,'a')
        log.write('Checking convergence...\n\n')
        log.write('%% Relative diff. in objective from last iteration: %e\n\n' % relDiff)    

        ind1 = max(0,k-1-tr.soft_convergence_limit)
        ind2 = len(fRelDiff) - 1
        converged = 1
        while ind2 >= ind1:
            if( np.abs(fRelDiff[ind1]) > tr.convergence_tolerance ):
                converged = 0
                break
            ind1 += 1
        if( converged and len(fRelDiff) >= tr.soft_convergence_limit):
            x.value[:] = copy.copy(xOpt)
            log.write('Optimization converged - soft convergence limit reached\n\n')  
            log.flush()
            log.close()
            terminateMessage = 'soft convergence limit reached'
            break
        
        # =========================================================================
        # Acceptance Test
        # =========================================================================
        accepted = 0
        log = open(log_file,'a')
        if( tr.acceptance_test == 'filter' ):
            if( Aviol2 + 1e-15 < Aviol ):
                log.write('Trust region update accepted since linear constraint violation is lower\n')
                log.write('con violation (new): %f\n' % Aviol2)
                log.write('con violation (old): %f\n' % Aviol)
                log.write('Design variable center updated to:\n')
                log.write('{}\n\n'.format(" ".join(str(e) for e in xOpt)))
                accepted = 1
            elif( fOpt_hi < f[-1] ):
                log.write('Trust region update accepted since objective value is lower\n')
                log.write('obj (new): %f\n' % fOpt_hi)
                log.write('obj (old): %f\n' % f[-1])
                log.write('Design variable center updated to:\n')
                log.write('{}\n\n'.format(" ".join(str(e) for e in xOpt)))
                accepted = 1
            elif( gviol2_hi < gviol ):
                log.write('Trust region update accepted since nonlinear constraint violation is lower\n')
                log.write('con violation (new): %f\n' % gviol2_hi)
                log.write('con violation (old): %f\n' % gviol)
                log.write('Design variable center updated to:\n')
                log.write('{}\n\n'.format(" ".join(str(e) for e in xOpt)))      
                accepted = 1
            else:
                log.write('Trust region update rejected (filter)\n')
                log.write('obj (new): %f\n' % fOpt_hi)
                log.write('obj (old): %f\n' % f[-1])
                if( mi + me > 0 ):
                    log.write('con violation (new): %f\n' % gviol2_hi)
                    log.write('con violation (old): %f\n' % gviol)
                log.write('Center remains unchanged at:\n')
                log.write('{}\n\n'.format(" ".join(str(e) for e in x.value)))
        elif( tr.acceptance_test == 'ratio' ):
            raise NotImplementedError('Ratio test not implemented yet for acceptance')
        else:
            raise NotImplementedError('Acceptance test %s not implemented' % tr.acceptance_test)        
        
        log.close()
        
        # =========================================================================
        # Update Trust Region Size
        # =========================================================================
        log = open(log_file,'a')
        tr_size_previous = tr.size
        tr_action = 0 # 1: shrink, 2: no change, 3: expand
        if( not accepted ): # shrink trust region
            tr.size = tr.size*tr.contraction_factor
            tr_action = 1
            log.write('Trust region shrunk from %f to %f\n\n' % (tr_size_previous,tr.size))        
        elif( rho < 0. ): # bad fit, shrink trust region
            tr.size = tr.size*tr.contraction_factor
            tr_action = 1
            log.write('Trust region shrunk from %f to %f\n\n' % (tr_size_previous,tr.size))
        elif( rho <= tr.contract_threshold ): # okay fit, shrink trust region
            tr.size = tr.size*tr.contraction_factor
            tr_action = 1
            log.write('Trust region shrunk from %f to %f\n\n' % (tr_size_previous,tr.size))
        elif( rho <= tr.expand_threshold ): # pretty good fit, retain trust region
            tr_action = 2
            log.write('Trust region size remains the same at %f\n\n' % tr.size)
        elif( rho <= 1.25 ): # excellent fit, expand trust region
            tr.size = tr.size*tr.expansion_factor
            tr_action = 3
            log.write('Trust region expanded from %f to %f\n\n' % (tr_size_previous,tr.size))
        else: # rho > 1.25, okay-bad fit, but good for us, retain trust region
            tr_action = 2
            log.write('Trust region size remains the same at %f\n\n' % tr.size)
        log.flush()
        log.close()
        
        # Terminate if trust region too small
        if( tr.size < tr.minimum_size ):
            log = open(log_file,'a')
            log.write('Trust region size is too small - terminating\n\n')
            log.flush()
            log.close()
            terminateMessage = 'trust region too small'
            break
            
        # Terminate if solution is infeasible, no change is detected, and trust region does not expand
        if( optInformValue == 13 and tr_action < 3 and \
            np.sum(np.isclose(xOpt.astype(np.float64),x.value.astype(np.float64),rtol=1e-15,atol=1e-14)) == x.value.size ):
            
            log = open(log_file,'a')
            log.write('Solution infeasible, no improvement can be made\n\n')  
            log.flush()
            log.close()
            terminateMessage = 'solution infeasible, no improvement can be made'
            break 

        # =========================================================================
        # Update Trust Region Center
        # =========================================================================            
        if( accepted ):
            x.value[:] = copy.copy(xOpt)
            t = M.increment_trust_region_center_index()
            trc = M.set_trust_region_center(np.hstack((x.value,y.value)))            
            log = open(log_file,'a')
            log.write('Trust region center updated\n\n')
            log.close()
        else:
            M.revert_user_data_update()
            log = open(log_file,'a')
            log.write('Changes to user global shared variables reverted\n\n')
            log.close()   
        
    #end
    endTime = time.time()
    
    # =============================================================================
    # Output Data
    # =============================================================================

    log = open(log_file,'a')
    log.write('SUMMARY\n\n')
    log.write('Trust region optimization terminated because %s\n' % terminateMessage)
    elapsedTime = endTime-startTime
    log.write('Elapsed time: %f sec\n' % elapsedTime)
    log.write('TRMM iterations: %i\n' % k)
    log.write('# Unique TRMM centers:  %i\n\n' % t)    
    for key in M.EVAL:
        log.write('%i total evals of truth %s function\n' % (M.EVAL[key],key))
    for key in M.SURR_EVAL:
        log.write('%i total evals of surrogate level %i\n' % (M.SURR_EVAL[key],key))
    log.write('\n')         

    log.write('Design Variables:\n')
    log.write('lower bounds:   {}\n'.format(" ".join(str(e) for e in x.lower_bound)))
    log.write('initial values: {}\n'.format(" ".join(str(e) for e in xi)))
    log.write('final values: {}\n'.format(" ".join(str(e) for e in x.value)))
    log.write('upper bounds:   {}\n'.format(" ".join(str(e) for e in x.upper_bound)))
    log.write('\n')
    
    log.write('Response at Optimum:\n')
    log.write('                  Initial              Final\n')
    log.write('Objective:     %0.16f %0.16f\n' % (fHistory[0],fOpt))
    for i in range(mi+me):
        log.write('Const. %i:     %0.16f %0.16f\n' % (i,gHistory[0][i],gOpt[i]))
    log.write('\n')
    
    log.write('Trust Region:\n')
    log.write('Final trust region size: %e\n\n' % tr.size)
  
    log.flush()
    
    log.write('HISTORY\n\n')
    log.write('Iter,     Obj,' + ' '*15 + 'TrSize,' + ' '*11 + '%% Diff Obj,' + ' '*9)
    for j in range(mi+me):
        log.write('Con %i,               ' % j)
    log.write('\n')
    log.flush()
    for i in range(len(fHistory)):
        log.write('%i, ' % i)
        log.write('%0.16f, ' % fHistory[i])
        if( i > 0 ):
            log.write('%0.16f, ' % M.TR_HIST['trSize'][i-1])
            log.write('%0.16f, ' % fRelDiff[i-1])
        else:
            log.write('                  , ')
            log.write('                  , ')
        for j in range(mi):
            log.write('%0.16f, ' % gHistory[i][j])
        for j in range(me):
            log.write('%0.16f, ' % gHistory[i][j])
        log.write('\n')
        log.flush()
    
    log.close()

    # =============================================================================
    # Save history
    # =============================================================================
    
    # Save surrogate history first
    for level in M.SURR_HIST:
        history = open('history_surr%i.dat' % level,'w')
        history.write('iter,'+'{},'.format(','.join('x'+str(i) for i in range(len(x.value))))+'{},'.format(','.join('y'+str(i) for i in range(len(y.value)))) + 'obj' + '{},'.format(','.join('con'+str(i) for i in range(mi+me))) + '\n')
        for i in range(len(M.SURR_HIST[level]['iter'])):
            history.write('%i,' % M.SURR_HIST[level]['iter'][i])
            history.write('{},'.format(','.join(str(e) for e in M.SURR_HIST[level]['x'][i])))
            history.write('{},'.format(','.join(str(e) for e in M.SURR_HIST[level]['y'][i])))
            history.write('%f,' % M.SURR_HIST[level]['objective'][i])
            history.write('{}\n'.format(','.join(str(e) for e in M.SURR_HIST[level]['constraints'][i])))
        history.close()
        
    # Save truth history next
    for tag in M.HIST:
        history = open('history_%s.dat' % tag,'w')
        history.write('{},'.format(','.join('x'+str(i) for i in range(len(x.value))))+'{},'.format(','.join('y'+str(i) for i in range(len(y.value))))+'response\n')
        for i in range(len(M.HIST[tag]['x'])):
            #print tag
            #print 'M.HIST[tag][response][i] = ', M.HIST[tag]['response'][i]
            #print 'type(M.HIST[tag][response][i]) = ', type(M.HIST[tag]['response'][i])
            history.write('{},'.format(','.join(str(e) for e in M.HIST[tag]['x'][i])))
            history.write('{},'.format(','.join(str(e) for e in M.HIST[tag]['y'][i])))
            #history.write('%f\n' % M.HIST[tag]['response'][i])
            history.write(str(M.HIST[tag]['response'][i]))
        history.close()        

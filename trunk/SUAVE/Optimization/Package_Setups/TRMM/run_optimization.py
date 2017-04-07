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
    
def evaluate_model(xval,y,log_file,flow,opt,level,ret, my_function):
    f, g = user_setup.function(xval,y,flow,opt,level,ret, my_function)[0:2]          
        
    return f, g
        
        
def evaluate_corrected_model(xval,y=None,corr=None,flow=None,opt=None,tr=None,level=None,ret='all', my_function = None,**kwargs):

    fail = 0
    
    
    A, b = corr
    x0 = tr.center[0:xval.size]
        
    f, g = user_setup.function(xval,y,flow,opt,level,ret, my_function)[0:2]
    f = f + np.dot(A[0,:],(xval-x0)) + b[0]
    g = g + np.dot(A[1:,:],(xval-x0)) + b[1:]    #g + np.matmul(A[1:,:],(xval-x0)) + b[1:]       
            
    return f, g, fail     
    
        
        
def calculate_correction(f,g,df,dg,tr):

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
    
    
    # =========================================================================
    # Begin Trust Region Iterations
    # =========================================================================

    startTime = time.time()
    while k < tr.max_iterations:
    
        k += 1
        M.increment_iteration()
        M.assign_to_trust_region_history(k,t,trc,tr.size)
        tr.set_center(trc)
    
        
        # Set necessary local variables
        xOpt = np.zeros(x.n) # values of design variables at optimum
        fOpt = None
        gOpt = np.zeros(mi+me) # values of nonlinear constraints at optimum

        # =========================================================================
        # Evaluate function values and derivatives @ center of trust region
        # =========================================================================
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
        
        # Record initial values to general history
        if( k == 1 ):
            fHistory.append(f[-1])
            gHistory.append(g[-1])
            
        # =========================================================================
        # Calculate correction for all responses
        # =========================================================================         
        
        corr = calculate_correction(f,g,df,dg,tr)

        # =========================================================================
        # Calculate constraint violation for all fidelities
        # =========================================================================  
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

        # Set trust region bounds
        xTrLower = np.max(np.vstack((x.lower_bound,x.value-tr.size)),axis=0)
        xTrUpper = np.min(np.vstack((x.upper_bound,x.value+tr.size)),axis=0)
        
        opt_prob = Optimization('TRMM Subproblem',evaluate_corrected_model)
        for i in range(x.n):
            variableName = 'x' + str(i)
            opt_prob.addVar(variableName,'c',lower=xTrLower[i],upper=xTrUpper[i],value=x.value[i])
        opt_prob.addObj('f',value=f[-1])
        
        for i in range(mi):
            constraintName = 'gi' + str(i)
            opt_prob.addCon(constraintName,'i',lower=opt.constraint_lower_bounds[i], \
                            upper=opt.constraint_upper_bounds[i],value=g[-1][i])
            
        print opt_prob   
            
        # =========================================================================
        # Solve Subproblem
        # =========================================================================        

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
            else:
                raise NotImplementedError('Only FD or user are currently accepted for pyOpt gradient calculations')

            optInformValue = opt_prob._solutions[0].opt_inform['value']
            optInformMessage = opt_prob._solutions[0].opt_inform['text']
            fOpt = opt_prob._solutions[0]._objectives[0].value
            for i in range(x.n):
                xOpt[i] = opt_prob._solutions[0]._variables[i].value
            for i in range(mi+me):
                gOpt[i] = opt_prob._solutions[0]._constraints[i].value

        else:
            raise NotImplementedError('Only SNOPT (using pyOpt) is currently implemented')        
         
        # Check nonlinear constraint violation for low-fidelity model
        if( mi + me > 0 ):
            gviol2_lo = calculate_constraint_violation(gOpt,opt)     
                    
            
        # =========================================================================
        # Check Hard Convergence
        # =========================================================================
        # If optimizer terminates successfully at initial value of design variables
        # then we have convergence
        if( optInformValue == 1 and \
            np.sum(np.isclose(xOpt,x.value,rtol=1e-14,atol=1e-12)) == x.value.size ):

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
            
            x2 = copy.copy(x)
            x2.value = xOpt
            setupSharedModelData(x2,y,log_file,flow)
                        
        # Evaluate highest-fidelity function at optimum
        fOpt_hi, gOpt_hi = evaluate_model(xOpt,y,log_file,flow,opt,np.max(flow.fidelity_levels),'val', my_function)
        
        # Append optimal value to history
        fHistory.append(fOpt_hi)
        gHistory.append(gOpt_hi)
                    
        # Calculate constraint violation for high-fidelity model
        if( mi + me > 0 ):
            gviol2_hi = calculate_constraint_violation(gOpt_hi,opt)
                    
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
                accepted = 1
            elif( fOpt_hi < f[-1] ):
                log.write('Trust region update accepted since objective value is lower\n')
                accepted = 1
            elif( gviol2_hi < gviol ):
                log.write('Trust region update accepted since nonlinear constraint violation is lower\n')    
                accepted = 1
            else:
                log.write('Trust region update rejected (filter)\n')
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
        
        # Terminate if trust region too small
        if( tr.size < tr.minimum_size ):
            terminateMessage = 'trust region too small'
            break
            
        # Terminate if solution is infeasible, no change is detected, and trust region does not expand
        if( optInformValue == 13 and tr_action < 3 and \
            np.sum(np.isclose(xOpt.astype(np.float64),x.value.astype(np.float64),rtol=1e-15,atol=1e-14)) == x.value.size ):
            terminateMessage = 'solution infeasible, no improvement can be made'
            break 

        # =========================================================================
        # Update Trust Region Center
        # =========================================================================            
        if( accepted ):
            x.value[:] = copy.copy(xOpt)
            t = M.increment_trust_region_center_index()
            trc = M.set_trust_region_center(np.hstack((x.value,y.value)))            
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


    # =============================================================================
    # Save history
    # =============================================================================      
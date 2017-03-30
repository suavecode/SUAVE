'''
Define functions used to call user functions.

Rick Fenrich 2/22/17
'''

import numpy as np
import multiprocessing
import sys
import os

#import my_function

import market as M

# Setup function called to prep function evaluations    
def setup(x,y,nCores, my_function):

    h, hs, d = my_function.modelSetup(x,y,nCores)

    # Add evaluations to history so they not be forgotten, arr!
    for tag in h:
        for e in range(len(h[tag]['x'])):
            M.assignToHistory(tag,h[tag]['x'][e],h[tag]['y'][e],h[tag]['response'][e])
            
    # Add evaluations to history so they not be forgotten, arr!
    for e in range(len(hs['x'])):
        M.assignToSurrogateHistory(hs['level'][e],M.K,hs['x'][e],hs['y'][e],hs['objective'][e],hs['constraints'][e])
            
    # Add shared user data to global market M
    for key in d:
        M.addUserData(key,d[key])
    
    return 0
    
    
def checkForDuplicateEvals(xlist,ylist,level,flow,surrHist,trhistory,k):

    # What to search
    scope = flow.function_dependency[level-1]
    
    # Extract history specific to model fidelity of given level
    history = surrHist[level]
    
    # Perform search
    evalFlag = []
    for qq in range(len(xlist)):
    
        xval = xlist[qq]
        if( len(ylist) == 1 ):
            yval = ylist[0]
        else:
            yval = ylist[qq]
    
        dup = 0
        
        # Search entire history in all trust regions
        if( scope == 'evaluation_point' ):
        
            for i in range(len(history['x'])):
                if( np.sum(np.isclose(history['x'][i],xval,rtol=1e-15,atol=1e-14)) == xval.size  and np.sum(np.isclose(history['y'][i],yval,rtol=1e-15,atol=1e-14)) == yval.size ):
                    evalFlag.append([history['objective'][i],history['constraints'][i]])
                    print 'Duplicate evaluation found for {}'.format([e for e in xval])
                    dup = 1
                    break
            if not dup:
                evalFlag.append(0)
        
        # Search history in current trust region only
        elif( scope == 'trust_region' ):
        
            for i in range(len(history['x'])):
                if( history['iter'][i] == k ): # same trust region
                    if( np.sum(np.isclose(history['x'][i],xval,rtol=1e-15,atol=1e-14)) == xval.size and np.sum(np.isclose(history['y'][i],yval,rtol=1e-15,atol=1e-14)) == yval.size ):
                        evalFlag.append([history['objective'][i],history['constraints'][i]])
                        print 'Duplicate evaluation found for {}'.format([e for e in xval])
                        dup = 1
                        break
            if not dup:
                evalFlag.append(0)                    
        
        # Search history in trust regions which have same center
        elif( scope == 'trust_region_center' ):
        
            # Make list of trust region iterations which have the same center as current region
            candidates = []
            center = trhistory['center'][k-1]
            for i in range(len(trhistory['iter'])):
                if( np.sum(np.isclose(trhistory['center'][k-1],trhistory['center'][i],rtol=1e-14,atol=1e-12)) == trhistory['center'][i].size ):
                    candidates.append(i+1)
        
            for i in range(len(history['x'])):
                if( history['iter'][i] in candidates ): # same trust region center
                    if( np.sum(np.isclose(history['x'][i],xval,rtol=1e-15,atol=1e-14)) == xval.size and np.sum(np.isclose(history['y'][i],yval,rtol=1e-15,atol=1e-14)) == yval.size ):
                        evalFlag.append([history['objective'][i],history['constraints'][i]])
                        print 'Duplicate evaluation found for {}'.format([e for e in xval])
                        dup = 1
                        break
            if not dup:
                evalFlag.append(0) 
        
        else:
            raise NotImplementedError('Function dependency of %s not implemented' % scope)
            
    #end        
    
    return evalFlag
    
    
def model_evaluation_setup(xval,yitem,data,level,dirname,link_files, my_function):

    # yitem can be a variables class instance or a numpy array

    if( dirname ):
    
        # Create and enter new directory        
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        else:
            print 'directory %s already exists' % dirname
            #raise EnvironmentError('directory %s already exists' % dirname)
            
        os.chdir(dirname)
        
        # Link necessary files
        for f in link_files:
            if( os.path.exists(f) ):
                pass
            else:
                try:
                    os.link(os.path.join('..','..',f),f)
                except:
                    os.link(os.path.join('..',f),f)
                
        output = my_function.model_evaluation(xval,yitem,data,level)
        
        os.chdir('..')
        
    else:
    
        output = my_function.model_evaluation(xval,yitem,data,level)
        
    return output


# Function called to evaluate responses (& derivatives) given inputs
def function(xval,y,flow,opt,level,ret, my_function):

    # Obtain necessary global variables
    n = M.SURR_EVAL[level] # index of last function evaluation of desired fidelity level
    surrHistory = M.SURR_HIST # surrogate function histories
    history = M.HIST # truth function histories
    trHistory = M.TR_HIST
    k = M.K
    data = M.USER_DATA

    # =========================================================================
    # Evaluate and return function values and derivatives
    # =========================================================================    
    if( ret == 'all' or ret == 'der' ):

        # =========================================================================
        # Build list of function evaluation points
        # =========================================================================    

        mEval = [] # multiprocessing array to record responses
        iEval = [] # number of function evaluation for specified model level
        hEval = [] # local truth function histories for each evaluation of model

        # Forward finite difference method
        if( flow.gradient_evaluation[level-1] == 'FD' ):
            
            xEval = [] # design variable values
            rEval = [] # response for each evaluation
            fd_step = opt.difference_interval
            
            # Evaluation to obtain value at center
            n += 1
            iEval.append(n)
            mEval.append(-1)
            rEval.append(-1)
            hEval.append(-1)
            xEval.append(xval[:])
            
            # Evaluations to obtain values at center + delta away
            for p in range(xval.size):
                n += 1
                iEval.append(n)
                mEval.append(-1)
                rEval.append(-1)
                hEval.append(-1)   
                d = np.zeros((xval.size,))
                d[p] = fd_step
                xEval.append(xval[:]+d)
                         
            #print xEval
            #print rEval
            #print iEval
           
        # Smart-pce method: used if QoI is a response level corresponding to a given
        # CDF probability: i.e. QoI g is calculated as dg/dx_i = dg/dF * dF/dx_i where
        # F is the CDF of g. Both derivatives are evaluated at the required CDF probability.
        # Hence the user must provide user data, namely a list of dg/dF for each g in
        # data['smart-pce-coef'] and a list of y-values corresponding to response at each
        # CDF threshold probability for each g in data['smart-pce-yeval']. dF/dx_i is 
        # calculated using forward finite difference.
        elif( flow.gradient_evaluation[level-1] == 'smart-pce' ):
            
            xEvalLocal = [] # design variable values
            yEvalLocal = [] # parameter values
            rEvalLocal = [] # response for each evaluation
            fd_step = opt.difference_interval
            
            for p in range(xval.size):
            
                d = np.zeros((xval.size,))
                d[p] = fd_step
                
                for q in range(opt.num_constraints+1):
                
                    # Evaluations at center
                    n += 1
                    iEval.append(n)
                    mEval.append(-1)
                    rEvalLocal.append(-1)
                    hEval.append(-1)
                    xEvalLocal.append(xval[:])
                    yEvalLocal.append(data['smart-pce-yeval'][q])
                    
                    # Evaluations at center + delta away
                    n += 1
                    iEval.append(n)
                    mEval.append(-1)
                    rEvalLocal.append(-1)
                    hEval.append(-1)
                    xEvalLocal.append(xval[:]+d)
                    yEvalLocal.append(data['smart-pce-yeval'][q])
                    
            derCoef = data['smart-pce-coef']
            
            #print xEvalLocal
            #print yEvalLocal
            #print rEvalLocal
            #print iEval
            
        else:
            raise NotImplementedError('Only gradient calculation via finite difference currently enabled')
        
        # =========================================================================
        # Check for duplicate evaluations
        # =========================================================================    
        #print 'Checking for duplicate evalutions of function level %i' % level
        if( flow.gradient_evaluation[level-1] == 'FD' ):
            duplicate = checkForDuplicateEvals(xEval,[y.value],level,flow,surrHistory,trHistory,k)
        elif( flow.gradient_evaluation[level-1] == 'smart-pce' ):
            duplicate = [0]*len(iEval) # disable duplicate evaluation checking
        
        # =========================================================================
        # Begin function evaluations
        # =========================================================================  
        
        # Serial evaluation  
        if( flow.n_cores == 1):
            
            for i in range(len(iEval)):
            
                if not duplicate[i]:
                    
                    if( flow.function_evals_in_unique_directory ):
                        dirname = os.path.join(os.getcwd(), 'F' + str(level) + '_' + str(iEval[i]))
                    else:
                        dirname = 0
                    
                    # Finite difference hands x values and y definition over
                    if( flow.gradient_evaluation[level-1] == 'FD' ):
                        rEval[i], hEval[i] = model_evaluation_setup(xEval[i],y,data, \
                                         level,dirname,flow.link_files, my_function)       
                    # Smart-pce hands x values and y value at CDF threshold over             
                    elif( flow.gradient_evaluation[level-1] == 'smart-pce' ):
                        rEvalLocal[i], hEval[i] = model_evaluation_setup(xEvalLocal[i],yEvalLocal[i],data, \
                                         level,dirname,flow.link_files)
                                         
                else: 
                
                    if( flow.gradient_evaluation[level-1] == 'FD' ):
                        rEval[i] = np.hstack(duplicate[i])
                    elif( flow.gradient_evaluation[level-1] == 'smart-pce' ):
                        rEvalLocal[i] = np.hstack(duplicate[i])
                    hEval[i] = 0
        
        # Parallel evaluation with multiprocessing pool
        else:
        
            print 'Starting multiprocessing pool with %i processes' % flow.n_cores
            pool = multiprocessing.Pool(processes=flow.n_cores)
            
            for i in range(len(iEval)):
            
                if not duplicate[i]:
                    if( flow.function_evals_in_unique_directory ):
                        dirname = os.path.join(os.getcwd(), 'F' + str(level) + '_' + str(iEval[i]))
                    else:
                        dirname = 0
                        
                    # Finite difference hands x values and y definition over    
                    if( flow.gradient_evaluation[level-1] == 'FD' ):
                        mEval[i] = pool.apply_async(model_evaluation_setup, \
                        (xEval[i],y,data,level,dirname,flow.link_files))
                    # Smart-pce hands x values and y value at CDF threshold over
                    elif( flow.gradient_evaluation[level-1] == 'smart-pce' ):
                        mEval[i] = pool.apply_async(model_evaluation_setup, \
                        (xEvalLocal[i],yEvalLocal[i],data,level,dirname,flow.link_files))
                        
                else:
                
                    if( flow.gradient_evaluation[level-1] == 'FD' ):
                        rEval[i] = np.hstack(duplicate[i])
                    elif( flow.gradient_evaluation[level-1] == 'smart-pce' ):
                        rEvalLocal[i] = np.hstack(duplicate[i])
                    hEval[i] = 0
                    
            pool.close()
            pool.join()
        
            # Obtain results of calculation
            for i in range(len(mEval)):
                if not duplicate[i]:
                    if( flow.gradient_evaluation[level-1] == 'FD' ):
                        rEval[i], hEval[i] = mEval[i].get()
                    elif( flow.gradient_evaluation[level-1] == 'smart-pce' ):
                        rEvalLocal[i], hEval[i] = mEval[i].get()
            
        #print rEval
        #print rEvalLocal
        #print nEval
        
        # =========================================================================
        # Assemble function values
        # ========================================================================= 
        
        # Smart-pce method uses point evaluations
        if( flow.gradient_evaluation[level-1] == 'smart-pce' ):
        
            if( ret == 'all' ): # must also return function values
                
                n += 1
                iEval.append(n)
                rEval = [-1]
                hEval.append(-1)                
                xEval = [xval[:]]
                
                # Check for duplicate evaluations
                #print 'Checking for duplicate evalutions of function level %i' % level
                duplicate = checkForDuplicateEvals(xEval,[y.value],level,flow,surrHistory,trHistory,k)
        
                if not duplicate[0]:
                
                    print('\nWARNING: An (expensive) function evaluation (might be able to) be saved for fidelity %i if functions are evaluated in my_function.modelSetup() and appended to surrogate history.\n\n')
                
                    if( flow.function_evals_in_unique_directory ):
                        dirname = os.path.join(os.getcwd(), 'F' + str(level) + '_' + str(iEval2))
                    else:
                        dirname = 0
                    rEval[0], hEval[-1] = model_evaluation_setup(xEval[0],y,data,level,dirname,flow.link_files)
                    
                else:
                
                    rEval[0] = np.hstack(duplicate[0])
                    hEval[-1] = 0
                    
                f = rEval[0][0]
                g = rEval[0][1:]        
                
            else:
                f = 0
                g = 0
                
        else:
        
            f = rEval[0][0]
            g = rEval[0][1:]

        # =========================================================================
        # Assemble gradients
        # ========================================================================= 

        # [ f_0   g_0   g_1   ... g_m   ]
        # [ f_0+  g_0+  g_1+  ... g_m+  ]
        # [ ...                   ...   ]
        # [ f_0n+ g_0n+ g_1n+ ... g_mn+ ]

        m = rEval[0].size - 1            
        df = np.zeros((1,xval.size))
        dg = np.zeros((m,xval.size))
        
        if( flow.gradient_evaluation[level-1] == 'FD' ):
        
            for i in range(df.size):
                df[0,i] = (rEval[i+1][0] - f)/fd_step
                
            for i in range(df.size):
                for j in range(m):
                    dg[j,i] = (rEval[i+1][j+1] - g[j])/fd_step
                    
        elif( flow.gradient_evaluation[level-1] == 'smart-pce' ):
        
            for i in range(df.size):
                df[0,i] = derCoef[0]*(rEvalLocal[2*i*(m+1)+1][0] - \
                          rEvalLocal[2*i*(m+1)][0])/fd_step
                
            for i in range(df.size):
                for j in range(m):
                    dg[j,i] = derCoef[j+1]*(rEvalLocal[2*i*(m+1)+3+2*j][j+1] - \
                              rEvalLocal[2*i*(m+1)+2+2*j][j+1])/fd_step            

    # =========================================================================
    # Evaluate and return function values only
    # =========================================================================    
    elif( ret == 'val' ):
    
        n += 1
        iEval = n
        rEval = [-1]
        hEval = [-1]
        xEval = [xval[:]]

        # Check for duplicate evaluations
        #print 'Checking for duplicate evalutions of function level %i' % level
        duplicate = checkForDuplicateEvals(xEval,[y.value[:]],level,flow,surrHistory,trHistory,k)
        
        if not duplicate[0]:
        
            if( flow.function_evals_in_unique_directory ):
                dirname = os.path.join(os.getcwd(), 'F' + str(level) + '_' + str(iEval))
            else:
                dirname = 0
                
            rEval[0], hEval[0] = model_evaluation_setup(xEval[0],y,data,level,dirname,flow.link_files, my_function)
            
        else:
        
            rEval[0] = np.hstack(duplicate[0])
            hEval[0] = 0
        
        # Put together evaluations
        f = rEval[0][0]
        g = rEval[0][1:]
        df = 0
        dg = 0
                
    else:
    
        raise ValueError("Only 'all', 'der', or 'val' are accepted as return keywords")
            
    #end

    # =========================================================================
    # Add evaluations to history so they not be forgotten, arr!
    # =========================================================================    
    
    for p in range(len(xEval)):
        M.assignToSurrogateHistory(level,k,xEval[p],y.value,rEval[p][0],rEval[p][1:])
    
    for h in hEval:
        if h: # skip assignment of duplicate values
            for tag in h:
                for e in range(len(h[tag]['x'])):
                    M.assignToHistory(tag,h[tag]['x'][e],h[tag]['y'][e],h[tag]['response'][e])
            
    return f, g, df, dg

""" Solvers.py: Methods for numerical solutions of systems """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import sys
import numpy as np
from SUAVE.Plugins.ADiPy import *

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

def newton(x,function,data,tol_f=1.0e-8,tol_x=1.0e-10,max_iters=1000,max_step=100.0):
  
    tol_min = 10*sys.float_info.epsilon

    N = len(x)
    check = False

    fv = function(x,data)
    if len(fv) != N:
        print "Newton solver: vector returned by function call is different length than x, exiting" 
        check = True
        return x0, None, check

    # Test for initial guess being a root  
    if (np.max(np.abs(fv)) < tol_f):
        print "Newton solver: Initial guess is a root, exiting..." 
        return x0, fv, check

    # initialize minimization function
    f = 0.5*np.dot(fv,fv)

    # Calculate stpmax for line searches
    x_norm = np.linalg.norm(x)    
    stpmax = max_step*np.max(np.append(x_norm,N))

    # iteration loop
    for i in xrange(0,max_iters):  

        # evaluate function
        fv = function(x,data)

        # evaluate Jacobian
        J = jacobian_AD(x,data,function)

        # gradient of f
        g = np.dot(np.transpose(J),fv) 

        x_old = x.copy()         # store x 
        f_old = f         # store f 

        # solve linear system
        p = np.linalg.solve(J,-fv)

        # perform line search
        x, fv, f, check = line_search(x_old,f_old,g,p,function,data,stpmax) 
        # line_search(x_old,f_old,g,p,function,max_step,tol_x=1e-8,alpha=1e-4)
        # Lnsr chreturns new x and f. It also calculates fvec at the new x when it calls fmin. 

        # begin various convergence tests
        x_scale = np.max(np.append(np.abs(x),1.0))

        # test for convergence on function values 
        if (np.max(np.abs(fv)) < tol_f):
            print "Newton solver: Converged function values, exiting..."
            return x, fv, check

        # check for spurious convergence
        if check:
            g_max = np.max(np.abs(g))           
            f_scale = np.max([f,0.5*n])
            check = (g_max/(f_scale/x_scale) < tol_min)
            return x, fv, check

        # Test for convergence on dx
        if np.max(np.abs(x - x_old))/x_scale < tol_x:
            print "Newton solver: Convergence change in variable values, exiting..."
            return x, fv, check 

    # end iteration loop
    print "Newton solver: maximum iterations exceeded, exiting..." 

    return x, fv, check
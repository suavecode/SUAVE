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

def line_search(x_old,f_old,g,p,function,data,max_step,tol_x=1e-8,alpha=1e-4): 

    check = False  

    # Scale if attempted step is too big
    if np.linalg.norm(p) > max_step: 
        p *= max_step/np.norm(p)  

    slope = np.dot(g,p) 
    if (slope >= 0.0):
        print "Newton solver: roundoff problem in line search, exiting..."
        return x_old, None, f_old, check

    x_scale = np.max(np.append(np.abs(x_old),1.0))
    lamda_norm = np.max(np.abs(p)/x_scale)

    alamin = tol_x/lamda_norm 
    alam = 1.0

    while True:
        
        # take step
        x = x_old + alam*p
        
        # evaluate function     
        fv = function(x,data) 
        f = 0.5*np.dot(fv,fv)
            
        if alam < alamin:                               # convergence on dx 
            x = xold 
            check = True 
            return x, fv, f, check

        elif f <= f_old + alpha*alam*slope:             # sufficient function decrease, backtrack
            return x, fv, f, check
            
        else:
            if alam == 1.0:
                tmplam = -slope/(2.0*(f-f_old-slope))   # first attempt
            else:                                       # subsequent backtracks
                rhs1 = f - f_old - alam*slope 
                rhs2 = f2-fold-alam2*slope; 
                a = (rhs1/(alam*alam) - rhs2/(alam2*alam2))/(alam - alam2) 
                b = (-alam2*rhs1/(alam*alam) + alam*rhs2/(alam2*alam2))/(alam - alam2) 
                if a == 0.0:
                    tmplam = -slope/(2.0*b) 
                else: 
                    disc = b*b - 3.0*a*slope 
                    if (disc < 0.0):
                        tmplam = 0.5*alam 
                    elif (b <= 0.0): 
                        tmplam = (-b + np.sqrt(disc))/(3.0*a)
                    else: 
                        tmplam = -slope/(b + np.sqrt(disc)) 

                if (tmplam > 0.5*alam):
                    tmplam = 0.5*alam 

        alam2 = alam; 
        f2 = f; 
        alam = np.max([tmplam,O.l*alam])                # try again

def jacobian_AD(x,data,function):

    xi = ad(x,np.eye(len(x)))
    res = jacobian(function(xi,data))

    return res


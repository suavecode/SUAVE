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

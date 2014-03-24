""" test_AD.py: test AD functionality with toy problem """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import sys
import numpy as np
import matplotlib.pyplot as plt
from adipy import *
from time import time
# from SUAVE.Plugins.ADiPy import *
# from SUAVE.Methods.Solvers import newton

# ----------------------------------------------------------------------
#  Main
# ----------------------------------------------------------------------

def main():

    # control points
    N = 64

    # solution tolerance
    tol = 1.0e-8

    # create x vector and d/dx^2, I operators
    x, D, I = chebyshev(N)
    D2 = np.dot(D,D)

    # compute objective function
    # alpha = ad(np.pi)
    alpha = np.pi + 0j
    f = objective(alpha,N,tol,x,D2,I)
    # print f.d(alpha)
    
    return

def objective(alpha,N,tol,x,D2,I,plot=True):

    # initial guess
    f0 = np.zeros(N - 2)

    # pack data
    data = [alpha, D2]

    # call solver
    # t0 = time()
    # solution = root(residuals, f0, args=(alpha,D2), method="hybr", tol=tol, jac=J)
    f, R, check = newton(f0,residuals,data)
    # dt = time() - t0
    # print N, dt
    
    # append BCs
    f = np.append(0.0,f)
    f = np.append(f,0.0)

    if plot:
        # plot results     
        plt.plot(x, f, 'o-')
        title = r"$\frac{d^2f}{d x^2} = e^{\pi f}$"
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title(title, fontsize=18)
        plt.grid(True)       
        plt.show()

    return np.dot(I,f)[-1]

def residuals(f,data):

    # unpack
    alpha = data[0]
    D2 = data[1]

    # append BCs
    f = np.append(0.0,f)
    f = np.append(f,0.0)

    # compute residuals
    R = np.dot(D2,f) - exp(alpha*f)

    return R[1:-1]

def J(x,data,function):

    xi = ad(x,np.eye(len(x)))
    res = jacobian(function(xi,data))

    return res

def chebyshev(N,integration=True):
    N = int(N)

    # error checking:
    if N <= 0:
        print "N must be > 0"
        return []   

    # initialize
    D = np.zeros((N, N))

    # x array
    x = 0.5*(1 - np.cos(np.pi*np.arange(0, N)/(N - 1)))

    # D operator
    c = np.array(2)
    c = np.append(c, np.ones(N - 2))
    c = np.append(c, 2)
    c = c*((-1)**np.arange(0, N))
    A = np.tile(x, (N, 1)).transpose()
    dA = A - A.transpose() + np.eye(N)
    cinv = 1/c

    for i in range(N):
        for j in range(N):
            D[i, j] = c[i]*cinv[j]/dA[i, j]

    D = D - np.diag(np.sum(D.transpose(),axis=0))

    # I operator
    if integration:
        I = np.linalg.inv(D[1:, 1:])
        I = np.append(np.zeros((1, N - 1)), I, axis=0)
        I = np.append(np.zeros((N, 1)), I, axis=1)
        return x, D, I
    else:
        return x, D

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
        jac = J(x,data,function)

        # gradient of f
        g = np.dot(np.transpose(jac),fv) 

        x_old = x.copy()         # store x 
        f_old = f         # store f 

        # solve linear system
        p = np.linalg.solve(jac,-fv)

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

# call main
if __name__ == '__main__':
    main()

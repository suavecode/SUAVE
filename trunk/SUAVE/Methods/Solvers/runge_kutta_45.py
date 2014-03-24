""" Utilities.py: Mathematical tools and numerical integration methods for ODEs """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
#import ad
from scipy.optimize import root   #, fsolve, newton_krylov
from SUAVE.Structure import Data
from SUAVE.Attributes.Results import Segment

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------


def runge_kutta_45(problem,options):

    """  t, z = runge_kutta_45(problem,options): integrate a system of ODEs using Runge-Kutta-Fehlberg 4th order
    
         Inputs:    problem.f = function handle to ODE function of the form z' = f(t,y)     (required)                          (function handle)    
                    problem.t0 = initial time                                               (required)                          (float)
                    problem.tf = final time                                                 (required)                          (float)
                    problem.z0 = array of initial conditions                                (required)                          (floats)
                    problem.zmin = array of minimum conditions                              (required / optional if tf defined) (floats)
                    problem.zmax = array of maximum conditions                              (required / optional if tf defined) (floats)
                    problem.h0 = initial time step                                          (required)                          (float)
                    problem.config = vehicle configuration instance                         (required for Mission / Segment)    (class instance)

                    options.tol_solution = solution tolerance                               (required)                          (float)
                    options.tol_BCs = boundary condition tolerance                          (required)                          (float)

         Outputs:   solution.t = time vector                                                                                             (floats)
                    solution.z = m-column array of y and all its derivatives                                                     (floats)

        """

    # check input data
    m = len(problem.z0); err = False
    if len(problem.zmin) is not m:
        print "z0 and zmin are different lengths, please check inputs."; err = True
    if len(problem.zmax) is not m:
        print "z0 and zmax are different lengths, please check inputs."; err = True
    if problem.tf is not None:
        if problem.tf <= problem.t0:
            print "Final time is <= initial time, please check inputs."; err = True
    if problem.h0 is None or problem.h0 == 0.0:
        print "Initial time step is zero or undefined, please check inputs."; err = True
    if problem.tf is None:
        err = True
        for j in range(m): 
            if not np.isnan(problem.zmin[j]) and np.isfinite(problem.zmin[j]):
                err = False
            if not np.isnan(problem.zmax[j]) and np.isfinite(problem.zmax[j]):
                err = False
        if err:
            print "No final time or final conditions detected, please check inputs."
    for j in range(m): 
        if problem.z0[j] < problem.zmin[j]:
            print "Initial condition " + str(j) + " is < minimum conditions, please check inputs."; err = True
        elif problem.z0[j] > problem.zmax[j]:
            print "Initial condition " + str(j) + " is > maximum conditions, please check inputs."; err = True
        elif problem.zmin[j] > problem.zmax[j]:
            print "Minimum condition " + str(j) + " is > maximum condition, please check inputs."; err = True
    if err:
        return None

    # RKF coefficients (to prevent repeated operations)
    h = problem.h0
    hk = np.array([0.0, 0.25, 0.375, 12.0/13, 1.0, 0.5])
    ck2 = np.array([3.0/32, 9.0/32])
    ck3 = np.array([1932.0/2197, -7200.0/2197, 7296.0/2197])
    ck4 = np.array([439.0/216, -8.0, 3680.0/513, -845.0/4104])
    ck5 = np.array([-8.0/27, 2.0, -3544.0/2565, 1859.0/4104, -11.0/40])
    cz  = np.array([25.0/216, 1408.0/2565, 2197.0/4104, -1.0/5])
    cz_ = np.array([16.0/135, 6656.0/12825, 28561.0/56430, -9.0/50, 2.0/55])

    # set up ICs
    z = np.zeros((1,m)); z[0,:] = np.array(problem.z0); 
    
    # initialize arrays
    t = np.array([problem.t0]); k = np.zeros((6,m))
    z_new = np.zeros((1,m)); z_new_ = np.zeros(m)

    # run integration
    stop = False; mins = [False]*m; maxes = [False]*m; i = 0; solution = Solution()
    while not stop:

        # Fehlberg RK45 step
        dt = t[i] + h*hk
        k[0,:] = h*problem.f(dt[0], z[i,:])        
        k[1,:] = h*problem.f(dt[1], z[i,:] + k[0,:]/4)
        k[2,:] = h*problem.f(dt[2], z[i,:] + ck2[0]*k[0,:] + ck2[1]*k[1,:])
        k[3,:] = h*problem.f(dt[3], z[i,:] + ck3[0]*k[0,:] + ck3[1]*k[1,:] + ck3[2]*k[2,:])
        k[4,:] = h*problem.f(dt[4], z[i,:] + ck4[0]*k[0,:] + ck4[1]*k[1,:] + ck4[2]*k[2,:] + ck4[3]*k[3,:])
        k[5,:] = h*problem.f(dt[5], z[i,:] + ck5[0]*k[0,:] + ck5[1]*k[1,:] + ck5[2]*k[2,:] + ck5[3]*k[3,:] + ck5[4]*k[4,:])
        z_new[0,:]  = z[i,:] + cz[0]*k[0,:] + cz[1]*k[2,:] + cz[2]*k[3,:] + cz[3]*k[4,:]
        z_new_      = z[i,:] + cz_[0]*k[0,:] + cz_[1]*k[2,:] + cz_[2]*k[3,:] + cz_[3]*k[4,:] + cz_[4]*k[5,:]
            
        # adapt time step
        R = np.linalg.norm(z_new[0,:] - z_new_)/h
        d = 0.84*(options.tol_solution/R)**0.25     
        t_new = t[i] + h

        if R <= options.tol_solution:

            append = True

            # test for final time
            if problem.tf is not None: 
                if np.abs(t_new - problem.tf) < options.tol_BCs:
                    stop = True; h = 0.0; print "h ---> 0 due to tf"
                    solution.exit.j = m; solution.exit.err = np.abs(t_new - problem.tf) 
                    solution.exit.reason = "maximum time reached"
                elif t_new > problem.tf:
                    h = problem.tf - t[i]; append = False
                    # print "over end time"
            
            # in endpoint mode
            if (any(mins) or any(maxes)) and not stop:
                print "endpoint mode"
                for j in range(m):
                    if mins[j]:
                        dz = np.abs(problem.zmin[j] - z_new[0,j])
                        if dz < options.tol_BCs:
                            solution.exit.j = j; solution.exit.err = dz 
                            solution.exit.reason = "minimum value reached"
                            stop = True; h = 0.0; print "h ---> 0 due to zmin"; break
                        else:
                            slope = problem.f(t_new,z_new[0,:])
                            h += (problem.zmin[j] - z_new[0,j])/slope[j]
                            append = False
                    elif maxes[j]:
                        dz = np.abs(problem.zmax[j] - z_new[0,j])
                        if dz < options.tol_BCs:
                            solution.exit.j = j; solution.exit.err = dz 
                            solution.exit.reason = "maximum value reached"
                            stop = True; h = 0.0; print "h ---> 0 due to zmax"; break
                        else:
                            slope = problem.f(t_new,z_new[0,:])
                            h += (problem.zmax[j] - z_new[0,j])/slope[j]
                            append = False

            # check stopping criteria
            if not stop:
                for j in range(m):
                    if not np.isnan(problem.zmin[j]) and np.isfinite(problem.zmin[j]):
                        dz = np.abs(problem.zmin[j] - z_new[0,j])
                        if dz < options.tol_BCs:
                            solution.exit.j = j; solution.exit.err = dz 
                            solution.exit.reason = "minimum value reached"
                            stop = True; break
                        else:
                            if z_new[0,j] < problem.zmin[j]:
                                slope = (z_new[0,j] - z[i,j])/h
                                h = (problem.zmin[j] - z[i,j])/slope           # do a linear interpolation to estimate crossing time
                                # print "min end condition found"
                                append = False; mins[j] = True;
                    if not np.isnan(problem.zmax[j]) and np.isfinite(problem.zmax[j]):
                        dz = np.abs(problem.zmax[j] - z_new[0,j])
                        if dz < options.tol_BCs:
                            solution.exit.j = j; solution.exit.err = dz 
                            solution.exit.reason = "maximum value reached" 
                            stop = True; break
                        else:
                            if z_new[0,j] > problem.zmax[j]:
                                slope = (z_new[0,j] - z[i,j])/h
                                h = (problem.zmax[j] - z[i,j])/slope           # do a linear interpolation to estimate crossing time
                                # print "max end condition found"
                                append = False; maxes[j] = True;
               
            if append:
                # advance to next step, append to time and state vectors
                i += 1; 
                z = np.append(z,z_new,axis=0) 
                t = np.append(t,t_new)
                if not (any(mins) or any(maxes)):
                    h = d*h

        else:
            h = d*h

        #print i, h, R, d, z_new[0,:]
        #raw_input()

    # package results and return
    solution.t = t
    solution.z = z

    return solution
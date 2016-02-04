""" Utilities.py: Mathematical tools and numerical integration methods for ODEs """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import autograd.numpy as np 
#import ad
##from scipy.optimize import root   #, fsolve, newton_krylov
from SUAVE.Core import Data

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

def pseudospectral(problem):

    """  solution = pseudospectral(problem,options): integrate a system of ODEs using Chebyshev pseudospectral method
    
         Inputs:    problem.f = function handle to ODE function of the form z' = f(t,y)     (required)                          (function handle)    
                    problem.FC = function handle to final condition of the form FC(z) = 0   (required)                          (function handle)
                    problem.t0 = initial time                                               (required)                          (float)
                    problem.tf = final time estimate                                        (required)                          (float)
                    problem.z0 = array of initial conditions                                (required)                          (floats)
                    problem.config = vehicle configuration instance                         (required for Mission / Segment)    (class instance)

                    options.tol_solution = solution tolerance                               (required)                          (float)
                    options.tol_BCs = boundary condition tolerance                          (required)                          (float)
                    options.Npoints = number of control points                              (required)                          (int)

         Outputs:   solution.t = time vector                                                                                             (floats)
                    solution.z = m-column array of state variables                                                     (floats)

        """

    # some packing and error checking (needs more work - MC)
    err = False
    if not problem.unpack:
        print "Error: no unpacking function provided. Exiting..."
        err = True; return []

    err = problem.initialize()
    if err:
        print "Error: problem reported with initialization. Exiting..."
        return[]

    x_state_0, x_control_0, dt = problem.unpack(problem.guess)
    
    try: 
        problem.options.N
    except AttributeError:
        print "Warning: number of control points not specified. Using size of initial guess."
        if len(x_state) == 0:
            problem.Nstate = 0
        else:
            if len(np.shape(x_state_0)) == 2:
                problem.options.N, problem.Nstate = np.shape(x_state_0)
            elif len(np.shape(x_state_0)) == 1:
                problem.options.N = np.shape(x_state_0)
                problem.Nstate = 1
    else:
        if len(x_state_0) == 0:
            problem.Nstate = 0
        else:
            if len(np.shape(x_state_0)) == 2:
                rows, problem.Nstate = np.shape(x_state_0)
            elif len(np.shape(x_state_0)) == 1:
                rows = np.shape(x_state_0)
                problem.Nstate = 1
        
                if problem.options.N != rows:
                    print "Warning: number of control points specified does not match size of initial guess. Overriding with size of guess."
                    problem.options.N = rows

    if len(x_control_0) == 0:
        problem.Ncontrol = 0
    else:
        if len(np.shape(x_control_0)) == 2:
            rows, problem.Ncontrol = np.shape(x_control_0)
        elif len(np.shape(x_control_0)) == 1:
            rows = np.shape(x_control_0)

        if problem.options.N != rows:
            print "Warning: number of control points does not match between state and control variables. Exiting..."
            err = True; return []

    if not dt:
        problem.variable_final_time = False
    else:
        problem.variable_final_time = True

    problem.Nvars = problem.Ncontrol + problem.Nstate

    # create "raw" Chebyshev data (0 ---> 1)  
    problem.numerics.t, problem.numerics.D, problem.numerics.I = \
        chebyshev_data(problem.options.N,integration=True)                    
 
    # print residuals(problem.guess,problem)
    #np.savetxt("analytical_J.txt", J, fmt='%.18e', delimiter=' ', newline='\n', header='', footer='', comments='')
    #problem.jacobian = "complex"; Ji = jacobian(problem.guess,problem)
    #np.savetxt("complex_J.txt", Ji, fmt='%.18e', delimiter=' ', newline='\n', header='', footer='', comments='')
    #return
    # solve system
    if problem.jacobian.lower() == "analytic":
        solution = root(residuals,problem.guess,args=problem, \
            method="hybr",jac=True,tol=problem.options.tol_solution)
    elif problem.jacobian.lower() == "complex":
        solution = root(residuals,problem.guess,args=problem, \
            method="hybr",jac=jacobian_complex,tol=problem.options.tol_solution)
    elif problem.jacobian.lower() == "AD":
        # dRdx, d2Rdx2 = ad.gh(residuals0)
        solution = root(residuals0,problem.guess,args=problem, \
            method="hybr",jac=jacobian_AD,tol=problem.options.tol_solution)
    else:
        solution = root(residuals,problem.guess,args=problem, \
            method="hybr",jac=False,tol=problem.options.tol_solution)

    # pack solution
    problem.solution(solution.x)

    return

def chebyshev_data(N,integration=False):

    N = int(N)

    # error checking:
    if N <= 0:
        print "N must be > 0"
        return []   

    D = np.zeros((N,N));

    # x array
    x = 0.5*(1 - np.cos(np.pi*np.arange(0,N)/(N-1)))

    # D operator
    c = np.array(2)
    c = np.append(c,np.ones(N-2))
    c = np.append(c,2)
    c = c*((-1)**np.arange(0,N));
    A = np.tile(x,(N,1)).transpose(); 
    dA = A - A.transpose() + np.eye(N); 
    cinv = 1/c; 

    for i in range(N):
        for j in range(N):
            D[i][j] = c[i]*cinv[j]/dA[i][j]

    D = D - np.diag(np.sum(D.transpose(),axis=0));

    # I operator
    if integration:
        I = np.linalg.inv(D[1:,1:]); 
        I = np.append(np.zeros((1,N-1)),I,axis=0)
        I = np.append(np.zeros((N,1)),I,axis=1)
        return x, D, I
    else:
        return x, D
    
def cosine_space(N,x0,xf):

    N = int(N)

    # error checking:
    if N <= 0:
        print "N must be > 0"
        return []   

    # x array
    x = x0 + 0.5*(1 - np.cos(np.pi*np.arange(0,N)/(N-1)))*(xf - x0)

    return x

def chebyshev_fit(x0,xf,f):

    N = len(f)

    # error checking:
    if N < 2:
        print "N must be > 1"
        return []   
    else:
        c = np.zeros(N)

    fac = 2.0/N
    for j in range(N):
        sum = 0.0
        for k in range(N):
            sum += f[N-1-k]*np.cos(np.pi*j*(k+0.5)/N)
        c[j] = fac*sum

    return c

def chebyshev_interpolate(x0,xf,c,x):

    N = len(c)

    # error checking:
    if N < 2:
        print "N must be > 1"
        return []   
    else:
        f = np.zeros(N)

    y = (2.0*x-x0-xf)/(xf-x0); y2 = 2.0*y
    d = 0.0; dd = 0.0
    for j in range(N-1,0,-1):
        sv = d
        d = y2*d - dd + c[j]
        dd = sv

    return y*d - dd + 0.5*c[0]

def chebyshev_basis_function(n,x):

    n = int(n)

    # base cases
    if n == 0:
        return np.ones(len(x))
    elif n == 1:
        return x
    else:
        return 2*x*chebyshev_basis_function(n-1,x) - \
            chebyshev_basis_function(n-2,x)

def residuals(x,problem):

    # preliminaries 
    N = problem.options.N
    if problem.jacobian.lower() == "analytic":
        jacobian = True
        Nvars = (N-1)*problem.Nstate  ############ FIX
        if problem.variable_final_time:
            Nvars += 1
        J = np.zeros((Nvars,Nvars))
    else:
        jacobian = False

    # unpack vector
    x_state, x_control, dt = problem.unpack(x)

    # differentiation & integration operators (non-dim t)
    if not problem.variable_final_time:
        dt = problem.dt
    D = problem.numerics.D/dt; I = problem.numerics.I*dt    

    if problem.Nstate > 0:

        # call user-supplied dynamics function
        if jacobian:
            rhs, drhs_dx_state, drhs_dx_control \
                = problem.dynamics(x_state,x_control,D,I)
        else:
            rhs = problem.dynamics(x_state,x_control,D,I)

        # evaluate residuals of EOMs
        if problem.complex:
            Rs = np.zeros_like(x_state) + 0j
        else:
            Rs = np.zeros_like(x_state) 

        for j in range(problem.Nstate):
            Rs[:,j] = np.dot(D,x_state[:,j]) - rhs[:,j] 
        Rs = Rs[1:].flatten('F')

        # Jacobian of EOMs, if needed
        if jacobian:

            # d(EOMs)/d(x_state)
            for i in range(problem.Nstate):
                for j in range(problem.Nstate):
                    #irange = i*(N-1) + np.arange(0,N-1) 
                    #jrange = j*(N-1) + np.arange(0,N-1)
                    imin = i*(N-1); imax = imin + N-1
                    jmin = j*(N-1); jmax = jmin + N-1
                    if i == j:
                        J[imin:imax,jmin:jmax] = D[1:,1:] - drhs_dx_state[i,j,1:,1:]
                    else:
                        J[imin:imax,jmin:jmax] = -drhs_dx_state[i,j,1:,1:]
            #print J
            # d(EOMs)/d(x_control)
            # placeholder for eventual optimal control problems 

            # d(EOMs)/d(tf)
            if problem.variable_final_time:
                D_tf2 = problem.D/(tf**2)
                for i in range(problem.Nstate):
                    imin = i*(N-1); imax = imin + N-1
                    J[imin:imax,-1] = np.dot(-D_tf2,x_state[:,i])[1:]
            ioffset = imax

    else:
        Rs = []; ioffset = 0
    
    # call user-supplied constraint functions
    if problem.Ncontrol > 0:

        if jacobian:
            Rc, dRc_dx_state, dRc_dx_control \
                = problem.constraints(x_state,x_control,D,I)
        else:
            Rc = problem.constraints(x_state,x_control,D,I)
        Rc = Rc.flatten('F')

        # Jacobian of EOMs, if needed
        if jacobian:

            # d(constraints)/d(x_state)
            # placeholder for eventual optimal control problems

            # d(constraints)/d(x_control)
            for i in range(problem.Ncontrol):
                for j in range(problem.Ncontrol):
                    imin = ioffset + i*N; imax = imin + N
                    jmin = j*N; jmax = jmin + N
                    J[imin:imax,jmin:jmax] = dRc_dx_control[i,j,1:,1:]
            
            # d(constraints)/d(tf)
            # placeholder for eventual optimal control problems

    else:
        Rc = []

    # append constraints
    R = np.append(Rs,Rc)

    # append final condition if needed
    if problem.variable_final_time:  
        if jacobian:
            Rf, Jf = problem.final_condition(x_state,x_control,D,I)
            J[-1,:-1] = Jf[1:].flatten('F')
        else:
            Rf = problem.final_condition(x_state,x_control,D,I)
        R = np.append(R,Rf)

    if jacobian:
        return R, J
    else:
        return R

def residuals0(x,problem):

    # preliminaries 
    N = problem.Npoints

    # unpack vector
    x_state, x_control, tf = problem.unpack(x)
    x_state = ad.adnumber(x_state)
    # differentiation & integration operators (non-dim t)
    if not problem.variable_final_time:
        tf = problem.tf
    D = problem.D/tf; I = problem.I*tf
    # D = ad.adnumber(problem.D/tf); I = ad.adnumber(problem.I*tf)    

    if problem.Nstate > 0:

        # call user-supplied dynamics function
        rhs = problem.dynamics(x_state,x_control,D,I)
        print "begin dot product"
        # evaluate residuals of EOMs
        Rs = np.zeros_like(x_state) 
        for var in range(problem.Nstate):
            #Rs[:,j] = np.dot(D,x_state[:,j]) - rhs[:,j]
            for i in range(N):
                print i
                for j in range(N):
                    
                    Rs[i,var] += D[i,j]*x_state[j,var] 
                Rs[i,var] -= rhs[i,var]
        print "end dot product"
        Rs = Rs[1:].flatten('F')

    else:
        Rs = []; ioffset = 0
    
    # call user-supplied constraint functions
    if problem.Ncontrol > 0:

        Rc = problem.constraints(x_state,x_control,D,I)
        Rc = Rc.flatten('F')

    else:
        Rc = []

    # append constraints
    R = np.append(Rs,Rc)

    # append final condition if needed
    if problem.variable_final_time:  
        Rf = problem.final_condition(x_state,x_control,D,I)
        R = np.append(R,Rf)
    print "end R"
    return R

def residuals_unpowered(x,problem):

    # unpack vector
    z, u, tf = unpack(x,problem)

    # differentiation & integration operators (non-dim t)
    D = problem.D/tf; I = problem.I*tf

    # dimensionalize and get state data
    state = create_state_data(z,u,D,I,problem)

    # call user-supplied dynamics function (non-dim rhs)
    rhs = problem.dynamics(state)

    # call user-supplied constraint functions (non-dim constraints)
    Rc = np.zeros((problem.Npoints,problem.Ncontrols));
    for constraint in problem.constraints:
        Rc[:,j] = constraint(state)

    Rz = np.zeros_like(z); 
    for j in range(problem.Nstate):

        # equations of motion
        v = np.append(problem.z0[j],z[:,j])
        Rz[:,j] = np.dot(D,v)[1:] - rhs[:,j]

    # reshape
    R = np.reshape(Rz[1:],((problem.Npoints-1)*problem.Nstate,1),order="F")

    # append constraints, if necessary
    R = np.append(R,np.reshape(Rc,(problem.Npoints*problem.Ncontrol,1),order="F"))

    # append final condition
    R = np.append(R,problem.final_condition(state))

    return R

def residuals_uncontrolled(x,problem):

    # unpack vector
    z, tf = unpack(x,problem)

    # differentiation & integration operators
    D = problem.D/tf; I = problem.I*tf

    # dimensionalize and get state data
    state = create_state_data(z,problem)

    # call user-supplied dynamics function
    rhs = problem.dynamics(z,D,I)

    R = np.zeros_like(z)
    for j in range(problem.Nstate):

        # equations of motion
        v = np.append(problem.z0[j],z[:,j])
        R[:,j] = np.dot(D,v)[1:] - rhs[:,j]

    # reshape
    R = np.reshape(R,((problem.Npoints-1)*problem.Nvars,1),order="F")

    # append final condition
    R = np.append(R,problem.final_condition(z,D,I))

    return R

def jacobian_complex(x,problem):

    # Jacobian via complex step 
    problem.complex = True
    h = 1e-12; N = len(x)
    J = np.zeros((N,N))
    for i in range(N):
        xi = np.array(x, dtype=complex); xi[i] = np.complex(x[i],h)
        R = residuals(xi,problem)
        J[:,i] = np.imag(R)/h

    problem.complex = False
    return J

def jacobian_AD(x,problem):

    # Jacobian via AD 
    N = len(x); J = np.zeros((N,N)); xi = ad.adnumber(x)
    print "begin Jacobian"
    R = residuals0(xi,problem)  
    for i in range(N):
        for j in range(N):          
            J[i,j] = R[i].d(xi[j])
    print "end Jacobian"
    return J

def create_guess(problem,options):

    # d/dt, integration operators 
    D = problem.D/problem.tf; I = problem.I*problem.tf
    D[0,:] = 0.0; D[0,0] = 1.0

    zs = np.zeros((problem.Npoints,problem.Nstate))
    for j in range(problem.Nstate):
        zs[:,j] = problem.z0[j]*np.ones(problem.Npoints)
    
    if problem.Ncontrol > 0:
        zc = np.zeros((problem.Npoints,problem.Ncontrol))
        for j in range(problem.Ncontrol):
            zc[:,j] = problem.c0[j]*np.ones(problem.Npoints)

    dz = np.ones(problem.Nstate)
    zs_new = np.zeros_like(zs)

    # FPI with fixed final time
    while max(dz) > options.tol_solution:        

        if problem.Ncontrol > 0:
            rhs = problem.f(np.append(zs,zc,axis=1))
            rhs[0,range(problem.Nstate)] = problem.z0
        else:
            rhs = problem.dynamics(zs,D,I)
            rhs[0,:] = problem.z0

        for j in range(problem.Nstate):

            zs_new[:,j] = np.linalg.solve(D,rhs[:,j])
            dz[j] = np.linalg.norm(zs_new[:,j] - zs[:,j])
        
        zs = zs_new.copy()
    
    # flatten z array
    z = np.reshape(zs[1:,:],((options.Npoints-1)*problem.Nstate,1),order="F")

    # append control vars, if necessary
    if problem.Ncontrol > 0:
        z = np.append(z,np.reshape(zc,(problem.Npoints*problem.Ncontrol,1),order="F"))

    return z

def unpack(x,problem):

    # grab tf and trim x
    tf = x[-1]; x = x[0:-1]

    # get state data
    indices = range((problem.Npoints-1)*problem.Nstate)
    z = np.reshape(x[indices],(problem.Npoints-1,problem.Nstate),order="F") 

    # get control data if applicable
    if problem.Ncontrol > 0:

        # get control vector
        indices = indices[-1] + range(problem.Npoints*problem.Ncontrol)
        u = np.reshape(x[indices],(problem.Npoints,problem.Ncontrol),order="F") 

        # get throttle
        indices = indices[-1] + range(problem.Npoints)
        eta = x[indices]

        return z, u, eta, tf

    else:
        return z, tf

def create_state_data(z,u,problem,eta=[]):

    N, m = np.shape(z)

    # scale data
    if problem.dofs == 2:                      # 2-DOF
        z[:,0] *= problem.scale.L
        z[:,1] *= problem.scale.V
        z[:,2] *= problem.scale.L
        z[:,3] *= problem.scale.V
    elif problem.dofs == 3:                    # 3-DOF
        z[:,0] *= problem.scale.L
        z[:,1] *= problem.scale.V
        z[:,2] *= problem.scale.L
        z[:,3] *= problem.scale.V
        z[:,4] *= problem.scale.L
        z[:,5] *= problem.scale.V
    else:
        print "something went wrong in dimensionalize"
        return []
    
    state = State()
    if problem.powered:
        state.compute_state(z,u,problem.planet,problem.atmosphere, \
            problem.config.Functions.Aero,problem.config.Functions.Propulsion,eta,problem.flags)
    else:
        state.compute_state(z,u,problem.planet,problem.atmosphere, \
            problem.config.Functions.Aero,flags=problem.flags)

    return state

def assign_values(A,B,irange,jrange):

    n, m = np.shape(B)
    if n != len(irange):
        print "Error: rows do not match between A and B"
        return
    if m != len(jrange):
        print "Error: columsn do not match between A and B"
        return

    ii = 0
    for i in irange:
        jj = 0
        for j in jrange:
            A[i][j] = B[ii][jj]
            jj += 1
        ii += 1

    return

# ----------------------------------------------------------------------
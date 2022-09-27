## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method
# inviscid_functions.py
# 
# Created:  Aug 2022, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ---------------------------------------------------------------------- 
from SUAVE.Core import Units  
import numpy as np   

from SUAVE.Methods.Aerodynamics.Airfoil_Panel_Method.supporting_functions import * 
from SUAVE.Methods.Aerodynamics.Airfoil_Panel_Method.airfoil_paneling     import *  
from SUAVE.Methods.Aerodynamics.Airfoil_Panel_Method.post_processing      import *

## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method
def solve_inviscid(Airfoils):
    """Solves the inviscid system, rebuilds 0,90deg solutions
    
    Assumptions:
      Uses the angle of attack in Airfoils.oper.gamma
      Also initializes thermo variables for normalization 

    Source:   
                                                     
    Inputs:  
       Airfoils : data structure
                                                                           
    Outputs:  
       Inviscid vorticity distribution is computed
    
    Properties Used:
    N/A
    """          
    if Airfoils.foil.N>0: 
        assert('No panels')
    Airfoils.oper.viscous = False
    init_thermo(Airfoils)
    Airfoils.isol.sgnue = np.ones((1,Airfoils.foil.N)) # do not distinguish sign of ue if inviscid
    build_gamma(Airfoils, Airfoils.oper.alpha) 
    calc_force(Airfoils)
    Airfoils.glob.conv = True # no coupled system ... convergence is guaranteed
    return  

def get_ueinv(Airfoils): 
    """Computes invicid tangential velocity at every node
    
    Assumptions:
      The airfoil velocity is computed directly from gamma
      The tangential velocity is measured + in the streamwise direction

    Source:   
                                                     
    Inputs:
       Airfoils : data structure
                                                                           
    Outputs:
       ueinv : inviscid velocity at airfoil and wake (if exists) points
    
    Properties Used:
    N/A
    """ 
    alpha = Airfoils.oper.alpha  
    cs    = np.zeros((2,1))
    cs[0] = np.cos(alpha)
    cs[1] = np.sin(alpha) 
    uea   = (Airfoils.isol.sgnue*(np.matmul(Airfoils.isol.gamref,cs)).T ).T # airfoil
    if (Airfoils.oper.viscous) and (Airfoils.wake.N > 0):
        uew    = np.matmul(Airfoils.isol.uewiref,cs) # wake
        uew[0] = uea[-1] # ensures continuity of upper surface and wake ue
    else:
        uew = np.empty(shape=[0,1])
     
    ueinv= np.concatenate((uea, uew), axis = 0)# airfoil/wake edge velocity

    return ueinv

    
## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method 
def get_ueinvref(Airfoils):
    """Computes 0,90deg inviscid tangential velocities at every node 
    
    Assumptions:
      Uses gamref for the airfoil, uewiref for the wake (if exists)

    Source:   
                                                     
    Inputs:
       Airfoils : data structure
                                                                           
    Outputs    
       ueinvref : 0,90 inviscid tangential velocity at all points (N+Nw)x2
    
    Properties Used:
    N/A
    """   
    
    uearef = Airfoils.isol.sgnue.T*Airfoils.isol.gamref # airfoil 
    if (Airfoils.oper.viscous) and (Airfoils.wake.N > 0):
        uewref = Airfoils.isol.uewiref # wake
        uewref[0,:] = uearef[-1,:] # continuity of upper surface and wake
    else:
        uewref = np.empty([0,2]) 
    
    ueinvref =  np.concatenate((uearef, uewref), axis = 0) 

    return ueinvref


## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method 
def build_gamma(Airfoils, alpha): 
    """Builds and solves the inviscid linear system for alpha=0,90,input 
    
    Assumptions:
    None

    Source:   
      Uses streamdef approach: constant psi at each node
      Continuous linear vorticity distribution on the airfoil panels
      Enforces the Kutta condition at the TE
      Accounts for TE gap through const source/vorticity panels
      Accounts for sharp TE through gamma extrapolation
                                                     
    Inputs:
      Airfoils     : data structure
      alpha : angle of attack (degrees)
                                                                           
    Outputs     
      Airfoils.isol.gamref : 0,90deg vorticity distributions at each node (Nx2)
      Airfoils.isol.gam    : gamma for the particular input angle, alpha
      Airfoils.isol.AIC    : aerodynamic influence coefficient matrix, filled in
    
    Properties Used:
    N/A
    """      

    N                = Airfoils.foil.N         # number of points  
    A                = np.zeros((N+1,N+1))  # influence matrix
    rhs              = np.zeros((N+1,2))  # right-hand sides for 0,90
    _,hTE,_,tcp,tdp  = TE_info(Airfoils.foil.x) # trailing-edge info
    nogap            = (abs(hTE) < 1E-10*Airfoils.geom.chord) # indicates no TE gap 
    ep               = 1e-10  
    hTE              = np.array([hTE])
    tcp              = np.array([tcp])
    tdp              = np.array([tdp]) 
    A                = np.zeros((1,N+1,N+1))  # influence matrix
    rhs              = np.zeros((1,N+1,2))  # right-hand sides for 0,90
    j1               = np.arange(N)     
    j2               = np.append(np.arange(1,N),0)       
    xi               = Airfoils.foil.x.T[None,:,:] # coord of node i 
    xi_x_vec         = np.tile(xi[:,:,0][:,:,None],(1,1,N))
    xi_z_vec         = np.tile(xi[:,:,1][:,:,None],(1,1,N)) 

    # panel coordinates 
    xj1              = np.swapaxes(np.tile(xi[:,j1,0][:,:,None],(1,1,N)),1,2)
    zj1              = np.swapaxes(np.tile(xi[:,j1,1][:,:,None],(1,1,N)),1,2)
    xj2              = np.swapaxes(np.tile(xi[:,j2,0][:,:,None],(1,1,N)),1,2)
    zj2              = np.swapaxes(np.tile(xi[:,j2,1][:,:,None],(1,1,N)),1,2)

    # panel-aligned tangent and np.linalg.normal vectors
    diff_x           = (xj2-xj1)[:,:,:,None] 
    diff_z           = (zj2-zj1)[:,:,:,None] 
    t_vec            = np.concatenate((diff_x , diff_z ),axis = 3)
    norms            = np.tile(np.linalg.norm(t_vec,axis = 3)[:,:,:,None],(1,1,1,2))
    t                = t_vec/norms 
    t_norm_z         = -t[:,:,:,1][:,:,:,None] 
    t_norm_x         =  t[:,:,:,0][:,:,:,None]      
    n                = np.concatenate((t_norm_z, t_norm_x),axis = 3) 

    # control point relative to (xj1,zj1) 
    diff_xz_x        = (xi_x_vec-xj1)[:,:,:,None]
    diff_zz_x        = (xi_z_vec-zj1)[:,:,:,None]
    xz               = np.concatenate((diff_xz_x , diff_zz_x ),axis = 3)   
    x                = np.sum(xz*t, axis = 3) # np.dot(xz,t) # in panel-aligned coord system
    z                = np.sum(xz*n, axis = 3) # np.dot(xz,n)  # in panel-aligned coord system

    # distances and angles    
    d                = norms[:,:,:,0][:,:,:,None]
    new_x            = x[:,:,:,None]
    new_z            = z[:,:,:,None]
    r1_vec           = np.concatenate((new_x,new_z),axis = 3)    
    r1               = np.linalg.norm(r1_vec,axis = 3)          # left edge to control point
    r2_vec           = np.concatenate(((new_x-d),new_z),axis = 3)          
    r2               = np.linalg.norm(r2_vec,axis = 3)           # right edge to control point
    theta1           = np.arctan2(new_z,new_x)[:,:,:,0]          # left angle
    theta2           = np.arctan2(new_z,new_x-d)[:,:,:,0]        # right angle

    # check for r1, r2 zero 
    r1_dim_0         = np.shape(r1)[0]
    r1_dim_1         = np.shape(r1)[1] 
    r1_dim_2         = np.shape(r1)[2] 
    r1_flattened     = np.reshape(r1,(1,r1_dim_0*r1_dim_1*r1_dim_2))
    r2_flattened     = np.reshape(r2,(1,r1_dim_0*r1_dim_1*r1_dim_2))

    theta1_flattened = np.reshape(theta1,(1,r1_dim_0*r1_dim_1*r1_dim_2))
    theta2_flattened = np.reshape(theta2,(1,r1_dim_0*r1_dim_1*r1_dim_2))
    logr1_flattened  = np.log(r1_flattened) 
    logr2_flattened  = np.log(r2_flattened)    

    index_flag_1                   = np.where(r1_flattened < ep) 
    index_flag_2                   = np.where(r2_flattened < ep)   
    logr1_flattened[index_flag_1]  = 0
    theta1_flattened[index_flag_1] = np.pi
    theta2_flattened[index_flag_1] = np.pi 
    logr2_flattened[index_flag_2]  = 0   
    theta1_flattened[index_flag_2] = 0
    theta2_flattened[index_flag_2] = 0   

    logr1  = np.reshape(logr1_flattened,(r1_dim_0,r1_dim_1,r1_dim_2))
    logr2  = np.reshape(logr2_flattened,(r1_dim_0,r1_dim_1,r1_dim_2))
    theta1 = np.reshape(theta1_flattened,(r1_dim_0,r1_dim_1,r1_dim_2))
    theta2 = np.reshape(theta2_flattened,(r1_dim_0,r1_dim_1,r1_dim_2))

    # streamdef components
    P1     = (0.5/np.pi)*(z*(theta2-theta1) - d[:,:,:,0] + x*logr1 - (x-d[:,:,:,0])*logr2)
    P2     = x*P1 + (0.5/np.pi)*(0.5*r2**2*logr2 - 0.5*r1**2*logr1 - r2**2/4 + r1**2/4)

    # influence coefficients
    a      = P1-P2/d[:,:,:,0]
    b      =    P2/d[:,:,:,0] 

    # influence coefficients  af, i,j
    A[:,:-1,:-2]    = a[:,:,:-1]   
    A[:,:-1,1:-1]   = A[:,:-1,1:-1] + b[:,:,:-1] 
    A[:,:-1,N]      = -1 # last unknown = streamdef value on surf   

    # right-hand sides
    rhs[:,0:-1,0]   = -xi[:,:,1]
    rhs[:,0:-1,1]   =  xi[:,:,0] 

    P_mat           = (x*(theta1-theta2) + d[:,:,:,0]*theta2 + z*logr1 - z*logr2)/(2*np.pi) 
    dP              = d[:,:,:,0] # delta psi
    P               = P_mat + 0.75*dP 
    P_alt           = P_mat - 0.25*dP 
    index_flag_3    = np.where((theta1_flattened+theta2_flattened) > np.pi)  
    P_flattened     = np.reshape(P,(1,r1_dim_0*r1_dim_1*r1_dim_2))
    P_alt_flattened = np.reshape(P_alt,(1,r1_dim_0*r1_dim_1*r1_dim_2))   
    P_flattened[index_flag_3] = P_alt_flattened[index_flag_3]  
    P               = np.reshape(P_flattened,(r1_dim_0,r1_dim_1,r1_dim_2)) 

    # TE vortex panel 
    A[:,:-1,0]      = A[:,:-1,0]   - (a[:,:,-1]+b[:,:,-1])*(-0.5*np.tile(tdp[:,None],(1,N))) - P[:,:,-1]*(0.5*np.tile(tcp[:,None],(1,N)))
    A[:,:-1,N-1]    = A[:,:-1,N-1] + (a[:,:,-1]+b[:,:,-1])*(-0.5*np.tile(tdp[:,None],(1,N))) + P[:,:,-1]*(0.5*np.tile(tcp[:,None],(1,N)))  

    # special Nth equation (extrapolation of gamma differences) if no gap
    if (nogap):
        A[:,N-1,:]                   = 0    
        A[:,N-1,[0,1,2,N-3,N-2,N-1]] = [1,-2,1,-1,2,-1]  

    # Kutta condition
    A[:,N,0]  = 1
    A[:,N,-2] = 1    

    # Solve system for unknown vortex strengths
    Airfoils.isol.AIC    = A[0]
    g             = np.linalg.solve(Airfoils.isol.AIC,rhs[0])
    Airfoils.isol.gamref = g[:-1,:] # last value is surf streamdef   
    Airfoils.isol.gam    = Airfoils.isol.gamref[:,0]*np.cos(alpha) + Airfoils.isol.gamref[:,1]*np.sin(alpha)      
    
    return 


## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method
def inviscid_velocity(X, G, Vinf, alpha, x,nargout=1): 
    """Returns inviscid velocity at x due to gamma (G) on panels X, and Vinf
    
    Assumptions:
      Uses linear vortex panels on the airfoil
      Accounts for TE const source/vortex panel
      Includes the freestream contribution 

    Source:   
                                                     
    Inputs:   
      X     : coordinates of N panel nodes (N-1 panels) (Nx2)
      G     : vector of gamma values at each airfoil node (Nx1)
      Vinf  : freestream speed magnitude
      alpha : angle of attack (degrees)
      x     : location of point at which velocity vector is desired  
                                                                           
    Outputs: 
      V    : velocity at the desired point (2x1)
      V_G  : (optional) linearization of V w.r.t. G, (2xN)    
    
    Properties Used:
    N/A
    """      

    N     = len(X[1])   # number of points  
    V     = np.zeros(2)  # velocity
    dolin = False
    if (nargout > 1):
        dolin = True  # (nargout > 1) # linearization requested
    if (dolin):
        V_G = np.zeros((2,N))
    _,_,_,tcp,tdp  = TE_info(X) # trailing-edge info
    
    # assume x is not a midpoint of a panel (can check for this)
    for j in range(N-1): # loop over panels
        a, b = panel_linvortex_velocity(X[:,[j,j+1]], x, np.empty([0]), False)
        V = V + a*G[j] + b*G[j+1]
        if dolin:
            V_G[:,j]   = V_G[:,j] + a 
            V_G[:,j+1] = V_G[:,j+1] + b 
    
    # TE source influence
    a  = panel_constsource_velocity(X[:,[-1,0]], x, np.empty([0]))
    f1 = a*(-0.5*tcp) 
    f2 = a*0.5*tcp
    V  = V + f1*G[0] + f2*G[-1]
    if dolin:
        V_G[:,0] = V_G[:,0] + f1 
        V_G[:,-1] = V_G[:,-1] + f2 
        
    # TE vortex influence
    a,b = panel_linvortex_velocity(X[:,[-1,0]], x,np.empty([0]), False)
    f1  = (a+b)*(0.5*tdp) 
    f2  = (a+b)*(-0.5*tdp)
    V   = V + f1*G[0] + f2*G[-1]
    if (dolin):
        V_G[:,0]  = V_G[:,0] + f1 
        V_G[:,-1] = V_G[:,-1] + f2 
        
    # freestream influence
    alf    = np.zeros(2)
    alf[0] = np.cos(alpha)
    alf[1] = np.sin(alpha)
    V      = V+ Vinf*alf

    if dolin:
        return V_G  
    else: 
        return V
    
#-------------------------------------------------------------------------------
def rebuild_isol(M):
    ''' Rebuilds inviscid solution, after an angle of attack change 
    
    Assumptions: 
      None

    Source:  
       None 
                                                     
    Inputs:   
       M     : mfoil class with inviscid reference solution and angle of attack
                                                                           
    Outputs: 
       M.isol.gam : correct combination of reference gammas
       New stagnation point location if inviscid
       New wake and source influence matrix if viscous 
    
    Properties Used:
       N/A
    
    '''
    alpha      = M.oper.alpha
    M.isol.gam = M.isol.gamref[:,0]*np.cos(alpha) + M.isol.gamref[:,1]*np.sin(alpha)
    if not (M.oper.viscous):
        # viscous stagnation point movement is handled separately
        stagpoint_find(M)
    elif (M.oper.redowake):
        build_wake(M)
        identify_surfaces(M)
        calc_ue_m(M) # rebuild matrices due to changed wake geometry 
    return    
    

## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method
def build_wake(Airfoils): 
    """Builds wake panels from the inviscid solution
    
    Assumptions:
      Constructs the wake path through repeated calls to inviscid_velocity
      Uses a predictor-corrector method
      Point spacing is geometric prescribed wake length and number of points 

    Source:   
                                                     
    Inputs:              
      Airfoils     : data class with a valid inviscid solution (gam)    
      
    Outputs:   
      Airfoils.wake.N  : Nw, the number of wake points
      Airfoils.wake.x  : coordinates of the wake points (2xNw)
      Airfoils.wake.s  : s-values of wake points (continuation of airfoil) (1xNw)
      Airfoils.wake.t  : tangent vectors at wake points (2xNw)
    
    Properties Used:
    N/A
    """  
    N    = Airfoils.foil.N  # number of points on the airfoil
    Vinf = Airfoils.oper.Vinf    # freestream speed
    Nw   = int(np.ceil(N/10 + 10*Airfoils.geom.wakelen)) # number of points on wake
    S    = Airfoils.foil.s  # airfoil S values
    ds1  = 0.5*(S[1]-S[0] + S[-1]-S[-2]) # first nominal wake panel size
    sv   = space_geom(ds1, Airfoils.geom.wakelen*Airfoils.geom.chord, Nw) # geometrically-spaced points
    xyw  = np.zeros((2,Nw))
    tw   = np.zeros((2,Nw)) # arrays of x,y points and tangents on wake
    xy1  = Airfoils.foil.x[:,0] 
    xyN  = Airfoils.foil.x[:,-1] # airfoil TE points
    xyte = 0.5*(xy1 + xyN) # TE midpoint 
    n    = xyN-xy1 
    t    = np.array([n[1],-n[0]]) # normal and tangent
    if t[0] > 0:  
        assert('Wrong wake direction ensure airfoil points are CCW')
    xyw[:,0] = xyte + 1E-5*t*Airfoils.geom.chord # first wake point, just behind TE
    sw = S[-1] + sv # s-values on wake, measured as continuation of the airfoil

    # loop over rest of wake
    for i in range(Nw-1):
        v1         = inviscid_velocity(Airfoils.foil.x, Airfoils.isol.gam, Vinf, Airfoils.oper.alpha, xyw[:,i])
        v1         = v1/np.linalg.norm(v1) 
        tw[:,i]    = v1 # normalized
        xyw[:,i+1] = xyw[:,i] + (sv[i+1]-sv[i])*v1 # forward Euler (predictor) step
        v2         = inviscid_velocity(Airfoils.foil.x, Airfoils.isol.gam, Vinf, Airfoils.oper.alpha, xyw[:,i+1])
        v2         = v2/np.linalg.norm(v2) 
        tw[:,i+1]  = v2 # normalized
        xyw[:,i+1] = xyw[:,i] + (sv[i+1]-sv[i])*0.5*(v1+v2) # corrector step
    

    # determine inviscid ue in the wake, and 0,90deg ref ue too
    uewi    = np.zeros((Nw,1)) 
    uewiref = np.zeros((Nw,2))
    for i in range(Nw):
        v            = inviscid_velocity(Airfoils.foil.x, Airfoils.isol.gam, Vinf, Airfoils.oper.alpha, xyw[:,i])
        uewi[i]      = np.dot(v,tw[:,i])
        v            = inviscid_velocity(Airfoils.foil.x, Airfoils.isol.gamref[:,0], Vinf, 0, xyw[:,i])
        uewiref[i,0] = np.dot(v,tw[:,i])
        v            = inviscid_velocity(Airfoils.foil.x, Airfoils.isol.gamref[:,1], Vinf, np.pi/2, xyw[:,i])
        uewiref[i,1] = np.dot(v,tw[:,i])

    # set values
    Airfoils.wake.N       = Nw
    Airfoils.wake.x       = xyw
    Airfoils.wake.s       = sw
    Airfoils.wake.t       = tw
    Airfoils.isol.uewi    = uewi
    Airfoils.isol.uewiref = uewiref

    return 

## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method
def stagpoint_find(Airfoils):
    """Finds the LE stagnation point on the airfoil (using inviscid solution)
    
    Assumptions:
    None

    Source:   
                                                     
    Inputs:  
      Airfoils  : data class with inviscid solution, gam
                                                                           
    Outputs:    
      Airfoils.isol.sstag   : scalar containing s value of stagnation point
      Airfoils.isol.sstag_g : linearization of sstag w.r.t gamma (1xN)
      Airfoils.isol.Istag   : [i,i+1] node indices before/after stagnation (1x2)
      Airfoils.isol.sgnue   : sign conversion from CW to tangential velocity (1xN)
      Airfoils.isol.xi      : distance from stagnation point at each node (1xN)
    
    Properties Used:
    N/A
    """ 
    N = Airfoils.foil.N  # number of points on the airfoil
    J = np.where(Airfoils.isol.gam>0)[0] 
    if np.shape(J) == 0:
        print('no stagnation point')
    I                     = [J[0]-1, J[0]] 
    G                     = Airfoils.isol.gam[I]
    S                     = Airfoils.foil.s[I]
    Airfoils.isol.Istag   = I  # indices of neighboring gammas
    den                   = (G[1]-G[0]) 
    w1                    = G[1]/den 
    w2                    = -G[0]/den
    sst                   = w1*S[0] + w2*S[1]  # s location
    Airfoils.isol.sstag   = sst 
    W_vec                 = np.array([w1,w2])
    Airfoils.isol.xstag   = np.dot(Airfoils.foil.x[:,I],W_vec.T)  # x location
    st_g1                 = G[1]*(S[0]-S[1])/(den*den)
    Airfoils.isol.sstag_g = np.array([st_g1, -st_g1])
    sgnue                 = -1*np.ones((1,N))
    sgnue[:,J]            = 1 # upper/lower surface sign
    Airfoils.isol.sgnue   = sgnue
    Airfoils.isol.xi      = np.concatenate((abs(Airfoils.foil.s-Airfoils.isol.sstag), Airfoils.wake.s-Airfoils.isol.sstag),axis = 0)

    return 

## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method
def space_geom(dx0, L, Np):  
    """Spaces Np points geometrically from [0,L], with dx0 as first interval
    
    Assumptions:
    None

    Source:   
                                                     
    Input:
       dx0 : first interval length
       L   : total domain length
       Np  : number of points, including points at 0,L
                                                                           
    Outputs:
       x   : point locations (1xN)
    
    Properties Used:
    N/A
    """     
    if Np>1: 
        assert('Need at least two points for spacing.')
    N = Np - 1 # number of intervals
    d = L/dx0 
    a = N*(N-1.)*(N-2.)/6. 
    b = N*(N-1.)/2. 
    c = N-d
    disc = max(b*b-4*a*c, 0.) 
    r = 1 + (-b+np.sqrt(disc))/(2*a)
    for _ in range(10):
        R = r**N -1-d*(r-1) 
        R_r = N*r**(N-1)-d 
        dr = -R/R_r
        if (abs(dr)<1e-6): 
            break  
        r = r - R/R_r
    
    vec   = np.arange(N)
    xx    = dx0*r**vec
    x     = np.zeros(N+1)
    x[1:] = np.cumsum(xx)

    return x
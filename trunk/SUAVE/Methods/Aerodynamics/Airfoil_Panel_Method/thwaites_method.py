## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method
# thwaites_method.py 

# Created:  Mar 2021, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
from SUAVE.Core import Data 
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import odeint 

# ----------------------------------------------------------------------
# thwaites_method
# ----------------------------------------------------------------------  
## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method
def thwaites_method(nalpha,nRe,L,RE_L,X_I,VE_I, DVE_I,batch_analysis,THETA_0,n = 200,tol  = 1E0):
    """ Computes the boundary layer characteristics in laminar 
    flow pressure gradients
    
    Assumptions:
    None

    Inputs:  
    nalpha         - number of angle of attacks
    nRe            - number of reynolds numbers 
    batch_analysis - flag for batch analysis
    THETA_0        - intial momentum thickness  
    L              - normalized lenth of surface
    RE_L           - Reynolds number
    X_I            - x coordinated on surface of airfoil
    VE_I           - boundary layer velocity at transition location 
    DVE_I          - intial derivative value of boundary layer velocity at transition location 
    n              - number of points on surface 
    tol            - boundary layer error correction tolerance

    Outputs: 
    RESULTS.
      X_T          - reshaped distance along airfoil surface    
      THETA_T      - momentum thickness
      DELTA_STAR_T - displacement thickness
      H_T          - shape factor 
      CF_T         - friction coefficient 
      RE_THETA_T   - Reynolds number as a function of momentum thickness
      RE_X_T       - Reynolds number as a function of distance
      DELTA_T      - boundary layer thickness 

    Properties Used:
    N/A
    """     
    # Initialize vectors 
    X_T        = np.zeros((n,nalpha,nRe))
    THETA_T    = np.zeros_like(X_T)
    DELTA_STAR_T = np.zeros_like(X_T)
    H_T        = np.zeros_like(X_T)
    CF_T       = np.zeros_like(X_T)
    RE_THETA_T = np.zeros_like(X_T)
    RE_X_T     = np.zeros_like(X_T)
    DELTA_T    = np.zeros_like(X_T)  
    
    if batch_analysis:
        N_ALPHA = nalpha
    else:
        N_ALPHA = 1 
    
    for a_i in range(N_ALPHA):
        for re_i in range(nRe):    
            if batch_analysis: 
                l   = L[a_i,re_i]
            else:
                a_i = re_i 
            # compute laminar boundary layer properties  
            l           = L[a_i,re_i]
            theta_0     = THETA_0 
            Re_L        = RE_L[a_i,re_i]
            nu          = l/Re_L    
            x_i         = X_I.data[:,0,0][X_I.mask[:,0,0] ==False]
            Ve_i        = VE_I.data[:,0,0][VE_I.mask[:,0,0] ==False]
            dVe_i       = DVE_I.data[:,0,0][DVE_I.mask[:,0,0] ==False] 
            y0          = theta_0**2 * getVe(0,x_i,Ve_i)**6 
            xspan       = np.linspace(0,l,n)   
            theta2_Ve6  = odeint(odefcn, y0, xspan, args=(nu, x_i, Ve_i))  
            x           = np.linspace(0,l,n)  
            
            # Compute momentum thickness, theta 
            theta       = np.sqrt(theta2_Ve6[:,0]/ getVe(x, x_i, Ve_i)**6)
            
            # find theta values that do not converge and replace them with neighbor
            idx1        = np.where(abs((theta[1:] - theta[:-1])/theta[:-1]) > tol)[0] 
            if len(idx1)> 1:  
                np.put(theta,idx1 + 1, theta[idx1])
            
            # Thwaites separation criteria 
            lambda_val  = theta**2 * getdVe(x,x_i,dVe_i) / nu 
            
            # Compute H 
            H           = getH(lambda_val)
            H[H<0]      = 1E-6   # H cannot be negative 
            # find H values that do not converge and replace them with neighbor
            idx1        = np.where(abs((H[1:] - H[:-1])/H[:-1]) >tol)[0]
            if len(idx1)> 1: 
                np.put(H,idx1 + 1, H[idx1]) 
            
            # Compute Reynolds numbers based on momentum thickness  
            Re_theta    = getVe(x,x_i,Ve_i) * theta / nu
            
            # Compute Reynolds numbers based on distance along airfoil
            Re_x        = getVe(x,x_i,Ve_i) * x/ nu
            
            # Compute skin friction 
            cf          = getcf(lambda_val ,Re_theta)
            cf[cf<0]    = 1E-6 
            
            # Compute displacement thickness
            del_star    = H*theta   
            
            # Compute boundary layer thickness 
            delta       = 5.2*x/np.sqrt(Re_x)
            delta[0]    = 0   
            
            # Reynolds number at x=0 cannot be negative (give nans)
            Re_x[0]     = 1E-5
            
            # Store results 
            X_T[:,a_i,re_i]          = x
            THETA_T[:,a_i,re_i]      = theta
            DELTA_STAR_T[:,a_i,re_i] = del_star
            H_T[:,a_i,re_i]          = H
            CF_T[:,a_i,re_i]         = cf
            RE_THETA_T[:,a_i,re_i]   = Re_theta
            RE_X_T[:,a_i,re_i]       = Re_x
            DELTA_T[:,a_i,re_i]      = delta 
    
    RESULTS = Data(
        X_T          = X_T,      
        THETA_T      = THETA_T,   
        DELTA_STAR_T = DELTA_STAR_T,
        H_T          = H_T,       
        CF_T         = CF_T,      
        RE_THETA_T   = RE_THETA_T,   
        RE_X_T       = RE_X_T,    
        DELTA_T      = DELTA_T,  
    )    
    
    return RESULTS
    
def getH(lambda_val ): 
    """ Computes the shape factor, H

    Assumptions:
    None

    Source:
    None

    Inputs: 
    lamdda_val  - thwaites separation criteria 

    Outputs:  
    H           - shape factor

    Properties Used:
    N/A
    """       
    H       = 0.0731/(0.14 + lambda_val ) + 2.088 
    idx1    = (lambda_val>0.0)  
    H[idx1] = 2.61 - 3.75*lambda_val[idx1]  + 5.24*lambda_val[idx1]**2   
    return H 
    
def odefcn(y,x, nu,x_i,Ve_i):
    """ Computes bounday layer functions using SciPy ODE solver 

    Assumptions:
    None

    Source:
    None

    Inputs: 
    y           - initial conditions of functions 
    x           - new x values at which to solve ODE
    nu          - kinematic viscosity 
    x_i         - intial array of x values 
    Ve_i        - intial boundary layer velocity 
    
    Outputs:  
    dydx        - expression for the momentum thickness and velocity (theta**2/Ve**6)

    Properties Used:
    N/A 
    """        
    dydx = 0.45*getVe(x,x_i,Ve_i)**5*nu
    return dydx 
    
def getVe(x,x_i,Ve_i):
    """ Interpolates the bounday layer velocity over a new dimension of x 

    Assumptions:
    None

    Source:
    None

    Inputs: 
    x         - new x dimension
    x_i       - old x dimension 
    Ve_i      - old boundary layer velocity values  
    
    Outputs:  
    Ve        - new boundary layer velocity values 

    Properties Used:
    N/A 
    """
    Ve_func = interp1d(x_i,Ve_i, axis=0,fill_value = "extrapolate")
    Ve      = Ve_func(x)
    return Ve 


def getcos(x,x_i,cos_i):
    """ Interpolates the bounday layer velocity over a new dimension of x 

    Assumptions:
    None

    Source:
    None

    Inputs: 
    x         - new x dimension
    x_i       - old x dimension 
    cos_i      - old boundary layer velocity values  
    
    Outputs:  
    cos        - new boundary layer velocity values 

    Properties Used:
    N/A 
    """
    cos_func = interp1d(x_i,cos_i, axis=0,fill_value = "extrapolate")
    cos_t     = cos_func(x)
    return cos_t

def getdVe(x,x_i,dVe_i):
    """ Interpolates the derivatives of the bounday layer velocity over a new dimension of x 

    Assumptions:
    None

    Source:
    None

    Inputs: 
    x         - new x dimension
    x_i       - old x dimension 
    dVe_i     - old derivative of boundary layer velocity values  
    
    Outputs:  
    dVe       - new derivative of boundary layer velocity values 

    Properties Used:
    N/A 
    """
    dVe_func = interp1d(x_i,dVe_i,fill_value = "extrapolate")
    dVe      = dVe_func(x)
    return dVe 

def getcf(lambda_val , Re_theta):
    """ Computes the skin friction coefficient, cf

    Assumptions:
    None

    Source:
    None

    Inputs: 
    lambda_val - thwaites separation criteria 
    Re_theta   - Reynolds Number as a function of momentum thickness  

    Outputs:  
    cf       - skin friction coefficient

    Properties Used:
    N/A 
    """        
    l       = 0.22 + 1.402*lambda_val  + (0.018*lambda_val)/(0.107 + lambda_val ) 
    idx1    = (lambda_val>0.0)   
    l[idx1] = 0.22 + 1.57*lambda_val[idx1] - 1.8*lambda_val[idx1]**2 
    cf      = 2*l/Re_theta  
    return cf 
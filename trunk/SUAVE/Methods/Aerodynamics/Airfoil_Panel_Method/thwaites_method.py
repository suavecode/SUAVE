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
def thwaites_method(npanel,nalpha,nRe,L,RE_L,X_I,VE_I, DVE_I,batch_analysis,THETA_0,tol  = 1E0):
    """ Computes the boundary layer characteristics in laminar 
    flow pressure gradients
    
    Source:
    Thwaites, Bryan. "Approximate calculation of the laminar boundary layer." 
    Aeronautical Quarterly 1.3 (1949): 245-280.
    
    Assumptions:
    None  

    Inputs:  
    npanel         - number of points on surface                                                 [unitless]
    nalpha         - number of angle of attacks                                                  [unitless]
    nRe            - number of reynolds numbers                                                  [unitless]
    batch_analysis - flag for batch analysis                                                     [boolean]
    THETA_0        - initial momentum thickness                                                  [m]
    L              - normalized length of surface                                                [unitless]
    RE_L           - Reynolds number                                                             [unitless]
    X_I            - x coordinate on surface of airfoil                                          [unitless]
    VE_I           - boundary layer velocity at transition location                              [m/s] 
    DVE_I          - initial derivative value of boundary layer velocity at transition location  [m/s-m] 
    tol            - boundary layer error correction tolerance                                   [unitless]

    Outputs: 
    RESULTS.
      X_T          - reshaped distance along airfoil surface             [unitless]
      THETA_T      - momentum thickness                                  [m]
      DELTA_STAR_T - displacement thickness                              [m] 
      H_T          - shape factor                                        [unitless]
      CF_T         - friction coefficient                                [unitless]
      RE_THETA_T   - Reynolds number as a function of momentum thickness [unitless]
      RE_X_T       - Reynolds number as a function of distance           [unitless]
      DELTA_T      - boundary layer thickness                            [m]

    Properties Used:
    N/A
    """
    
    # Initialize vectors 
    X_T          = np.zeros((npanel,nalpha,nRe))
    THETA_T      = np.zeros_like(X_T)
    DELTA_STAR_T = np.zeros_like(X_T)
    H_T          = np.zeros_like(X_T)
    CF_T         = np.zeros_like(X_T)
    RE_THETA_T   = np.zeros_like(X_T)
    RE_X_T       = np.zeros_like(X_T)
    DELTA_T      = np.zeros_like(X_T)  
    
    if batch_analysis:
        N_ALPHA = nalpha
    else:
        N_ALPHA = 1 
    
    for a_i in range(N_ALPHA):
        for re_i in range(nRe):    
            if not batch_analysis:  
                a_i = re_i 
            # compute laminar boundary layer properties  
            l           = L[a_i,re_i]
            theta_0     = THETA_0 
            Re_L        = RE_L[a_i,re_i]
            nu          = l/Re_L    
            x_i         = X_I.data[:,a_i,re_i][X_I.mask[:,a_i,re_i] ==False]
            Ve_i        = VE_I.data[:,a_i,re_i][VE_I.mask[:,a_i,re_i] ==False]
            dVe_i       = DVE_I.data[:,a_i,re_i][DVE_I.mask[:,a_i,re_i] ==False] 
            y0          = theta_0**2 * getVe(0,x_i,Ve_i)**6   
            theta2_Ve6  = odeint(odefcn, y0,x_i , args=(nu, x_i, Ve_i))  
            
            # Compute momentum thickness, theta 
            theta       = np.sqrt(theta2_Ve6[:,0]/ Ve_i**6)
            
            # find theta values that do not converge and replace them with neighbor
            idx1        = np.where(abs((theta[1:] - theta[:-1])/theta[:-1]) > tol)[0] 
            if len(idx1)> 1:  
                np.put(theta,idx1 + 1, theta[idx1])
            
            # Thwaites separation criteria 
            lambda_val  = theta**2 * dVe_i / nu 
            
            # Compute H 
            H           = getH(lambda_val)
            H[H<0]      = 1E-6   # H cannot be negative 
            # find H values that do not converge and replace them with neighbor
            idx1        = np.where(abs((H[1:] - H[:-1])/H[:-1]) >tol)[0]
            if len(idx1)> 1: 
                np.put(H,idx1 + 1, H[idx1]) 
            
            # Compute Reynolds numbers based on momentum thickness  
            Re_theta    = Ve_i * theta / nu
            
            # Compute Reynolds numbers based on distance along airfoil
            Re_x        = Ve_i * x_i/ nu
            
            # Compute skin friction 
            cf          = getcf(lambda_val ,Re_theta)
            cf[cf<0]    = 1E-6 
            
            # Compute displacement thickness
            del_star    = H*theta   
            
            # Compute boundary layer thickness 
            delta       = 5.2*x_i/np.sqrt(Re_x)
            delta[0]    = 0   
            
            # Reynolds number at x=0 cannot be negative 
            Re_x[0]     = 1E-5
            
            # Find where matrices are not masked 
            indices = np.where(X_I.mask[:,a_i,re_i] == False)
            
            # Store results 
            np.put(X_T[:,a_i,re_i],indices,x_i)
            np.put(THETA_T[:,a_i,re_i],indices,theta)
            np.put(DELTA_STAR_T[:,a_i,re_i],indices,del_star)
            np.put(H_T[:,a_i,re_i],indices,H)
            np.put(CF_T[:,a_i,re_i],indices ,cf)
            np.put(RE_THETA_T[:,a_i,re_i],indices,Re_theta)
            np.put(RE_X_T[:,a_i,re_i],indices,Re_x)
            np.put(DELTA_T[:,a_i,re_i],indices,delta)
    
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
    lamdda_val  - thwaites separation criteria [unitless]

    Outputs:  
    H           - shape factor [unitless]

    Properties Used:
    N/A
    """       
    H       = 0.0731/(0.14 + lambda_val ) + 2.088 
    idx1    = (lambda_val>0.0)  
    H[idx1] = 2.61 - 3.75*lambda_val[idx1]  + 5.24*lambda_val[idx1]**2   
    return H 
    
def odefcn(y,x, nu,x_i,Ve_i):
    """ Computes boundary layer functions using SciPy ODE solver 

    Assumptions:
    None

    Source:
    None

    Inputs: 
    y           - initial conditions of functions    [unitless]
    x           - new x values at which to solve ODE [unitless]
    nu          - kinematic viscosity                [m^2/s]
    x_i         - intial array of x values           [unitless]
    Ve_i        - intial boundary layer velocity     [m/s]
    
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
    x         - new x dimension                    [unitless]
    x_i       - old x dimension                    [unitless]
    Ve_i      - old boundary layer velocity values [m/s] 
    
    Outputs:  
    Ve        - new boundary layer velocity values [m/s]

    Properties Used:
    N/A 
    """
    Ve_func = interp1d(x_i,Ve_i, axis=0,fill_value = "extrapolate")
    Ve      = Ve_func(x)
    return Ve  

def getdVe(x,x_i,dVe_i):
    """ Interpolates the derivatives of the bounday layer velocity over a new dimension of x 

    Assumptions:
    None

    Source:
    None

    Inputs: 
    x         - new x dimension                                   [unitless]
    x_i       - old x dimension                                   [unitless]
    dVe_i     - old derivative of boundary layer velocity values  [m/s-m]
    
    Outputs:  
    dVe       - new derivative of boundary layer velocity values  [m/s-m]

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
    lambda_val - thwaites separation criteria                        [unitless]
    Re_theta   - Reynolds Number as a function of momentum thickness [unitless]

    Outputs:  
    cf         - skin friction coefficient [unitless]

    Properties Used:
    N/A 
    """        
    l       = 0.22 + 1.402*lambda_val  + (0.018*lambda_val)/(0.107 + lambda_val ) 
    idx1    = (lambda_val>0.0)   
    l[idx1] = 0.22 + 1.57*lambda_val[idx1] - 1.8*lambda_val[idx1]**2 
    cf      = 2*l/Re_theta  
    return cf 
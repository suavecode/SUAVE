## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method
# heads_method.py 
# Created:  Mar 2021, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
from SUAVE.Core import Data 
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import odeint
# ----------------------------------------------------------------------
# heads_method.py 
# ----------------------------------------------------------------------   
## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method
def heads_method(nalpha,nRe,DEL_0,THETA_0,DELTA_STAR_0, L, RE_L, X_I,VE_I, DVE_I,X_TR,batch_analysis, n = 200):
    """ Computes the boundary layer characteristics in turbulent
    flow pressure gradients

    Assumptions:
    None

    Source:
    None

    Inputs: 
    del_0       - intital bounday layer thickness 
    theta_0     - intial momentum thickness 
    del_star_0  - initial displacement thickness 
    L           - normalized lenth of surface
    Re_L        - Reynolds number
    x_i         - x coordinated on surface of airfoil
    Ve_i        - boundary layer velocity at transition location 
    dVe_i       - intial derivative value of boundary layer velocity at transition location
    x_tr        - transition location on surface 
    n           - number of points on surface 

    Outputs: 
    x           - new dimension of x coordinated on surface of airfoil
    theta       - momentum thickness
    del_star    - displacement thickness
    H           - shape factor
    cf          - friction coefficient
    delta       - boundary layer thickness

    Properties Used:
    N/A
    """   
    
    X_H          = np.zeros((n,nalpha,nRe))
    THETA_H      = np.zeros_like(X_H)
    DELTA_STAR_H = np.zeros_like(X_H)
    H_H          = np.zeros_like(X_H)
    CF_H         = np.zeros_like(X_H) 
    RE_THETA_H   = np.zeros_like(X_H)
    RE_X_H       = np.zeros_like(X_H)
    DELTA_H      = np.zeros_like(X_H)       

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
            l            = L[a_i,re_i]
            theta_0      = THETA_0[a_i,re_i] 
            Re_L         = RE_L[a_i,re_i] 
            nu           = l/Re_L    
            x_i          = X_I.data[:,0,0][X_I.mask[:,0,0] ==False]
            Ve_i         = VE_I.data[:,0,0][VE_I.mask[:,0,0] ==False]
            dVe_i        = DVE_I.data[:,0,0][DVE_I.mask[:,0,0] ==False]
            del_0        = DEL_0[a_i,re_i]
            del_star_0   = DELTA_STAR_0[a_i,re_i]
            H_0          = del_star_0 / theta_0
            H1_0         = getH1(np.atleast_1d(H_0))[0]
            if np.isnan(H1_0):
                H1_0     = (del_0 - del_star_0) / theta_0 
            y0           = [theta_0, getVe(0,x_i,Ve_i)*theta_0*H1_0]    
            xspan        = np.linspace(0,l,n)   
            y            = odeint(odefcn,y0,xspan,args=(Re_L/l, x_i, Ve_i, dVe_i)) 
            theta       = y[:,0] 
            Ve_theta_H1 = y[:,1]   
            
            idx1            = np.where(abs((theta[1:] - theta[:-1])/theta[:-1]) > 2E1)[0]
            if len(idx1)> 1:
                next_idx        = idx1 + 1
                np.put(theta,next_idx, theta[idx1])   
            
                     
            idx1               = np.where(abs((Ve_theta_H1[1:] - Ve_theta_H1[:-1])/Ve_theta_H1[:-1]) > 2E1)[0]
            if len(idx1)> 1:
                next_idx           = idx1 + 1
                np.put(Ve_theta_H1,next_idx, Ve_theta_H1[idx1])   
            
            # compute flow properties    
            x            = np.linspace(0,l,n)       
            H1           = Ve_theta_H1/(theta*getVe(x, x_i, Ve_i))
            H            = getH(np.atleast_1d(H1))
            Re_theta     = Re_L/l * getVe(x,x_i,Ve_i) * theta 
            Re_x         = getVe(x,x_i,Ve_i) * x/ nu
            cf           = getcf(np.atleast_1d(Re_theta),np.atleast_1d(H))
            del_star     = H*theta   
            delta        = theta*H1 + del_star 
            delta[0]     = 0       
            Re_x[0]      = 1E-5
            
            X_H[:,a_i,re_i]          = x
            THETA_H[:,a_i,re_i]      = theta
            DELTA_STAR_H[:,a_i,re_i] = del_star
            H_H[:,a_i,re_i]          = H
            CF_H[:,a_i,re_i]         = cf 
            RE_THETA_H[:,a_i,re_i]   = Re_theta
            RE_X_H[:,a_i,re_i]       = Re_x
            DELTA_H[:,a_i,re_i]      = delta
        
        
    RESULTS = Data(
        X_H          = X_H,      
        THETA_H      = THETA_H,   
        DELTA_STAR_H = DELTA_STAR_H,
        H_H          = H_H,       
        CF_H         = CF_H,   
        RE_THETA_H   = RE_THETA_H,   
        RE_X_H       = RE_X_H,    
        DELTA_H      = DELTA_H,   
    )    
         
    return  RESULTS

def getH(H1):
    """ Computes the shape factor, H

    Assumptions:
    None

    Source:
    None

    Inputs: 
    H1       - mass flow shape factor

    Outputs:  
    H        - shape factor

    Properties Used:
    N/A
    """         
    H       = 0.6778 + 1.1536*(H1-3.3)**-0.326
    idx1    = (H1 < 3.3)
    H[idx1] = 3.0
    idx2    = (H1 > 5.3)
    H[idx2] = 1.1 + 0.86*(H1[idx2] - 3.3)**-0.777 
    #idx3    = (H<0) # this makes sure the values are sensical 
    #H[idx3] = 1E-6    
    return H 

def getH1(H) :    
    """ Computes the mass flow shape factor, H1

    Assumptions:
    None

    Source:
    None

    Inputs: 
    H        - shape factor

    Outputs:  
    H1       - mass flow shape factor

    Properties Used:
    N/A 
    """
    H1       = 3.3 + 0.8234*(H - 1.1)**-1.287  
    idx1     = (H > 1.6) 
    H1[idx1] = 3.3 + 1.5501*(H[idx1] - 0.6778)**-3.064
    return H1 

def odefcn(y,x,ReL_div_L, x_i, Ve_i, dVe_i): 
    """ Computes bounday layer functions using SciPy ODE solver 

    Assumptions:
    None

    Source:
    None

    Inputs: 
    y           - initial conditions of functions 
    x           - new x values at which to solve ODE
    ReL_div_L   - ratio of Reynolds number to length of surface 
    x_i         - intial array of x values 
    Ve_i        - intial boundary layer velocity
    dVe_i       - initial derivative of bounday layer velocity
    
    Outputs:  
    f           - 2D function of momentum thickness and the product of 
                  the velocity,momentum thickness and the mass flow shape factor

    Properties Used:
    N/A 
    """    
    theta       = y[0]
    Ve_theta_H1 = y[1]  
    
    if theta == 0:
        H1 = Ve_theta_H1 / (theta + 1e-6) / getVe(x,x_i,Ve_i)
    else:
        H1 = Ve_theta_H1 / theta / getVe(x,x_i,Ve_i)
    
    H           = getH(np.atleast_1d(H1))
    Re_theta    = ReL_div_L * theta
    cf          = getcf(np.atleast_1d(Re_theta),np.atleast_1d(H))
    dydx_1      = 0.5*cf-(theta/getVe(x,x_i,Ve_i))*(2+H)*getdVe(x, x_i, dVe_i)
    dydx_2      = getVe(x,x_i,Ve_i)*0.0306*(H1 - 3)**-0.6169 
    f           = [dydx_1,dydx_2] 
    return f 

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
    Ve_func = interp1d(x_i,Ve_i,fill_value = "extrapolate")
    Ve      = Ve_func(x)
    return Ve 

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

def getcf(Re_theta,H): 
    """ Computes the skin friction coefficient, cf

    Assumptions:
    None

    Source:
    None

    Inputs: 
    Re_theta - Reynolds Number as a function of momentum thickness 
    H        - shape factor

    Outputs:  
    cf       - skin friction coefficient

    Properties Used:
    N/A 
    """    
    cf       = 0.246*10**(-0.678*H)*(Re_theta)**-0.268 
    idx1     = (Re_theta == 0) 
    cf[idx1] = 0.246*10**(-0.678*H[idx1])*(1e-3)**-0.268 
    return cf 
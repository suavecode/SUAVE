## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method
# heads_method.py 
# Created:  Mar 2021, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import SUAVE 
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import odeint
# ----------------------------------------------------------------------
# heads_method.py 
# ----------------------------------------------------------------------   
## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method
def heads_method_old(del_0,theta_0, del_star_0, L, Re_L, x_i, Ve_i, dVe_i,x_tr, n = 200):
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
   
    nu           = L / Re_L
    H_0          = del_star_0 / theta_0
    H1_0         = getH1(np.atleast_1d(H_0))[0]
    if np.isnan(H1_0):
        H1_0     = (del_0 - del_star_0) / theta_0 
    y0           = [theta_0, getVe(0,x_i,Ve_i)*theta_0*H1_0]    
    xspan        = np.linspace(0,L,n)  
    ReL_div_L    = Re_L/L
    y            = odeint(odefcn,y0,xspan,args=(ReL_div_L, x_i, Ve_i, dVe_i)) 
    thetav       = y[:,0] 
    Ve_theta_H1v = y[:,1]   
    
    theta           = thetav
    idx1            = (abs((thetav[1:] - thetav[:-1])/thetav[:-1]) > 5E-1)
    theta[1:][idx1] = thetav[:-1][idx1] + 1E-12    
            
    Ve_theta_H1           = Ve_theta_H1v
    idx2                  = (abs((Ve_theta_H1v[1:] - Ve_theta_H1v[:-1])/Ve_theta_H1v[:-1]) > 5E-1)
    Ve_theta_H1[1:][idx2] = Ve_theta_H1v[:-1][idx2] + 1E-12      
    
    # compute flow properties    
    x            = np.linspace(0,L,n)       
    H1           = Ve_theta_H1/(theta*getVe(x, x_i, Ve_i))
    H            = getH(np.atleast_1d(H1))
    Re_theta     = Re_L/L * getVe(x,x_i,Ve_i) * theta 
    cf           = getcf(np.atleast_1d(Re_theta),np.atleast_1d(H))
    del_star     = H*theta   
    delta        = theta*H1 + del_star 
    delta[0]     = 0 
     
    return x, theta, del_star, H, cf ,delta  

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
    idx3    = (H<0) # this makes sure the values are sensical 
    H[idx3] = 1E-6    
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
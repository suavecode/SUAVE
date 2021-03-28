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
def heads_method(del_0,theta_0, del_star_0, L, Re_L, x_i, Ve_i, dVe_i):
    """ Computes the boundary layer characteristics in turbulen
    flow pressure gradients

    Assumptions:
    None

    Source:
    None

    Inputs: 

    Outputs: 

    Properties Used:
    N/A
    """        
    n            = 100
    nu           = L / Re_L
    H_0          = del_star_0 / theta_0
    H1_0         = getH1(np.atleast_1d(H_0))[0]
    if np.isnan(H1_0):
        H1_0     = (del_0 - del_star_0) / theta_0 
    y0           = [theta_0, getVe(0,x_i,Ve_i)*theta_0*H1_0]    
    xspan        = np.linspace(0,L,n)  
    ReL_div_L    = Re_L/L
    y            = odeint(odefcn,y0,xspan,args=(ReL_div_L, x_i, Ve_i, dVe_i)) 
    theta        = y[:,0] 
    Ve_theta_H1  = y[:,1]
    x            = np.linspace(0,L,n)  
    
    # compute flow properties 
    H1           = Ve_theta_H1/(theta*getVe(x, x_i, Ve_i))
    H            = getH(np.atleast_1d(H1))
    Re_theta     = Re_L/L * getVe(x,x_i,Ve_i) * theta
    Re_x         = getVe(x,x_i,Ve_i) * x/ nu
    cf           = getcf(np.atleast_1d(Re_theta),np.atleast_1d(H))
    del_star     = H*theta  
    delta        = 0.37*x/(Re_x**0.2)
    
    return x, theta, del_star, H, cf ,delta  

def getH(H1):
    H       = 0.6778 + 1.1536*(H1-3.3)**-0.326
    idx1    = (H1 < 3.3)
    H[idx1] = 3.0
    idx2    = (H1 > 5.3)
    H[idx2] = 1.1 + 0.86*(H1[idx2] - 3.3)**-0.777 
    idx3    = (H<0) # this makes sure the values are sensical 
    H[idx3] = 1E-6    
    return H 

def getH1(H) :    
    H1       = 3.3 + 0.8234*(H - 1.1)**-1.287 
    idx1     = (H > 1.6) 
    H1[idx1] = 3.3 + 1.5501*(H[idx1] - 0.6778)**-3.064
    return H1 

def odefcn(y,x,ReL_div_L, x_i, Ve_i, dVe_i): 
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
    Ve_func = interp1d(x_i,Ve_i,fill_value = "extrapolate")
    Ve      = Ve_func(x)
    return Ve 

def getdVe(x,x_i,dVe_i):
    dVe_func = interp1d(x_i,dVe_i,fill_value = "extrapolate")
    dVe      = dVe_func(x)
    return dVe  

def getcf(Re_theta,H): 
    cf       = 0.246*10**(-0.678*H)*(Re_theta)**-0.268 
    idx1     = (Re_theta == 0) 
    cf[idx1] = 0.246*10**(-0.678*H[idx1])*(Re_theta[idx1] + 1e-3)**-0.268 
    return cf 
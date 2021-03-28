## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method
# thwaites_method.py 

# Created:  Mar 2021, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import SUAVE 
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import odeint

# ----------------------------------------------------------------------
# thwaites_method
# ----------------------------------------------------------------------  
## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method
def thwaites_method(theta_0, L, Re_L, x_i, Ve_i, dVe_i):
    """ Computes the boundary layer characteristics in laminar 
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
    n           = 100
    nu          = L / Re_L
    y0          = theta_0**2 * getVe(0,x_i,Ve_i)**6 
    xspan       = np.linspace(0,L,n)  
    theta2_Ve6  = odeint(odefcn, y0, xspan, args=(nu, x_i, Ve_i)) 
    x           = np.linspace(0,L,n) 
    theta       = np.sqrt(theta2_Ve6[:,0]/ getVe(x, x_i, Ve_i)**6)
    
    # thwaites separation criteria 
    lambda_val  = theta**2 * getdVe(x,x_i,dVe_i) / nu 
    
    # compute flow properties 
    H           = getH(lambda_val )
    Re_theta    = getVe(x,x_i,Ve_i) * theta/ nu
    Re_x        = getVe(x,x_i,Ve_i) * x/ nu
    cf          = getcf(lambda_val ,Re_theta)
    del_star    = H *theta 
    delta       = 5.2*x/np.sqrt(Re_x)
    
    return x, theta, del_star, H, cf, Re_theta, Re_x , delta 
    
def getH(lambda_val ): 
    H       = 0.0731/(0.14 + lambda_val ) + 2.088 
    idx1    = (lambda_val>0.0)  
    H[idx1] = 2.61 - 3.75*lambda_val[idx1]  + 5.24*lambda_val[idx1]**2  
    idx2    = (H<0) # this makes sure the values are sensical 
    H[idx2] = 1E-6
    return H 
    
def odefcn(y,x, nu,x_i,Ve_i):
    dydx = 0.45*getVe(x,x_i,Ve_i)**5*nu
    return dydx 
    
def getVe(x,x_i,Ve_i):
    Ve_func = interp1d(x_i,Ve_i,fill_value = "extrapolate")
    Ve      = Ve_func(x)
    return Ve 

def getdVe(x,x_i,dVe_i):
    dVe_func = interp1d(x_i,dVe_i,fill_value = "extrapolate")
    dVe      = dVe_func(x)
    return dVe 

def getcf(lambda_val , Re_theta):
    l       = 0.22 + 1.402*lambda_val  + (0.018*lambda_val)/(0.107 + lambda_val ) 
    idx1    = (lambda_val>0.0)   
    l[idx1] = 0.22 + 1.57*lambda_val[idx1] - 1.8*lambda_val[idx1] **2 
    cf      = 2*l/Re_theta  
    return cf 
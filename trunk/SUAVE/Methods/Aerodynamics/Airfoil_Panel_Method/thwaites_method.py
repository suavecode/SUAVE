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
def thwaites_method(theta_0, L, Re_L, x_i, Ve_i, dVe_i,n = 200):
    """ Computes the boundary layer characteristics in laminar 
    flow pressure gradients
    
    Assumptions:
    None

    Inputs:  
    theta_0     - intial momentum thickness  
    L           - normalized lenth of surface
    Re_L        - Reynolds number
    x_i         - x coordinated on surface of airfoil
    Ve_i        - boundary layer velocity at transition location 
    dVe_i       - intial derivative value of boundary layer velocity at transition location 
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
        
    # remove non converged points in ODE solver 
    for i in range(len(del_star)-1): 
        if (abs((del_star[i+1] - del_star[i])/del_star[i]) > 1E2): 
            del_star[i+1]  = del_star[i] + 1E-12 
            
    delta       = 5.2*x/np.sqrt(Re_x)
    delta[0]    = 0 
    Re_x[0]     = 1e-12
    
    return x, theta, del_star, H, cf, Re_theta, Re_x , delta 
    
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
    idx2    = (H<0) # this makes sure the values are sensical 
    H[idx2] = 1E-6
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
    l[idx1] = 0.22 + 1.57*lambda_val[idx1] - 1.8*lambda_val[idx1] **2 
    cf      = 2*l/Re_theta  
    return cf 
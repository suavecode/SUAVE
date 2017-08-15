## @ingroup Methods-Flight_Dynamics-Dynamic_Stability-Approximations
# spiral.py
# 
# Created:  Apr 2014, A. Wendorff
# Modified: Jan 2016, E. Botero

# ----------------------------------------------------------------------
#   Method
# ----------------------------------------------------------------------

## @ingroup Methods-Flight_Dynamics-Dynamic_Stability-Approximations
def spiral(mass, velocity, density, S_gross_w, Cl_p, Cn_Beta, Cy_phi, Cl_Beta, Cn_r, Cl_r):
    """ This calcualtes the approximate time constant for the spiral mode         
  
    Assumptions:
        Linearized equations of motion
        X-Z axis is plane of symmetry
        Constant mass of aircraft
        Origin of axis system at c.g. of aircraft
        Aircraft is a rigid body
        Earth is inertial reference frame
        Perturbations from equilibrium are small
        Flow is Quasisteady      
                   
    Source:
        J.H. Blakelock, "Automatic Control of Aircraft and Missiles" Wiley & Sons, Inc. New York, 1991, p 142.        
        
    Inputs:
        mass - mass of the aircraft                                            [kilograms]
        velocity - flight velocity at the condition being considered           [meters/seconds]
        density - flight density at condition being considered                 [kg/meters**3]
        S_gross_w - area of the wing                                           [meters**2]
        Cl_p - change in rolling moment due to the rolling velocity            [dimensionless]
        Cn_Beta - coefficient for change in yawing moment due to sideslip      [dimensionless]
        Cy_phi - coefficient for change in sideforce due to aircraft roll      [dimensionless] (Usually equals C_L)
        Cl_Beta - coefficient for change in rolling moment due to sideslip     [dimensionless]
        Cn_r - coefficient for change in yawing moment due to yawing velocity  [dimensionless]
        Cl_r - coefficient for change in rolling moment due to yawing velocity [dimensionless] (Usually equals C_L/4)
    
    Outputs:
        spiral_tau - time constant for the spiral mode                         [seconds] (positive values are bad)
     
    Properties Used:
        N/A           
    """ 
    
    #process
    spiral_tau = mass * velocity / S_gross_w / (0.5 * density * velocity **2.) * (Cl_p*Cn_Beta/(Cy_phi*(Cl_Beta*Cn_r-Cn_Beta*Cl_r)))
    
    return spiral_tau
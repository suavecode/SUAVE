## @ingroup Plots
# Airfoil_Plots.py
#
# Created: Mar 2021, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import matplotlib.pyplot as plt  

## @ingroup Plots
def plot_airfoil_properties(ap,line_style= 'k-',arrow_color = 'r',plot_pressure_vectors = False ):  
  
    fig  = plt.figure('Airfoil')
    axis1 = fig.add_subplot(1,1,1)     
    axis1.plot(ap.x, ap.y,line_style) 
    axis1.set_xlabel('x')
    axis1.set_ylabel('y')  
    
    fig  = plt.figure('Pressure_Coefficient')
    axis2 = fig.add_subplot(1,1,1)     
    axis2.invert_yaxis() 
    axis2.plot(ap.x, ap.Cp,line_style) 
    axis2.set_xlabel('x')
    axis2.set_ylabel('$C_p$') 
    
    fig  = plt.figure('Boundary_Layer_Edge_Velocity')
    axis3 = fig.add_subplot(1,1,1)     
    axis3.plot(ap.x, abs(ap.Ue_Vinf),line_style)
    axis3.set_xlabel('x')
    axis3.set_ylabel('$Ue/V_{inf}$')   
    
    fig  = plt.figure('Momentum_Thickness')
    axis = fig.add_subplot(1,1,1)   
    axis.plot(ap.x, ap.theta,line_style, label = r'$\theta$' ) 
    axis.set_xlabel('x')
    axis.set_ylabel('Momentum Thickness') 
    axis.legend(loc='upper right')   
    
    fig  = plt.figure('Displacement Thickness')
    axis4 = fig.add_subplot(1,1,1)    
    axis4.plot(ap.x,ap.delta_star,line_style, label = r'$\delta$*')
    axis4.set_xlabel('x')
    axis4.set_ylabel('Displacement Thickness') 
    axis4.legend(loc='upper right')    
    
    if plot_pressure_vectors: 
        fig  = plt.figure('Airfoil_Pressure_Normals')
        axis5 = fig.add_subplot(1,1,1)     
        axis5.plot(ap.x, ap.y,line_style) 
        for i in range(len(ap.x)):
            dx_val = ap.normals[i][0]*abs(ap.Cp[i])*0.1
            dy_val = ap.normals[i][1]*abs(ap.Cp[i])*0.1
            if ap.Cp[i] < 0:
                plt.arrow(x= ap.x[i], y=ap.y[i] , dx= dx_val , dy = -dy_val , 
                          fc=arrow_color, ec=arrow_color,head_width=0.005, head_length=0.01 )   
            else:
                plt.arrow(x= ap.x[i]+dx_val , y= ap.y[i]-dy_val , dx= -dx_val , dy = dy_val , 
                          fc=arrow_color, ec=arrow_color,head_width=0.005, head_length=0.01 )   
    
    return 
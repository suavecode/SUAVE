## @ingroup Plots
# Airfoil_Plots.py
#
# Created: Mar 2021, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import SUAVE
from SUAVE.Core import Units
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.import_airfoil_polars \
     import import_airfoil_polars 
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.compute_airfoil_polars \
     import compute_airfoil_polars 
import numpy as np 
import matplotlib.pyplot as plt  
import matplotlib.cm as cm
import os

## @ingroup Plots
def plot_airfoil_analysis_boundary_layer_properties(ap,show_legend = True ):  
    """ This plots the boundary layer properties of an airfoil
        or group of airfoils
    
        Assumptions:
        None
        
        Inputs: 
        ap       - data stucture of airfoil boundary layer properties  
        
        Outputs: 
        None 
        
        Properties Used:
        N/A
        """        
    
    # determine dimension of angle of attack and reynolds number 
    nAoA = len(ap.AoA)
    nRe  = len(ap.Re)
    
    # create array of colors for difference reynolds numbers 
    colors  = cm.rainbow(np.linspace(0, 1,nAoA))
    markers = ['o','v','s','P','p','^','D','X','*']
    
    fig1  = plt.figure('Airfoil Geometry',figsize=(8,6)) 
    axis1 = fig1.add_subplot(1,1,1)     
    axis1.set_xlabel('x')
    axis1.set_ylabel('y')   
    axis1.set_ylim(-0.2, 0.2)  
    
    fig2  = plt.figure('Airfoil Boundary Layer Properties',figsize=(12,8))
    axis2 = fig2.add_subplot(2,3,1)      
    axis2.set_ylabel('$Ue/V_{inf}$')    
     
    axis3 = fig2.add_subplot(2,3,2)      
    axis3.set_ylabel('$dV_e/dx$')   
    axis3.set_ylim(-1, 10)  
      
    axis4 = fig2.add_subplot(2,3,3)   
    axis4.set_ylabel(r'$\theta$')  
     
    axis5 = fig2.add_subplot(2,3,4)    
    axis5.set_xlabel('x')
    axis5.set_ylabel('$H$')   
     
    axis6 = fig2.add_subplot(2,3,5)     
    axis6.set_xlabel('x')
    axis6.set_ylabel(r'$\delta$*')  
     
    axis7 = fig2.add_subplot(2,3,6)     
    axis7.set_xlabel('x')
    axis7.set_ylabel(r'$\delta$')   
     
    fig3  = plt.figure('Airfoil Cp',figsize=(8,6)) 
    axis8 = fig3.add_subplot(1,1,1)      
    axis8.set_ylabel('$C_p$') 
    axis8.set_ylim(1.2,-7)  
     
    mid = int(len(ap.x)/2)
    
    for i in range(nRe): 
        
        for j in range(nAoA):
        
            tag = 'AoA: ' + str(round(ap.AoA[j][0]/Units.degrees,2)) + '$\degree$, Re: ' + str(round(ap.Re[i][0]/1000000,2)) + 'E6'
            
            axis1.plot(ap.x[:,j,i], ap.y[:,j,i],'k-') 
            axis1.plot(ap.x_bl[:,j,i],ap.y_bl[:,j,i],color = colors[j], linestyle = '-' ,marker =  markers[j%9] , label = tag)            
             
            axis2.plot(ap.x[:mid,j,i], abs(ap.Ue_Vinf)[:mid,j,i],color = colors[j], linestyle = '-' ,marker =  markers[j%9] , label= tag )  
            axis2.plot(ap.x[mid:,j,i], abs(ap.Ue_Vinf)[mid:,j,i],color = colors[j], linestyle = '--' ,marker =  markers[j%9])   
           
            axis3.plot(ap.x[:mid,j,i], abs(ap.dVe)[:mid,j,i],color = colors[j], linestyle = '-' ,marker =  markers[j%9] )
            axis3.plot(ap.x[mid:,j,i], abs(ap.dVe)[mid:,j,i],color = colors[j], linestyle = '--' ,marker =  markers[j%9])
             
            axis4.plot(ap.x[:mid,j,i], ap.theta[:mid,j,i],color = colors[j], linestyle = '-' ,marker =  markers[j%9] )  
            axis4.plot(ap.x[mid:,j,i], ap.theta[mid:,j,i],color = colors[j], linestyle = '--' ,marker =  markers[j%9])    
                     
            axis5.plot(ap.x[:mid,j,i], ap.H[:mid,j,i],color = colors[j], linestyle = '-' ,marker =  markers[j%9]  )  
            axis5.plot(ap.x[mid:,j,i], ap.H[mid:,j,i],color = colors[j], linestyle = '--' ,marker =  markers[j%9] )    
            
            axis6.plot(ap.x[:mid,j,i],ap.delta_star[:mid,j,i],color = colors[j], linestyle = '-' ,marker =  markers[j%9] ) 
            axis6.plot(ap.x[mid:,j,i],ap.delta_star[mid:,j,i],color = colors[j], linestyle = '--' ,marker =  markers[j%9])  
            
            axis7.plot(ap.x[:mid,j,i],ap.delta[:mid,j,i],color = colors[j], linestyle = '-' ,marker =  markers[j%9] )    
            axis7.plot(ap.x[mid:,j,i],ap.delta[mid:,j,i],color = colors[j], linestyle = '--' ,marker =  markers[j%9]) 
            
            axis8.plot(ap.x[:mid,j,i], ap.Cp[:mid,j,i] ,color = colors[j], linestyle = '-' ,marker =  markers[j%9] , label= tag) 
            axis8.plot(ap.x[mid:,j,i], ap.Cp[ mid:,j,i],color = colors[j], linestyle = '--' ,marker =  markers[j%9])    
             
                           
    # add legends for plotting
    plt.tight_layout()
    if show_legend:
        lines1, labels1 = fig2.axes[0].get_legend_handles_labels()
        fig2.legend(lines1, labels1, loc='upper center', ncol=5)
        plt.tight_layout()
        axis8.legend(loc='upper right')     
    return   
 

## @ingroup Plots
def plot_airfoil_analysis_polars(ap,show_legend = True):  
    """ This plots the polars of an airfoil or group of airfoils
    
        Assumptions:
        None
        
        Inputs: 
        ap       - data stucture of airfoil boundary layer properties and polars 
         
        Outputs: 
        None 
        
        Properties Used:
        N/A
        """        
    
    # determine dimension of angle of attack and reynolds number 
    nAoA = len(ap.AoA)
    nRe  = len(ap.Re)
    
    # create array of colors for difference reynolds numbers 
    colors  = cm.rainbow(np.linspace(0, 1,nAoA))
    markers = ['o','v','s','P','p','^','D','X','*']
    
    fig1  = plt.figure('Airfoil Geometry',figsize=(8,6)) 
    axis1 = fig1.add_subplot(1,1,1)     
    axis1.set_xlabel('x')
    axis1.set_ylabel('y')   
    axis1.set_ylim(-0.2, 0.2)  
     
    
    fig4   = plt.figure('Airfoil Polars',figsize=(12,5))
    axis12 = fig4.add_subplot(1,3,1)     
    axis12.set_title('Lift Coefficients')
    axis12.set_xlabel('AoA')
    axis12.set_ylabel(r'$C_l$') 
    axis12.set_ylim(-1,2)  
    
    axis13 = fig4.add_subplot(1,3,2)    
    axis13.set_title('Drag Coefficient') 
    axis13.set_xlabel('AoA')
    axis13.set_ylabel(r'$C_d$') 
    axis13.set_ylim(0,0.1)  
    
    axis14 = fig4.add_subplot(1,3,3)   
    axis14.set_title('Moment Coefficient')  
    axis14.set_xlabel('AoA')
    axis14.set_ylabel(r'$C_m$')    
    axis14.set_ylim(-0.1,0.1)  
      
    for i in range(nRe):  
        
        Re_tag  = 'Re: ' + str(round(ap.Re[i][0]/1000000,2)) + 'E6'
        
        # Lift Coefficient
        axis12.plot(ap.AoA[:,0]/Units.degrees,ap.Cl[:,i],color = colors[i], linestyle = '-' ,marker =  markers[i], label= Re_tag )
        
        # Drag Coefficient
        axis13.plot(ap.AoA[:,0]/Units.degrees,ap.Cd[:,i],color = colors[i], linestyle = '-',marker =  markers[i], label =  Re_tag)  
        
        # Moment Coefficient
        axis14.plot(ap.AoA[:,0]/Units.degrees, ap.Cm[:,i],color = colors[i], linestyle = '-',marker =  markers[i], label =  Re_tag)     
        plt.tight_layout() 
    
    # add legends for plotting 
    if show_legend:
        axis12.legend(loc='upper left')   
        axis13.legend(loc='upper left')      
        axis14.legend(loc='upper left')  
        
    return   
 
## @ingroup Plots
def plot_airfoil_analysis_surface_forces(ap,show_legend= True,arrow_color = 'r'):  
    """ This plots the forces on an airfoil surface
    
        Assumptions:
        None
        
        Inputs: 
        ap       - data stucture of airfoil boundary layer properties and polars 
         
        Outputs: 
        None 
        
        Properties Used:
        N/A
        """        
    
    # determine dimension of angle of attack and reynolds number 
    nAoA = len(ap.AoA)
    nRe  = len(ap.Re)
     
    for i in range(nRe): 
        for j in range(nAoA): 
      
            label =  '_AoA_' + str(round(ap.AoA[j][0]/Units.degrees,2)) + '_deg_Re_' + str(round(ap.Re[i][0]/1000000,2)) + 'E6'
            fig   = plt.figure('Airfoil_Pressure_Normals' + label )
            axis15 = fig.add_subplot(1,1,1)      
            axis15.plot(ap.x[:,j,i], ap.y[:,j,i],'k-') 
            
            for k in range(len(ap.x)):
                dx_val = ap.normals[k,0,j,i]*abs(ap.Cp[k,j,i])*0.1
                dy_val = ap.normals[k,1,j,i]*abs(ap.Cp[k,j,i])*0.1
                if ap.Cp[k,j,i] < 0:
                    plt.arrow(x= ap.x[k,j,i], y=ap.y[k,j,i] , dx= dx_val , dy = dy_val , 
                              fc=arrow_color, ec=arrow_color,head_width=0.005, head_length=0.01 )   
                else:
                    plt.arrow(x= ap.x[k,j,i]+dx_val , y= ap.y[k,j,i]+dy_val , dx= -dx_val , dy = -dy_val , 
                              fc=arrow_color, ec=arrow_color,head_width=0.005, head_length=0.01 )   
    
    return   



def plot_airfoil_polar_files(airfoil_path, airfoil_polar_paths, line_color = 'k-', use_surrogate = False, 
                display_plot = False, save_figure = False, save_filename = "Airfoil_Polars", file_type = ".png"):
    """This plots all airfoil polars in the list "airfoil_polar_paths" 

    Assumptions:
    None

    Source:
    None

    Inputs:
    airfoil_polar_paths   [list of strings]

    Outputs: 
    Plots

    Properties Used:
    N/A	
    """
    shape = np.shape(airfoil_polar_paths)
    n_airfoils = shape[0]
    n_Re       = shape[1]
    
    if use_surrogate:
        # Compute airfoil surrogates
        a_data = compute_airfoil_polars(airfoil_path, airfoil_polar_paths, use_pre_stall_data=False)
        CL_sur = a_data.lift_coefficient_surrogates
        CD_sur = a_data.drag_coefficient_surrogates
        
        alpha   = np.asarray(a_data.aoa_from_polar)
        n_alpha = len(alpha.T)
        alpha   = np.reshape(alpha,(n_airfoils,1,n_alpha))
        alpha   = np.repeat(alpha, n_Re, axis=1)
        
        Re      = a_data.re_from_polar
        Re      = np.reshape(Re,(n_airfoils,n_Re,1))
        Re      = np.repeat(Re, n_alpha, axis=2)

        CL = np.zeros_like(Re)
        CD = np.zeros_like(Re)

    else:
        # Use the raw data polars
        airfoil_polar_data = import_airfoil_polars(airfoil_polar_paths)
        CL = airfoil_polar_data.lift_coefficients
        CD = airfoil_polar_data.drag_coefficients
        
        n_alpha = np.shape(CL)[2]
        Re = airfoil_polar_data.reynolds_number
        Re = np.reshape(Re, (n_airfoils,n_Re,1))
        Re = np.repeat(Re, n_alpha, axis=2)


    for i in range(n_airfoils):
        airfoil_name = os.path.basename(airfoil_path[i][0])
        if use_surrogate:
            CL[i,:,:] = CL_sur[airfoil_path[i]](Re[i,:,:],alpha[i,:,:],grid=False)
            CD[i,:,:] = CD_sur[airfoil_path[i]](Re[i,:,:],alpha[i,:,:],grid=False)
            
        # plot all Reynolds number polars for ith airfoil
        fig  = plt.figure(save_filename +'_'+ str(i))
        fig.set_size_inches(10, 4)
        axes = fig.add_subplot(1,1,1)
        axes.set_title(airfoil_name)            
        for j in range(n_Re):
            Re_val = str(round(Re[i,j,0]))
            axes.plot(CD[i,j,:], CL[i,j,:], label='Re='+Re_val)
            
        axes.set_xlabel('$C_D$')  
        axes.set_ylabel('$C_L$')  
        axes.legend(bbox_to_anchor=(1,1), loc='upper left', ncol=1)
        
        if save_figure:
            plt.savefig(save_filename +'_' + str(i) + file_type)   
        if display_plot:
            plt.show()
    return
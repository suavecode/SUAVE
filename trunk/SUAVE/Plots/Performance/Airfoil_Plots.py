## @ingroup Plots
# Airfoil_Plots.py
#
# Created:  Mar 2021, M. Clarke
# Modified: Feb 2022, M. Clarke
#           Aug 2022, R. Erhard

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
from SUAVE.Core import Units
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.import_airfoil_polars   import import_airfoil_polars 
import numpy as np 
import matplotlib.pyplot as plt  
import matplotlib.cm as cm

## @ingroup Plots-Performance
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
    
    for i in range(nAoA):
        for j in range(nRe): 
        
            tag = 'AoA: ' + str(round(ap.AoA[i][0]/Units.degrees,2)) + '$\degree$, Re: ' + str(round(ap.Re[j][0]/1000000,2)) + 'E6'
            
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
 

## @ingroup Plots-Performance
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
 
## @ingroup Plots-Performance
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
    nAoA   = len(ap.AoA)
    nRe    = len(ap.Re)
    n_cpts = len(ap.x[0,0,:])
    

    for i in range(nAoA):     
        for j in range(nRe): 
            label =  '_AoA_' + str(round(ap.AoA[i][0]/Units.degrees,2)) + '_deg_Re_' + str(round(ap.Re[j][0]/1000000,2)) + 'E6'
            fig   = plt.figure('Airfoil_Pressure_Normals' + label )
            axis = fig.add_subplot(1,1,1) 
            axis.plot(ap.x[0,0,:], ap.y[0,0,:],'k-')   
            for k in range(n_cpts):
                dx_val = ap.normals[i,j,k,0]*abs(ap.Cp[i,j,k])*0.1
                dy_val = ap.normals[i,j,k,1]*abs(ap.Cp[i,j,k])*0.1
                if ap.Cp[i,j,k] < 0:
                    plt.arrow(x= ap.x[i,j,k], y=ap.y[i,j,k] , dx= dx_val , dy = dy_val , 
                              fc=arrow_color, ec=arrow_color,head_width=0.005, head_length=0.01 )   
                else:
                    plt.arrow(x= ap.x[i,j,k]+dx_val , y= ap.y[i,j,k]+dy_val , dx= -dx_val , dy = -dy_val , 
                              fc=arrow_color, ec=arrow_color,head_width=0.005, head_length=0.01 )   
    
    return   



## @ingroup Plots-Performance
def plot_airfoil_polars(airfoil_polar_data, aoa_sweep, Re_sweep, display_plot = False, 
                        save_figure = False, save_filename = "Airfoil_Polars", file_type = ".png"):
    """This plots all airfoil polars in the list "airfoil_polar_paths" 

    Assumptions:
    None

    Source:
    None

    Inputs:
    airfoil_data   - airfoil geometry and polar data (see outputs of compute_airfoil_polars.py)
    aoa_sweep      - angles over which to plot the polars                [rad]
    Re_sweep       - Reynolds numbers over which to plot the polars      [-]
    
    Outputs: 
    Plots

    Properties Used:
    N/A	
    """
    # Extract surrogates from airfoil data
    airfoil_names   = airfoil_polar_data.airfoil_names
    airfoil_cl_surs = airfoil_polar_data.lift_coefficient_surrogates
    airfoil_cd_surs = airfoil_polar_data.drag_coefficient_surrogates

    #----------------------------------------------------------------------------
    # plot airfoil polar surrogates
    #----------------------------------------------------------------------------

    col_raw = ['black','firebrick', 'darkorange', 'gold','forestgreen','teal','deepskyblue', 'blue',
               'blueviolet', 'fuchsia', 'deeppink', 'gray'] 
    for jj in range(len(airfoil_names)):
        fig, (ax,ax2,ax3) = plt.subplots(1,3)
        fig.set_figheight(4)
        fig.set_figwidth(12)
        for ii in range(len(Re_sweep)):
            cl_sur = airfoil_cl_surs[airfoil_names[jj]](
                (Re_sweep[ii], aoa_sweep)
            )
            cd_sur = airfoil_cd_surs[airfoil_names[jj]](
                (Re_sweep[ii], aoa_sweep)
            )
            ax.plot(aoa_sweep / Units.deg, cl_sur, col_raw[ii], label="Re="+str(Re_sweep[ii]))
            ax.set_xlabel("Alpha (deg)")
            ax.set_ylabel("Cl")
            ax.legend()
            ax.grid()
            
            ax2.plot(aoa_sweep / Units.deg, cd_sur, col_raw[ii])
            ax2.set_xlabel("Alpha (deg)")
            ax2.set_ylabel("Cd")
            ax2.set_title(airfoil_names[jj])
            ax2.grid()    
            
            ax3.plot(cd_sur, cl_sur, col_raw[ii])
            ax3.set_xlabel("Cd")
            ax3.set_ylabel("Cl")
            ax3.grid() 
        
        plt.tight_layout()

        if save_figure:
            plt.savefig(save_filename + '_' + str(jj) + file_type)   
        if display_plot:
            plt.show()
    return


## @ingroup Plots-Performance
def plot_raw_data_airfoil_polars(airfoil_names, airfoil_polars_path, display_plot = False, 
                        save_figure = False, save_filename = "Airfoil_Polars", file_type = ".png"):
    """This plots all airfoil polars in the list "airfoil_polar_paths" 

    Assumptions:
    None

    Source:
    None

    Inputs:
    airfoil_data   - airfoil geometry and polar data (see outputs of compute_airfoil_polars.py)
    aoa_sweep      - angles over which to plot the polars                [rad]
    Re_sweep       - Reynolds numbers over which to plot the polars      [-]
    
    Outputs: 
    Plots

    Properties Used:
    N/A	
    """
    airfoil_data = import_airfoil_polars(airfoil_polars_path, airfoil_names)

    #----------------------------------------------------------------------------
    # plot airfoil polar surrogates
    #----------------------------------------------------------------------------
    col_raw = ['black','firebrick', 'darkorange', 'gold','forestgreen','teal','deepskyblue', 'blue',
               'blueviolet', 'fuchsia', 'deeppink', 'gray'] 
    for jj in range(len(airfoil_names)):
        Re_sweep = airfoil_data.reynolds_number[airfoil_names[jj]]
        aoa_sweep = airfoil_data.angle_of_attacks[airfoil_names[jj]]
        CL = airfoil_data.lift_coefficients[airfoil_names[jj]]
        CD = airfoil_data.drag_coefficients[airfoil_names[jj]]
        
        fig, (ax,ax2,ax3) = plt.subplots(1,3)
        fig.set_figheight(4)
        fig.set_figwidth(12)
        for ii in range(len(Re_sweep)):
            
            ax.plot(np.array(aoa_sweep[ii]), np.array(CL[ii]), col_raw[ii], label='Re='+str(Re_sweep[ii]))
            ax.set_xlabel("Alpha (deg)")
            ax.set_ylabel("Cl")
            ax.legend()
            ax.grid()
            
            ax2.plot(np.array(aoa_sweep[ii]), np.array(CD[ii]), col_raw[ii])
            ax2.set_xlabel("Alpha (deg)")
            ax2.set_ylabel("Cd")
            ax2.set_title(airfoil_names[jj] + " (Raw Polar Data)")
            ax2.grid()    
            
            ax3.plot(np.array(CD[ii]), np.array(CL[ii]), col_raw[ii])
            ax3.set_xlabel("Cd")
            ax3.set_ylabel("Cl")
            ax3.grid() 
            
        plt.tight_layout()

        if save_figure:
            plt.savefig(save_filename +'_' + str(jj) + file_type)   
        if display_plot:
            plt.show()
    return
## @ingroup Plots
# Airfoil_Plots.py
#
# Created:  Mar 2021, M. Clarke
# Modified: Feb 2022, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import SUAVE
from SUAVE.Core import Units
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.import_airfoil_polars \
     import import_airfoil_polars 
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.compute_airfoil_properties \
     import compute_airfoil_properties 
import numpy as np 
import matplotlib.pyplot as plt  
import matplotlib.cm as cm
import os

## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method
def plot_airfoil_panels(airfoil_results):
    """Plots the airfoil and wake panels
    
    Assumptions:
      None

    Source:  
      None
                                                     
    Inputs:
      airfoil_results : data structure
                                                                           
    Outputs 
      plot of panels
    
    Properties Used:
    N/A
    """          
    fig    = plt.figure()
    axis   = fig.add_subplot(1,1,1)   
    cases  = len(airfoil_results.foil_pts)  
    
    # create array of colors for difference reynolds numbers  
    blacks = cm.copper(np.linspace(0, 0.75,cases)) 
    
    for case in range(cases): 
        axis.plot(airfoil_results.foil_pts[case][:,0], airfoil_results.foil_pts[case][:,1], color = blacks[case] , marker = 'o', linestyle = '-', linewidth =  2, label = f"Airfoil No. {case + 1}")  
        if  airfoil_results.num_wake_pts > 0: 
            axis.plot(airfoil_results.wake_pts[case][:,0], airfoil_results.wake_pts[case][:,1], color = blacks[case] , marker = 'o', linestyle = '-', linewidth = 2, label = f"Airfoil No. {case + 1}")        
   
    axis.set_ylabel(r'$y$') 
    axis.set_xlabel(r'$x$') 
    axis.legend(loc='upper right')             
    return


## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method
def plot_airfoil_cp(airfoil_results):
    """Makes a cp plot with outputs printed
    
    Assumptions:
       None

    Source:   
       None 
                                                     
    Inputs: 
       airfoil_results : data structure
                                                                           
    Outputs:
       cp plot on current axes
    
    Properties Used:
    N/A
    """          
    
    cases  = len(airfoil_results.foil_pts)  
    # create array of colors for difference reynolds numbers
    blues  = cm.winter(np.linspace(0, 0.75,cases))
    reds   = cm.autumn(np.linspace(0, 0.75,cases))
    blacks = cm.copper(np.linspace(0, 0.75,cases))
    
    fig    = plt.figure('CP')
    axis   = fig.add_subplot(1,1,1)   
    for case in range(cases):
        case_label = 'AoA: ' + str(round(airfoil_results.AoA[case][0]/Units.degrees, 2)) + ', Re: ' + str(airfoil_results.Re[case][0])
        if len(airfoil_results.wake_pts[case][:,1]) > 0: 
            xz     = np.hstack((airfoil_results.foil_pts,airfoil_results.wake_pts))  
        else: 
            xz     = airfoil_results.foil_pts 
        if airfoil_results.viscous_flag: 
            for iss in range(3):
                if iss == 0:
                    col = reds[case]
                if iss == 1:
                    col = blues[case]
                if iss == 2:
                    col = blacks[case]
                Is = airfoil_results.vsol_Is[case][iss]
                axis.plot(xz[case,:,0][Is], airfoil_results.cp[case][Is] , color = col, linestyle = '-' , linewidth =  2, label = case_label)  
                axis.plot(xz[case,:,0][Is], airfoil_results.cpi[case][Is], color = col, linestyle = ':', linewidth =  2 ) 
            
        else:
            axis.plot(xz[case,:,0], airfoil_results.cp[case], color = blues[case], linestyle = '-', linewidth = 2, label = case_label)
            
    axis.set_ylabel(r'$c_p$') 
    axis.set_xlabel(r'$x$') 
    axis.legend(loc='upper right')   
    axis.set_ylim([1.2,1.1*np.min(airfoil_results.cp)])   # inverse axis  
     
    return 


## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method
def plot_airfoil(airfoil_results):
    """ Makes an airfoil plot  
    
    Assumptions:
    None

    Source:   
                                                     
    Inputs:
       airfoil_results : data structure
                                                                           
    Outputs   
       airfoil plot on current axes
    
    Properties Used:
    N/A
    """          

    cases  = len(airfoil_results.foil_pts)  
    xz     = np.concatenate((airfoil_results.foil_pts, airfoil_results.wake_pts),axis = 1)       
    
    # create array of colors for difference reynolds numbers  
    blacks = cm.copper(np.linspace(0, 0.75,cases)) 
    fig    = plt.figure()
    axis   = fig.add_subplot(1,1,1)  
    for case in range(cases): 
        axis.plot(xz[case,:,0], xz[case,:,1], color = blacks[case], linestyle = '-', marker = 'o', linewidth =  1, label = f"Airfoil No. {case + 1}")
        
    axis.set_ylabel(r'$y$') 
    axis.set_xlabel(r'$x$') 
    axis.legend(loc='upper right')     
    return 

## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method   
def plot_airfoil_boundary_layers(airfoil_results):
    """Makes a plot of the boundary layer 
    
    Assumptions:
    None

    Source:   
                                                     
    Inputs:
       airfoil_results : data structure
                                                                           
    Outputs   
       Boundary layer plot on current axes
    
    Properties Used:
    N/A
    """          

    if airfoil_results.viscous_flag == False:
        return 
    fig    = plt.figure()
    axis   = fig.add_subplot(1,1,1)  
    xz     = np.concatenate((np.atleast_2d(airfoil_results.foil_pts),np.atleast_2d(airfoil_results.wake_pts)) ,axis = 1)
    N      = airfoil_results.num_foil_pts
    ds     = airfoil_results.delta # displacement thickness  
    cases  = len(airfoil_results.foil_pts)

    # create array of colors for difference reynolds numbers     
    blues  = cm.winter(np.linspace(0, 0.75,cases))
    reds   = cm.autumn(np.linspace(0, 0.75,cases))
    blacks = cm.copper(np.linspace(0, 0.75,cases)) 
    
    for case in range(cases):
        case_label = 'AoA: ' + str(round(airfoil_results.AoA[case][0]/Units.degrees, 2)) + ', Re: ' + str(airfoil_results.Re[case][0])
         
        rl   = 0.5*(1+(ds[case][0]-ds[case][-1])/ds[case][N+1])
        ru   = 1-rl 
        n    = airfoil_results.normals[case] # outward normals
        d_s  = np.tile(ds[case][:,None],(1,2))
        
        for iss in range(4): 
            if iss == 0:
                col = reds[case]
                xzd  = xz[case] + n*d_s # airfoil + delta*
                Is = airfoil_results.vsol_Is[case][iss]
                axis.plot(xzd[Is,0], xzd[Is,1], color = col, marker = 'o', linestyle = '-', linewidth = 2, label = case_label)
            if iss == 1:
                col = blues[case]
                xzd  = xz[case] + n*d_s # airfoil + delta*
                Is = airfoil_results.vsol_Is[case][iss]
                axis.plot(xzd[Is,0], xzd[Is,1], color = col, marker = 'o', linestyle = '-', linewidth = 2)
            if (iss==2):
                col = blacks[case]
                xzd = xz[case] + n*d_s *ru 
                Is = airfoil_results.vsol_Is[case][iss]
                axis.plot(xzd[Is,0], xzd[Is,1], color = col, marker = 'o', linestyle = '-', linewidth = 2,)
            if (iss==3):
                col = blacks[case]
                xzd = xz[case] - n*d_s*rl 
                iss = 2 
                Is  = airfoil_results.vsol_Is[case][iss]
                axis.plot(xzd[Is,0], xzd[Is,1], color = col, marker = 'o', linestyle = '-', linewidth = 2)
    axis.legend(loc='upper right') 
    return 

## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method
def plot_quantity(airfoil_results, q, qaxis, qname):
    """Plots a quantity q over lower/upper/wake surfaces
    
    Assumptions:
    None

    Source:   
                                                     
    Inputs:
       airfoil_results  : data class with valid airfoil/wake points
       q         : vector of values to plot, on all points (wake too if present)
       qaxis     : name of quantity, for axis labeling
       qname     : name of quantity, for title labeling
                                                                           
    Outputs:
       figure showing q versus x
    
    Properties Used:
    N/A
    """          

    fig   = plt.figure()
    axis  = fig.add_subplot(1,1,1)       
    xy    = np.concatenate((np.atleast_2d(airfoil_results.foil_pts),np.atleast_2d(airfoil_results.wake_pts)) ,axis = 1)  
    cases = len(airfoil_results.foil_pts)

    # create array of colors for difference reynolds numbers     
    blues  = cm.winter(np.linspace(0, 0.75,cases))
    reds   = cm.autumn(np.linspace(0, 0.75,cases))
    blacks = cm.copper(np.linspace(0, 0.75,cases))
    
    for case in range(cases): 
        case_label = 'AoA: ' + str(round(airfoil_results.AoA[case][0]/Units.degrees, 2)) + ', Re: ' + str(airfoil_results.Re[case][0])
        if airfoil_results.viscous_flag == False: 
            axis.plot(xy[case][:,0], q[case], color = blues[case] , marker = 'o', linewidth = 2)
        else: 
            for iss in range(3): 
                if iss == 0:
                    col = reds[case]
                if iss == 1:
                    col = blues[case]
                if iss == 2:
                    col = blacks[case]                
                Is = airfoil_results.vsol_Is[case][iss]
                axis.plot(xy[case][Is,0], q[case][Is], color = col , marker = 'o', linewidth = 2, label = case_label ) 
            
    axis.set_title(qname)            
    axis.set_ylabel(qaxis) 
    axis.set_xlabel(r'$x$') 
    return 

 
## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method
def plot_airfoil_distributions(airfoil_results):
    """Plots viscous distributions
    
    Assumptions:
    None

    Source:   
                                                     
    Inputs:
       airfoil_results  : data class with solution
                                                                           
    Outputs:
    figures of viscous distributions
    
    Properties Used:
    N/A
    """          
    plot_quantity(airfoil_results, airfoil_results.ue,  r'$u_e$'  , 'edge velocity')
    plot_quantity(airfoil_results, airfoil_results.ue_inv, r'$u_{e_inv}$'  ,'inviscid edge velocity')
    plot_quantity(airfoil_results, airfoil_results.sa,  r'$c_{\tau}^{1/2}$'  ,'amplification')
    plot_quantity(airfoil_results, airfoil_results.H,  r'$H$'  ,'kinematic shape parameter')
    plot_quantity(airfoil_results, airfoil_results.delta_star, r'$\delta*$' ,'displacement thickness')
    plot_quantity(airfoil_results, airfoil_results.theta, r'$\theta$' ,'momentum thickness')
    plot_quantity(airfoil_results, airfoil_results.cf, r'$c_f $'  ,   'skin friction coefficient')
    plot_quantity(airfoil_results, airfoil_results.Re_theta,  r'$Re_{\theta}$'  ,'theta Reynolds number') 
    return    

## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method
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
        a_data = compute_airfoil_properties(airfoil_path, airfoil_polars
 = airfoil_polar_paths,npoints = 200, use_pre_stall_data=False)
        CL_sur = a_data.lift_coefficient_surrogates
        CD_sur = a_data.drag_coefficient_surrogates
        
        alpha   = np.asarray(a_data.aoa_from_polar) * Units.degrees
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
            CL[i,:,:] = CL_sur[airfoil_path[i]]((Re[i,:,:],alpha[i,:,:]))
            CD[i,:,:] = CD_sur[airfoil_path[i]]((Re[i,:,:],alpha[i,:,:]))
            
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


def plot_airfoil_aerodynamic_coefficients(airfoil_path, airfoil_polar_paths, line_color = 'k-', use_surrogate = True, 
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

    col_raw = ['m-', 'b-', 'r-', 'g-', 'k-','m-','b-','r-','g-', 'k-']    
    if use_surrogate:
        col_sur = ['m--', 'b--', 'r--', 'g--', 'k--','m--','b--','r--', 'g--', 'k--']
        # Compute airfoil surrogates
        a_data = compute_airfoil_properties(airfoil_path, airfoil_polars = airfoil_polar_paths,npoints = 200, use_pre_stall_data=True)
        CL_sur = a_data.lift_coefficient_surrogates
        CD_sur = a_data.drag_coefficient_surrogates
        
        alpha   = np.asarray(a_data.aoa_from_polar) * Units.deg
        n_alpha = len(alpha.T)
        alpha   = np.reshape(alpha,(n_airfoils,1,n_alpha))
        alpha   = np.repeat(alpha, n_Re, axis=1)
        
        Re      = a_data.re_from_polar
        Re      = np.reshape(Re,(n_airfoils,n_Re,1))
        Re      = np.repeat(Re, n_alpha, axis=2)

        CL = np.zeros_like(Re)
        CD = np.zeros_like(Re)
    
        for i in range(n_airfoils):
            CL[i,:,:] = CL_sur[airfoil_path[i]]((Re[i,:,:],alpha[i,:,:]))
            CD[i,:,:] = CD_sur[airfoil_path[i]]((Re[i,:,:],alpha[i,:,:]))      
    
    # Get raw data polars
    airfoil_polar_data = import_airfoil_polars(airfoil_polar_paths)
    CL_raw      = airfoil_polar_data.lift_coefficients
    CD_raw      = airfoil_polar_data.drag_coefficients
    alpha_raw   = airfoil_polar_data.angle_of_attacks
    n_alpha_raw = len(alpha_raw)        
    
    # reshape into Re and n_airfoils
    alpha_raw  = np.tile(alpha_raw, (n_airfoils,n_Re,1))
    
    Re = airfoil_polar_data.reynolds_number
    Re = np.reshape(Re, (n_airfoils,n_Re,1))
    Re = np.repeat(Re, n_alpha_raw, axis=2)
    
    for i in range(n_airfoils):
        airfoil_name = os.path.basename(airfoil_path[i])
            
        # plot all Reynolds number polars for ith airfoil
        fig  = plt.figure(airfoil_name[:-4], figsize=(8,2*n_Re))
          
        for j in range(n_Re):
            ax1    = fig.add_subplot(n_Re,2,1+2*j)
            ax2    = fig.add_subplot(n_Re,2,2+2*j)  
            
            Re_val = str(round(Re[i,j,0])/1e6)+'e6'
            ax1.plot(alpha_raw[i,j,:], CL_raw[i,j,:], col_raw[j], label='Re='+Re_val)
            ax2.plot(alpha_raw[i,j,:], CD_raw[i,j,:], col_raw[j], label='Re='+Re_val)
            if use_surrogate:
                ax1.plot(alpha[i,j,:]/Units.deg, CL[i,j,:], col_sur[j])
                ax2.plot(alpha[i,j,:]/Units.deg, CD[i,j,:], col_sur[j])
             
            ax1.set_ylabel('$C_l$')   
            ax2.set_ylabel('$C_d$')  
            ax1.legend(loc='best')
            
        ax1.set_xlabel('AoA [deg]') 
        ax2.set_xlabel('AoA [deg]') 
        fig.tight_layout()
        
        if save_figure:
            plt.savefig(save_filename +'_' + str(i) + file_type)   
        if display_plot:
            plt.show()
    return
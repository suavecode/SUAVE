## @ingroup Plots
# Airfoil_Plots.py
#
# Created: Mar 2021, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import SUAVE
from SUAVE.Core import Units
import numpy as np 
import matplotlib.pyplot as plt  
import matplotlib.cm as cm

## @ingroup Plots
def plot_airfoil_properties(ap,arrow_color = 'r',plot_pressure_vectors = False ):  
    
    # determine dimension of angle of attack and reynolds number 
    nAoA = len(ap.AoA)
    nRe  = len(ap.Re)
    
    # create array of colors for difference reynolds numbers 
    colors  = cm.rainbow(np.linspace(0, 1,nAoA))
    markers = ['o','v','s','P','p','^','D','X','*']
    
    fig  = plt.figure('Airfoil',figsize=(8,6)) 
    axis1 = fig.add_subplot(1,1,1)     
    axis1.set_xlabel('x')
    axis1.set_ylabel('y')   
    axis1.set_ylim(-0.2, 0.2)  
    
    fig  = plt.figure('Figure_1',figsize=(12,8))
    axis2 = fig.add_subplot(2,3,1)      
    axis2.set_ylabel('$Ue/V_{inf}$')    
     
    axis3 = fig.add_subplot(2,3,2)      
    axis3.set_ylabel('$dV_e/dx$')   
    axis3.set_ylim(-5, 5)  
      
    axis4 = fig.add_subplot(2,3,3)   
    axis4.set_ylabel(r'$\theta$')  
     
    axis5 = fig.add_subplot(2,3,4)     
    axis5.set_xlabel('x')
    axis5.set_ylabel(r'$\delta$*')  
     
    axis6 = fig.add_subplot(2,3,5)     
    axis6.set_xlabel('x')
    axis6.set_ylabel(r'$\delta$')  
     
    axis7 = fig.add_subplot(2,3,6)    
    axis7.set_xlabel('x')
    axis7.set_ylabel('$C_p$')  
    axis7.set_ylim(1,-1.5)
    
    fig  = plt.figure('Figure_2',figsize=(8,6))
    axis8 = fig.add_subplot(2,2,1)      
    axis8.set_ylabel('Transition Top Surface')  
    axis8.set_ylim(0, 5000)  
     
    axis9 = fig.add_subplot(2,2,2)      
    axis9.set_ylabel('Transition Bottom Surface')     
    axis9.set_ylim(0, 2000)  
    
    axis10 = fig.add_subplot(2,2,3)      
    axis10.set_ylabel('H')  
    
    axis11 = fig.add_subplot(2,2,4)      
    axis11.set_ylabel('$C_f$')      
        
    
    #fig  = plt.figure('Airfoil',figsize=(8,6)) 
    #axis8 = fig.add_subplot(1,1,1)      
    #axis8.set_xlabel('x')
    #axis8.set_ylabel('y')   
    #axis8.set_ylim(-0.2, 0.2) 
    #
    #fig  = plt.figure('Lift Coefficient')
    #axis9 = fig.add_subplot(1,1,1)     
    #axis9.set_xlabel('AoA')
    #axis9.set_ylabel(r'Lift Coefficient, Cl') 
    #
    #fig  = plt.figure('Drag Coefficient')
    #axis10 = fig.add_subplot(1,1,1)     
    #axis10.set_xlabel('AoA')
    #axis10.set_ylabel(r'Drag Coefficient, Cd') 
    #
    #fig  = plt.figure('Moment Coefficient')
    #axis11 = fig.add_subplot(1,1,1)     
    #axis11.set_xlabel('AoA')
    #axis11.set_ylabel(r'Moment Coefficient, Cm ')    
     
    mid = int(len(ap.x)/2)
    
    for i in range(nRe): 
        for j in range(nAoA):
        
            tag = 'AoA: ' + str(round(ap.AoA[j][0]/Units.degrees,2)) + 'deg, Re: ' + str(round(ap.Re[i][0]/1000000,2)) + 'E6'
            
            axis1.plot(ap.x[:,j,i], ap.y[:,j,i],'k-') 
            axis1.plot(ap.x_bl[:,j,i],ap.y_bl[:,j,i],color = colors[j], linestyle = '-' ,marker =  markers[i] , label = tag)            
             
            axis2.plot(ap.x[:mid,j,i], abs(ap.Ue_Vinf)[:mid,j,i],color = colors[j], linestyle = '-' ,marker =  markers[i] , label= tag )  
            axis2.plot(ap.x[mid:,j,i], abs(ap.Ue_Vinf)[mid:,j,i],color = colors[j], linestyle = '--' ,marker =  markers[i], label= tag )   
           
            axis3.plot(ap.x[:mid,j,i], abs(ap.dVe)[:mid,j,i],color = colors[j], linestyle = '-' ,marker =  markers[i] , label= tag )
            axis3.plot(ap.x[mid:,j,i], abs(ap.dVe)[mid:,j,i],color = colors[j], linestyle = '--' ,marker =  markers[i], label= tag )
             
            axis4.plot(ap.x[:mid,j,i], ap.theta[:mid,j,i],color = colors[j], linestyle = '-' ,marker =  markers[i] , label =  tag)  
            axis4.plot(ap.x[mid:,j,i], ap.theta[mid:,j,i],color = colors[j], linestyle = '--' ,marker =  markers[i], label =  tag)   
            
            axis5.plot(ap.x[:mid,j,i],ap.delta_star[:mid,j,i],color = colors[j], linestyle = '-' ,marker =  markers[i] , label =  tag) 
            axis5.plot(ap.x[mid:,j,i],ap.delta_star[mid:,j,i],color = colors[j], linestyle = '--' ,marker =  markers[i], label =  tag) 
        
            axis6.plot(ap.x[:mid,j,i],ap.delta[:mid,j,i],color = colors[j], linestyle = '-' ,marker =  markers[i] , label =  tag)    
            axis6.plot(ap.x[mid:,j,i],ap.delta[mid:,j,i],color = colors[j], linestyle = '--' ,marker =  markers[i], label =  tag)      
                     
            axis7.plot(ap.x[:mid,j,i], ap.Cp[:mid,j,i] ,color = colors[j], linestyle = '-' ,marker =  markers[i] ,  label = tag) 
            axis7.plot(ap.x[mid:,j,i], ap.Cp[ mid:,j,i],color = colors[j], linestyle = '--' ,marker =  markers[i], label = tag)             
            
            trans_bot = 1.174*(1 + 224000/ap.Re_x[1:mid,j,i])*(ap.Re_x[1:mid,j,i]**0.46) 
            axis8.plot(ap.x[:mid,j,i],ap.Re_theta[:mid,j,i] ,color = colors[j], linestyle = '-' ,marker =  markers[i] , label =  r'$Re_\theta$')    
            axis8.plot(ap.x[1:mid,j,i], trans_bot           ,color = colors[j], linestyle = '--' ,marker =  markers[i],  label = r'1.174(1+22400/Re_x)Re_x^{0.46}$')  
            
            trans_top = 1.174*(1 + 224000/ap.Re_x[(mid+1):,j,i])*(ap.Re_x[(mid+1):,j,i]**0.46) 
            axis9.plot(ap.x[mid:,j,i],ap.Re_theta[mid:,j,i],color = colors[j], linestyle = '-' ,marker =  markers[i] ,label =  r'$Re_\theta$')    
            axis9.plot(ap.x[(mid+1):,j,i], trans_top       ,color = colors[j], linestyle = '--' ,marker =  markers[i], label = r'1.174(1+22400/Re_x)Re_x^{0.46}$')  
            
            axis10.plot(ap.x[:mid,j,i], ap.H[:mid,j,i],color = colors[j], linestyle = '-' ,marker =  markers[i] , label= tag )  
            axis10.plot(ap.x[mid:,j,i], ap.H[mid:,j,i],color = colors[j], linestyle = '--' ,marker =  markers[i], label= tag )   
            
            axis11.plot(ap.x[:mid,j,i], ap.Cf[:mid,j,i],color = colors[j], linestyle = '-' ,marker =  markers[i] , label= tag )  
            axis11.plot(ap.x[mid:,j,i], ap.Cf[mid:,j,i],color = colors[j], linestyle = '--' ,marker =  markers[i], label= tag )                    
            
    
            #axis8.plot(ap.x_bl[:,j,i], ap.y_bl[:,j,i],color = colors[j], linestyle = '--')      
            
            #if plot_pressure_vectors: 
                #label =  '_AoA_' + str(round(ap.AoA[j][0]/Units.degrees,2)) + '_deg_Re_' + str(round(ap.Re[i][0]/1000000,2)) + 'E6'
                #fig   = plt.figure('Airfoil_Pressure_Normals' + label )
                #axis7 = fig.add_subplot(1,1,1)      
                #axis7.plot(ap.x[:,j,i], ap.y[:,j,i],line_style) 
                
                #for k in range(len(ap.x)):
                    #dx_val = ap.normals[k,0,j,i]*abs(ap.Cp[k,j,i])*0.1
                    #dy_val = ap.normals[k,1,j,i]*abs(ap.Cp[k,j,i])*0.1
                    #if ap.Cp[k,j,i] < 0:
                        #plt.arrow(x= ap.x[k,j,i], y=ap.y[k,j,i] , dx= dx_val , dy = dy_val , 
                                  #fc=arrow_color, ec=arrow_color,head_width=0.005, head_length=0.01 )   
                    #else:
                        #plt.arrow(x= ap.x[k,j,i]+dx_val , y= ap.y[k,j,i]+dy_val , dx= -dx_val , dy = -dy_val , 
                                  #fc=arrow_color, ec=arrow_color,head_width=0.005, head_length=0.01 )   
            
        #Re_tag  = 'Re: ' + str(round(ap.Re[i][0]/1000000,2)) + 'E6'
        
        ## Lift Coefficient
        #axis9.plot(ap.AoA[:,0]/Units.degrees,ap.Cl[:,i],color = colors[i], linestyle = '-' ,marker =  markers[i], label= Re_tag )
        
        ## Drag Coefficient
        #axis10.plot(ap.AoA[:,0]/Units.degrees,ap.Cd[:,i],color = colors[i], linestyle = '-',marker =  markers[i], label =  Re_tag)  
        
        ## Moment Coefficient
        #axis11.plot(ap.AoA[:,0]/Units.degrees, ap.Cm[:,i],color = colors[i], linestyle = '-',marker =  markers[i], label =  Re_tag)  
        
    ## append                      
    #axis2.legend(loc='upper right', ncol=2)   
    #axis3.legend(loc='upper right', ncol=2)  
    #axis4.legend(loc='upper left', ncol=2)   
    #axis5.legend(loc='upper left', ncol=2)  
    #axis6.legend(loc='upper left', ncol=2)
    #if plot_pressure_vectors: 
        #axis7.legend(loc='upper left')      
    #axis9.legend(loc='upper left')   
    #axis10.legend(loc='upper left')      
    #axis11.legend(loc='upper left')  
    
    plt.tight_layout()
    return  

## @ingroup Plots
def plot_airfoil_properties_old(ap,line_style= 'k-',arrow_color = 'r',plot_pressure_vectors = False ):  
    
    # determine dimension of angle of attack and reynolds number 
    nAoA = len(ap.AoA)
    nRe  = len(ap.Re)
    
    # create array of colors for difference reynolds numbers 
    colors  = cm.rainbow(np.linspace(0, 1,nAoA))
    markers = ['o','v','s','P','p','^','D','X','*']
    
    fig  = plt.figure('Airfoil',figsize=(8,6)) 
    axis1 = fig.add_subplot(1,1,1)     
    axis1.plot(ap.x, ap.y,line_style) 
    axis1.set_xlabel('x')
    axis1.set_ylabel('y')   
    axis1.set_ylim(-0.2, 0.2) 
    
    fig  = plt.figure('Pressure_Coefficient',figsize=(8,6))
    axis2 = fig.add_subplot(1,1,1)    
    axis2.set_xlabel('x')
    axis2.set_ylabel('$C_p$')  
    axis2.set_ylim(2, -10)     # invery CP plot to convention               
    
    fig  = plt.figure('Boundary_Layer_Edge_Velocity',figsize=(8,6))
    axis3 = fig.add_subplot(1,1,1)     
    axis3.set_xlabel('x')
    axis3.set_ylabel('$Ue/V_{inf}$')       
    
    fig  = plt.figure('Momentum_Thickness',figsize=(8,6))
    axis4 = fig.add_subplot(1,1,1)    
    axis4.set_xlabel('x')
    axis4.set_ylabel(r'Momentum Thickness, $\theta$') 
    axis4.set_ylim(-0.005, 0.03)  
    
    fig  = plt.figure('Displacement Thickness',figsize=(8,6))
    axis5 = fig.add_subplot(1,1,1)     
    axis5.set_xlabel('x')
    axis5.set_ylabel(r'Displacement Thickness, $\delta$*') 
    axis5.set_ylim(-0.005, 0.10)  
    
    fig  = plt.figure('Boundary Layer Thickness',figsize=(8,6))
    axis6 = fig.add_subplot(1,1,1)     
    axis6.set_xlabel('x')
    axis6.set_ylabel(r'Boundary Layer Thickness, $\delta$') 
    axis6.set_ylim(-0.005, 0.15)  
    
    fig  = plt.figure('Airfoil',figsize=(8,6)) 
    axis8 = fig.add_subplot(1,1,1)     
    axis8.plot(ap.x, ap.y,line_style) 
    axis8.set_xlabel('x')
    axis8.set_ylabel('y')   
    axis8.set_ylim(-0.2, 0.2) 
    
    fig  = plt.figure('Lift Coefficient')
    axis9 = fig.add_subplot(1,1,1)     
    axis9.set_xlabel('AoA')
    axis9.set_ylabel(r'Lift Coefficient, Cl') 
    
    fig  = plt.figure('Drag Coefficient')
    axis10 = fig.add_subplot(1,1,1)     
    axis10.set_xlabel('AoA')
    axis10.set_ylabel(r'Drag Coefficient, Cd') 
    
    fig  = plt.figure('Moment Coefficient')
    axis11 = fig.add_subplot(1,1,1)     
    axis11.set_xlabel('AoA')
    axis11.set_ylabel(r'Moment Coefficient, Cm ')    
    
    for i in range(nRe): 
        for j in range(nAoA):
            tag = 'AoA: ' + str(round(ap.AoA[j][0]/Units.degrees,2)) + 'deg, Re: ' + str(round(ap.Re[i][0]/1000000,2)) + 'E6'
            # Pressure Coefficient 
            axis2.plot(ap.x, ap.Cp[:,j,i], line_style  , label = tag) 
            axis2.plot(ap.x_bl, ap.Cp[:,j,i], 'ro' , label = tag)
            
            # Boundary Layer Edge Velocity 
            axis3.plot(ap.x, abs(ap.Ue_Vinf)[:,j,i], line_style , label= tag )
            
            # Momentum Thickness  
            axis4.plot(ap.x, ap.theta[:,j,i], line_style , label =  tag)  
            
            # Displacement thickenss 
            axis5.plot(ap.x,ap.delta_star[:,j,i], line_style , label =  tag)  
    
            # Boundary Layer thickenss 
            axis6.plot(ap.x,ap.delta[:,j,i], line_style , label =  tag)      
            
            if plot_pressure_vectors: 
                label =  '_AoA_' + str(round(ap.AoA[j][0]/Units.degrees,2)) + '_deg_Re_' + str(round(ap.Re[i][0]/1000000,2)) + 'E6'
                fig   = plt.figure('Airfoil_Pressure_Normals' + label )
                axis7 = fig.add_subplot(1,1,1)     
                axis7.plot(ap.x, ap.y,line_style) 
                for k in range(len(ap.x)):
                    dx_val = ap.normals[k][0]*abs(ap.Cp[k,j,i])*0.1
                    dy_val = ap.normals[k][1]*abs(ap.Cp[k,j,i])*0.1
                    if ap.Cp[k,j,i] < 0:
                        plt.arrow(x= ap.x[k], y=ap.y[k] , dx= dx_val , dy = dy_val , 
                                  fc=arrow_color, ec=arrow_color,head_width=0.005, head_length=0.01 )   
                    else:
                        plt.arrow(x= ap.x[k]+dx_val , y= ap.y[k]+dy_val , dx= -dx_val , dy = -dy_val , 
                                  fc=arrow_color, ec=arrow_color,head_width=0.005, head_length=0.01 )   
                    
            axis8.plot(ap.x_bl[:,j,i], ap.y_bl[:,j,i],color = colors[j], linestyle = '--')      
            
        Re_tag  = 'Re: ' + str(round(ap.Re[i][0]/1000000,2)) + 'E6'
        
        # Lift Coefficient
        axis9.plot(ap.AoA[:,0]/Units.degrees,ap.Cl[:,i],color = colors[i], linestyle = '-' ,marker =  markers[i], label= Re_tag )
        
        # Drag Coefficient
        axis10.plot(ap.AoA[:,0]/Units.degrees,ap.Cd[:,i],color = colors[i], linestyle = '-',marker =  markers[i], label =  Re_tag)  
        
        # Moment Coefficient
        axis11.plot(ap.AoA[:,0]/Units.degrees, ap.Cm[:,i],color = colors[i], linestyle = '-',marker =  markers[i], label =  Re_tag)  
        
    # append                      
    axis2.legend(loc='upper right', ncol=2)   
    axis3.legend(loc='upper right', ncol=2)  
    axis4.legend(loc='upper left', ncol=2)   
    axis5.legend(loc='upper left', ncol=2)  
    axis6.legend(loc='upper left', ncol=2)
    if plot_pressure_vectors: 
        axis7.legend(loc='upper left')      
    axis9.legend(loc='upper left')   
    axis10.legend(loc='upper left')      
    axis11.legend(loc='upper left')  
    
    return 
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
    
    fig1  = plt.figure('Airfoil',figsize=(8,6)) 
    axis1 = fig1.add_subplot(1,1,1)     
    axis1.set_xlabel('x')
    axis1.set_ylabel('y')   
    axis1.set_ylim(-0.2, 0.2)  
    
    fig2  = plt.figure('Figure_1',figsize=(12,8))
    axis2 = fig2.add_subplot(2,3,1)      
    axis2.set_ylabel('$Ue/V_{inf}$')    
     
    axis3 = fig2.add_subplot(2,3,2)      
    axis3.set_ylabel('$dV_e/dx$')   
    axis3.set_ylim(-1, 10)  
      
    axis4 = fig2.add_subplot(2,3,3)   
    axis4.set_ylabel(r'$\theta$')  
     
    axis5 = fig2.add_subplot(2,3,4)     
    axis5.set_xlabel('x')
    axis5.set_ylabel(r'$\delta$*')  
     
    axis6 = fig2.add_subplot(2,3,5)     
    axis6.set_xlabel('x')
    axis6.set_ylabel(r'$\delta$')  
     
    axis7 = fig2.add_subplot(2,3,6)    
    axis7.set_xlabel('x')
    axis7.set_ylabel('$C_p$')  
    axis7.set_ylim(1.2,-7)
    
    fig3  = plt.figure('Figure_2',figsize=(8,6))
    axis8 = fig3.add_subplot(2,2,1)      
    axis8.set_ylabel('Transition Top Surface')  
    axis8.set_ylim(0, 5000)  
     
    axis9 = fig3.add_subplot(2,2,2)      
    axis9.set_ylabel('Transition Bottom Surface')     
    axis9.set_ylim(0, 2000)  
    
    axis10 = fig3.add_subplot(2,2,3)      
    axis10.set_ylabel('H')  
    
    axis11 = fig3.add_subplot(2,2,4)      
    axis11.set_ylabel('$C_f$')      
    
    fig4   = plt.figure('Figure_3',figsize=(12,5))
    axis12 = fig4.add_subplot(1,3,1)     
    axis12.set_title('Aero Coefficients')
    axis12.set_xlabel('AoA')
    axis12.set_ylabel(r'Lift Coefficient, Cl') 
    axis12.set_ylim(-1,2)  
    
    axis13 = fig4.add_subplot(1,3,2)    
    axis13.set_title('Drag Coefficient') 
    axis13.set_xlabel('AoA')
    axis13.set_ylabel(r'Drag Coefficient, Cd') 
    axis13.set_ylim(0,0.25)  
    
    axis14 = fig4.add_subplot(1,3,3)   
    axis14.set_title('Moment Coefficient')  
    axis14.set_xlabel('AoA')
    axis14.set_ylabel(r'Moment Coefficient, Cm ')    
    axis14.set_ylim(-0.1,0.1)  
     
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
            
            axis5.plot(ap.x[:mid,j,i],ap.delta_star[:mid,j,i],color = colors[j], linestyle = '-' ,marker =  markers[j%9] ) 
            axis5.plot(ap.x[mid:,j,i],ap.delta_star[mid:,j,i],color = colors[j], linestyle = '--' ,marker =  markers[j%9]) 
        
            axis6.plot(ap.x[:mid,j,i],ap.delta[:mid,j,i],color = colors[j], linestyle = '-' ,marker =  markers[j%9] )    
            axis6.plot(ap.x[mid:,j,i],ap.delta[mid:,j,i],color = colors[j], linestyle = '--' ,marker =  markers[j%9])      
                     
            axis7.plot(ap.x[:mid,j,i], ap.Cp[:mid,j,i] ,color = colors[j], linestyle = '-' ,marker =  markers[j%9] ) 
            axis7.plot(ap.x[mid:,j,i], ap.Cp[ mid:,j,i],color = colors[j], linestyle = '--' ,marker =  markers[j%9])             
            
            trans_bot = 1.174*(1 + 224000/ap.Re_x[1:mid,j,i])*(ap.Re_x[1:mid,j,i]**0.46) 
            axis8.plot(ap.x[:mid,j,i],ap.Re_theta[:mid,j,i] ,color = colors[j], linestyle = '-' ,marker =  markers[j%9] , label =  r'$Re_\theta$')    
            axis8.plot(ap.x[1:mid,j,i], trans_bot           ,color = colors[j], linestyle = '--' ,marker =  markers[j%9],  label = r'1.174(1+22400/Re_x)Re_x^{0.46}$')  
            
            trans_top = 1.174*(1 + 224000/ap.Re_x[(mid+1):,j,i])*(ap.Re_x[(mid+1):,j,i]**0.46) 
            axis9.plot(ap.x[mid:,j,i],ap.Re_theta[mid:,j,i],color = colors[j], linestyle = '-' ,marker =  markers[j%9] ,label =  r'$Re_\theta$')    
            axis9.plot(ap.x[(mid+1):,j,i], trans_top       ,color = colors[j], linestyle = '--' ,marker =  markers[j%9], label = r'1.174(1+22400/Re_x)Re_x^{0.46}$')  
            
            axis10.plot(ap.x[:mid,j,i], ap.H[:mid,j,i],color = colors[j], linestyle = '-' ,marker =  markers[j%9]  )  
            axis10.plot(ap.x[mid:,j,i], ap.H[mid:,j,i],color = colors[j], linestyle = '--' ,marker =  markers[j%9] )   
            
            axis11.plot(ap.x[:mid,j,i], ap.Cf[:mid,j,i],color = colors[j], linestyle = '-' ,marker =  markers[j%9] )  
            axis11.plot(ap.x[mid:,j,i], ap.Cf[mid:,j,i],color = colors[j], linestyle = '--' ,marker =  markers[j%9] )                    
            
            plt.tight_layout()
     
            if plot_pressure_vectors: 
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
                          
                                  
        Re_tag  = 'Re: ' + str(round(ap.Re[i][0]/1000000,2)) + 'E6'
        
        # Lift Coefficient
        axis12.plot(ap.AoA[:,0]/Units.degrees,ap.Cl[:,i],color = colors[i], linestyle = '-' ,marker =  markers[i], label= Re_tag )
        
        # Drag Coefficient
        axis13.plot(ap.AoA[:,0]/Units.degrees,ap.Cd[:,i],color = colors[i], linestyle = '-',marker =  markers[i], label =  Re_tag)  
        
        # Moment Coefficient
        axis14.plot(ap.AoA[:,0]/Units.degrees, ap.Cm[:,i],color = colors[i], linestyle = '-',marker =  markers[i], label =  Re_tag)     
        plt.tight_layout() 
    
    # add legends for plotting
    plt.tight_layout()
    lines1, labels1 = fig2.axes[0].get_legend_handles_labels()
    fig2.legend(lines1, labels1, loc='upper center', ncol=3)
     
    if plot_pressure_vectors: 
        axis7.legend(loc='upper left')      
        
    axis12.legend(loc='upper left')   
    axis13.legend(loc='upper left')      
    axis14.legend(loc='upper left')  
    
    return   
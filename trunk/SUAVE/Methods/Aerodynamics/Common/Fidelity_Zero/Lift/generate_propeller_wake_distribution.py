## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
#  generate_propeller_wake_distribution.py
# 
# Created:  Jul 2020, M. Clarke 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# package imports
import numpy as np
from SUAVE.Core import Data 
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.compute_wake_contraction_matrix import compute_wake_contraction_matrix
## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift  

def generate_propeller_wake_distribution(prop,m,VD):
    # to put in settings 
    time = 0.1
    
    # Unpack unknowns  
    R            = prop.outputs.propeller_radius
    r            = prop.outputs.blade_radial_distribution 
    c            = prop.outputs.blade_chord_distribution 
    N            = len(r)
    omega        = prop.outputs.omega                        
    vt           = prop.outputs.tangential_induced_velocity_2d         
    va           = prop.outputs.axial_induced_velocity_2d  
    V_inf        = prop.outputs.velocity
    MCA          = prop.mid_chord_aligment 
    B            = prop.outputs.num_blades  
    gamma        = prop.outputs.blade_Gamma_2d
    blade_angles = np.linspace(0,2*np.pi,B+1)[:-1]        
    dt           = 0.0025 # (2*np.pi/N)/omega[0]
    nts          = int(time/dt)
    ts           = np.linspace(0,time,nts)
    num_prop     = len(prop.origin) 
    
    
    # define points ( control point, time step , blade number , location on blade )
    # compute lambda and mu 
    mean_induced_velocity  = np.mean( np.mean(va,axis = 1),axis = 1)  
    
    lambda_tot   =  (V_inf[:,0]  + mean_induced_velocity)/(omega*R)       # inflow advance ratio (page 30 Leishman)
    mu_prop      =  V_inf[:,2] /(omega*R)                              # rotor advance ratio  (page 30 Leishman) 
    V_prop       = np.sqrt((V_inf[:,0]  + mean_induced_velocity)**2 + (V_inf[:,2])**2)
    
    # wake skew angle 
    wake_skew_angle = np.arctan(mu_prop/lambda_tot)
    
    # reshape gamma to fund the average between stations 
    gamma_new = np.zeros((m,N,(N-1)))
    gamma_new = (gamma[:,:,:-1] + gamma[:,:,1:])*0.5
    
    # define empty arrays 
    WD      = Data()
    n = N-1
    WD_XA1   = np.zeros((m,num_prop,nts-1,B,n))
    WD_YA1   = np.zeros((m,num_prop,nts-1,B,n))
    WD_ZA1   = np.zeros((m,num_prop,nts-1,B,n))
    WD_XA2   = np.zeros((m,num_prop,nts-1,B,n))
    WD_YA2   = np.zeros((m,num_prop,nts-1,B,n))
    WD_ZA2   = np.zeros((m,num_prop,nts-1,B,n))    
    WD_XB1   = np.zeros((m,num_prop,nts-1,B,n))
    WD_YB1   = np.zeros((m,num_prop,nts-1,B,n))
    WD_ZB1   = np.zeros((m,num_prop,nts-1,B,n))
    WD_XB2   = np.zeros((m,num_prop,nts-1,B,n)) 
    WD_YB2   = np.zeros((m,num_prop,nts-1,B,n)) 
    WD_ZB2   = np.zeros((m,num_prop,nts-1,B,n))   
    WD_GAMMA = np.zeros((m,num_prop,nts-1,B,n))   
    
    WD.XA1   = np.zeros((m,num_prop*(nts-1)*B*n))
    WD.YA1   = np.zeros((m,num_prop*(nts-1)*B*n))
    WD.ZA1   = np.zeros((m,num_prop*(nts-1)*B*n))
    WD.XA2   = np.zeros((m,num_prop*(nts-1)*B*n))
    WD.YA2   = np.zeros((m,num_prop*(nts-1)*B*n))
    WD.ZA2   = np.zeros((m,num_prop*(nts-1)*B*n))   
    WD.XB1   = np.zeros((m,num_prop*(nts-1)*B*n))
    WD.YB1   = np.zeros((m,num_prop*(nts-1)*B*n))
    WD.ZB1   = np.zeros((m,num_prop*(nts-1)*B*n))
    WD.XB2   = np.zeros((m,num_prop*(nts-1)*B*n))
    WD.YB2   = np.zeros((m,num_prop*(nts-1)*B*n))
    WD.ZB2   = np.zeros((m,num_prop*(nts-1)*B*n))  
    
    
    VD.Wake = Data()
    VD.Wake.XA1   = np.zeros((num_prop,(nts-1),B,n))
    VD.Wake.YA1   = np.zeros((num_prop,(nts-1),B,n))
    VD.Wake.ZA1   = np.zeros((num_prop,(nts-1),B,n))
    VD.Wake.XA2   = np.zeros((num_prop,(nts-1),B,n))
    VD.Wake.YA2   = np.zeros((num_prop,(nts-1),B,n))
    VD.Wake.ZA2   = np.zeros((num_prop,(nts-1),B,n))   
    VD.Wake.XB1   = np.zeros((num_prop,(nts-1),B,n))
    VD.Wake.YB1   = np.zeros((num_prop,(nts-1),B,n))
    VD.Wake.ZB1   = np.zeros((num_prop,(nts-1),B,n))
    VD.Wake.XB2   = np.zeros((num_prop,(nts-1),B,n))
    VD.Wake.YB2   = np.zeros((num_prop,(nts-1),B,n))
    VD.Wake.ZB2   = np.zeros((num_prop,(nts-1),B,n))    
     
    for i in range(num_prop): 
        Gamma  = np.zeros((m,nts-1,B,n))   
        num = int(N/B) -1 
        for t_idx in range(nts-1): 
            for B_idx in range(B): 
                B_loc = (B_idx*num + t_idx)%N 
                Gamma[:,t_idx,B_idx,:] = gamma_new[:,B_loc,:]  
        
        #( control point, time step , blade number , location on blade )
        sx_inf0 = np.multiply(np.atleast_2d(V_prop*np.cos(wake_skew_angle)).T,np.atleast_2d(ts))
        sx_inf  = np.repeat(np.repeat(sx_inf0[:, :,  np.newaxis], N, axis = 2)[:,  : ,np.newaxis,  :], B, axis = 2)  
        
        sy_inf0 = np.multiply(np.atleast_2d(V_inf[:,1]).T,np.atleast_2d(ts)) # = zero since no crosswind
        sy_inf  = np.repeat(np.repeat(sy_inf0[:, :,  np.newaxis], N, axis = 2)[:,  : ,np.newaxis,  :], B, axis = 2)   
        
        sz_inf0 = np.multiply(np.atleast_2d(V_prop*np.sin(wake_skew_angle)).T,np.atleast_2d(ts))
        sz_inf  = np.repeat(np.repeat(sz_inf0[:, :,  np.newaxis], N, axis = 2)[:,  : ,np.newaxis,  :], B, axis = 2)           
        
        omega_t = np.repeat(np.repeat(np.multiply(np.atleast_2d(omega),np.atleast_2d(ts))[:, :,  np.newaxis],B, axis = 2)[:, :,:, np.newaxis],N, axis = 3) 
        ba      = np.repeat(np.repeat(np.tile(np.atleast_2d(blade_angles),(m,1))[:,  np.newaxis, :],nts, axis = 1) [:, :,:, np.newaxis],N, axis = 3) 
        
        azi_y   = np.sin(ba + omega_t)  
        azi_z   = np.cos(ba + omega_t)  
        
        #adjust for clockwise/counter clockwise rotation
        if (prop.rotation != None) and (prop.rotation[i] == -1):        
            azi_y   = -azi_y 
         
        x0_pts = np.tile(np.atleast_2d(MCA-c/4),(B,1)) 
        x_pts  = np.repeat(np.repeat(x0_pts[np.newaxis,:,  :], nts, axis=0)[ np.newaxis, : ,:, :,], m, axis=0) 
        X_pts  = prop.origin[i][0] +  x_pts + sx_inf   

        # compute wake contraction CURRENTLY INCORRECT
        wake_contraction , r_div_r_prime , Kd = compute_wake_contraction_matrix(i,ts,prop,N,m,nts,X_pts)          
        
        y0_pts = np.tile(np.atleast_2d(r),(B,1))
        y_pts  = np.repeat(np.repeat(y0_pts[np.newaxis,:,  :], nts, axis=0)[ np.newaxis, : ,:, :,], m, axis=0) 
        Y_pts  = prop.origin[i][1] + (y_pts*wake_contraction)*azi_y  + sy_inf    
    
        z0_pts = np.tile(np.atleast_2d(r),(B,1))
        z_pts  = np.repeat(np.repeat(z0_pts[np.newaxis,:,  :], nts, axis=0)[ np.newaxis, : ,:, :,], m, axis=0)
        Z_pts  = prop.origin[i][2] + (z_pts*wake_contraction)*azi_z + sz_inf     
        
        # Store points  
        # ( control point,  prop ,  time step , blade number , location on blade )
        WD_XA1[:,i,:,:,:] =  X_pts[: , :-1 , : , :-1 ]
        WD_YA1[:,i,:,:,:] =  Y_pts[: , :-1 , : , :-1 ]
        WD_ZA1[:,i,:,:,:] =  Z_pts[: , :-1 , : , :-1 ]
        WD_XA2[:,i,:,:,:] =  X_pts[: ,  1: , : , :-1 ]
        WD_YA2[:,i,:,:,:] =  Y_pts[: ,  1: , : , :-1 ]
        WD_ZA2[:,i,:,:,:] =  Z_pts[: ,  1: , : , :-1 ]
        WD_XB1[:,i,:,:,:] =  X_pts[: , :-1 , : , 1:  ]
        WD_YB1[:,i,:,:,:] =  Y_pts[: , :-1 , : , 1:  ]
        WD_ZB1[:,i,:,:,:] =  Z_pts[: , :-1 , : , 1:  ]
        WD_XB2[:,i,:,:,:] =  X_pts[: ,  1: , : , 1:  ]
        WD_YB2[:,i,:,:,:] =  Y_pts[: ,  1: , : , 1:  ]
        WD_ZB2[:,i,:,:,:] =  Z_pts[: ,  1: , : , 1:  ] 
        WD_GAMMA[:,i,:,:,:] = Gamma 
        
        # store points for plotting 
        VD.Wake.XA1[i,:,:,:] =  X_pts[0 , :-1 , : , :-1 ]
        VD.Wake.YA1[i,:,:,:] =  Y_pts[0 , :-1 , : , :-1 ]
        VD.Wake.ZA1[i,:,:,:] =  Z_pts[0 , :-1 , : , :-1 ]
        VD.Wake.XA2[i,:,:,:] =  X_pts[0 ,  1: , : , :-1 ]
        VD.Wake.YA2[i,:,:,:] =  Y_pts[0 ,  1: , : , :-1 ]
        VD.Wake.ZA2[i,:,:,:] =  Z_pts[0 ,  1: , : , :-1 ]
        VD.Wake.XB1[i,:,:,:] =  X_pts[0 , :-1 , : , 1:  ]
        VD.Wake.YB1[i,:,:,:] =  Y_pts[0 , :-1 , : , 1:  ]
        VD.Wake.ZB1[i,:,:,:] =  Z_pts[0 , :-1 , : , 1:  ]
        VD.Wake.XB2[i,:,:,:] =  X_pts[0 ,  1: , : , 1:  ]
        VD.Wake.YB2[i,:,:,:] =  Y_pts[0 ,  1: , : , 1:  ]
        VD.Wake.ZB2[i,:,:,:] =  Z_pts[0 ,  1: , : , 1:  ]  
          
    
    # Compress Data into 1D Arrays  
    WD.XA1  =  np.reshape(np.reshape(np.reshape(WD_XA1,(m,num_prop,(nts-1),B*n)),(m,num_prop,(nts-1)*B*n)),(m,num_prop*(nts-1)*B*n))
    WD.YA1  =  np.reshape(np.reshape(np.reshape(WD_YA1,(m,num_prop,(nts-1),B*n)),(m,num_prop,(nts-1)*B*n)),(m,num_prop*(nts-1)*B*n))
    WD.ZA1  =  np.reshape(np.reshape(np.reshape(WD_ZA1,(m,num_prop,(nts-1),B*n)),(m,num_prop,(nts-1)*B*n)),(m,num_prop*(nts-1)*B*n))
    WD.XA2  =  np.reshape(np.reshape(np.reshape(WD_XA2,(m,num_prop,(nts-1),B*n)),(m,num_prop,(nts-1)*B*n)),(m,num_prop*(nts-1)*B*n))
    WD.YA2  =  np.reshape(np.reshape(np.reshape(WD_YA2,(m,num_prop,(nts-1),B*n)),(m,num_prop,(nts-1)*B*n)),(m,num_prop*(nts-1)*B*n))
    WD.ZA2  =  np.reshape(np.reshape(np.reshape(WD_ZA2,(m,num_prop,(nts-1),B*n)),(m,num_prop,(nts-1)*B*n)),(m,num_prop*(nts-1)*B*n))
    WD.XB1  =  np.reshape(np.reshape(np.reshape(WD_XB1,(m,num_prop,(nts-1),B*n)),(m,num_prop,(nts-1)*B*n)),(m,num_prop*(nts-1)*B*n))
    WD.YB1  =  np.reshape(np.reshape(np.reshape(WD_YB1,(m,num_prop,(nts-1),B*n)),(m,num_prop,(nts-1)*B*n)),(m,num_prop*(nts-1)*B*n))
    WD.ZB1  =  np.reshape(np.reshape(np.reshape(WD_ZB1,(m,num_prop,(nts-1),B*n)),(m,num_prop,(nts-1)*B*n)),(m,num_prop*(nts-1)*B*n))
    WD.XB2  =  np.reshape(np.reshape(np.reshape(WD_XB2,(m,num_prop,(nts-1),B*n)),(m,num_prop,(nts-1)*B*n)),(m,num_prop*(nts-1)*B*n))
    WD.YB2  =  np.reshape(np.reshape(np.reshape(WD_YB2,(m,num_prop,(nts-1),B*n)),(m,num_prop,(nts-1)*B*n)),(m,num_prop*(nts-1)*B*n))
    WD.ZB2  =  np.reshape(np.reshape(np.reshape(WD_ZB2,(m,num_prop,(nts-1),B*n)),(m,num_prop,(nts-1)*B*n)),(m,num_prop*(nts-1)*B*n))
    WD.GAMMA=  np.reshape(np.reshape(np.reshape(WD_GAMMA,(m,num_prop,(nts-1),B*n)),(m,num_prop,(nts-1)*B*n)),(m,num_prop*(nts-1)*B*n))
    
    return WD,ts,B,N 
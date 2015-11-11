# Angle_vector_noise.py
# 
# Created:  Nov 2015, Carlos Ilario

# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import SUAVE

import numpy as np

def noise_geometric(noise_segment,analyses):
    
    #We need to define a way to set this automaticaly
    sideline = analyses.noise.settings.sideline
    flyover  = analyses.noise.settings.flyover
    approach = analyses.noise.settings.approach
    
    #unpack
    position_vector = noise_segment.conditions.frames.inertial.position_vector 
    altitute        = noise_segment.conditions.freestream.altitude[:,0] 
    climb_angle     = 6. #How can I get this from the code at this point?
    
    s = position_vector[:,0]
    n_steps = len(altitute)  #number of time steps (space discretization)
       
    if approach==1:
        
        #--------------------------------------------------------
        #-------------------APPROACH CALCULATION-----------------
        #--------------------------------------------------------
        
        phi=np.zeros(n_steps)    
        
        #Microphone position from the threshold
        x0= 2000.
       
        #Calculation of the distance vector and emission angle
        dist=np.sqrt(altitute**2+(s-x0)**2)
        theta=np.arctan(np.abs(altitute/(s-x0)))
    
    elif flyover==1:
        
        #--------------------------------------------------------
        #------------------ FLYOVER CALCULATION -----------------
        #--------------------------------------------------------
        
        phi=np.zeros(n_steps)    
        
        #Lift-off position from the brake release
        S_0=1061 
        
        #Microphone position from the threshold
        x0= 6500. - S_0      
        
        #Calculation of the distance vector and emission angle
        dist=np.sqrt(altitute**2+(s-x0)**2)
        theta=np.arctan(np.abs(altitute/(s-x0)))
        
    else:
        
        #--------------------------------------------------------
        #-------------------SIDELINE CALCULATION-----------------
        #--------------------------------------------------------        

        z0 = 450.  #position on the z-direction of the sideline microphone
        y0=0.    #position on the y-direction of the sideline microphone
        
        #Lift-off position from the brake release
        S_0=1061.        
        
        x0=S_0+(304.8-altitute[0])/np.tan(climb_angle*np.pi/180) #Position of the sideline microphone for the maximum take-off noise assumed to be at 1000ft of altitute
     
    
        phi=np.arctan(z0/altitute)
        dist=np.sqrt((z0/np.sin(phi))**2+(s-x0)**2)
        theta=np.arccos(np.abs((x0-s)/dist))
        
    
    return (dist,theta,phi)

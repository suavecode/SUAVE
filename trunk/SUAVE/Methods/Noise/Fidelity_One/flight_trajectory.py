#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:     
#
# Author:      CARIDSIL
#
# Created:     20/07/2015
# Copyright:   (c) CARIDSIL 2015
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import numpy as np
from SUAVE.Core            import Units

def flight_trajectory(configs):
    
     #unpack
    s0  = configs.flight.initial_position   
    t0  = configs.flight.initial_time
    velocity=configs.flight.velocity
    altitute=configs.flight.altitute
    
    
    approach=0
    flyover=0
    takeoff=1
    
    #Necessary input for determination of noise trajectory
    
    x0=0 #microphone reference position
    dt=0.5*Units.s  #time step for noise calculation
    total_time = 18*Units.s #total time for noise calculation
    
    n_steps = np.int(total_time/dt +1)  #number of time steps (space discretization)
        
   
    
    theta=np.zeros(n_steps)
    s=np.zeros(n_steps)
    dist=np.zeros(n_steps)
    time=np.zeros(n_steps)
    
    #initial conditions
    time[0]=t0
    s[0]=s0
        
    if flyover==1:
        dist[0]=np.sqrt(altitute**2+(s[0]-x0)**2)
        theta[0]=np.arctan(np.abs(altitute/s0))
    #Calculate flight path
        for i in range(1,n_steps):           
            s[i]=s[i-1]+velocity*dt
            dist[i]=np.sqrt(altitute**2+(s[i]-x0)**2)
            theta[i]=np.arctan(np.abs(altitute/(s[i]-x0)))
            time[i]=time[i-1]+dt
    
    if approach==1:
        velocity_x=velocity*np.cos(3*np.pi/180)
        velocity_y=velocity*np.sin(3*np.pi/180)
        
        total_time=np.int(4000/velocity_x)    
        n_steps = np.int(total_time/dt +1)  #number of time steps (space discretization)
        
        theta=np.zeros(n_steps)
        s=np.zeros(n_steps)
        dist=np.zeros(n_steps)
        time=np.zeros(n_steps)
        
        #initial conditions
        time[0]=t0
        s[0]=s0
        
        altitute=np.zeros(n_steps)
        altitute[0]=120+2000*np.tan(3*np.pi/180) 
        dist[0]=np.sqrt(altitute[0]**2+(s[0]-x0)**2)
        theta[0]=np.arctan(np.abs(altitute[0]/s0))
        
        for i in range(1,n_steps):
            s[i]=s[i-1]+velocity_x*dt
            altitute[i]=altitute[i-1]-velocity_y*dt
            dist[i]=np.sqrt(altitute[i]**2+(s[i]-x0)**2)
            theta[i]=np.arctan(np.abs(altitute[i]/(s[i]-x0)))
            time[i]=time[i-1]+dt
            
    if takeoff==1:
        x0=6500*Units.m #Position of the Flyover microphone
        gama=7      #Angle of takeoff
        
        velocity_x=velocity*np.cos(gama*np.pi/180)
        velocity_y=velocity*np.sin(gama*np.pi/180)
        
        total_time=np.int((x0+500)/velocity_x)    
        n_steps = np.int(total_time/dt +1)  #number of time steps (space discretization)
        
        theta=np.zeros(n_steps)
        s=np.zeros(n_steps)
        dist=np.zeros(n_steps)
        time=np.zeros(n_steps)
        altitute=np.zeros(n_steps)
        
        #initial conditions
        time[0]=t0
        s[0]=1500*Units.m #Lift-off position from the brake release
        altitute[0]=35*Units.ft 
        
       
                
        dist[0]=np.sqrt(altitute[0]**2+(x0-s[0])**2)
        theta[0]=np.arctan(np.abs(altitute[0]/(x0-s[0])))
        
        for i in range(1,n_steps):
            s[i]=s[i-1]+velocity_x*dt
            altitute[i]=altitute[i-1]+velocity_y*dt
            dist[i]=np.sqrt(altitute[i]**2+(s[i]-x0)**2)
            theta[i]=np.arctan(np.abs(altitute[i]/(s[i]-x0)))
            time[i]=time[i-1]+dt
            

    
    return(time,altitute,dist,theta)
        


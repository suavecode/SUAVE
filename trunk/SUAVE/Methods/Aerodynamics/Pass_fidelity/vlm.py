# vlm.py
# 
# Created:  Your Name, Dec 2013
# Modified:         

""" vlm(wing,segment)
    """

# ----------------------------------------------------------------------
#  Imports
#

# suave imports
#from SUAVE.Attributes.Gases.Air import compute_speed_of_sound
import numpy
from SUAVE.Attributes.Gases import Air
aero_atm=Air()

# python imports
import os, sys, shutil
from copy import deepcopy
from warnings import warn

# package imports
import numpy as np
import scipy as sp

def vlm(Wing,segment,Sref,alpha,n,nn):
#def vlm(span,root_chord,tip_chord,n,nn,sweep,taper,alpha_rc,alpha_tc,Vinf,Sref,rho,sym_para,Mc,aoa,AR):
    """ SUAVE.Methods.parasite_drag_aircraft(aircraft,segment,Cl,cdi_inv,cdp,fd_ws)
        computes the parasite_drag_aircraftassociated with an aircraft 
        
        Inputs:
            Wing - Wing object 
            segment - the segment object contains information regarding the mission segment
            alpha -  angle of attack  
            n -  no of panels in y direction
            nn -  no of panels in x direction
        Outpus:
            Cl_comp - lift
            Cd_comp - drag         

        Assumptions:
            if needed
        
    """    

    #Unpack
    span=Wing.span
    root_chord=Wing.chord_root
    tip_chord=Wing.chord_tip
    sweep=Wing.sweep
    taper=Wing.taper
    alpha_rc=Wing.alpha_rc
    alpha_tc=Wing.alpha_tc
    sym_para=Wing.symmetric
    AR=Wing.ar


    R=287    

    #Sref=Sref
    roc=segment.rho
    muc=segment.mew
    Tc=segment.T 
    Mc=segment.M
    pc=segment.p
    
    gm=aero_atm.compute_gamma(Tc,pc)
    Vinf=Mc*numpy.sqrt(gm*R*Tc)

    aoa=alpha
    
    
  
    #eeta=0.97
  
    dchord=(root_chord-tip_chord)
    if sym_para is True :

        span=span/2
    
    deltax=span/n
    
    section_length= numpy.empty(n)
    area_section=numpy.empty(n)
    sl=numpy.empty(n)
    
    xpos=numpy.empty(n)
    
    ya=numpy.empty(n)
    yb=numpy.empty(n)
    xa=numpy.empty(n)
    yab=numpy.empty(n)
    ybb=numpy.empty(n)
    y2=numpy.empty(n)   
    
    x=numpy.empty(n)
    y=numpy.empty(n)
    twist_distri=numpy.empty(n)
    xloc_leading=numpy.empty(n)
    xloc_trailing=numpy.empty(n)  
    RHS=numpy.empty([n,1])
    w=numpy.empty([n,n])
    wb=numpy.empty([n,n])
    A=numpy.empty([n,n])
    L=numpy.empty(n)
    T=numpy.empty(n)
    
    A_v=numpy.empty([n,n])
    v=numpy.empty(n)


    Lfi=numpy.empty(n)
    Lfk=numpy.empty(n)
    
    Lft=numpy.empty(n)
    Dg=numpy.empty(n)   
    D=numpy.empty(n)    
    
   
    
    
    #--discretizing the wing sections into panels--------------------------------
    for i in range(0,n):
    
        section_length[i]= dchord/span*(span-(i+1)*deltax+deltax/2) + tip_chord
        area_section[i]=section_length[i]*deltax
        sl[i]=section_length[i]
        twist_distri[i]=alpha_rc + i/float(n)*(alpha_tc-alpha_rc)
        xpos[i]=(i)*deltax
        
        ya[i]=(i)*deltax
        yb[i]=(i+1)*deltax
        xa[i]=((i+1)*deltax-deltax/2)*numpy.tan(sweep)+ 0.25*sl[i]
        
        x[i]=((i+1)*deltax-deltax/2)*numpy.tan(sweep) + 0.75*sl[i]
        y[i]=((i+1)*deltax-deltax/2)
        xloc_leading[i]=((i+1)*deltax)*numpy.tan(sweep)
        xloc_trailing[i]=((i+1)*deltax)*numpy.tan(sweep)+sl[i]

    
  
    #------Influence coefficient computation-----------------------
    
    
    
    for i in range(0,n):

        RHS[i,0]=Vinf*numpy.sin(twist_distri[i]+aoa)  #)

        for j in range(0,n):
          
       
            yad=y[i]-ya[j]
            xd=x[i]-xa[j]
            ybd=y[i]-yb[j]
            
            yadd=y[i]-yab[j]
            ybdd=y[i]-ybb[j]
            
     
            A[i,j]=whav(x[i],y[i],xa[j],ya[j])-whav(x[i],y[i],xa[j],yb[j])-whav(x[i],y[i],xa[j],-ya[j])+whav(x[i],y[i],xa[j],-yb[j])
            A[i,j]=A[i,j]*0.25/numpy.pi
    
     
    #---------------Vortex strength computation by matrix inversion-----------
    
    #if sym_para is True :

        #A=w+wb
    
    #else :
        #A=w
        

    T=numpy.linalg.solve(A,RHS)
    
    
    #---Calculating the effective velocty--------------------------

    for i in range(0,n):
       v[i]=0.0      
       for j in range(0,n):
   
           A_v[i,j]=whav(xa[i],y[i],xa[j],ya[j])-whav(xa[i],y[i],xa[j],yb[j])-whav(xa[i],y[i],xa[j],-ya[j])+whav(xa[i],y[i],xa[j],-yb[j])
           A_v[i,j]=A[i,j]*0.25/numpy.pi*T[j]
           v[i]=v[i]+A_v[i,j]
   

       Lfi[i]=-T[i]*(Vinf*numpy.sin(alpha_tc)-v[i])
       Lfk[i]=T[i]*Vinf*numpy.cos(alpha_tc)

       Lft[i]=(-Lfi[i]*numpy.sin(alpha_tc)+Lfk[i]*numpy.cos(alpha_tc))
       Dg[i]=(Lfi[i]*numpy.cos(alpha_tc)+Lfk[i]*numpy.sin(alpha_tc))
       

       
   
    #---------Lift computation from elements---------------------------------
    LT=0.0
    DT=0.0
    arsec=0.0
    
    for i in range(0,n):

        L[i]=deltax*roc*Lft[i]   #T(i)*v(i)*sin(alpha)
        D[i]=deltax*roc*Dg[i]    
        
        LT=LT+L[i]
        DT=DT+D[i]
        
        arsec=arsec+area_section[i]

   
    Cl=2*LT/(0.5*roc*Vinf**2*Sref)
    Cl_comp=Cl/(numpy.sqrt(1-Mc**2))


    Cd=2*DT/(0.5*roc*Vinf**2*Sref)       
    Cd_comp=Cd/(numpy.sqrt(1-Mc**2))
    
    #--Karman Tsien compressibility correction to be added to prevent singularity at M=1
    #Cl=LT/((0.5*rho*Vinf**2*Sref)*(numpy.sqrt(1-Mc**2)))    

    return Cl_comp, Cd_comp
    



def whav(x1,y1,x2,y2):

    if x1==x2:
        whv=1/(y1-y2)
    else:  
        whv=1/(y1-y2)*(1+ (numpy.sqrt((x1-x2)**2+(y1-y2)**2)/(x1-x2)))
    
    return whv
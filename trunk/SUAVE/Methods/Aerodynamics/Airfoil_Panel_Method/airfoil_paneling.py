## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method
# airfoil_paneling.py
# 
# Created:  Aug 2022, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ---------------------------------------------------------------------- 
import SUAVE  
import numpy as np        
import math

## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method
def make_panels(Airfoils, npanel):  
    """Places panels on the current airfoil, as described by Airfoils.geom.xpoint
    
    Assumptions:
      Uses curvature-based point distribution on a spline of the points

    Source:   
                                                     
    Inputs: 
      Airfoils      : data class
      npanel : number of panels
                                                                           
    Outputs: 
      Airfoils.foil.N : number of panel points
      Airfoils.foil.x : coordinates of panel nodes (2xN)
      Airfoils.foil.s : arclength values at nodes (1xN)
      Airfoils.foil.t : tangent vectors, not np.linalg.normalized, dx/ds, dy/ds (2xN)
    
    Properties Used:
    N/A
    """    
    X        = Airfoils.geom.xpoint
    S        = np.append(0,np.cumsum(np.sqrt(np.diff(X[0,:])**2 + np.diff(X[1,:])**2 )))
    dXdS     = np.gradient(X[0,:],S)  # dX/dS
    dYdS     = np.gradient(X[1,:],S)  # dY/dS
    XS       = np.concatenate((dXdS[None,:],dYdS[None,:]),axis = 0) 
    
    Airfoils.foil.x = X 
    Airfoils.foil.s = S
    Airfoils.foil.t = XS  
    Airfoils.foil.N = len(Airfoils.foil.x[1])

    return 

## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method
def TE_info(X):  
    """Returns trailing-edge information for an airfoil with node coords X
    
    Assumptions:
    None

    Source:   
                                                     
    Inputs:
      X : node coordinates, ordered clockwise (2xN)
                                                                           
    Outputs: 
      t    : bisector vector = average of upper/lower tangents, np.linalg.normalized
      hTE  : trailing edge gap, measured as a cross-section
      dtdx : thickness slope = d(thickness)/d(wake x)
      tcp  : |t cross p|, used for setting TE source panel strength
      tdp  : t dot p, used for setting TE vortex panel strength
    
    Properties Used:
    N/A
    """ 

    t1   = X[:, 0]-X[:,1]
    t1   = t1/np.linalg.norm(t1) # lower tangent vector
    t2   = X[:,-1]-X[:,-2] 
    t2   = t2/np.linalg.norm(t2) # upper tangent vector
    t    = 0.5*(t1+t2) 
    t    = t/np.linalg.norm(t) # average tangent gap bisector
    s    = X[:,-1]-X[:,0] # lower to upper connector vector
    hTE  = -s[0]*t[1] + s[1]*t[0] # TE gap
    dtdx = t1[0]*t2[1] - t2[0]*t1[1] # sin(theta between t1,t2) approx dt/dx
    p    = s/np.linalg.norm(s) # unit vector along TE panel
    tcp  = abs(t[0]*p[1]-t[1]*p[0]) 
    tdp  = np.dot(t,p)

    return t, hTE, dtdx, tcp, tdp


## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method
def panel_linvortex_velocity(Xj, xi, vdir, onmid): 
    """Calculates the velocity coefficients for a linear vortex panel
    
    Assumptions:
    None

    Source:   
                     
    Inputs: 
      x       : x coordinates of panels
      z       : z coordinates of paneels
      theta1  : angle between panel and control point 1
      theta2  : angle between panel and control point 2
      t       : nornal vectors 
      n       : unit nornals 
      d       : distances between control points on panel
      r1      : position vector to point 1 
      r2      : position vector to point 2                     
      vdir    : direction of dot product  
      onmid   : true means xi is on the panel midpoint
                                                                           
    Outputs:
      a,b   : velocity influence coefficients of the panel
    
    Properties Used:
    N/A
    """  

    # panel coordinates
    xj1 = Xj[0,0]  
    zj1 = Xj[1,0]
    xj2 = Xj[0,1]  
    zj2 = Xj[1,1]

    # panel-aligned tangent and np.linalg.normal vectors
    t = np.array([xj2-xj1 , zj2-zj1])
    t = t/np.linalg.norm(t)
    n = np.array([-t[1] ,t[0]]) 

    # control point relative to (xj1,zj1)
    xz = np.array([(xi[0]-xj1), (xi[1]-zj1)]).T
    x = np.dot(xz,t)  # in panel-aligned coord system
    z = np.dot(xz,n)   # in panel-aligned coord system

    # distances and angles
    d      = np.linalg.norm([xj2-xj1, zj2-zj1]) # panel length
    r1     = np.linalg.norm([x,z])             # left edge to control point
    r2     = np.linalg.norm([x-d,z])           # right edge to control point
    theta1 = math.atan2(z,x)          # left angle
    theta2 = math.atan2(z,x-d)        # right angle

    # velocity in panel-aligned coord system
    if (onmid):
        ug1 = 1/2 - 1/4   
        ug2 = 1/4
        wg1 = -1/(2*np.pi)   
        wg2 = 1/(2*np.pi)
    else:
        temp1 = (theta2-theta1)/(2*np.pi)
        temp2 = (2*z*np.log(r1/r2) - 2*x*(theta2-theta1))/(4*np.pi*d)
        ug1 =  temp1 + temp2
        ug2 =        - temp2
        temp1 = np.log(r2/r1)/(2*np.pi)
        temp2 = (x*np.log(r1/r2) - d + z*(theta2-theta1))/(2*np.pi*d)
        wg1 =  temp1 + temp2
        wg2 =        - temp2   

    # velocity influence in original coord system
    a = np.array([ug1*t[0]+wg1*n[0], ug1*t[1]+wg1*n[1]]) # point 1
    b = np.array([ug2*t[0]+wg2*n[0], ug2*t[1]+wg2*n[1]]) # point 2
    if np.shape(vdir)[0] != 0:
        a = a.T*vdir
        b = b.T*vdir 

    return a,b


## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method
def panel_linvortex_stream(Xj, xi,ep = 1e-10):   
    """Calculates the streamdef coefficients for a linear vortex panel
    
    Assumptions:
    None

    Source:   
                     
    Inputs: 
      x       : x coordinates of panels
      z       : z coordinates of paneels
      theta1  : angle between panel and control point 1
      theta2  : angle between panel and control point 2
      t       : nornal vectors 
      n       : unit nornals 
      d       : distances between control points on panel
      r1      : position vector to point 1 
      r2      : position vector to point 2                     
      vdir    : direction of dot product   
      
    Outputs: 
      a,b     : streamdef influence coefficients
    
    Properties Used:
    N/A
    """  

    # panel coordinates
    xj1 = Xj[0,0]  
    zj1 = Xj[1,0]
    xj2 = Xj[0,1]  
    zj2 = Xj[1,1]

    # panel-aligned tangent and np.linalg.normal vectors
    t = np.array([[xj2-xj1 ],[ zj2-zj1]])
    t = t/np.linalg.norm(t)
    n = np.array([-t[1], t[0]])

    # control point relative to (xj1,zj1)
    xz = np.array([(xi[0]-xj1), (xi[1]-zj1)]).T
    x = np.dot(xz,t) # in panel-aligned coord system
    z = np.dot(xz,n)  # in panel-aligned coord system

    # distances and angles
    d      = np.linalg.norm([xj2-xj1, zj2-zj1]) # panel length
    r1     = np.linalg.norm([x,z])             # left edge to control point
    r2     = np.linalg.norm([x-d,z])           # right edge to control point
    theta1 = math.atan2(z,x)          # left angle
    theta2 = math.atan2(z,x-d)        # right angle

    # check for r1, r2 zero 
    if (r1 < ep):
        logr1 =  np.array([0]) 
    else:
        logr1 = np.log(r1) 
    if (r2 < ep):
        logr2 = np.array([0]) 
    else:
        logr2 = np.log(r2) 

    # streamdef components
    P1 = (0.5/np.pi)*(z*(theta2-theta1) - d + x*logr1 - (x-d)*logr2)
    P2 = x*P1 + (0.5/np.pi)*(0.5*r2**2*logr2 - 0.5*r1**2*logr1 - r2**2/4 + r1**2/4)

    # influence coefficients
    a = P1-P2/d
    b =    P2/d

    return a, b

## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method
def panel_constsource_velocity(Xj, xi, vdir,ep = 1e-10):  
    """Calculates the velocity coefficient for a constant source panel
    
    Assumptions:
    None

    Source:   
                     
    Inputs: 
      x       : x coordinates of panels
      z       : z coordinates of paneels
      theta1  : angle between panel and control point 1
      theta2  : angle between panel and control point 2
      t       : nornal vectors 
      n       : unit nornals 
      d       : distances between control points on panel
      r1      : position vector to point 1 
      r2      : position vector to point 2                     
      vdir    : direction of dot product 
                                                                           
    Outputs: 
      a       : velocity influence coefficient of the panel
    
    Properties Used:
    N/A
    """  

    # panel coordinates
    xj1 = Xj[0,0]  
    zj1 = Xj[1,0]
    xj2 = Xj[0,1]  
    zj2 = Xj[1,1]

    # panel-aligned tangent and np.linalg.normal vectors
    t = np.array([xj2-xj1 , zj2-zj1])
    t = t/np.linalg.norm(t)
    n = np.array([-t[1] ,t[0]])

    # control point relative to (xj1,zj1)
    xz = np.array([(xi[0]-xj1), (xi[1]-zj1)])
    x  = np.dot(xz,t) # in panel-aligned coord system
    z  = np.dot(xz,n)  # in panel-aligned coord system    

    # distances and angles
    d      = np.linalg.norm([xj2-xj1, zj2-zj1]) # panel length
    r1     = np.linalg.norm([x,z])             # left edge to control point
    r2     = np.linalg.norm([x-d,z])           # right edge to control point
    theta1 = math.atan2(z,x)          # left angle
    theta2 = math.atan2(z,x-d)        # right angle 
    
    if (r1 < ep):
        logr1 = 0 
        theta1=np.pi 
        theta2=np.pi 
    else:
        logr1 = np.log(r1) 
    if (r2 < ep):
        logr2 = 0 
        theta1=0 
        theta2=0 
    else:
        logr2 = np.log(r2) 


    # velocity in panel-aligned coord system
    u = (0.5/np.pi)*(logr1 - logr2)
    w = (0.5/np.pi)*(theta2-theta1)

    # velocity in original coord system dotted with given vector
    a = np.array([u*t[0]+w*n[0],u*t[1]+w*n[1]])
    if np.shape(vdir)[0]!= 0:
        a = np.dot(a,vdir)
    return a


## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method
def panel_constsource_stream(Xj, xi,ep = 1e-10):   
    """Calculates the streamdef coefficient for a constant source panel
    
    Assumptions:
    None

    Source:   
                   
    Inputs: 
      x       : x coordinates of panels
      z       : z coordinates of paneels
      theta1  : angle between panel and control point 1
      theta2  : angle between panel and control point 2
      t       : nornal vectors 
      n       : unit nornals 
      d       : distances between control points on panel
      r1      : position vector to point 1 
      r2      : position vector to point 2                     
      vdir    : direction of dot product 
                                             
    Outputs: 
      a       : streamdef influence coefficient of the panel
                                                                           
    Outputs: 
    
    Properties Used:
    N/A
    """ 

    # panel coordinates
    xj1 = Xj[0,0]  
    zj1 = Xj[1,0]
    xj2 = Xj[0,1]  
    zj2 = Xj[1,1]

    # panel-aligned tangent and np.linalg.normal vectors
    t = np.array([xj2-xj1, zj2-zj1 ])
    t = t/np.linalg.norm(t)
    n = np.array([-t[1],t[0]])

    # control point relative to (xj1,zj1)
    xz = np.array([(xi[0]-xj1), (xi[1]-zj1)])
    x = np.dot(xz,t) # in panel-aligned coord system
    z = np.dot(xz,n)  # in panel-aligned coord system        

    # distances and angles
    d  = np.linalg.norm([xj2-xj1, zj2-zj1]) # panel length
    r1 = np.linalg.norm([x,z])             # left edge to control point
    r2 = np.linalg.norm([x-d,z])           # right edge to control point

    theta1 = math.atan2(z,x)          # left angle
    theta2 = math.atan2(z,x-d)        # right angle

    # streamdef 
    if (r1 < ep): 
        logr1  = np.array([0]) 
        theta1 = np.array([np.pi]) 
        theta2 = np.array([np.pi]) 
    else: 
        logr1 = np.log(r1) 
    if (r2 < ep): 
        logr2  = np.array([0]) 
        theta1 = np.array([0]) 
        theta2 = np.array([0]) 
    else: 
        logr2 = np.log(r2) 
    P = (x*(theta1-theta2) + d*theta2 + z*logr1 - z*logr2)/(2*np.pi)

    dP = d # delta psi
    if ((theta1+theta2) > np.pi):
        P = P - 0.25*dP 
    else:
        P = P + 0.75*dP 

    # influence coefficient
    a = P
     
    return a 


## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method
def panel_linsource_velocity(Xj, xi, vdir): 
    """ Calculates the streamdef coefficients for a linear velocity panel
    
    Assumptions:
    None

    Source:   
                                                     
    Inputs: 
      x       : x coordinates of panels
      z       : z coordinates of paneels
      theta1  : angle between panel and control point 1
      theta2  : angle between panel and control point 2
      t       : nornal vectors 
      n       : unit nornals 
      d       : distances between control points on panel
      r1      : position vector to point 1 
      r2      : position vector to point 2                     
      vdir    : direction of dot product 
                                             
    Outputs: 
      a,b     : velocity influence coefficients of the panel
    
    Properties Used:
    N/A
    """  

    # panel coordinates
    xj1 = Xj[0,0]  
    zj1 = Xj[1,0]
    xj2 = Xj[0,1]  
    zj2 = Xj[1,1]

    # panel-aligned tangent and np.linalg.normal vectors
    t = np.array([xj2-xj1 ,zj2-zj1])
    t = t/np.linalg.norm(t)
    n = np.array([-t[1],t[0]])

    # control point relative to (xj1,zj1
    xz = np.array([(xi[0]-xj1), (xi[1]-zj1)])
    x  = np.dot(xz,t)  # in panel-aligned coord system
    z  = np.dot(xz,n)   # in panel-aligned coord system

    # distances and angles
    d      = np.linalg.norm([xj2-xj1, zj2-zj1]) # panel length
    r1     = np.linalg.norm([x,z])             # left edge to control point
    r2     = np.linalg.norm([x-d,z])           # right edge to control point
    theta1 = math.atan2(z,x)          # left angle
    theta2 = math.atan2(z,x-d)        # right angle

    # velocity in panel-aligned coord system
    temp1 = np.log(r1/r2)/(2*np.pi)
    temp2 = (x*np.log(r1/r2) - d + z*(theta2-theta1))/(2*np.pi*d)
    ug1   =  temp1 - temp2
    ug2   =          temp2
    temp1 = (theta2-theta1)/(2*np.pi)
    temp2 = (-z*np.log(r1/r2) + x*(theta2-theta1))/(2*np.pi*d)
    wg1   =  temp1 - temp2
    wg2   =          temp2
  
    # velocity influence in original coord system
    a = np.array([ug1*t[0]+wg1*n[0], ug1*t[1]+wg1*n[1]]) # point 1
    b = np.array([ug2*t[0]+wg2*n[0], ug2*t[1]+wg2*n[1]]) # point 2
    if np.shape(vdir) !=0: 
        a = np.dot(a,vdir)  
        b = np.dot(b,vdir)          

    return a, b 

## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method
def panel_linsource_stream(Xj, xi,ep = 1e-10): 
    """Calculates the streamdef coefficients for a linear source panel
     
     Assumptions:
       None
 
     Source:    
                                                       
     Inputs: 
       x       : x coordinates of panels
       z       : z coordinates of paneels
       theta1  : angle between panel and control point 1
       theta2  : angle between panel and control point 2
       t       : nornal vectors 
       n       : unit nornals 
       d       : distances between control points on panel
       r1      : position vector to point 1 
       r2      : position vector to point 2                     
       vdir    : direction of dot product 
                                                                            
     Outputs: 
       a,b : streamdef influence coefficients
     
     Properties Used:
     N/A
    """ 

    # panel coordinates
    xj1 = Xj[0,0]  
    zj1 = Xj[1,0]
    xj2 = Xj[0,1]  
    zj2 = Xj[1,1]

    # panel-aligned tangent and np.linalg.normal vectors
    t_s = np.array([[xj2-xj1],[ zj2-zj1]])
    t   = t_s/np.linalg.norm(t_s)
    n   = np.array([-t[1],t[0]])

    # control point relative to (xj1,zj1)
    xz = np.array([(xi[0]-xj1), (xi[1]-zj1)])
    x  = np.dot(xz,t)  # in panel-aligned coord system
    z  = np.dot(xz,n)   # in panel-aligned coord system

    # distances and angles
    d      = np.linalg.norm([xj2-xj1, zj2-zj1]) # panel length
    r1     = np.linalg.norm([x,z])             # left edge to control point
    r2     = np.linalg.norm([x-d,z])           # right edge to control point
    theta1 = math.atan2(z,x)          # left angle
    theta2 = math.atan2(z,x-d)        # right angle

    # make branch cut at theta = 0
    if (theta1<0): 
        theta1 = theta1 + 2*np.pi 
    if (theta2<0): 
        theta2 = theta2 + 2*np.pi 

    # check for r1, r2 zero 
    if (r1 < ep): 
        logr1 = 0 
        theta1 = np.pi 
        theta2 = np.pi 
    else: 
        logr1 = np.log(r1) 
    if (r2 < ep): 
        logr2  = 0 
        theta1 = 0 
        theta2 = 0 
    else: 
        logr2 = np.log(r2) 

    # streamdef components
    P1 = (0.5/np.pi)*(x*(theta1-theta2) + theta2*d + z*logr1 - z*logr2)
    P2 = x*P1 + (0.5/np.pi)*(0.5*r2**2*theta2 - 0.5*r1**2*theta1 - 0.5*z*d)

    # influence coefficients
    a = P1-P2/d
    b =    P2/d
 
    return a, b 



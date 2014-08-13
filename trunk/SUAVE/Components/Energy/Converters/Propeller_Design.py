#Prop_Design.py
# 
# Created:  Emilio Botero, Jul 2014
# Modified:  

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
import SUAVE

# package imports
import numpy as np
import scipy as sp

from SUAVE.Attributes import Units

from SUAVE.Structure import (
Data, Container, Data_Exception, Data_Warning,
)

    #def __defaults__(self):
    
        ## Default values
        #Tc = 0.0
        #Pc = 0.0

# ----------------------------------------------------------------------
#  Main
# ----------------------------------------------------------------------
    
def Propeller_Design(Prop_attributes):
    """ Optimizes propeller chord and twist given input parameters.
          
          Inputs:
              Either design power or thrust
              hub radius
              tip radius
              rotation rate
              freestream velocity
              number of blades
              number of stations
              design lift coefficient
              airfoil data

          Outputs:
              Twist distribution
              Chord distribution
              
          Assumptions:
              Based on Design of Optimum Propellers by Adkins and Liebeck

    """    
    #Unpack
    Pc    = Prop_attributes.Pc     # Design Power coefficient
    Tc    = Prop_attributes.Tc     # Design Thrust coefficient
    B     = Prop_attributes.B      # Number of Blades
    R     = Prop_attributes.R      # Tip Radius
    Rh    = Prop_attributes.Rh     # Hub Radius
    omega = Prop_attributes.omega  # Rotation Rate in rad/s
    V     = Prop_attributes.V      # Freestream Velocity
    Cl    = Prop_attributes.Des_CL # Design Lift Coefficient
    nu    = Prop_attributes.nu     # Kinematic Viscosity
    rho   = Prop_attributes.rho    # Density
    
    tol   = 1e-10# Convergence tolerance
    N     = 30 # Number of Stations
    
    #Figure out how to enter airfoil data

    #Step 1, assume a zeta
    zeta = 0.1 # Assume to be small initially
    
    #Step 2, determine F and phi at each blade station
    
    chi0    = Rh/R # Where the propeller blade actually starts
    chi     = np.linspace(chi0,1,N+1) # Vector of nondimensional radii
    chi     = chi[0:N]
    lamda   = V/(omega*R)           # Speed ratio
    r       = chi*R                 # Radial coordinate
    x       = omega*r/V             # Nondimensional distance
    diff    = 1.0                   # Difference between zetas
    n       = omega/(2*np.pi)       # Cycles per second
    D       = 2.*R
    J       = V/(D*n)
    
    
    
    while diff>tol:
        #Things that need a loop
        Tcnew   = Tc
        tanphit = lamda*(1.+zeta/2.)   # Tangent of the flow angle at the tip
        phit    = np.arctan(tanphit) # Flow angle at the tip
        tanphi  = tanphit/chi        # Flow angle at every station
        f       = (B/2.)*(1.-chi)/np.sin(phit) 
        F       = (2./np.pi)*np.arccos(np.exp(-f)) #Prandtl momentum loss factor
        phi     = np.arctan(tanphi)  #Flow angle at every station
        
        #Step 3, determine the product Wc, and RE
        
        G       = F*x*np.cos(phi)*np.sin(phi) #Circulation function
        Wc      = 4.*np.pi*lamda*G*V*R*zeta/(Cl*B)
        RE      = Wc/nu

        #Step 4, determine epsilon and alpha from airfoil data

        Cd    = 0.01385 #From xfoil of the DAE51 at RE=150k, Cl=0.7
        alpha = Cl/(2.*np.pi)
        epsilon   = Cd/Cl
        
        #Step 5, change Cl and repeat steps 3 and 4 until epsilon is minimized
        
        #Step 6, determine a and a', and W
        
        a       = (zeta/2.)*(np.cos(phi)**2.)*(1.-epsilon*np.tan(phi))
        aprime  = (zeta/(2.*x))*np.cos(phi)*np.sin(phi)*(1.+epsilon/np.tan(phi))
        W       = V*(1.+a)/np.sin(phi)
        
        #Step 7, compute the chord length and blade twist angle    
        
        c       = Wc/W
        beta    = alpha + phi # Blade twist angle
    
        #Step 8, determine 4 derivatives in I and J
    
        Iprime1 = 4.*chi*G*(1.-epsilon*np.tan(phi))
        Iprime2 = lamda*(Iprime1/(2.*chi))*(1.+epsilon/np.tan(phi)
                                            )*np.sin(phi)*np.cos(phi)
        Jprime1 = 4.*chi*G*(1.+epsilon/np.tan(phi))
        Jprime2 = (Jprime1/2.)*(1.-epsilon*np.tan(phi))*(np.cos(phi)**2.)
        
        dR      = (r[1]-r[0])*np.ones_like(Jprime1)
        dchi    = (chi[1]-chi[0])*np.ones_like(Jprime1)
        
        #Integrate derivatives from chi=chi0 to chi=1
        
        I1      = np.dot(Iprime1,dchi)
        I2      = np.dot(Iprime2,dchi)
        J1      = np.dot(Jprime1,dchi)
        J2      = np.dot(Jprime2,dchi)        

        #Step 9, determine zeta and and Pc or zeta and Tc
        
        if (Pc==0.)&(Tc!=0.): 
            #First Case, Thrust is given
            #Check to see if Tc is feasible, otherwise try a reasonable number
            if Tcnew>=I2*(I1/(2.*I2))**2.:
                Tcnew = I2*(I1/(2.*I2))**2.
            zetan    = (I1/(2.*I2)) - ((I1/(2.*I2))**2.-Tcnew/I2)**0.5

        elif (Pc!=0.)&(Tc==0.): 
            #Second Case, Thrust is given
            zetan    = -(J1/(J2*2.)) + ((J1/(J2*2.))**2.+Pc/J2)**0.5
            
        else:
            print('Power and thrust are both specified!')
    
        #Step 10, repeat starting at step 2 with the new zeta
        diff = abs(zeta-zetan)
        
        zeta = zetan
    
    #Step 11, determine propeller efficiency etc...
    
    if (Pc==0.)&(Tc!=0.): 
        if Tcnew>=I2*(I1/(2.*I2))**2.:
            Tcnew = I2*(I1/(2.*I2))**2.
            print('Tc infeasible, reset to:')
            print(Tcnew)        
        #First Case, Thrust is given
        zeta    = (I1/(2.*I2)) - ((I1/(2.*I2))**2.-Tcnew/I2)**0.5
        Pc      = J1*zeta + J2*(zeta**2.)
        Tc      = I1*zeta - I2*(zeta**2.)
        
    elif (Pc!=0.)&(Tc==0.): 
        #Second Case, Thrust is given
        zeta    = -(J1/(2.*J2)) + ((J1/(2.*J2))**2.+Pc/J2)**0.5
        Tc      = I1*zeta - I2*(zeta**2.)
        Pc      = J1*zeta + J2*(zeta**2.)
        
    else:
        print('Power and thrust are both specified!')    
        
    #efficiency = Tc/Pc
    #Cp         = np.pi*Pc/(8.*(J**3.))
    #Ct         = np.pi*Tc/(8.*(J**2.))
    #eta        = Ct*J/Cp 
    
    
    Power = Pc*rho*(V**3)*np.pi*(R**2)/2
    Cp   = Power/(rho*(n**3)*(D**5))

    Prop_attributes.c    = c
    Prop_attributes.beta = beta
    Prop_attributes.Cp   = Cp
    
    #These are used to check, the values here were used to verify against
    #AIAA 89-2048 for their propeller
    #print(2*Pc)
    #print(2*Tc)
    #print(V/(omega*R))
    #print(eta)
    #print(efficiency)
    
    return Prop_attributes
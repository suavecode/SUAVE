#propeller.py
# 
# Created:  Emilio Botero, Jun 2014
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
from SUAVE.Components.Energy.Energy_Component import Energy_Component

# ----------------------------------------------------------------------
#  Propeller Class
# ----------------------------------------------------------------------    
 
class Propeller(Energy_Component):
    
    def __defaults__(self):
        
        self.Ct     = 0.0
        self.Cp     = 0.0
        self.radius = 0.0
    
    def spin(self,conditions):
        """ Analyzes a propeller given geometry and operating conditions
                 
                 Inputs:
                     hub radius
                     tip radius
                     rotation rate
                     freestream velocity
                     number of blades
                     number of stations
                     chord distribution
                     twist distribution
                     airfoil data
       
                 Outputs:
                     Power coefficient
                     Thrust coefficient
                     
                 Assumptions:
                     Based on Qprop Theory document
       
           """
           
        #Unpack    
        B     = self.Prop_attributes.B      # Number of Blades
        R     = self.Prop_attributes.R      # Tip Radius
        Rh    = self.Prop_attributes.Rh     # Hub Radius
        beta  = self.Prop_attributes.beta   # Twist
        c     = self.Prop_attributes.c      # Chord distribution
        omega = self.inputs.omega           # Rotation Rate in rad/s
        rho   = conditions.freestream.density
        mu    = conditions.freestream.viscosity
        V     = conditions.freestream.velocity[:,0]
        a     = conditions.freestream.speed_of_sound
        
        nu    = mu/rho
        tol   = 1e-8 # Convergence tolerance
           
        ######
        #Figure out how to enter airfoil data
        ######
        
        #Things that don't change with iteration
        N       = len(c) #Number of stations
        chi0    = Rh/R # Where the propeller blade actually starts
        chi     = np.linspace(chi0,1,N+1) # Vector of nondimensional radii
        chi     = chi[0:N]
        lamda   = V/(omega*R)           # Speed ratio
        r       = chi*R                 # Radial coordinate

        x       = r*np.dot(omega,1/V)             # Nondimensional distance
        n       = omega/(2.*np.pi)      # Cycles per second
        J       = V/(2.*R*n)    
    
        sigma   = np.multiply(B*c,1./(2.*np.pi*r))   
    
        #I make the assumption that externally-induced velocity at the disk is zero
        #This can be easily changed if needed in the future:
        ua = 0.0
        ut = 0.0
        
        omegar = np.outer(omega,r)
        Ua = np.outer((V + ua),np.ones_like(r))
        Ut = omegar - ut
        U  = np.sqrt(Ua**2. + Ut**2.)
        
        #Things that will change with iteration
    
        #Setup a Newton iteration
        psi    =  np.ones_like(c)
        psiold = np.zeros_like(c)
        diff   = np.ones_like(c)
        
        while (np.any(diff>tol)):
            #print(psi)
            Wa    = 0.5*Ua + 0.5*U*np.sin(psi)
            Wt    = 0.5*Ut + 0.5*U*np.cos(psi)           
            #va    = Wa - Ua
            vt    = Ut - Wt
            alpha = beta - np.arctan2(Wa,Wt)
            W     = np.sqrt(Wa**2. + Wt**2.)
            Re    = W*c*nu
            #Ma    = W/a #a is the speed of sound
            
            lamdaw = r*Wa/(R*Wt)
            f      = (B/2.)*(1.-r/R)/lamdaw
            piece  = np.exp(-f)
            piece[piece>1] = 1.0
            #print(piece)
            F      = 2.*np.arccos(piece)/np.pi
            Gamma  = vt*(4.*np.pi*r/B)*F*np.sqrt(1.+(4.*lamdaw*R/(np.pi*B*r))**2.)
            
            #Ok, from the airfoil data, given Re, Ma, alpha we need to find Cl
            Cl = 2.*np.pi*alpha
            
            Rsquiggly = Gamma - 0.5*W*c*Cl   
            
            #An analytical derivative for dR_dpsi, this is derived by taking a derivative of the above equations
            #This was solved symbolically in Matlab and exported        
            dR_dpsi = ((4.*U*r*np.arccos(piece)*np.sin(psi)*((16.*(Ua + U*np.sin(psi))**2.)/(B**2.*np.pi**2.*(2*Wt)**2.) + 
                      1.)**(0.5))/B - (np.pi*U*(Ua*np.cos(psi) - Ut*np.sin(psi))*(beta - np.arctan((2*Wa)/(2*Wt))))/(2.*((2*Wt)**2. +
                      (2*Wa)**2.)**(0.5)) + (np.pi*U*((2*Wt)**2. +(2*Wa)**2.)**(0.5)*(U + Ut*np.cos(psi) + 
                      Ua*np.sin(psi)))/(2.*((2*Wa)**2./(2*Wt)**2. + 1.)*(Ut + U*np.cos(psi))**2.) - (4.*U*piece*((16.*(Ua +
                      U*np.sin(psi))**2.)/(B**2.*np.pi**2.*(2*Wt)**2.) + 1.)**(0.5)*(R - r)*(Ut/2. - (U*np.cos(psi))/2.)*(U + 
                      Ut*np.cos(psi) + Ua*np.sin(psi)))/((2*Wa)**2.*(1. - np.exp(-(B*(2*Wt)*(R - r))/(r*(Ua + U*np.sin(psi)))))**(0.5)) + 
                      (128.*U*r*np.arccos(piece)*(Ua + U*np.sin(psi))*(Ut/2. - (U*np.cos(psi))/2.)*(U + Ut*np.cos(psi) + 
                      Ua*np.sin(psi)))/(B**3.*np.pi**2.*(Ut + U*np.cos(psi))**3.*((16.*(2*Wa)**2.)/(B**2.*np.pi**2.*(2*Wt)**2.) + 1.)**(0.5))) 
                      
            dpsi = -Rsquiggly/dR_dpsi
            
            psi = psi + dpsi
            diff = abs(psiold-psi)
    
            psiold = psi
    
        Cd       = 0.01385 #From xfoil of the DAE51 at RE=150k, Cl=0.7
        epsilon  = Cd/Cl
        deltar   = (r[1]-r[0])
        thrust   = rho[:,0]*B*np.transpose(np.sum(Gamma*(Wt-epsilon*Wa)*deltar,axis=1))   #T
        torque   = rho[:,0]*B*np.sum(Gamma*(Wa+epsilon*Wt)*r*deltar,axis=1) #Q
        power    = torque*omega       
       
        D        = 2*R
        Cp       = power/(rho[:,0]*(n**3)*(D**5))

        thrust[conditions.propulsion.throttle[:,0]  <=0.0] = 0.0
        power[conditions.propulsion.throttle[:,0]  <=0.0] = 0.0
        
        #etap     = V*thrust/(power)

        return thrust, torque, power, Cp
    
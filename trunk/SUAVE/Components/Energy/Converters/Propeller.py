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
from SUAVE.Core import Units
from SUAVE.Components.Energy.Energy_Component import Energy_Component
from SUAVE.Core import (
Data, Container, Data_Exception, Data_Warning,
)
from warnings import warn

# ----------------------------------------------------------------------
#  Propeller Class
# ----------------------------------------------------------------------    
 
class Propeller(Energy_Component):
    
    def __defaults__(self):
        
        self.prop_attributes = Data
        self.prop_attributes.number_blades      = 0.0
        self.prop_attributes.tip_radius         = 0.0
        self.prop_attributes.hub_radius         = 0.0
        self.prop_attributes.twist_distribution = 0.0
        self.prop_attributes.chord_distribution = 0.0
        
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
        B     = self.prop_attributes.number_blades
        R     = self.prop_attributes.tip_radius
        Rh    = self.prop_attributes.hub_radius
        beta  = self.prop_attributes.twist_distribution
        c     = self.prop_attributes.chord_distribution
        omega = self.inputs.omega
        rho   = conditions.freestream.density[:,0,None]
        mu    = conditions.freestream.dynamic_viscosity[:,0,None]
        V     = conditions.freestream.velocity[:,0,None]
        a     = conditions.freestream.speed_of_sound[:,0,None]
        T     = conditions.freestream.temperature[:,0,None]
        
        nu    = mu/rho
        tol   = 1e-5 # Convergence tolerance
           
        ######
        # Enter airfoil data in a better way, there is currently Re and Ma scaling from DAE51 data
        ######

        #Things that don't change with iteration
        N       = len(c) #Number of stations
        chi0    = Rh/R # Where the propeller blade actually starts
        chi     = np.linspace(chi0,1,N+1) # Vector of nondimensional radii
        chi     = chi[0:N]
        lamda   = V/(omega*R)           # Speed ratio
        r       = chi*R                 # Radial coordinate

        x       = r*np.multiply(omega,1/V)             # Nondimensional distance
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
        psi    = np.ones_like(c)
        psiold = np.zeros_like(c)
        diff   = np.ones_like(c)
        
        ii = 0
        while (np.any(diff>tol)):
            Wa    = 0.5*Ua + 0.5*U*np.sin(psi)
            Wt    = 0.5*Ut + 0.5*U*np.cos(psi)  
            #va    = Wa - Ua
            vt    = Ut - Wt
            alpha = beta - np.arctan2(Wa,Wt)
            W     = np.sqrt(Wa**2. + Wt**2.)
            Re    = (W*c)/nu
            Ma    = (W)/a #a is the speed of sound
            
            if np.any(Ma> 1.0):
                warn('Propeller blade tips are supersonic.', Warning)
            
            lamdaw = r*Wa/(R*Wt)
            f      = (B/2.)*(1.-r/R)/lamdaw
            piece  = np.exp(-f)
            #piece[piece>1] = 1.0
            F      = 2.*np.arccos(piece)/np.pi
            Gamma  = vt*(4.*np.pi*r/B)*F*np.sqrt(1.+(4.*lamdaw*R/(np.pi*B*r))**2.)
            
            # Ok, from the airfoil data, given Re, Ma, alpha we need to find Cl
            Clvals = 2.*np.pi*alpha
            
            Cl     = np.zeros_like(Clvals)
            # Scale for Mach, this is Karmen_Tsien
            Cl[Ma[:,:]<1.] = Clvals[Ma[:,:]<1.]/(np.sqrt(1-Ma[Ma[:,:]<1.]**2)+((Ma[Ma[:,:]<1.]**2)/(1+np.sqrt(1-Ma[Ma[:,:]<1.]**2)))*Clvals[Ma<1.]/2)
            
            # If the blade segments are supersonic, don't scale
            Cl[Ma[:,:]>=1.] = Clvals[Ma[:,:]>=1.] 
            
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
            dR_dpsi[np.isnan(dR_dpsi)] = 0.1
                      
            dpsi   = -Rsquiggly/dR_dpsi
            psi    = psi + dpsi
            diff   = abs(psiold-psi)
            psiold = psi
            
            ii += 1
            if ii>100:
                break
    
        #This is an atrocious fit of DAE51 data at RE=50k for Cd
        #There is also RE scaling
        Cdval = (0.108*(Cl**4)-0.2612*(Cl**3)+0.181*(Cl**2)-0.0139*Cl+0.0278)*((50000./Re)**0.2)
        
        #More Cd scaling from Mach from AA241ab notes for turbulent skin friction
        Tw_Tinf = 1. + 1.78*(Ma**2)
        Tp_Tinf = 1. + 0.035*(Ma**2) + 0.45*(Tw_Tinf-1.)
        Tp      = (Tp_Tinf)*T
        Rp_Rinf = (Tp_Tinf**2.5)*(Tp+110.4)/(T+110.4)
        
        Cd = ((1/Tp_Tinf)*(1/Rp_Rinf)**0.2)*Cdval
        
        epsilon  = Cd/Cl
        deltar   = (r[1]-r[0])
        thrust   = rho*B*(np.sum(Gamma*(Wt-epsilon*Wa)*deltar,axis=1)[:,None])
        torque   = rho*B*np.sum(Gamma*(Wa+epsilon*Wt)*r*deltar,axis=1)[:,None]
        power    = torque*omega       
       
        D        = 2*R
        Cp       = power/(rho*(n**3)*(D**5))


        thrust[omega<0.0] = 0.0
        power[omega<0.0]  = 0.0

        thrust[conditions.propulsion.throttle[:,0] <=0.0] = 0.0
        power[conditions.propulsion.throttle[:,0]  <=0.0] = 0.0

        etap     = V*thrust/(power)
        etap[omega<0.0] = 0.0
        
        #Cp[omega<0.0] = self.prop_attributes.Cp
        
        conditions.propulsion.etap = etap
        
        return thrust, torque, power, Cp
    
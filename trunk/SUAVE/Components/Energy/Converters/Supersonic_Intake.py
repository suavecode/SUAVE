# Supersonic_Nozzle.py
#
# Created:  Aug 2017, P. Goncalves

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import SUAVE

# python imports
from warnings import warn
from scipy.optimize import fsolve

# package imports
import numpy as np

from SUAVE.Components.Energy.Energy_Component import Energy_Component

# ----------------------------------------------------------------------
#  Compression Nozzle Component
# ----------------------------------------------------------------------
def detached_shock_limit(M0, gamma):
    theta_limit = (4/(np.sqrt(3)*3*(gamma+1)))*(M0**2-1)**1.5/M0**2
                  
    return theta_limit

def theta_beta_mach(M0, theta, gamma, delta):
#    func = lambda beta : (np.tan(beta-theta)/np.tan(beta))-((2+(gamma-1)*(M0*np.sin(beta))**2)/((gamma+1)*(M0*np.sin(beta))**2))
#    beta_guess_1 = 0.1
#    beta_guess_2 = np.deg2rad(90) 
#    beta_weak = fsolve(func,beta_guess_1)
#    beta_strong = fsolve(func,beta_guess_2)
    
    l1 = np.sqrt(((M0**2-1)**2)-3*((1+((gamma-1)/2)*M0**2)*(1+((gamma+1)/2)*M0**2))*(np.tan(theta))**2)
    l2 = ((M0**2-1)**3-9*(1+(gamma-1)/2*M0**2)*(1+(gamma-1)/2*M0**2+(gamma+1)/4*M0**4)*(np.tan(theta))**2)/(l1**3)

    beta = np.arctan((M0**2-1+2*l1*np.cos((4*np.pi*delta+np.arccos(l2))/3))/(3*(1+(gamma-1)/2*M0**2)*np.tan(theta)))
    return beta

def oblique_shock_relation(M0, gamma, theta, beta):

    M0_n  = M0*np.sin(beta)
    M1_n  = np.sqrt((1+(gamma-1)/2*M0_n**2)/(gamma*M0_n**2-(gamma-1)/2))
    M1    = M1_n/np.sin(beta-theta)
    P_r   = 1+(2*gamma/(gamma+1))*(M1_n**2-1)
    T_r   = P_r*(((gamma-1)*M1_n**2+2)/((gamma+1)*M1_n**2))
    Pt_r  = (((gamma+1)*(M0*np.sin(beta))**2)/((gamma-1)*(M0*np.sin(beta))**2+2))**(gamma/(gamma-1))*((gamma+1)/(2*gamma*(M0*np.sin(beta))**2-(gamma-1)))**(1/(gamma-1)) 
    
    return M1, T_r, P_r, Pt_r

def normal_shock_relation(M0, gamma):
    M1    = np.sqrt((1+(gamma-1)/2*M0**2)/(gamma*M0**2-(gamma-1)/2))
    P_r   = 1+(2*gamma/(gamma+1))*(M1**2-1)
    T_r   = P_r*(((gamma-1)*M1**2+2)/((gamma+1)*M1**2))
    Pt_r  = ((((gamma+1)*(M0**2))/((gamma-1)*M0**2+2))**(gamma/(gamma-1)))*((gamma+1)/(2*gamma*M0**2-(gamma-1)))**(1/(gamma-1))
    return M1, T_r, P_r, Pt_r


class Supersonic_Intake(Energy_Component):
    """ SUAVE.Components.Energy.Gas_Turbine.Nozzle
        a nozzle component
        
        this class is callable, see self.__call__
        
        """
    
    def __defaults__(self):
        
        #setting the default values 
        self.tag = 'Nozzle'
        self.polytropic_efficiency           = 1.0
        self.pressure_ratio                  = 1.0
        self.theta1 = 1.0
        self.theta2 = 1.0
        self.inputs.stagnation_temperature   = 0.
        self.inputs.stagnation_pressure      = 0.
        self.outputs.stagnation_temperature  = 0.
        self.outputs.stagnation_pressure     = 0.
        self.outputs.stagnation_enthalpy     = 0.
        self.outputs.mach_number             = -1.
    
    def size(self, conditions):
        
        #unpack from conditions
        gamma   = conditions.freestream.isentropic_expansion_factor
        M0      = conditions.freestream.mach_number
        
        
        #unpack from self
        eff =  self.polytropic_efficiency
        Theta1_inlet = self.theta1
        Theta2_inlet = self.theta2
            
        #MIL specification for pressure recovery factor
        Prf = 1.0 * M0 / M0
        eff = 0.96
        if np.any(M0 <= 1):
            Prf = eff     
        else:
            Prf = eff*(1-0.075*(M0-1)**1.35)
            
        
        M_inlet = M0 
        
        i = 10  #initial beta estimate
        firstSolution = True
        threshold = 0.05    #threshold for pressure recovery factor estimation
        
        while (i<60):
            #-- Enter 1st oblique shock
            beta = np.deg2rad(i)
            theta = np.arctan((2/np.tan(beta))*((M0*np.sin(beta))**2-1)/(M0**2*(gamma+np.cos(2*beta))+2))
            
            if theta > 0 :
                #-- Only physically correct solutions
                M1, Tr, Pr, Ptr = oblique_shock_relation(M0,gamma, theta, beta)
                
         
                j=10
                #-- Enter 2nd oblique shock
                while (j<60):
                    beta2 = np.deg2rad(j)
                    theta2 = np.arctan((2/np.tan(beta2))*((M1*np.sin(beta2))**2-1)/(M1**2*(gamma+np.cos(2*beta2))+2))
                    
                    if theta2 > 0 :
                        #-- Only physically correct solutions
                        M2, Tr2, Pr2, Ptr2 = oblique_shock_relation(M1,gamma, theta2, beta2)
                 
                        #-- In case of strong shock
                        if M2 <= 1.0 :
                            Ptr3 = 1.0
                            M3 = M2
                        else:
                            M3, Tr3, Pr3, Ptr3 = normal_shock_relation(M2,gamma)
                            PtrT = Ptr3*Ptr2*Ptr
                            
                        #-- Calculated Total pressure ratio below defined threshold  
                        if (np.abs(Prf-PtrT)/Prf < threshold):
                            
                            #-- Obtain the first valid solution, regardless of output Mach number
                            if firstSolution :
                                print 'T1 ', round(np.rad2deg(beta),3), 'T2 ', round(np.rad2deg(beta2),3), 'M2 ', round(M2,3), 'M3', round(M3,3)
                                error = np.abs(Prf-PtrT)/Prf
                                Ptr_inlet = PtrT
                                Theta1_inlet = theta
                                Theta2_inlet = theta2
                                M_inlet =  M3
                                firstSolution = False
                                
                            #-- Optimization to obtain a lower Mach number entering the combustor
                            if (M3<M_inlet):
                                print 'T1 ', round(np.rad2deg(beta),3), 'T2 ', round(np.rad2deg(beta2),3), 'M2 ', round(M2,3), 'M3', round(M3,3)
                                error = np.abs(Prf-PtrT)/Prf
                                Ptr_inlet = PtrT
                                Theta1_inlet = theta
                                Theta2_inlet = theta2
                                M_inlet=  M3
                            
                    j=j+1
                        
                    
            i=i+1
            
        
            
        if firstSolution == False :
            print 'Erro: ', error
            print 'PTR', Ptr_inlet, ' vs PRF REAL ', Prf
            print 'THETA', np.rad2deg(Theta1_inlet)
            print 'THETA2', np.rad2deg(Theta2_inlet)
            print 'FINAL MACH', M_inlet
        
        else :
            print 'DEU MERDA'
        
        #pack outputs
        self.theta1            = Theta1_inlet
        self.theta2            = Theta2_inlet

        print '++ INSIDE COMPUTE +++++++++'
        self.compute(conditions)
        
        print '+++++++++++++++++++ END SIZE'        
        return
            
        
            
            
    def compute(self,conditions):
        
        #unpack the values
        
        #unpack from conditions
        gamma   = conditions.freestream.isentropic_expansion_factor
        Cp      = conditions.freestream.specific_heat_at_constant_pressure
        Po      = conditions.freestream.pressure
        R       = conditions.freestream.universal_gas_constant
        M0      = conditions.freestream.mach_number
    
        
        #unpack from inpust
        Tt_in   = self.inputs.stagnation_temperature
        Pt_in   = self.inputs.stagnation_pressure
        
        #unpack from self
        pid     =  self.pressure_ratio
        etapold =  self.polytropic_efficiency
        theta1  =  self.theta1
        theta2  =  self.theta2
        
        #Method to compute the output variables

        if theta1 > detached_shock_limit(M0[0],gamma):
            #-- Detached normal shock, apply normal shock equations
            M1, Tr, Pr, Ptr = normal_shock_relation(M0,gamma)
        
            M3 = M1
            PtrT = Ptr
            TrT  = Tr
        
        else:
            beta1 = theta_beta_mach(M0, theta1, gamma,1)
            M1, Tr, Pr, Ptr = oblique_shock_relation(M0,gamma, theta1, beta1)

            if theta2 > detached_shock_limit(M1[0],gamma):
                #-- Detached normal shock, apply normal shock equations
                M2, Tr2, Pr2, Ptr2 = normal_shock_relation(M1,gamma)
            
                M3 = M2
                PtrT = Ptr2*Ptr
                TrT  = Tr2*Tr
            
            else:         
                beta2 = theta_beta_mach(M1,theta2,gamma,1)
                M2, Tr2, Pr2, Ptr2 = oblique_shock_relation(M1,gamma, theta2, beta2)
        
                if M2[0] > 1.0 :
                    M3, Tr3, Pr3, Ptr3 = normal_shock_relation(M2,gamma)
                    PtrT = Ptr3*Ptr2*Ptr
                    TrT  = Tr3*Tr2*Tr
        
                else :
                    M3 = M2
                    PtrT = Ptr2*Ptr
                    TrT  = Tr2*Tr
                    
    

        
        #compute the output Mach number, static quantities and the output velocity
        Mach    = M3
        Pt_out  = Pt_in*PtrT
        Tt_out  = Tt_in
        T_out   = Tt_out/(1+(gamma-1)/2*Mach*Mach)
        h_out   = Cp*T_out
        ht_out  = Cp*Tt_out
        u_out   = np.sqrt(2*(ht_out-h_out))
          
        #pack computed quantities into outputs
        self.outputs.stagnation_temperature  = Tt_out
        self.outputs.stagnation_pressure     = Pt_out
        self.outputs.stagnation_enthalpy     = ht_out
        self.outputs.mach_number             = Mach
        self.outputs.static_temperature      = T_out
        self.outputs.static_enthalpy         = h_out
        self.outputs.velocity                = u_out
    

    __call__ = compute
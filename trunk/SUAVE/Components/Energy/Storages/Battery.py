#Battery.py
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
#  Battery Class
# ----------------------------------------------------------------------    

class Battery(Energy_Component):
    
    def __defaults__(self):
        
        self.type = 'Li-Ion'
        self.Mass_Props.mass = 0.0
        self.CurrentEnergy = 0.0
        self.R0 = 0.07446
        
    def max_energy(self):
        """ The maximum energy the battery can hold
            
            Inputs:
                battery mass
                battery type
               
            Outputs:
                maximum energy the battery can hold
               
            Assumptions:
                This is a simple battery
               
        """
        
        #These need to be fixed
        if self.type=='Li-Ion':
            return 0.90*(10**6)*self.Mass_Props.mass
        
        elif self.type=='Li-Po':
            return 0.90*(10**6)*self.Mass_Props.mass
        
        elif self.type=='Li-S':
            return 500.*3600.*self.Mass_Props.mass        
    
    def energy_calc(self,numerics):
        
        #Unpack
        Ibat  = self.inputs.batlogic.Ibat
        pbat  = self.inputs.batlogic.pbat
        edraw = self.inputs.batlogic.e
        
        Rbat  = self.R0
        I     = numerics.integrate_time
        
        #X value from Mike V.'s battery model
        x = np.divide(self.CurrentEnergy,self.max_energy())[:,0]
        
        #C rate from Mike V.'s battery model
        C = 3600.*pbat/self.max_energy()
        
        f = 1-np.exp(-20*x)-np.exp(-20*(1-x)) #empirical value for discharge
        
        f[x<0.0] = 0.0
        
        R = Rbat*(1+C*f)       #model discharge characteristics based on changing resistance
        Ploss = (Ibat**2)*R       #calculate resistive losses
        eloss = np.dot(I,Ploss)

        self.CurrentEnergy = self.CurrentEnergy[0] + edraw + eloss
        self.CurrentEnergy[self.CurrentEnergy>self.max_energy()] = self.max_energy()
                    
        return  
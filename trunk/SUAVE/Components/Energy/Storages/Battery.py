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
        self.mass_properties.mass = 0.0
        self.CurrentEnergy = 0.0
        self.resistance = 0.0
        
    def max_energy(self):
        """ The maximum energy the battery can hold
            
            Inputs:
                battery mass
                battery type
               
            Outputs:
                maximum energy the battery can hold
               
            Assumptions:
                This is a simple battery, based on the model by:
                AIAA 2012-5045 by Anubhav Datta/Johnson
               
        """
        
        #These need to be fixed
        if self.type=='Li-Ion':
            return 0.90*(10**6)*self.mass_properties.mass
        
        elif self.type=='Li-Po':
            return 0.90*(10**6)*self.mass_properties.mass
        
        elif self.type=='Li-S':
            return 500.*3600.*self.mass_properties.mass        
    
    def energy_calc(self,numerics):
        
        # Unpack
        Ibat  = self.inputs.batlogic.Ibat
        pbat  = self.inputs.batlogic.pbat
        edraw = self.inputs.batlogic.e
        Rbat  = self.resistance
        I     = numerics.integrate_time
        
        # X value
        x = np.divide(self.CurrentEnergy,self.max_energy())[:,0,None]
        
        # C rate from 
        C = 3600.*pbat/self.max_energy()
        
        # Empirical value for discharge
        x[x<-35.] = -35. # Fix x so it doesn't warn
        
        f = 1-np.exp(-20*x)-np.exp(-20*(1-x)) 
        
        f[x<0.0] = 0.0 # Negative f's don't make sense
        
        # Model discharge characteristics based on changing resistance
        R = Rbat*(1+C*f)
        
        #Calculate resistive losses
        Ploss = (Ibat**2)*R
        
        # Energy loss from power draw
        eloss = np.dot(I,Ploss)
        
        # Pack up
        self.CurrentEnergy = self.CurrentEnergy[0] + edraw + eloss
        self.CurrentEnergy[self.CurrentEnergy>self.max_energy()] = self.max_energy()
                    
        return  
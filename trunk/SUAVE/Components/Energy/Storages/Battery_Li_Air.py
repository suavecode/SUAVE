"""Battery_Li_Air.Py: calculates battery discharge losses and mass gain when run """
# M. Vegh
#

# ------------------------------------------------------------
#  Imports 
# ------------------------------------------------------------

from Storage import Storage
from Battery import Battery
import numpy as np
# ------------------------------------------------------------
#  Battery
# ------------------------------------------------------------
    
class Battery_Li_Air(Storage):
    """ SUAVE.Attributes.Components.Energy.Storage.Battery_Li_Air()
    Models for a Lithium Air Battery
    """
    def __defaults__(self):
      
        self.MassDensity = 1000.0     # kg/m^3
        self.SpecificEnergy = 2000.0  # W-hr/kg
        self.SpecificPower = .66      # kW/kg
        self.MaxPower=0.0             # W
        self.TotalEnergy = 0.0        # J
        self.CurrentEnergy=0.0        # J
        self.Volume = 0.0             # m^3
        self.R0=.07446                #base resistance (ohms)
        
    def __call__(self,power, t, Icur=90):
        """
        Inputs:
        power=power requirements for the time step [W]
        t=time step to calculate energy consumption in the battery [s]
        
        Reads:
        power
        t
    
        Returns:
        Ploss= additional discharge power losses from the battery [W]
        mdot= mass gain rate of the lithium-air battery (if applicable) [W]
        """
    
        x=np.divide(self.CurrentEnergy,self.TotalEnergy)
        
        if self.MaxPower<power:
            print "Warning, battery not powerful enough"          
        C=3600*power/self.TotalEnergy        #C rate of the power output
        Eloss_ideal=power*t                  #ideal power losses
        f=1-np.exp(-20*x)-np.exp(-20*(1-x))  #empirical value for discharge
        
        if x<0:                              #reduce discharge losses when model no longer makes sense
            f=0
        R=self.R0*(1+C*f)                    #model discharge characteristics based on changing resistance
        Ploss=(Icur**2)*R                    #calculate resistive losses
        if power!=0:
            self.CurrentEnergy=self.CurrentEnergy-Eloss_ideal-Ploss*t
            
        if self.CurrentEnergy<0:
            print 'Warning, battery out of energy'
       
        #model taken from EADS Voltair Paper
        mdot=-(power+Ploss) *(1.92E-4)*(1./3600.)      #weight gain of battery (positive means mass loss)
        return Ploss, mdot

    def find_mass_gain(self):               #find total potential mass gain of a lithium air battery
        mgain=self.TotalEnergy*(1.92E-4)*(1./3600.)
        return mgain

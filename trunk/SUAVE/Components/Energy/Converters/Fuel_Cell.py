#Fuel_Cell.py
# 
# Created:  Michael Vegh, Apr 2015
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
from SUAVE.Attributes.Gases import Steam, Air
from SUAVE.Attributes.Propellants import Gaseous_H2
from SUAVE.Methods.Power.Fuel_Cell.Discharge import zero_fidelity
# ----------------------------------------------------------------------
#  Fuel_Cell Class
# ----------------------------------------------------------------------
class Fuel_Cell(Energy_Component):
    
    def __defaults__(self):
        self.propellant     = Gaseous_H2()
        self.oxidizer       = Air()
        self.efficiency     = .65                                 #normal fuel cell operating efficiency at sea level
        self.specific_power = 2.08        *Units.kW/Units.kg      #specific power of fuel cell [kW/kg]; default is Nissan 2011 level
        self.mass_density   = 1203.208556 *Units.kg/Units.m**3.   #take default as specs from Nissan 2011 fuel cell            
        self.volume         = 0.0
        self.max_power      = 0.0
        self.discharge_model= zero_fidelity
        
    def energy_calc(self,conditions,numerics):
        mdot=self.discharge_model(self, conditions, numerics)
        return mdot  

    
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
from SUAVE.Structure import (
Data, Container, Data_Exception, Data_Warning,
)
from warnings import warn

# ----------------------------------------------------------------------
#  Propeller Class
# ----------------------------------------------------------------------    
 
class Propeller_Lo_Fid(Energy_Component):
    
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
           
        # Unpack    
        B     = self.prop_attributes.number_blades
        R     = self.prop_attributes.tip_radius
        Rh    = self.prop_attributes.hub_radius
        beta  = self.prop_attributes.twist_distribution
        c     = self.prop_attributes.chord_distribution
        etap  = self.propulsive_efficiency
        omega = self.inputs.omega
        Qm    = self.inputs.torque
        rho   = conditions.freestream.density[:,0,None]
        mu    = conditions.freestream.viscosity[:,0,None]
        V     = conditions.freestream.velocity[:,0,None]
        a     = conditions.freestream.speed_of_sound[:,0,None]
        T     = conditions.freestream.temperature[:,0,None]
        
        # Do very little calculations
        power  = Qm*omega
        n      = omega/(2.*np.pi) 
        D      = 2*R
        
        thrust = etap*power/V
        
        Cp     = power/(rho*(n**3)*(D**5))
        conditions.propulsion.etap = etap
        
        return thrust, Qm, power, Cp
    
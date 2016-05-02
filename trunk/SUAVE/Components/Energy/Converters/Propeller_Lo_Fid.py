# Propeller_Lo_Fid.py
#
# Created:  Jun 2014, E. Botero
# Modified: Jan 2016, T. MacDonald

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
import SUAVE

# package imports
import numpy as np
from SUAVE.Components.Energy.Energy_Component import Energy_Component
from warnings import warn

# ----------------------------------------------------------------------
#  Propeller Class
# ----------------------------------------------------------------------    
 
class Propeller_Lo_Fid(Energy_Component):
    
    def __defaults__(self):
        
        self.tip_radius            = 0.0
        self.propulsive_efficiency = 0.0

        
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
        R     = self.tip_radius
        etap  = self.propulsive_efficiency
        omega = self.inputs.omega
        Qm    = self.inputs.torque
        rho   = conditions.freestream.density[:,0,None]
        mu    = conditions.freestream.dynamic_viscosity[:,0,None]
        V     = conditions.freestream.velocity[:,0,None]
        a     = conditions.freestream.speed_of_sound[:,0,None]
        T     = conditions.freestream.temperature[:,0,None]
        
        # Do very little calculations
        power  = Qm*omega
        n      = omega/(2.*np.pi) 
        D      = 2*R
        
        thrust = etap*power/V
        
        Cp     = power/(rho*(n*n*n)*(D*D*D*D*D))
        conditions.propulsion.etap = etap
        
        return thrust, Qm, power, Cp
    
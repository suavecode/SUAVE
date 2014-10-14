# Gas_Turbine.py
#
# Created:  Anil, July 2014

#--put in a folder

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUAVE imports

import SUAVE

from SUAVE.Structure import Data
from SUAVE.Attributes import Units

# python imports
import os, sys, shutil
from copy import deepcopy
from warnings import warn

# package imports
import numpy as np
import scipy as sp

from SUAVE.Structure import (
                             Data, Container, Data_Exception, Data_Warning,
                             )

from SUAVE.Structure import Data, Data_Exception, Data_Warning
from SUAVE.Components import Component, Physical_Component, Lofted_Body
from SUAVE.Components.Energy.Energy_Component import Energy_Component
from SUAVE.Components import Component_Exception
#from SUAVE.Components.Energy.Gas_Turbine import Network
from SUAVE.Components.Propulsors.Propulsor import Propulsor

def fm_id(M):
    
    R=287.87
    g=1.4
    m0=(g+1)/(2*(g-1))
    m1=((g+1)/2)**m0
    m2=(1+(g-1)/2*M**2)**m0
    fm=m1*M/m2
    return fm


class Expansion_Nozzle(Energy_Component):
    """ SUAVE.Components.Energy.Gas_Turbine.Nozzle
        a nozzle component
        
        this class is callable, see self.__call__
        
        """
    
    def __defaults__(self):
        
        
        self.tag = 'Nozzle'
        
        self.polytropic_efficiency = 1.0
        self.pressure_ratio = 1.0
        
        self.inputs.stagnation_temperature=0.
        self.inputs.stagnation_pressure=0.
        
        self.outputs.stagnation_temperature=0.
        self.outputs.stagnation_pressure=0.
        self.outputs.stagnation_enthalpy=0.
    
    
    
    def compute(self,conditions):
        
        #unpack the variables
        gamma=conditions.freestream.gamma
        Cp = conditions.freestream.Cp
        Po = conditions.freestream.pressure
        Pto = conditions.freestream.stagnation_pressure
        Tto = conditions.freestream.stagnation_temperature
        R = conditions.freestream.R
        Mo =  conditions.freestream.mach_number
        Tt_in = self.inputs.stagnation_temperature
        Pt_in = self.inputs.stagnation_pressure
        pid = self.pressure_ratio
        etapold = self.polytropic_efficiency
        
        
        #Computing the output modules
        
        #--Getting the outptu stagnation quantities
        Pt_out=Pt_in*pid
        Tt_out=Tt_in*pid**((gamma-1)/(gamma)*etapold)
        ht_out=Cp*Tt_out
        
        
        #compute the output Mach number, static quantities and the output velocity
        Mach=np.sqrt((((Pt_out/Po)**((gamma-1)/gamma))-1)*2/(gamma-1))
        T_out=Tt_out/(1+(gamma-1)/2*Mach**2)
        h_out=Cp*T_out
        u_out=np.sqrt(2*(ht_out-h_out))
        
        i_low = Mach < 1.0
        i_high = Mach>=1.0
        
        P_out = 1.0 *Mach/Mach
        
        
        
        P_out[i_low]=Po[i_low]
        Mach[i_low]=np.sqrt((((Pt_out[i_low]/Po[i_low])**((gamma-1)/gamma))-1)*2/(gamma-1))
        
        
        Mach[i_high]=1.0*Mach[i_high]/Mach[i_high]
        P_out[i_high]=Pt_out[i_high]/(1+(gamma-1)/2*Mach[i_high]**2)**(gamma/(gamma-1))
        
        
        #print 'P_out array' ,P_out
        
        #if np.linalg.norm(Mach) < 1.0:
        ## nozzle unchoked
        
        #P_out=Po
        
        #Mach=np.sqrt((((Pt_out/Po)**((gamma-1)/gamma))-1)*2/(gamma-1))
        
        
        #else:
        #Mach=1.0
        #P_out=Pt_out/(1+(gamma-1)/2*Mach**2)**(gamma/(gamma-1))
        #print 'pout current',P_out
        
        #
        T_out=Tt_out/(1+(gamma-1)/2*Mach**2)
        h_out=Cp*T_out
        u_out=np.sqrt(2*(ht_out-h_out))
        rho_out=P_out/(R*T_out)
        
        
        area_ratio=(fm_id(Mo)/fm_id(Mach)*(1/(Pt_out/Pto))*(np.sqrt(Tt_out/Tto)))
        
        
        
        
        #pack outputs
        self.outputs.stagnation_temperature=Tt_out
        self.outputs.stagnation_pressure=Pt_out
        self.outputs.stagnation_enthalpy=ht_out
        self.outputs.mach_number = Mach
        self.outputs.static_temperature = T_out
        self.outputs.static_enthalpy = h_out
        self.outputs.velocity = u_out
        self.outputs.static_pressure = P_out
        self.outputs.area_ratio = area_ratio
    
    
    
    
    
    
    __call__ = compute
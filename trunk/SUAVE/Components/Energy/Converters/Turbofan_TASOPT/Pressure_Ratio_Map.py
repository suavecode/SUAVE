# Pressure_Ratio_Map.py
#
# Created:  Jul 2016, T. MacDonald
# Modified:

# Built from TASOPT maps, based on Anil's TASOPT code

# SUAVE imports

import SUAVE

# package imports
import numpy as np

from SUAVE.Core import Data

class Pressure_Ratio_Map(Data):
    
    def __defaults__(self):
        self.pressure_ratio = 1.7
        self.a              = 3.0
        self.b              = 0.85
        self.k              = 0.03
        
    def compute_speed(self,pi,md):
        a = self.a
        b = self.b
        k = self.k
        #pi = self.pressure_ratio
        piD = self.design_pressure_ratio
        mD  = self.design_mass_flow
        #md = self.inputs.mass_flow
        
        mb = md/mD
        
        Nd = self.Nd
        mb0 = mD
        
        pb = (pi-1.)/(piD-1.)
        R = 1.0 # base residual
        Nb = 0.5*np.ones(pb.shape)# N tilde
        dN = 1.e-8
        
        while (np.linalg.norm(R)>1e-8):
            
            ms = Nb**b
            ps = ms**a  
            
            Nb_1 = Nb*(1.+dN)
            ms_1 = Nb_1**b
            ps_1 = ms_1**a
            
            R  = ms + k*(1. - np.exp((pb - ps)/(2.*Nb*k))) - mb
            R1 = ms_1 + k*(1. - np.exp((pb - ps_1)/(2.*Nb_1*k))) - mb
            dR = (R1-R)/(dN*Nb)            
    
            delN = -R/dR #-R/dR
            Nb += delN
            
        grad_n = 0.5/Nb*np.exp((pb - ps)/(2.*Nb*k)) 
        grad_d = b*Nb**(b-1.0) + 0.5/Nb**2.0*np.exp((pb - ps)/(2.*Nb*k))*(a*b*Nb**(a*b) + pb - ps)
    
        dN_dpi = Nd*(grad_n/grad_d)/(piD-1.0)
        dN_dm  = Nd/(mb0*grad_d)
    
    
    
        N_corrected = Nb*Nd
    
        return N_corrected,dN_dpi,dN_dm        
    
    
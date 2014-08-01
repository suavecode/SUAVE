# tail.py
# 
# Created:  Emilio Botero, Jun 2014
# Modified:

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Attributes import Units as Units
import numpy as np
from SUAVE.Structure import (
    Data, Container, Data_Exception, Data_Warning,
)

def tail(Sts,bts,cts,Ntsr,t_cts,qm):      
    
    """ weight = SUAVE.Methods.Weights.Correlations.Solar_HPA_weights.tail(Sts,bts,cts,deltats,Ntsr,t_cts)     
        
        Inputs:
            Sts -      tail surface area (m)
            bts -      tail surface span (m)
            cts -      average tail surface chord (m)
            deltats -  average rib spacing to average chord ratio
            Ntsr -     number of tail surface ribs (bts^2)/(deltats*Sts)
            t_cts -    tail airfoil thickness to chord ratio
            qm -       dynamic pressure at maneuvering speed (N/m2)
    
        Outputs:
            Wtss -     weight of tail surface spar (kg)
            Wtsr -     weight of tail surface ribs (kg)
            WtsLE -    weight of tail surface leading edge (kg)
            Wtsc -     weight of tail surface covering (kg)
                
        Assumptions:
            All of this is from AIAA 89-2048, units are in kg. These weight estimates
            are from the MIT Daedalus and are valid for very lightweight
            carbon fiber composite structures. This may need to be solved iteratively since
            gross weight is an input.
            
    """    
    deltats = (bts**2)/(Sts*Ntsr)
    
    #Rudder & Elevator Primary Structure:
    Wtss = (bts * 4.15e-2 + (bts**2) * 3.91e-3) * (1.0 + ((qm * Sts)/78.5 - 1.0)/12.0)
    
    #Rudder & Elevator Secondary Structure:
    Wtsr = Ntsr * (cts**2 * t_cts * 1.16e-1 + cts * 4.01e-3)
    Wts=0.174*((Sts**2)*(deltats**(4./3.))/bts)
    Wtsc=Sts * 1.93e-2
    
    weight = Wtss + Wtsr + Wts + Wtsc
    
    return weight
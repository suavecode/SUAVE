# wing.py
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

def wing(Sw,bw,cw,Nwr,t_cw,Nwer,nult,GW):
    """ weight = SUAVE.Methods.Weights.Correlations.Solar_HPA_weights.wing(Sw,bw,cw,deltaw,nwr,t_cw,Nwer,nult,Gw)     
        
        Inputs:
            Sw -       wing area [m^2]
            bw -       wing span [m]
            cw -       average wing chord [m]
            deltaw -   average rib spacing to average chord ratio
            Nwr -      number of wing or tail surface ribs (bw^2)/(deltaw*Sw)
            t_cw -     wing airfoil thickness to chord ratio  
            Nwer -     number of wing end ribs (2*number of individual wing panels -2)
            nult -     ultimate load factor
            GW -       aircraft gross weight
    
        Outputs:
            Wws -      weight of wing spar (kg)
            Wwr -      weight of wing ribs (kg)
            Wwer -     weight of wing end ribs (kg)
            WwLE -     weight of wing leading edge (kg)
            WwTE -     weight of wing trailing edge (kg)
            Wwc -      weight of wing covering (kg)
                
        Assumptions:
            All of this is from AIAA 89-2048, units are in kg. These weight estimates
            are from the MIT Daedalus and are valid for very lightweight
            carbon fiber composite structures. This may need to be solved iteratively since
            gross weight is an input.
            
    """
    
    deltaw = (bw**2)/(Sw*Nwr)
    
    #Wing One Wire Main Spar:
    Wws    = (bw * (3.10e-2) + (7.56e-3) * (bw**2)) * (1.0 + (nult * GW /100.0 - 2.0) / 4.0)
    
    #Wing Secondary Structure:
    Wwr    = Nwr * ((cw**2) * t_cw * 5.50e-2 + cw * 1.91e-3)
    Wwer   = Nwer * ((cw**2) * t_cw * 6.62e-1 + cw * 6.57e-3)
    WwLE   = 0.456 * ((Sw**2)*(deltaw**(4./3.))/bw)
    WwTE   = bw * 2.77e-2
    Wwc    = Sw * 3.08e-2
    
    print(Wws)
    
    weight = Wws + Wwr + Wwer + WwLE + WwTE + Wwc
    
    return weight
#fuselage.py
# 
# Created:  Emilio Botero, Jun 2014
# Modified:

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Attributes import Units as Units
import numpy as np
from SUAVE.Core import (
    Data, Container, Data_Exception, Data_Warning,
)

def fuselage(Sts,qm,Ltb):
    
    """ weight = SUAVE.Methods.Weights.Correlations.Solar_HPA_weights.fuselage(Sts,qm,Ltb): 
            
            Inputs:
                Sts -      tail surface area (m)
                qm -       dynamic pressure at maneuvering speed (N/m2)
                Ltb -      tailboom length (m)
        
            Outputs:
                Wtb -      tailboom weight (kg)
                    
            Assumptions:
                All of this is from AIAA 89-2048, units are in kg. These weight estimates
                are from the MIT Daedalus and are valid for very lightweight
                carbon fiber composite structures. This may need to be solved iteratively since
                gross weight is an input.
                
        """    
    
    #Fuselage:
    Wtb=(Ltb*1.14e-1 +(1.96e-2)*(Ltb**2))*(1.0+((qm*Sts)/78.5-1.0)/2.0)
    
    return Wtb
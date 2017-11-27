## @ingroup Methods-Weights-Correlations-Human_Powered 
# fuselage.py
# 
# Created:  Jun 2014, E. Botero
# Modified: Feb 2016, E. Botero

# ----------------------------------------------------------------------
#  Fuselage
# ----------------------------------------------------------------------

## @ingroup Methods-Weights-Correlations-Human_Powered
def fuselage(Sts,qm,Ltb):   
    """ Compute weifht estimate of human-powered aircraft fuselage 
    
    Assumptions:
       All of this is from AIAA 89-2048, units are in kg. These weight estimates
       are from the MIT Daedalus and are valid for very lightweight
       carbon fiber composite structures. This may need to be solved iteratively since
       gross weight is an input.
       
    Source: 
        MIT Daedalus
        
    Inputs:
        Sts -      tail surface area                     [meters]
        qm -       dynamic pressure at maneuvering speed [Pascals]
        Ltb -      tailboom length                       [meters]
    
    Outputs:
        Wtb -      tailboom weight                       [kilogram]
        
    Properties Used:
        N/A
    """ 
    #Fuselage:
    Wtb=(Ltb*1.14e-1 +(1.96e-2)*(Ltb**2))*(1.0+((qm*Sts)/78.5-1.0)/2.0)
    
    return Wtb
## @ingroup Methods-Weights-Correlations-Human_Powered
# tail.py
# 
# Created:  Jun 2014, E. Botero
# Modified: Feb 2016, E. Botero

# ----------------------------------------------------------------------
#  Tail
# ----------------------------------------------------------------------

## @ingroup Methods-Weights-Correlations-Human_Powered
def tail(Sts,bts,cts,Ntsr,t_cts,qm):        
    """ Compute weight of human-powered aircraft tail     
    
    Assumptions:
       All of this is from AIAA 89-2048, units are in kg. These weight estimates
       are from the MIT Daedalus and are valid for very lightweight
       carbon fiber composite structures. This may need to be solved iteratively since
       gross weight is an input.
       
    Source: 
        MIT Daedalus
                
    Inputs:
        Sts -      tail surface area                                [meters]
        bts -      tail surface span                                [meters]
        cts -      average tail surface chord                       [meters]
        deltats -  average rib spacing to average chord ratio       [dimensionless]
        Ntsr -     number of tail surface ribs (bts^2)/(deltats*Sts)[dimensionless]
        t_cts -    tail airfoil thickness to chord ratio            [dimensionless]
        qm -       dynamic pressure at maneuvering speed            [Pascals]
    
    Outputs:
        Wtss -     weight of tail surface spar                      [kilogram]
        Wtsr -     weight of tail surface ribs                      [kilogram]
        WtsLE -    weight of tail surface leading edge              [kilogram]
        Wtsc -     weight of tail surface covering                  [kilogram]
            
    Properties Used:
        N/A
    """     
    deltats = (bts**2)/(Sts*Ntsr)
    
    #Rudder & Elevator Primary Structure:
    Wtss = (bts * 4.15e-2 + (bts**2) * 3.91e-3) * (1.0 + ((qm * Sts)/78.5 - 1.0)/12.0)
    
    #Rudder & Elevator Secondary Structure:
    Wtsr = Ntsr * (cts**2 * t_cts * 1.16e-1 + cts * 4.01e-3)
    Wts  = 0.174*((Sts**2)*(deltats**(4./3.))/bts)
    Wtsc = Sts * 1.93e-2
    
    weight = Wtss + Wtsr + Wts + Wtsc
    
    return weight
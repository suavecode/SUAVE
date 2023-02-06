## @ingroup Methods-Weights-Correlations-Human_Powered
# wing.py
# 
# Created:  Jun 2014, E. Botero
# Modified: Feb 2016, E. Botero

# ----------------------------------------------------------------------
#  Wing
# ----------------------------------------------------------------------

## @ingroup Methods-Weights-Correlations-Human_Powered
def wing(Sw,bw,cw,Nwr,t_cw,Nwer,nult,GW):
    """ Compute weight of human-powered aircraft wing 
    
    Assumptions:
       All of this is from AIAA 89-2048, units are in kg. These weight estimates
       are from the MIT Daedalus and are valid for very lightweight
       carbon fiber composite structures. This may need to be solved iteratively since
       gross weight is an input.
       
    Source: 
        MIT Daedalus
        
    Inputs:
        Sw -       wing area                                                       [meters**2]
        bw -       wing span                                                       [meters]
        cw -       average wing chord                                              [meters]
        eltaw -   average rib spacing to average chord ratio                       [dimensionless]
        Nwr -      number of wing or tail surface ribs (bw^2)/(deltaw*Sw)          [dimensionless]
        t_cw -     wing airfoil thickness to chord ratio                           [dimensionless]
        Nwer -     number of wing end ribs (2*number of individual wing panels -2) [dimensionless]
        nult -     ultimate load factor                                            [dimensionless]
        GW -       aircraft gross weight                                           [kilogram]
    
    Outputs:
        Wws -      weight of wing spar                                             [kilogram]
        Wwr -      weight of wing ribs                                             [kilogram]
        Wwer -     weight of wing end ribs                                         [kilogram]
        WwLE -     weight of wing leading edge                                     [kilogram]
        WwTE -     weight of wing trailing edge                                    [kilogram]
        Wwc -      weight of wing covering                                         [kilogram]

    Properties Used:
        N/A
    """ 
    
    deltaw = (bw**2)/(Sw*Nwr)
    
    #Wing One Wire Main Spar:
    #Wws    = (bw * (3.10e-2) + (7.56e-3) * (bw**2)) * (1.0 + (nult * GW /100.0 - 2.0) / 4.0)
    
    #Wing Cantilever Main Spar:
    Wws    = (bw * (1.17e-1) + (1.1e-2) * (bw**2)) * (1.0 + (nult * GW /100.0 - 2.0) / 4.0)    
    
    #Wing Secondary Structure:
    Wwr    = Nwr * ((cw**2) * t_cw * 5.50e-2 + cw * 1.91e-3)
    Wwer   = Nwer * ((cw**2) * t_cw * 6.62e-1 + cw * 6.57e-3)
    WwLE   = 0.456 * ((Sw**2)*(deltaw**(4./3.))/bw)
    WwTE   = bw * 2.77e-2
    Wwc    = Sw * 3.08e-2
    
    weight = Wws + Wwr + Wwer + WwLE + WwTE + Wwc
    
    return weight
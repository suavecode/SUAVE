# empty.py
# 
# Created:  Emilio Botero, Jun 2014
# Modified:  

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUave imports
import SUAVE

# package imports
import numpy as np
import tail as tail
import wing as wing
import fuselage as fuselage

from SUAVE.Attributes import Units as Units
from SUAVE.Structure import (
    Data, Container, Data_Exception, Data_Warning,
    )

def empty(vehicle):
    """ weight = SUAVE.Methods.Weights.Correlations.Solar_HPA_weights.empty(wing,aircraft,horizontal,vertical): 
            
        Inputs:
            wing - a data dictionary with the fields:
                Sw -       wing area [m**2]
                bw -       wing span [m]
                cw -       average wing chord [m]
                deltaw -   average rib spacing to average chord ratio
                Nwr -      number of wing surface ribs (bw**2)/(deltaw*Sw)
                t_cw -     wing airfoil thickness to chord ratio
                Nwer -     number of wing end ribs (2*number of individual wing panels -2)
                
            horizontal - a data dictionary with the fields:
                Sts -      tail surface area (m)
                bts -      tail surface span (m)
                cts -      average tail surface chord (m)
                deltawts - average rib spacing to average chord ratio
                Ntsr -     number of tail surface ribs (bts^2)/(deltats*Sts)
                t_cts -    tail airfoil thickness to chord ratio
                
            vertical - a data dictionary with the fields:
                Sts -      tail surface area (m)
                bts -      tail surface span (m)
                cts -      average tail surface chord (m)
                deltawts - average rib spacing to average chord ratio
                Ntsr -     number of tail surface ribs (bts**2)/(deltats*Sts)
                t_cts -    tail airfoil thickness to chord ratio
                
            aircraft - a data dictionary with the fields:    
                nult -     ultimate load factor
                GW -       aircraft gross weight
                qm -       dynamic pressure at maneuvering speed (N/m2)
                Ltb -      tailboom length (m)
        
            Outputs:
                Wws -      weight of wing spar (kg)
                Wtss -     weight of tail surface spar (kg)
                Wwr -      weight of wing ribs (kg)
                Wtsr -     weight of tail surface ribs (kg)
                Wwer -     weight of wing end ribs (kg)
                WwLE -     weight of wing leading edge (kg)
                WtsLE -    weight of tail surface leading edge (kg)
                WwTE -     weight of wing trailing edge (kg)
                Wwc -      weight of wing covering (kg)
                Wtsc -     weight of tail surface covering (kg)
                Wtb -      tailboom weight (kg)
                    
            Assumptions:
                All of this is from AIAA 89-2048, units are in kg. These weight estimates
                are from the MIT Daedalus and are valid for very lightweight
                carbon fiber composite structures. This may need to be solved iteratively since
                gross weight is an input.
                
        """
    
    #Unpack

    Sw     = vehicle.Wings['Main Wing'].sref
    bw     = vehicle.Wings['Main Wing'].span
    cw     = vehicle.Wings['Main Wing'].chord_mac
    Nwr    = vehicle.Wings['Main Wing'].Nwr
    t_cw   = vehicle.Wings['Main Wing'].t_c
    Nwer   = vehicle.Wings['Main Wing'].Nwer
    
    #S_h    = vehicle.Wings['Horizontal Stabilizer'].sref
    #b_h    = vehicle.Wings['Horizontal Stabilizer'].span
    #chs    = vehicle.Wings['Horizontal Stabilizer'].chord_mac
    #Nhsr   = vehicle.Wings['Horizontal Stabilizer'].Nwr
    #t_ch   = vehicle.Wings['Horizontal Stabilizer'].t_c
    
    #S_v    = vehicle.Wings['Vertical Stabilizer'].sref
    #b_v    = vehicle.Wings['Vertical Stabilizer'].span
    #cvs    = vehicle.Wings['Vertical Stabilizer'].chord_mac
    #Nvsr   = vehicle.Wings['Vertical Stabilizer'].Nwr
    #t_cv   = vehicle.Wings['Vertical Stabilizer'].t_c
    
    nult   = vehicle.Ultimate_Load
    gw     = vehicle.Mass_Props.m_full
    qm     = vehicle.qm
    #Ltb    = vehicle.Ltb    
    
    #Wing weight
    wt_wing = wing.wing(Sw,bw,cw,Nwr,t_cw,Nwer,nult,gw)

    ##Horizontal weight  
    #wt_ht   = tail.tail(S_h,b_h,chs,Nhsr,t_ch,qm)
    
    ##Vertical weight
    #wt_vt   = tail.tail(S_v,b_v,cvs,Nvsr,t_cv,qm)
    
    ##Fuselage weight
    #wt_tb   = fuselage.fuselage(S_h,qm,Ltb)
    
    vehicle.Wings['Main Wing'].Mass_Props.mass             = wt_wing
    #vehicle.Wings['Horizontal Stabilizer'].Mass_Props.mass = wt_ht
    #vehicle.Wings['Vertical Stabilizer'].Mass_Props.mass   = wt_vt
    #vehicle.Fuselages.Fuselage.Mass_Props.mass = wt_tb
    
    weight                 = Data()
    weight.wing            = wt_wing
    #weight.fuselage        = wt_tb
    #weight.horizontal_tail = wt_ht
    #weight.vertical_tail   = wt_vt
    
    return weight
    
    
    
    
    
    
    
    
    
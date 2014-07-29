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

def empty(mainwing,aircraft,horizontal,vertical):
    """ weight = SUAVE.Methods.Weights.Correlations.Solar_HPA_weights.empty(wing,aircraft,horizontal,vertical): 
            
        Inputs:
            wing - a data dictionary with the fields:
                Sw -       wing area [m^2]
                bw -       wing span [m]
                cw -       average wing chord [m]
                deltaw -   average rib spacing to average chord ratio
                Nwr -      number of wing surface ribs (bw^2)/(deltaw*Sw)
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
                Ntsr -     number of tail surface ribs (bts^2)/(deltats*Sts)
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
    Sw     = mainwing.sref
    bw     = mainwing.span
    cw     = mainwing.mac #close enough
    deltaw = mainwing.deltaw
    Nwr    = mainwing.Nwr
    t_cw   = mainwing.t_c
    Nwer   = mainwing.Nwer
    
    S_h    = horizontal.area
    b_h    = horizontal.span
    chs    = horizontal.mac
    deltah = horizontal.deltah
    Nhsr   = horizontal.Nwr
    t_ch   = horizontal.t_c
    
    S_v    = vertical.area
    b_v    = vertical.span
    cvs    = vertical.mac
    deltav = vertical.deltah
    Nvsr   = vertical.Nwr
    t_cv   = vertical.t_c
    
    nult   = aircraft.nult
    gw     = aircraft.gw
    qm     = aircraft.qm
    Ltb    = aircraft.Ltb

    #Wing weight
    wt_wing = wing.wing(Sw,bw,cw,deltaw,Nwr,t_cw,Nwer,nult,gw)

    #Horizontal weight  
    wt_ht   = tail.tail(S_h,b_h,chs,deltah,Nhsr,t_ch,qm)
    
    #Vertical weight
    wt_vt   = tail.tail(S_v,b_v,cvs,deltav,Nvsr,t_cv,qm)
    
    #Fuselage weight
    wt_tb   = fuselage.fuselage(S_h,qm,Ltb)
    
    weight                 =  Data()
    weight.wing            = wt_wing
    weight.fuselage        = wt_tb
    weight.horizontal_tail = wt_ht
    weight.vertical_tail   = wt_vt
    
    return weight
    
    
    
    
    
    
    
    
    
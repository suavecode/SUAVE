## @ingroup Methods-Weights-Correlations-Human_Powered
# empty.py
# 
# Created:  Jun 2014, E. Botero
# Modified: Feb 2016, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from . import tail as tail
from . import wing as wing
from . import fuselage as fuselage
import warnings

from SUAVE.Core import Data

# ----------------------------------------------------------------------
#  Empty
# ----------------------------------------------------------------------

## @ingroup Methods-Weights-Correlations-Human_Powered
def empty(vehicle,settings=None):
    """ Computes weights estimates for human powered aircraft
    
    Assumptions:
       All of this is from AIAA 89-2048, units are in kg. These weight estimates
       are from the MIT Daedalus and are valid for very lightweight
       carbon fiber composite structures. This may need to be solved iteratively since
       gross weight is an input.
       
    Source: 
        MIT Daedalus
                       
    Inputs:
        wing - a data dictionary with the fields:
            Sw -       wing area                                                       [meters**2]
            bw -       wing span                                                       [meters]
            cw -       average wing chord                                              [meters]
            deltaw -   average rib spacing to average chord ratio                      [dimensionless]
            Nwr -      number of wing surface ribs (bw**2)/(deltaw*Sw)                 [dimensionless]
            t_cw -     wing airfoil thickness to chord ratio                           [dimensionless]
            Nwer -     number of wing end ribs (2*number of individual wing panels -2) [dimensionless]
            
        horizontal - a data dictionary with the fields:
            Sts -      tail surface area                                               [meters]
            bts -      tail surface span                                               [meters]
            cts -      average tail surface chord                                      [meters]
            deltawts - average rib spacing to average chord ratio                      [dimensionless]
            Ntsr -     number of tail surface ribs (bts^2)/(deltats*Sts)               [dimensionless]
            t_cts -    tail airfoil thickness to chord ratio                           [dimensionless]
            
        vertical - a data dictionary with the fields:
            Sts -      tail surface area                                               [meters]
            bts -      tail surface span                                               [meters]
            cts -      average tail surface chord                                      [meters]
            deltawts - average rib spacing to average chord ratio                      [dimensionless]
            Ntsr -     number of tail surface ribs (bts**2)/(deltats*Sts)              [dimensionless]
            t_cts -    tail airfoil thickness to chord ratio                           [dimensionless]
            
        aircraft - a data dictionary with the fields:    
            nult -     ultimate load factor                                            [dimensionless]
            GW -       aircraft gross weight                                           [kilogram]
            qm -       dynamic pressure at maneuvering speed                           [Pascals]
            Ltb -      tailboom length                                                 [meters]
    
    Outputs:
        Wws -      weight of wing spar                                                 [kilogram]
        Wtss -     weight of tail surface spar                                         [kilogram]
        Wwr -      weight of wing ribs                                                 [kilogram]
        Wtsr -     weight of tail surface ribs                                         [kilogram]
        Wwer -     weight of wing end ribs                                             [kilogram]
        WwLE -     weight of wing leading edge                                         [kilogram]
        WtsLE -    weight of tail surface leading edge                                 [kilogram]
        WwTE -     weight of wing trailing edge                                        [kilogram]
        Wwc -      weight of wing covering                                             [kilogram]
        Wtsc -     weight of tail surface covering                                     [kilogram]
        Wtb -      tailboom weight                                                     [kilogram]
                
    Properties Used:
        N/A
    """ 
    
    #Unpack
    
    nult   = vehicle.envelope.ultimate_load
    gw     = vehicle.mass_properties.max_takeoff
    qm     = vehicle.envelope.maximum_dynamic_pressure
    
    # Wing weight
    if 'main_wing' not in vehicle.wings:
        wt_wing = 0.0
        warnings.warn("There is no Wing Weight being added to the Configuration", stacklevel=1)
    else:
        Sw      = vehicle.wings['main_wing'].areas.reference
        bw      = vehicle.wings['main_wing'].spans.projected
        cw      = vehicle.wings['main_wing'].chords.mean_aerodynamic
        Nwr     = vehicle.wings['main_wing'].number_ribs
        t_cw    = vehicle.wings['main_wing'].thickness_to_chord
        Nwer    = vehicle.wings['main_wing'].number_end_ribs
        wt_wing = wing.wing(Sw,bw,cw,Nwr,t_cw,Nwer,nult,gw)
        vehicle.wings['main_wing'].mass_properties.mass = wt_wing
    
    # Horizontal Tail weight
    if 'horizontal_stabilizer' not in vehicle.wings:
        wt_ht = 0.0
        warnings.warn("There is no Horizontal Tail Weight being added to the Configuration", stacklevel=1)
    else:      
        S_h    = vehicle.wings['horizontal_stabilizer'].areas.reference
        b_h    = vehicle.wings['horizontal_stabilizer'].spans.projected
        chs    = vehicle.wings['horizontal_stabilizer'].chords.mean_aerodynamic
        Nhsr   = vehicle.wings['horizontal_stabilizer'].number_ribs
        t_ch   = vehicle.wings['horizontal_stabilizer'].thickness_to_chord
        wt_ht  = tail.tail(S_h,b_h,chs,Nhsr,t_ch,qm)
        vehicle.wings['horizontal_stabilizer'].mass_properties.mass = wt_ht
    
    # Vertical Tail weight
    if 'vertical_stabilizer' not in vehicle.wings:   
        wt_vt = 0.0
        warnings.warn("There is no Vertical Tail Weight being added to the Configuration", stacklevel=1)    
    else:    
        S_v    = vehicle.wings['vertical_stabilizer'].areas.reference
        b_v    = vehicle.wings['vertical_stabilizer'].spans.projected
        cvs    = vehicle.wings['vertical_stabilizer'].chords.mean_aerodynamic
        Nvsr   = vehicle.wings['vertical_stabilizer'].number_ribs
        t_cv   = vehicle.wings['vertical_stabilizer'].thickness_to_chord
        wt_vt   = tail.tail(S_v,b_v,cvs,Nvsr,t_cv,qm)
        vehicle.wings['vertical_stabilizer'].mass_properties.mass = wt_vt

    # Fuselage weight
    if 'fuselage' not in vehicle.fuselages:   
        wt_tb = 0.0
        warnings.warn("There is no Fuselage Weight being added to the Configuration", stacklevel=1)    
    else: 
        Ltb     = vehicle.Ltb  
        wt_tb   = fuselage.fuselage(S_h,qm,Ltb)
        vehicle.Fuselages['fuselage'].mass_properties.mass = wt_tb
    
    weight                 = Data()
    weight.wing            = wt_wing
    weight.fuselage        = wt_tb
    weight.horizontal_tail = wt_ht
    weight.vertical_tail   = wt_vt
    
    weight.empty = wt_ht + wt_tb + wt_vt + wt_wing
    
    return weight
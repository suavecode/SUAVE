## @ingroup Methods-Weights-Correlations-BWB
# systems.py
# 
# Created:  Jan 2014, A. Wendorff
# Modified: Jul 2014, A. Wendorff
#           Feb 2016, E. Botero     

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Units, Data

# ----------------------------------------------------------------------
#   Systems
# ----------------------------------------------------------------------

## @ingroup Methods-Weights-Correlations-BWB
def systems(num_seats,  ctrl_type,S_h,S_gross_w, ac_type):
    """ Calculate the weight of the different engine systems on the aircraft
    
    Assumptions:
        numbers based on FAA regulations and correlations from previous aircraft
                
    Source:
        N/A  
    
    Inputs:
        num_seats - total number of seats on the aircraft                                                   [dimensionless]
        ctrl_type - specifies if the control system is fully power, partially powered, or not powered       [dimensionless]
        S_h - area of the BWB outer aileron                                                                 [meters**2]
        S_gross_w - area of the wing                                                                        [meters**2]
        ac_type - determines type of instruments, electronics, and operating items based on type of vehicle [dimensionless]
    
    Outputs:
        output - a data dictionary with fields:
            wt_flt_ctrl - weight of the flight control system                                               [kilograms]
            wt_apu - weight of the apu                                                                      [kilograms]
            wt_hyd_pnu - weight of the hydraulics and pneumatics                                            [kilograms]
            wt_instruments - weight of the instruments and navigational equipment                           [kilograms]
            wt_avionics - weight of the avionics                                                            [kilograms]
            wt_opitems - weight of the optional items based on the type of aircraft                         [kilograms]
            wt_elec - weight of the electrical items                                                        [kilograms]
            wt_ac - weight of the air conditioning and anti-ice system                                      [kilograms]
            wt_furnish - weight of the furnishings in the fuselage                                          [kilograms]
        
    Properties Used:
    N/A
    """ 
    # unpack inputs
    sref   = S_gross_w / Units.ft**2 # Convert meters squared to ft squared
    area_h = S_h / Units.ft**2 # Convert meters squared to ft squared
    
    # process
    # Flight Controls Group Wt
    if ctrl_type == "fully powered":       #fully powered controls 
        flt_ctrl_scaler = 3.5
    elif ctrl_type == "partially powered":     #partially powered controls
        flt_ctrl_scaler = 2.5
    else:
        flt_ctrl_scaler = 1.7 # fully aerodynamic controls
    flt_ctrl_wt = (flt_ctrl_scaler*area_h) * Units.lb

    # APU Group Wt   
    if num_seats >= 6.:
        apu_wt = 7.0 * num_seats *Units.lb
    else:
        apu_wt = 0.0 * Units.lb #no apu if less than 9 seats
    apu_wt = max(apu_wt,70.)
    # Hydraulics & Pneumatics Group Wt
    hyd_pnu_wt = (0.65 * sref) * Units.lb
     
    # Electrical Group Wt
    elec_wt = (13.0 * num_seats) * Units.lb

    # Furnishings Group Wt
    furnish_wt = ((43.7 - 0.037*min(num_seats,300.))*num_seats + 46.0*num_seats) * Units.lb
       
    # Environmental Control
    ac_wt = (15.0 * num_seats) * Units.lb

    # Instruments, Electronics, Operating Items based on Type of Vehicle
             
    if ac_type == "short-range": # short-range domestic, austere accomodations
        instruments_wt = 800.0 * Units.lb
        avionics_wt    = 900.0 * Units.lb
        opitems_wt     = 17.0  * num_seats * Units.lb
    elif ac_type == "medium-range": #medium-range domestic
        instruments_wt = 800.0 * Units.lb
        avionics_wt    = 900.0 * Units.lb
        opitems_wt     = 28.0  * num_seats * Units.lb          
    elif ac_type == "long-range": #long-range overwater
        instruments_wt = 1200.0 * Units.lb
        avionics_wt    = 1500.0 * Units.lb
        opitems_wt     = 28.0   * num_seats * Units.lb
        furnish_wt    += 23.0   * num_seats * Units.lb #add aditional seat wt                         
    elif ac_type == "business": #business jet
        instruments_wt = 100.0 * Units.lb
        avionics_wt    = 300.0 * Units.lb
        opitems_wt     = 28.0  * num_seats * Units.lb                     
    elif ac_type == "cargo": #all cargo
        instruments_wt = 800.0  * Units.lb
        avionics_wt    = 900.0  * Units.lb
        elec_wt        = 1950.0 * Units.lb # for cargo a/c  
        opitems_wt     = 56.0   * Units.lb                     
    elif ac_type == "commuter": #commuter
        instruments_wt = 300.0 * Units.lb
        avionics_wt    = 500.0 * Units.lb
        opitems_wt     = 17.0  * num_seats * Units.lb                        
    elif ac_type == "sst": #sst
        instruments_wt = 1200.0 * Units.lb 
        avionics_wt    = 1500.0 * Units.lb
        opitems_wt     = 40.0   * num_seats * Units.lb
        furnish_wt    += 23.0   * num_seats * Units.lb #add aditional seat wt                  
    else:
        instruments_wt = 800.0 * Units.lb 
        avionics_wt    = 900.0 * Units.lb 
        opitems_wt     = 28.0  * num_seats * Units.lb
    
    # packup outputs
    output = Data()   
    output.wt_flt_ctrl    = flt_ctrl_wt
    output.wt_apu         = apu_wt 
    output.wt_hyd_pnu     = hyd_pnu_wt
    output.wt_instruments = instruments_wt
    output.wt_avionics    = avionics_wt
    output.wt_opitems     = opitems_wt
    output.wt_elec        = elec_wt
    output.wt_ac          = ac_wt
    output.wt_furnish     = furnish_wt    
    output.wt_systems     = output.wt_flt_ctrl + output.wt_apu + output.wt_hyd_pnu \
                            + output.wt_ac + output.wt_avionics + output.wt_elec \
                            + output.wt_furnish + output.wt_instruments + output.wt_opitems
    
    return output
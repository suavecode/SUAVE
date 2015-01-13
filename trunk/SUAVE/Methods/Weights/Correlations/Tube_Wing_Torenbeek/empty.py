# empty.py
# 
# Created:  Andrew Wendorff, May 2014
# Modified: 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import SUAVE
import numpy as np
from wing_structure import wing_structure

from SUAVE.Attributes import Units as Units
from SUAVE.Core import (
    Data, Container, Data_Exception, Data_Warning,
)

# ----------------------------------------------------------------------
#  Method
# ----------------------------------------------------------------------

def empty(engine,wing,aircraft,fuselage,horizontal,vertical):
    """ output = SUAVE.Methods.Weights.Correlations.Tube_Wing.empty(engine,wing,aircraft,fuselage,horizontal,vertical)
        This is for a standard Tube and Wing aircraft configuration.        
        
        Inputs:
            engine - a data dictionary with the fields:                    
                thrust_sls - sea level static thrust of a single engine [Newtons]
            
            wing - a data dictionary with the fields:
                gross_area - wing gross area [meters**2]
                span - span of the wing [meters]
                taper - taper ratio of the wing [dimensionless]
                t_c - thickness-to-chord ratio of the wing [dimensionless]
                sweep - sweep angle of the wing [radians]
                mac - mean aerodynamic chord of the wing [meters]
                r_c - wing root chord [meters]
                
            aircraft - a data dictionary with the fields:                    
                Nult - ultimate load of the aircraft [dimensionless]
                Nlim - limit load factor at zero fuel weight of the aircraft [dimensionless]
                TOW - maximum takeoff weight of the aircraft [kilograms]
                zfw - maximum zero fuel weight of the aircraft [kilograms]
                num_eng - number of engines on the aircraft [dimensionless]
                num_eng_wing - number of wing-mounted engines [dimensionless]
                num_pax - number of passengers on the aircraft [dimensionless]
                wt_cargo - weight of the bulk cargo being carried on the aircraft [kilograms]
                num_seats - number of seats installed on the aircraft [dimensionless]
                ctrl - specifies if the control system is "fully powered", "partially powered", or not powered [dimensionless]
                ac - determines type of instruments, electronics, and operating items based on types: 
                    "short-range", "medium-range", "long-range", "business", "cargo", "commuter", "sst" [dimensionless]
                w2h - tail length (distance from the airplane c.g. to the horizontal tail aerodynamic center) [meters]
                
            fuselage - a data dictionary with the fields:
                area - fuselage wetted area [meters**2]
                diff_p - Maximum fuselage pressure differential [Pascal]
                width - width of the fuselage [meters]
                height - height of the fuselage [meters]
                length - length of the fuselage [meters]                     
            
            horizontal
                area - area of the horizontal tail [meters**2]
                span - span of the horizontal tail [meters]
                sweep - sweep of the horizontal tail [radians]
                mac - mean aerodynamic chord of the horizontal tail [meters]
                t_c - thickness-to-chord ratio of the horizontal tail [dimensionless]
                exposed - exposed area ratio for the horizontal tail [dimensionless]
            
            vertical
                area - area of the vertical tail [meters**2]
                span - sweight = weight * Units.lbpan of the vertical [meters]
                t_c - thickness-to-chord ratio of the vertical tail [dimensionless]
                sweep - sweep angle of the vertical tail [radians]
                t_tail - factor to determine if aircraft has a t-tail, "yes" [dimensionless]
    
        Outputs:
            output - a data dictionary with fields:
                wt_payload - weight of the passengers plus baggage and paid cargo [kilograms]
                wt_pax - weight of all the passengers [kilogram]
                wt_bag - weight of all the baggage [kilogram]
                wt_fuel - weight of the fuel carried[kilogram]
                wt_empty - operating empty weight of the aircraft [kilograms]
                
        Assumptions:
            
    """
    
    # Unpack inputs
    S_gross_w  = wing.gross_area
    b          = wing.span
    lambda_w   = wing.taper
    t_c_w      = wing.t_c
    sweep_w    = wing.sweep
    mac_w      = wing.mac
    wing_c_r   = wing.c_r
    sweep_LE   = wing.sweep_LE
    sweep_half = wing.sweep_half
    n_s        = wing.n_s
    braced     = wing.braced
    S_f        = wing.S_f
    b_f        = wing.b_f
    d_f        = wing.d_f
    sweep_f    = wing.sweep_f
    t_c_f      = wing.t_c_f
    flap_type  = wing.flap_type
    variable_flap = wing.variable
    
    Nult          = aircraft.Nult
    Nlim          = aircraft.Nlim
    TOW           = aircraft.TOW
    wt_zf         = aircraft.zfw
    num_eng       = aircraft.num_eng
    num_pax       = aircraft.num_pax
    wt_cargo      = aircraft.wt_cargo
    num_seats     = aircraft.num_seats
    ctrl_type     = aircraft.ctrl
    ac_type       = aircraft.ac  
    l_w2h         = aircraft.w2h    
    eng_wing      = aircraft.num_eng_wing
    V_D           = aircraft.V_D
    W_Wing        = aircraft.weight.wing    
    V_stall       = aircraft.V_stall
    undercarriage = aircraft.undercarriage
      
    # Calculating Empty Weight of Aircraft 
    output = wing_structure(S_gross_w,b,lambda_w,t_c_w,Nult,TOW,eng_wing,sweep_LE,sweep_half,V_D,n_s,W_Wing,braced,S_f,b_f,V_stall,d_f,sweep_f,t_c_f,flap_type,variable_flap,undercarriage)
    
    # packup outputs
    
    return output
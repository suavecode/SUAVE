# empty.py
# 
# Created:  Andrew Wendorff, Jan 2014
# Modified: Andrew Wendorff, Feb 2014 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import SUAVE
import numpy as np
from tube import tube
from landing_gear import landing_gear
from payload import payload
from systems import systems
from tail_horizontal import tail_horizontal
from tail_vertical import tail_vertical
from wing_main import wing_main
from SUAVE.Methods.Weights.Correlations import Propulsion as Propulsion
from SUAVE.Attributes import Units as Units
from SUAVE.Structure import (
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
            calculated aircraft weight from correlations created per component of historical aircraft
    """     

    # Unpack inputs
    thrust_sls = engine.thrust_sls
    
    S_gross_w  = wing.gross_area
    b          = wing.span
    lambda_w   = wing.taper
    t_c_w      = wing.t_c
    sweep_w    = wing.sweep
    mac_w      = wing.mac
    wing_c_r   = wing.c_r
    
    Nult       = aircraft.Nult
    Nlim       = aircraft.Nlim
    TOW        = aircraft.TOW
    wt_zf      = aircraft.zfw
    num_eng    = aircraft.num_eng
    num_pax    = aircraft.num_pax
    wt_cargo   = aircraft.wt_cargo
    num_seats  = aircraft.num_seats
    ctrl_type  = aircraft.ctrl
    ac_type    = aircraft.ac  
    l_w2h      = aircraft.w2h
    
    S_fus      = fuselage.area
    diff_p_fus = fuselage.diff_p
    w_fus      = fuselage.width
    h_fus      = fuselage.height
    l_fus      = fuselage.length
    
    S_h            = horizontal.area
    b_h            = horizontal.span
    sweep_h        = horizontal.sweep
    mac_h          = horizontal.mac
    t_c_h          = horizontal.t_c
    h_tail_exposed = horizontal.exposed
    
    S_v        = vertical.area
    b_v        = vertical.span
    t_c_v      = vertical.t_c
    sweep_v    = vertical.sweep
    t_tail     = vertical.t_tail     

    # process
    # Calculating Empty Weight of Aircraft
    wt_wing            = wing_main(S_gross_w,b,lambda_w,t_c_w,sweep_w,Nult,TOW,wt_zf)
    wt_engine_jet      = Propulsion.engine_jet(thrust_sls)
    wt_landing_gear    = landing_gear(TOW)
    wt_propulsion      = Propulsion.integrated_propulsion(wt_engine_jet,num_eng)
    wt_fuselage        = tube(S_fus, diff_p_fus,w_fus,h_fus,l_fus,Nlim,wt_zf,wt_wing,wt_propulsion, wing_c_r)
    output_2           = systems(num_seats, ctrl_type, S_h, S_v, S_gross_w, ac_type)
    wt_tail_horizontal = tail_horizontal(b_h,sweep_h,Nult,S_h,TOW,mac_w,mac_h,l_w2h,t_c_h, h_tail_exposed)
    output_3           = tail_vertical(S_v,Nult,b_v,TOW,t_c_v,sweep_v,S_gross_w,t_tail)

    # Calculate the equipment empty weight of the aircraft
    wt_empty           = (wt_wing + wt_fuselage + wt_landing_gear + wt_propulsion + output_2.wt_systems + \
                          wt_tail_horizontal + output_3.wt_tail_vertical + output_3.wt_rudder) 
    
    # packup outputs
    output             = payload(TOW, wt_empty, num_pax,wt_cargo)
    
    return output

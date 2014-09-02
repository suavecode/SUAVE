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

def empty(vehicle):
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
    thrust_sls = vehicle.Propulsors['Turbo Fan'].Thrust.design
    
    S_gross_w  = vehicle.reference_area
    #S_gross_w  = vehicle.Wings['Main Wing'].Areas.reference
    b          = vehicle.Wings['Main Wing'].Spans.projected
    lambda_w   = vehicle.Wings['Main Wing'].taper
    t_c_w      = vehicle.Wings['Main Wing'].thickness_to_chord
    sweep_w    = vehicle.Wings['Main Wing'].sweep
    mac_w      = vehicle.Wings['Main Wing'].Chords.mean_aerodynamic
    wing_c_r   = vehicle.Wings['Main Wing'].Chords.root
    
    Nult       = vehicle.Envelope.ultimate_load
    Nlim       = vehicle.Envelope.limit_load
    TOW        = vehicle.Mass_Properties.max_takeoff
    wt_zf      = vehicle.Mass_Properties.max_zero_fuel
    num_eng    = vehicle.Propulsors['Turbo Fan'].number_of_engines
    num_pax    = vehicle.passengers
    wt_cargo   = vehicle.Mass_Properties.cargo
    num_seats  = vehicle.Fuselages.Fuselage.number_coach_seats
    ctrl_type  = vehicle.Systems.control
    ac_type    = vehicle.Systems.accessories  
    l_w2h      = vehicle.Wings['Horizontal Stabilizer'].position[0] + vehicle.Wings['Horizontal Stabilizer'].aerodynamic_center[0] - vehicle.Wings['Main Wing'].position[0] - vehicle.Wings['Main Wing'].aerodynamic_center[0] #Need to check this is the length of the horizontal tail moment arm
    
    S_fus      = vehicle.Fuselages.Fuselage.Areas.wetted
    diff_p_fus = vehicle.Fuselages.Fuselage.differential_pressure
    w_fus      = vehicle.Fuselages.Fuselage.width
    h_fus      = vehicle.Fuselages.Fuselage.Heights.maximum
    l_fus      = vehicle.Fuselages.Fuselage.Lengths.total
    
    S_h            = vehicle.Wings['Horizontal Stabilizer'].Areas.reference
    b_h            = vehicle.Wings['Horizontal Stabilizer'].Spans.projected
    sweep_h        = vehicle.Wings['Horizontal Stabilizer'].sweep
    mac_h          = vehicle.Wings['Horizontal Stabilizer'].Chords.mean_aerodynamic
    t_c_h          = vehicle.Wings['Horizontal Stabilizer'].thickness_to_chord
    h_tail_exposed = vehicle.Wings['Horizontal Stabilizer'].Areas.exposed / vehicle.Wings['Horizontal Stabilizer'].Areas.wetted
    
    S_v        = vehicle.Wings['Vertical Stabilizer'].Areas.reference
    b_v        = vehicle.Wings['Vertical Stabilizer'].Spans.projected
    t_c_v      = vehicle.Wings['Vertical Stabilizer'].thickness_to_chord
    sweep_v    = vehicle.Wings['Vertical Stabilizer'].sweep
    t_tail     = vehicle.Wings['Vertical Stabilizer'].t_tail     

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
    
    vehicle.Wings['Main Wing'].Mass_Properties.mass = wt_wing
    vehicle.Wings['Horizontal Stabilizer'].Mass_Properties.mass = wt_tail_horizontal
    vehicle.Wings['Vertical Stabilizer'].Mass_Properties.mass = output_3.wt_tail_vertical + output_3.wt_rudder
    vehicle.Fuselages.Fuselage.Mass_Properties.mass = wt_fuselage
    vehicle.Propulsors['Turbo Fan'].Mass_Properties.mass = wt_engine_jet
    
    # packup outputs
    output             = payload(TOW, wt_empty, num_pax,wt_cargo)
    
    output.wing              = wt_wing
    output.fuselage          = wt_fuselage
    output.propulsion        = wt_propulsion
    output.landing_gear      = wt_landing_gear
    output.systems           = output_2.wt_systems
    output.wt_furnish        = output_2.wt_furnish
    output.horizontal_tail   = wt_tail_horizontal
    output.vertical_tail     = output_3.wt_tail_vertical
    output.rudder            = output_3.wt_rudder    
    
    return output
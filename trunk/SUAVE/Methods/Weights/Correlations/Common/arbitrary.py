## @ingroup Methods-Weights-Correlations-Common
# arbitrary.py
# 
# Created:  Nov 2018, E. Botero 
# Modified: 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import SUAVE
from SUAVE.Core import Data
#from .cabin          import cabin
#from .aft_centerbody import aft_centerbody
#from .systems        import systems
from SUAVE.Methods.Weights.Correlations.Tube_Wing.tail_horizontal import tail_horizontal
from SUAVE.Methods.Weights.Correlations.Tube_Wing.tail_vertical import tail_vertical
from SUAVE.Methods.Weights.Correlations.Tube_Wing.tube import tube
from SUAVE.Methods.Weights.Correlations.Common import wing_main as wing_main
from SUAVE.Methods.Weights.Correlations.Common import landing_gear as landing_gear
from SUAVE.Methods.Weights.Correlations.Common import payload as payload
from SUAVE.Methods.Weights.Correlations.Common.systems import systems
from SUAVE.Methods.Weights.Correlations import Propulsion as Propulsion
import SUAVE.Components.Energy.Networks as Nets
import SUAVE.Components.Wings as Wings

# ----------------------------------------------------------------------
#  Empty
# ----------------------------------------------------------------------


def arbitrary(vehicle,settings=None):
    """ This is for an arbitrary aircraft configuration. 
    
    Assumptions:
         N/A
      
    Source: 
        N/A
         
    Inputs:
        N/A
        
    Outputs:
        output - a data dictionary with fields:
            wt_payload - weight of the passengers plus baggage and paid cargo                              [kilograms]
            wt_pax - weight of all the passengers                                                          [kilogram]
            wt_bag - weight of all the baggage                                                             [kilogram]
            wt_fuel - weight of the fuel carried                                                           [kilogram]
            wt_empty - operating empty weight of the aircraft                                              [kilograms]
    
    Properties Used:
    N/A    
    """
    
    # Unpack inputs
    Nult       = vehicle.envelope.ultimate_load
    Nlim       = vehicle.envelope.limit_load
    TOW        = vehicle.mass_properties.max_takeoff
    wt_zf      = vehicle.mass_properties.max_zero_fuel
    wt_cargo   = vehicle.mass_properties.cargo
    num_pax    = vehicle.passengers
    ctrl_type  = vehicle.systems.control
    ac_type    = vehicle.systems.accessories
    S_gross_w  = vehicle.reference_area
    
    # Set the factors
    if settings == None:
        wt_factors = Data()
        wt_factors.main_wing = 0.
        wt_factors.empennage = 0.
        wt_factors.fuselage  = 0.
    else:
        wt_factors = settings.weight_reduction_factors   
        
    # Prime the totals
    wt_main_wing       = 0.0
    wt_tail_horizontal = 0.0
    wt_vtail_tot       = 0.0
    wt_propulsion      = 0.0
    wt_fuselage        = 0.0
    wt_rudder          = 0.0
    s_tail             = 0.0
    
    # Work through the propulsions systems
    for prop in vehicle.propulsors:
        if isinstance(prop,Nets.Turbofan) or isinstance(prop,Nets.Turbojet_Super):
            num_eng                   = prop.number_of_engines
            thrust_sls                = prop.sealevel_static_thrust
            wt_engine_jet             = Propulsion.engine_jet(thrust_sls)
            wt_prop                   = Propulsion.integrated_propulsion(wt_engine_jet,num_eng)
            prop.mass_properties.mass = wt_prop
            
            wt_propulsion             += wt_prop
    
    # Go through all the wings
    for wing in vehicle.wings:
        
        # Common Wing Parameters
        S     = wing.areas.reference
        mac   = wing.chords.mean_aerodynamic
        b     = wing.spans.projected
        t_c   = wing.thickness_to_chord
        sweep = wing.sweeps.quarter_chord
        
        # Main Wing
        if isinstance(wing,Wings.Main_Wing):

            # Unpack main wing specific parameters
            lambda_w = wing.taper
            
            # Calculate the weights
            wt_wing  = wing_main.wing_main(S_gross_w,b,lambda_w,t_c,sweep,Nult,TOW,wt_zf)
            
            # Apply weight factor
            wt_wing  = wt_wing*(1.-wt_factors.main_wing)
            
            # Pack and sum
            wing.mass_properties.mass = wt_wing
            wt_main_wing += wt_main_wing
            
        # Horizontal Tail    
        if isinstance(wing,Wings.Horizontal_Tail):
            
            # Unpack horizontal tail specific parameters
            h_tail_exposed = wing.areas.exposed / wing.areas.wetted
            l_w2h          = wing.origin[0] + wing.aerodynamic_center[0] - vehicle.wings['main_wing'].origin[0] - vehicle.wings['main_wing'].aerodynamic_center[0]
            mac_w          = vehicle.wings['main_wing'].chords.mean_aerodynamic
            
            # Calculate the weights
            wt_horiz = tail_horizontal(b,sweep,Nult,S,TOW,mac_w,mac,l_w2h,t_c, h_tail_exposed)   
            
            # Apply weight factor
            wt_horiz = wt_horiz*(1.-wt_factors.empennage)
            
            # Pack and sum
            wing.mass_properties.mass = wt_horiz
            wt_tail_horizontal += wt_horiz
            s_tail             += S
            
            
        # Vertical Tail
        if isinstance(wing,Wings.Vertical_Tail):       
            
            # Unpack vertical tail specific parameters
            t_tail = wing.t_tail  
            
            # Calculate the weights
            tv     = tail_vertical(S,Nult,b,TOW,t_c,sweep,S_gross_w,t_tail)
            
            # Apply weight factor
            wt_rud      = tv.wt_rudder*(1.-wt_factors.empennage)
            wt_vert_str = tv.wt_tail_vertical*(1.-wt_factors.empennage)
            
            # Pack and sum
            wt_vertical = wt_rud + wt_vert_str
            wing.mass_properties.mass = wt_vertical         
            wt_vtail_tot += wt_vertical
            wt_rudder    += wt_rud
            s_tail       += S
            
    # Fuselages
    for fuse in vehicle.fuselages:
        
        # Unpack
        S_fus      = fuse.areas.wetted
        diff_p_fus = fuse.differential_pressure
        w_fus      = fuse.width
        h_fus      = fuse.heights.maximum
        l_fus      = fuse.lengths.total
        wing_c_r   = vehicle.wings.main_wing.chords.root
        
        #
        wt_fuse = tube(S_fus,diff_p_fus,w_fus,h_fus,l_fus,Nlim,wt_zf,wt_wing,wt_propulsion,wing_c_r) 
        wt_fuse = wt_fuse*(1.-wt_factors.fuselage)
        fuse.mass_properties.mass = wt_fuse
        
        wt_fuselage += wt_fuse 
        
    # Landing Gear
    wt_landing_gear = landing_gear.landing_gear(TOW)   
        
    # Systems
    output_2        = systems(num_pax, ctrl_type, s_tail, S_gross_w, ac_type)
    
    # Calculate the equipment empty weight of the aircraft
    wt_empty        = (wt_wing + wt_fuselage + wt_landing_gear + wt_propulsion + output_2.wt_systems + \
                          wt_tail_horizontal + wt_vtail_tot)     

    # packup outputs
    output                   = payload.payload(TOW, wt_empty, num_pax,wt_cargo)
    output.wing              = wt_wing
    output.fuselage          = wt_fuselage
    output.propulsion        = wt_propulsion
    output.landing_gear      = wt_landing_gear
    output.horizontal_tail   = wt_tail_horizontal
    output.vertical_tail     = wt_vtail_tot
    output.rudder            = wt_rudder
    output.systems           = output_2.wt_systems       
    output.systems_breakdown = Data()
    output.systems_breakdown.control_systems   = output_2.wt_flt_ctrl    
    output.systems_breakdown.apu               = output_2.wt_apu         
    output.systems_breakdown.hydraulics        = output_2.wt_hyd_pnu     
    output.systems_breakdown.instruments       = output_2.wt_instruments 
    output.systems_breakdown.avionics          = output_2.wt_avionics    
    output.systems_breakdown.optionals         = output_2.wt_opitems     
    output.systems_breakdown.electrical        = output_2.wt_elec        
    output.systems_breakdown.air_conditioner   = output_2.wt_ac          
    output.systems_breakdown.furnish           = output_2.wt_furnish    
    
    #define weights components
    try: 
        landing_gear_component=vehicle.landing_gear #landing gear previously defined
    except AttributeError: # landing gear not defined
        landing_gear_component=SUAVE.Components.Landing_Gear.Landing_Gear()
        vehicle.landing_gear=landing_gear_component
    
    control_systems   = SUAVE.Components.Physical_Component()
    electrical_systems= SUAVE.Components.Physical_Component()
    passengers        = SUAVE.Components.Physical_Component()
    furnishings       = SUAVE.Components.Physical_Component()
    air_conditioner   = SUAVE.Components.Physical_Component()
    fuel              = SUAVE.Components.Physical_Component()
    apu               = SUAVE.Components.Physical_Component()
    hydraulics        = SUAVE.Components.Physical_Component()
    optionals         = SUAVE.Components.Physical_Component()
    rudder            = SUAVE.Components.Physical_Component()
    avionics          = SUAVE.Components.Energy.Peripherals.Avionics()
    
    #assign output weights to objects
    landing_gear_component.mass_properties.mass                      = output.landing_gear
    control_systems.mass_properties.mass                             = output.systems_breakdown.control_systems
    electrical_systems.mass_properties.mass                          = output.systems_breakdown.electrical
    passengers.mass_properties.mass                                  = output.pax + output.bag
    furnishings.mass_properties.mass                                 = output.systems_breakdown.furnish
    avionics.mass_properties.mass                                    = output.systems_breakdown.avionics \
        + output.systems_breakdown.instruments                  
    air_conditioner.mass_properties.mass                             = output.systems_breakdown.air_conditioner
    fuel.mass_properties.mass                                        = output.fuel
    apu.mass_properties.mass                                         = output.systems_breakdown.apu
    hydraulics.mass_properties.mass                                  = output.systems_breakdown.hydraulics
    optionals.mass_properties.mass                                   = output.systems_breakdown.optionals
    rudder.mass_properties.mass                                      = output.rudder
    
    #assign components to vehicle
    vehicle.control_systems                     = control_systems
    vehicle.electrical_systems                  = electrical_systems
    vehicle.avionics                            = avionics
    vehicle.furnishings                         = furnishings
    vehicle.passenger_weights                   = passengers 
    vehicle.air_conditioner                     = air_conditioner
    vehicle.fuel                                = fuel
    vehicle.apu                                 = apu
    vehicle.hydraulics                          = hydraulics
    vehicle.optionals                           = optionals
    vehicle.landing_gear                        = landing_gear_component

    return output    
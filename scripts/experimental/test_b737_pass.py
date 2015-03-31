# A B737-800 SUave Example
# J. Sinsay
# Created 4/5/2013

# Write into the Vehicle Database directly to get started w/o File I/O
# Future File I/O will read from an .av file

import SUAVE

def create_aerovehicle():
    
    #=============================================================================
    #    Vehicle Input Parameters (to be replaced by File I/O in the future)
    #=============================================================================
    
    mainWing = SUAVE.Components.Wings.Wing(
        tag = 'main_wing',
        symmetric          = True,         #new SUAVE variable (vertical tail is false)
        type_wt            = 'pass_wing',  #new SUAVE variable (identify component weight equation)
        ref_area           = 1344.0,       #sq-ft      #PASS name:"sref"     #IS SREF REALLY A WING PARAMETER, EXAMPLE BIPLANE?
        aspect_ratio       = 10.19,        #PASS name:"arw"
        sweep              = 25.0,         #PASS name:"sweepw"
        taper              = 0.159,        #PASS name:"taperw"
        dihedral           = 6.0,          #PASS name:"wingdihedral"
        lex                = 0.1,          #PASS name:"lex"
        tex                = 0.21,         #PASS name:"tex"
        span_chordext      = 0.33,         #PASS name:"chordextspan"
        wing_zND           = 0.0,          #PASS name:"wingheight"
        wing_xND           = 0.35,         #PASS name:"wingxposition"
        airfoil_thickness  = 0.1,          #PASS name:"tovercw"
        airfoil_type       = 1,            #PASS name:"supercritical?"
        airfoil_transition = 0.03,         #PASS name:"x/ctransition"
        fspan_flap         = 0.6,          #PASS name:"flapspan/b"
        fchord_flap        = 0.32659602,   #PASS name:"flapchord/c"          
    )
    
    horzTail = SUAVE.Components.Wings.Wing(
        tag = 'Horizontal Tail',
        symmetric         = True,          #new SUAVE variable (vertical tail is false)
        type_wt           = 'pass_htail',  #new SUAVE variable (identify component weight equation)
        fSref             = 0.2625305,     #PASS name:"sh/sref"
        aspect_ratio      = 6.16,          #PASS name:"arh"
        sweep             = 30.0,          #PASS name:"sweeph"
        airfoil_thickness = 0.08,          #PASS name:"toverch"
        airfoil_clmax     = 1.2,           #PASS name:"clhmax"
        taper             = 0.4,           #PASS name:"taperh"
        dihedral          = 7.0,           #PASS name:"dihedralh"
    )
    
    vertTail = SUAVE.Components.Wings.Wing( 
        tag = 'Vertical Tail',
        symmetric         = False,         #new SUAVE variable (vertical tail is false)
        type_wt           = 'pass_vtail',  #new SUAVE variable 
        fSref             = 0.2117543,     #PASS name:"sv/sref"
        aspect_ratio      = 1.91,          #PASS name:"arv"
        sweep             = 35.0,          #PASS name:"sweepv"
        airfoil_thickness = 0.08,          #PASS name:"tovercv"
        taper             = 0.25,          #PASS name:"taperv"
        ttail             = 0.0,           #PASS name:"ttail?"
    )
    
    fuselage = SUAVE.Components.Fuselages.Fuselage( 
        tag = 'fuselage',
        num_coach_seats   = 160,   #PASS name:"#coachseats"
        seat_layout_lower = 33,    #PASS name:"seatlayout1"
        seat_width        = 18.5,  # in      #PASS name:"seatwidth"
        seat_layout_upper = 0,     #PASS name:"seatlayout2"
        aisle_width       = 20.0,  # in     #PASS name:"aislewidth"
        seat_pitch        = 32.0,  # in      #PASS name:"seatpitch"
        cabin_alt         = 7000,  # ft       #PASS name:"altitude.cabin"
        xsection_ar       = 1.07,  #PASS name:"fuseh/w"
        fineness_nose     = 1.5,   #PASS name:"nosefineness"
        fineness_tail     = 2.5,   #PASS name:"tailfineness"
        windshield_height = 2.5,   #ft #PASS name:"windshieldht"
        cockpit_length    = 6.0,   #ft   #PASS name:"pilotlength"
        fwd_space         = 6.3,   #ft        #PASS name:"fwdspace"
        aft_space         = 0.0,   #ft        #PASS name:"aftspace"
        num_crew          = 2,     #PASS name:"#crew"
        num_attendants    = 5,     #PASS name:"#attendants"
    )
    
    turboFan = SUAVE.Components.Propulsors.Turbo_Fan(
        tag = 'TheTurboFan',
        symmetric   = True,     #new SUAVE variable (vertical tail is false)
        type_engine = 1,        #PASS name:"enginetype"  #This may go away in the future, decided by component declaration
        thrust_sls  = 27300.0,  # lbf #PASS name:"slsthrust"
        sfc_TF      = 1.0092,   #PASS name:"sfc/sfcref"
        kwt0_eng    = 0.2007,   #PASS name:"wdryengine/slst">
        pos_dx      = 0.0,      #PASS name:"dxengine1"  #Need to redo component positioning.
        pos_dy      = 0.0,      #PASS name:"dyengine1"
        pos_dz      = 0.0,      #PASS name:"dzengine1"
    )
    
    cost = SUAVE.Components.Cost( 
        tag = 'B737 Cost',
        depreciate_years = 12,   #PASS name:"yearstozero"
        fuel_price       = 2.0,  #$/gal #PASS name:"fuel-$pergal"  #Treat as unitless (no conversion)
        oil_price        = 10.0, #$/lb  #PASS name:"oil-$perlb" #Treat as unitless (no conversion)
        insure_rate      = 0.02, #PASS name:"insurerate"
        labor_rate       = 35.0, #PASS name:"laborrate"
        inflator         = 6.5,  #PASS name:"inflation"
    )
    
    systems = SUAVE.Components.Systems.System( 
        tag = 'B737 Systems',
        flt_ctrl_type     = 3,    #PASS name:"controlstype"
        main_ldg_width_ND = 1.55, #PASS name:"ygear/fusewidth"
    )
    
    massprops = SUAVE.Components.Mass_Props( 
        tag = 'B737 Vehicle Mass_Props',
        mtow              = 174200.0,   #lb #PASS name="weight.maxto"
        max_extra_payload = 0.0,        #lb #PASS name="maxextrapayload"
        other_wt          = -2343.7996, #lb #PASS name="wother"
        fmzfw             = 0.784,      #PASS name="mzfw/mtow"
        fmlw              = 0.84,       #PASS name="mlw/mtow"
        structure_wt_TF   = 1.0,        #PASS name="structwtfudge"  #Structural Weight Tech Factor
    )
    
    envelope = SUAVE.Components.Envelope( 
        tag = 'B737 Flight Envelope',
        alpha_limit = 180.0,    #deg  #PASS name="alphalimit"
        cg_ctrl     = 0.0,      #PASS name="cgcontrol"
        alt_vc      = 26000.0,  #ft      #PASS name="altitude.vc"
        alt_gust    = 20000.0,  #ft    #PASS name="altitude.strdes"
        max_ceiling = 41000.0,  #ft #PASS name="altitude.maxalt"
        mlafactor   = 1.0,      #PASS name="mlafactor"
        glafactor   = 1.0,      #PASS name="glafactor"
    )
    
    #pass_params = SUAVE.Components.Component( 
        #tag = 'PASS Modeling Parameters',
        #fother_drag  = 1.0,   # sq-ft #PASS name="fother"
        #fmarkup_drag = 1.01,  #PASS name:"fmarkup"
        #area_rule    = 1.0,   #PASS name:"arearulefactor"
        #dB_reduce    = 0.0,   #PASS name:"noisereduction"
        #type_av      = 2,     #PASS name:"aircrafttype"
    #)
    
    #-----------------------------------------------------------------------------
    #    Create a Vehicle and store all the "inputs"
    #-----------------------------------------------------------------------------
    
    The_AeroVehicle = SUAVE.Vehicle()
    
    The_AeroVehicle.add_component(fuselage)
    
    The_AeroVehicle.add_component(mainWing)
    The_AeroVehicle.add_component(horzTail)
    The_AeroVehicle.add_component(vertTail)

    The_AeroVehicle.add_component(turboFan)
    
    The_AeroVehicle.add_component(cost)
    The_AeroVehicle.add_component(systems)
    The_AeroVehicle.add_component(massprops)
    The_AeroVehicle.add_component(envelope)
    #The_AeroVehicle.add_component(pasas_params)
    
    
    # ------------------------------------------------------------------
    # Configuration - Take Off
    # ------------------------------------------------------------------
    The_AeroVehicle.new_configuration('TakeOff')
    the_config = The_AeroVehicle.Configs[0]  # this is a linked copy to The_AeroVehicle
    the_config.Wings['main_wing'].flaps = True
    the_config.Functions['Total_Lift'] = dummy_function
    the_config.Functions['Total_Drag'] = dummy_function

    # ------------------------------------------------------------------
    # Configuration - Cruise
    # ------------------------------------------------------------------    
    the_config = The_AeroVehicle.new_configuration('Cruise')
    the_config.Wings['main_wing'].flaps = False
        
    return The_AeroVehicle
    
def create_mission():    
    #=============================================================================
    #    Vehicle Input Parameters (to be replaced by File I/O in the future)
    #=============================================================================
    
    #These parameters apply the mission as a whole not a specific mission segment
    miss_param = SUAVE.Analyses.Mission.Segments.Segment( 
        tag = 'Design Mission',
        wt_cargo = 10117.4, #lb #PASS name:"wcargo"
        num_pax  = 160,     #PASS name:"#passengers"
    )
    
    # choose an atmosphere
    isa = None  # TWL - broke atmosphere... 
    #isa = SUAVE.Atmosphere.EarthIntlStandardAtmosphere()
    
    to_seg = SUAVE.Analyses.Mission.Segments.Segment( 
        tag = 'Take-Off',
        seg_type     = 'to',       #Flag to set kind of segment
        reserve      = False,      #This is not a reserve fuel segment
        mach         = 0.19770733, #Wow that's precise! #PASS name: "machnumber.to"
        alt          = 0,          #ft
        atmo         = isa,        # an atmosphere
        time         = 0,          #min
        adjust       = False,      #This segment is not adjustable
        flap_setting = 15.0,       #PASS name: "flapdeflection_to"
        slat_setting = 15.0,       #PASS name: "slatdeflection_to"
    )
    
    climb_seg = SUAVE.Analyses.Mission.Segments.Segment( 
        tag = 'Climb-Out',
        seg_type = 'climb', #Flag to set kind of segment
        reserve  = False,   #This is not a reserve fuel segment
        mach     = 0.650, 
        atmo     = isa,     # an atmosphere
        time     = 0.033,   #hr
        adjust   = False,   #This segment is not adjustable
    )
    
    init_cr = SUAVE.Analyses.Mission.Segments.Segment( 
        tag ='Initial Cruise',
        seg_type   = 'cruise',
        reserve    = False,    #This is not a reserve fuel segment
        mach       = 0.785,    #PASS name: "machnumber.initcr"
        alt        = 31000,    #ft #PASS name:"altitude.initcr"
        atmo       = isa,      # an atmosphere
        adjust     = True,     #This segment time/distance is adjustable
        cdi_factor = 1.035,    #PASS name: "cdi_factor.initcr"
    )
    
    final_cr = SUAVE.Analyses.Mission.Segments.Segment( 
        tag ='Final Cruise',
        seg_type   = 'cruise',
        reserve    = False,    #This is not a reserve fuel segment
        mach       = 0.785,    #PASS name: "machnumber.finalcr"
        alt        = 39000,    #ft #PASS name:"altitude.finalcr"
        atmo       = isa,      # an atmosphere
        adjust     = True,     #This segment time/distance is adjustable
        cdi_factor = 1.035,    #PASS name: "cdi_factor.finalcr"
    )
    
    ldg_seg = SUAVE.Analyses.Mission.Segments.Segment( 
        tag = 'Landing',
        seg_type     = 'ldg',      #Flag to set kind of segment
        reserve      = False,      #This is not a reserve fuel segment
        mach         = 0.19770733, #Wow that's precise! #PASS name: "machnumber.landing"
        alt          = 0,          #ft
        atmo         = isa,        # an atmosphere
        time         = 0,          #min
        adjust       = False,      #This segment is not adjustable
        flap_setting = 40.0,       #PASS name: "flapdeflection.landing"
        slat_setting = 15.924825,  #PASS name: "slatdeflection.landing"  #Another really precise number           
    )
    
    res = SUAVE.Analyses.Mission.Segments.Segment( 
        tag ='Reserve',
        seg_type = 'cruise',
        reserve  = True,     #This is a reserve fuel segment
        fFuelRes = 0.15,     #A different way to do fuel reserves as fraction of fuel burn
        atmo     = isa,      # an atmosphere
    )
    
    #-----------------------------------------------------------------------------
    #    Create a Mission and store all the "inputs"
    #-----------------------------------------------------------------------------
    
    The_Mission = SUAVE.Analyses.Mission.Mission(miss_param)
    
    #Add the mission segments
    The_Mission.add_segment(to_seg)
    The_Mission.add_segment(climb_seg)
    The_Mission.add_segment(init_cr)
    The_Mission.add_segment(final_cr)
    The_Mission.add_segment(ldg_seg)
    The_Mission.add_segment(res)
    
    The_Mission.Segments[0]
    The_Mission.Segments['Take-Off']
    
    return The_Mission

def create_analysis(Vehicle,Mission):

    Analysis = SUAVE.Attributes.Analysis()
    Analysis.Vehicle = Vehicle
    Analysis.Mission = Mission
    
    Configs   = Vehicle.Configs
    Segments  = Mission.Segments
    Procedure = Analysis.Procedure

    Procedure[0] = [ Segments['Take-Off']       , Configs['TakeOff'] ]
    Procedure[1] = [ Segments['Climb-Out']      , Configs['Cruise']  ]
    Procedure[2] = [ Segments['Initial Cruise'] , Configs['Cruise']  ]
    
    return Analysis
    

def dummy_function():
    pass

def main():
    
    The_AeroVehicle = create_aerovehicle()
    The_Mission     = create_mission()
    The_Analysis    = create_analysis(The_AeroVehicle,The_Mission)
    
    print The_AeroVehicle
    print The_Mission
    print The_Analysis
    
    return

# call main
if __name__ == '__main__':
    main()
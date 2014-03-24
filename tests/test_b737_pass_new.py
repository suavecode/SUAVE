# A B737-800 SUave Example
# 
# Created 4/5/2013 - J. Sinsay
# Updated 6/21/2013 - M. Colonno

# Write into the Vehicle Database directly to get started w/o File I/O
# Future File I/O will read from an .av file

import SUAVE

def create_vehicle():
    
    #=============================================================================
    #    Vehicle Input Parameters (to be replaced by File I/O in the future)
    #=============================================================================
    
    # Create a Vehicle
    B737 = SUAVE.Vehicle()
    
    # start here...
    myAV.AddComponent('Wing', vertTail)
    
    #myAV.AddComponent('Propulsion', propulsion)
    myAV.AddComponent('Turbo_Fan', turboFan)

    myAV.AddComponent('Cost', cost)
    myAV.AddComponent('Systems', systems)
    myAV.AddComponent('Mass_Props', massprops)
    myAV.AddComponent('Envelope', envelope)
    myAV.AddComponent('PASS', pass_params)

    # Fuselage 
    fuselage = SUAVE.Attributes.Components.Fuselage(tag = 'Fuselage')

    fuselage.seats = 160                                                #PASS name: "#coachseats"
    fuselage.seat_layout_lower = 33                                     #PASS name: "seatlayout1"
    fuselage.seat_width = SUAVE.Units.ConvertLength(18.5,'in','m')      #PASS name: "seatwidth" in ---> m
    fuselage.seat_layout_upper = 0                                      #PASS name: "seatlayout2"
    fuselage.aisle_width = SUAVE.Units.ConvertLength(20.0,'in','m')     #PASS name: "aislewidth" in ---> m
    fuselage.seat_pitch = SUAVE.Units.ConvertLength(32.0,'in','m')      #PASS name: "seatpitch" in ---> m
    fuselage.cabin_alt = SUAVE.Units.ConvertLength(7000.0,'ft','m')     #PASS name: "altitude.cabin" ft ---> m
    
    fuselage.xsection_ar = 1.07                                         #PASS name: "fuseh/w"
    fuselage.fineness_nose = 1.5                                        #PASS name: "nosefineness"
    fuselage.fineness_tail = 2.5                                        #PASS name: "tailfineness"

    fuselage.windshield_height = SUAVE.Units.ConvertLength(2.5,'ft','m')#PASS name: "windshieldht" ft ---> m
    fuselage.cockpit_length = SUAVE.Units.ConvertLength(6.0,'ft','m')   #PASS name: "pilotlength" ft ---> m
    fuselage.fwd_space = SUAVE.Units.ConvertLength(6.3,'ft','m')        #PASS name: "fwdspace" ft ---> m
    fuselage.aft_space = SUAVE.Units.ConvertLength(0.0,'ft','m')        #PASS name: "aftspace" ft ---> m
    fuselage.num_crew = 2                                               #PASS name: "#crew"
    fuselage.num_attendants = 5                                         #PASS name: "#attendants"
    B737.AddComponent('Fuselage', fuselage)
    
    # Main wing
    mainWing = SUAVE.Attributes.Components.Wing(tag = 'Main Wing')
    s = SUAVE.Attributes.Components.Segment()
    a = SUAVE.Attributes.Components.XSection.Airfoil()

    mainWing.symmetry = 'XY'
    mainWing.weight_eqn = 'PASS_wing'
    mainWing.ref.area = 1344.0*0.09290304           #PASS name: "sref", ft^2 ---> m^2
    mainWing.position = "Low"                       #PASS name: "wingheight"
    mainWing.origin = [0.35, 0, 0]                  #PASS name: "wingxposition" (need fuselage length)
    
    s.AR = 10.19                                    #PASS name: "arw"
    s.sweep = 25.0                                  #PASS name: "sweepw" (deg)
    s.TR = 0.159                                    #PASS name: "taperw"
    s.dihedral = 6.0                                #PASS name: "wingdihedral" (deg)

    a.leading_edge_extension = 0.10                 #PASS name: "lex" (in units of trapazoidal root chord apparently)       
    a.trailing_edge_extension = 0.21                #PASS name: "tex" (in units of trapazoidal root chord apparently)
    a.span_chord_extension = 0.33                   #PASS name: "chordextspan" (in units of span apparently)
    a.t = 0.10                                      #PASS name: "tovercw" (thickness / chord)
    a.type = "Supercritical"                        #PASS name: "supercritical?"
    a.x_transition = 0.03                           #PASS name: "x/ctransition"
    a.flap_span = 0.60                              #PASS name: "flapspan/b"
    a.flap_chord = 0.32659602                       #PASS name: "flapchord/c"      
    
    mainWing.sections.append(s)
    mainWing.airfoils.append(a)
    B737.AddComponent('Wing', mainWing)

    # Horizontal tail
    horzTail = SUAVE.Attributes.Components.Wing(tag = 'Horizontal Tail')
    s = SUAVE.Attributes.Components.Segment()
    a = SUAVE.Attributes.Components.XSection.Airfoil()

    horzTail.symmetry = "XY"
    horzTail.weight_eqn = 'PASS_htail'
    horzTail.ref.area = 0.2625305*mainWing.ref.area     #PASS name: "sh/sref" --> dimensional 

    s.AR = 6.16                                         #PASS name: "arh"
    s.sweep = 30.0                                      #PASS name: "sweeph" (deg)                                         
    s.TR = 0.40                                         #PASS name: "taperh"
    s.dihedral = 7.0                                    #PASS name: "dihedralh" (deg) 
          
    a.t = 0.08                                          #PASS name: "toverch"
    a.Cl_max = 1.20                                     #PASS name: "clhmax"

    horzTail.sections.append(s)
    horzTail.airfoils.append(a)
    B737.AddComponent('Wing', horzTail)

    # Vertical tail
    vertTail = SUAVE.Attributes.Components.Wing(tag = 'Vertical Tail')
    s = SUAVE.Attributes.Components.Segment()
    a = SUAVE.Attributes.Components.XSection.Airfoil()

    vertTail.symmetry = "None"
    vertTail.weight_eqn = 'PASS_vtail'
    vertTail.ref.area = 0.2117543*mainWing.ref.area     #PASS name: "sv/sref" --> dimensional 
    vertTail.ttail = False                              #PASS name: "ttail?"

    s.AR = 1.91                                         #PASS name: "arv"
    s.sweep = 35.0                                      #PASS name: "sweepv" (deg)
    s.TR = 0.25                                         #PASS name: "taperv"
    s.dihedral = 6.0                                    #PASS name: "dihedralh" (deg)
    vertTail.sections.append(s)

    a.t = 0.08                                          #PASS name: "tovercv"
    vertTail.airfoils.append(a)

    # Engines
    turboFan = SUAVE.Attributes.Components.Turbo_Fan(tag = 'Engine')

    # from PASS: "The following allow for changes in the assumed engine locations. Engines are numbered left to right."
    turboFan.origin = [0.0, 0.0, 0.0] # Need data
    turboFan.symmetric = 'XY'                                           
    turboFan.N = 2                                                      #PASS name: "#engines"
    turboFan.type_engine = "Turbofan"                                   #PASS name: "enginetype"  #This may go away in the future, decided by component declaration
    turboFan.thrust_sls = SUAVE.Units.ConvertForce(27300.0,'lbf','N')   #PASS name: "slsthrust" lbf ---> N
    turboFan.sfc_TF = 1.0092*0.326  # assuming lb/lbf-hr                #PASS name: "sfc/sfcref" * "high bypass ratio turbofan (uninstalled sls SFC = .326)"                                          
    turbofan.mass_props.mass = 0.2007*turboFan.thrust_sls               #PASS name: "wdryengine/slst" * "slsthrust" (N)
    
    #'tag' : 'B737 Propulsion', 
    #'num_eng_wing' : 2, #PASS name:"#wingengines"
    #'num_eng_tail' : 0, #PASS name:"#tailengines"

    cost = SUAVE.Attributes.Components.Cost(tag = 'B737 Cost Model')

    cost.depreciate_years = 12                                          #PASS name: "yearstozero"
    cost.fuel_price = 2.0                                              # $/gal #PASS name:"fuel-$pergal"  #Treat as unitless (no conversion)
    cost.oil_price = 10.0                                               # $/lb  #PASS name:"oil-$perlb" #Treat as unitless (no conversion)
    cost.insure_rate = 0.02                                             #PASS name: "insurerate"
    cost.labor_rate = 35.0                                              #PASS name: "laborrate"
    cost.inflator = 6.5                                                 #PASS name: "inflation"

    systems = SUAVE.Components.Systems({ 
        'tag' : 'B737 Systems',
        'flt_ctrl_type' : 3, #PASS name:"controlstype"
        'main_ldg_width_ND' : 1.55, #PASS name:"ygear/fusewidth"
    })
    
    massprops = SUAVE.Components.Mass_Props({ 
        'tag' : 'B737 Vehicle Mass_Props',
        'mtow' : 174200.0, #lb #PASS name="weight.maxto"
        'max_extra_payload' : 0.0, #lb #PASS name="maxextrapayload"
        'other_wt' : -2343.7996, #lb #PASS name="wother"
        'fmzfw' : 0.784, #PASS name="mzfw/mtow"
        'fmlw' : 0.84,   #PASS name="mlw/mtow"
        'structure_wt_TF' : 1.0, #PASS name="structwtfudge"  #Structural Weight Tech Factor
    })
    
    envelope = SUAVE.Components.Envelope({ 
        'tag' : 'B737 Flight Envelope',
        'alpha_limit' : 180.0, #deg  #PASS name="alphalimit"
        'cg_ctrl' : 0.0,            #PASS name="cgcontrol"
        'alt_vc' : 26000.0, #ft      #PASS name="altitude.vc"
        'alt_gust' : 20000.0, #ft    #PASS name="altitude.strdes"
        'max_ceiling' : 41000.0, #ft #PASS name="altitude.maxalt"
        'mlafactor' : 1.0,          #PASS name="mlafactor"
        'glafactor' : 1.0,          #PASS name="glafactor"
    })
    
    pass_params = SUAVE.Components.PASS({ 
        'tag' : 'PASS Modeling Parameters',
        'fother_drag' : 1.0, # sq-ft #PASS name="fother"
        'fmarkup_drag' : 1.01,     #PASS name:"fmarkup"
        'area_rule' : 1.0,         #PASS name:"arearulefactor"
        'dB_reduce' : 0.0,         #PASS name:"noisereduction"
        'type_av' : 2,             #PASS name:"aircrafttype"
    })
    
    #-----------------------------------------------------------------------------
    #    Create a Vehicle and store all the "inputs"
    #-----------------------------------------------------------------------------
    
    vehicle = SUAVE.Vehicle()
    
    myAV.AddComponent('Fuselage', fuselage)
    
    myAV.AddComponent('Wing', mainWing)
    myAV.AddComponent('Wing', horzTail)
    myAV.AddComponent('Wing', vertTail)
    
    #myAV.AddComponent('Propulsion', propulsion)
    myAV.AddComponent('Turbo_Fan', turboFan)

    myAV.AddComponent('Cost', cost)
    myAV.AddComponent('Systems', systems)
    myAV.AddComponent('Mass_Props', massprops)
    myAV.AddComponent('Envelope', envelope)
    myAV.AddComponent('PASS', pass_params)
        
    return vehicle
    
def create_mission():    
    #=============================================================================
    #    Vehicle Input Parameters (to be replaced by File I/O in the future)
    #=============================================================================
    
    #These parameters apply the mission as a whole not a specific mission segment
    miss_param = SUAVE.Components.Segment({ 
        'tag' : 'Design Mission',
        'wt_cargo' : 10117.4, #lb #PASS name:"wcargo"
        'num_pax' : 160, #PASS name:"#passengers"
    })
    
    # choose an atmosphere
    isa = None  # TWL - broke atmosphere... 
    #isa = SUAVE.Atmosphere.EarthIntlStandardAtmosphere()
    
    to_seg = SUAVE.Components.Segment({ 
        'tag' : 'Take-Off',
        'seg_type' : 'to', #Flag to set kind of segment
        'reserve' : False, #This is not a reserve fuel segment
        'mach' : 0.19770733, #Wow that's precise! #PASS name: "machnumber.to"
        'alt' : 0, #ft
        'atmo' : isa, # an atmosphere
        'time' : 0, #min
        'adjust' : False, #This segment is not adjustable
        'flap_setting' : 15.0, #PASS name: "flapdeflection_to"
        'slat_setting' : 15.0, #PASS name: "slatdeflection_to"
    })
    
    climb_seg = SUAVE.Components.Segment({ 
        'tag' : 'Climb-Out',
        'seg_type' : 'climb', #Flag to set kind of segment
        'reserve' : False, #This is not a reserve fuel segment
        'mach' : 0.650, 
        'atmo' : isa, # an atmosphere
        'time' : 0.033, #hr
        'adjust' : False, #This segment is not adjustable
    })
    
    init_cr = SUAVE.Components.Segment({ 
        'tag' :'Initial Cruise',
        'seg_type' : 'cruise',
        'reserve' : False, #This is not a reserve fuel segment
        'mach' : 0.785, #PASS name: "machnumber.initcr"
        'alt' : 31000, #ft #PASS name:"altitude.initcr"
        'atmo' : isa, # an atmosphere
        'adjust' : True, #This segment time/distance is adjustable
        'cdi_factor' : 1.035, #PASS name: "cdi_factor.initcr"
    })
    
    final_cr = SUAVE.Components.Segment({ 
        'tag' :'Final Cruise',
        'seg_type' : 'cruise',
        'reserve' : False, #This is not a reserve fuel segment
        'mach' : 0.785, #PASS name: "machnumber.finalcr"
        'alt' : 39000, #ft #PASS name:"altitude.finalcr"
        'atmo' : isa, # an atmosphere
        'adjust' : True, #This segment time/distance is adjustable
        'cdi_factor' : 1.035, #PASS name: "cdi_factor.finalcr"
    })
    
    ldg_seg = SUAVE.Components.Segment({ 
        'tag' : 'Landing',
        'seg_type' : 'ldg', #Flag to set kind of segment
        'reserve' : False, #This is not a reserve fuel segment
        'mach' : 0.19770733, #Wow that's precise! #PASS name: "machnumber.landing"
        'alt' : 0, #ft
        'atmo' : isa, # an atmosphere
        'time' : 0, #min
        'adjust' : False, #This segment is not adjustable
        'flap_setting' : 40.0, #PASS name: "flapdeflection.landing"
        'slat_setting' : 15.924825, #PASS name: "slatdeflection.landing"  #Another really precise number           
    })
    
    res = SUAVE.Components.Segment({ 
        'tag' :'Reserve',
        'seg_type' : 'cruise',
        'reserve' : True, #This is a reserve fuel segment
        'fFuelRes' : 0.15, #A different way to do fuel reserves as fraction of fuel burn
        'atmo' : isa, # an atmosphere
    })
    
    #-----------------------------------------------------------------------------
    #    Create a Mission and store all the "inputs"
    #-----------------------------------------------------------------------------
    
    myMiss = SUAVE.Mission(miss_param)
    
    #Add the mission segments (must be done in order, for now)
    # todo: insert inside list function
    myMiss.AddSegment(to_seg)
    myMiss.AddSegment(climb_seg)
    myMiss.AddSegment(init_cr)
    myMiss.AddSegment(final_cr)
    myMiss.AddSegment(ldg_seg)
    myMiss.AddSegment(res)
    
    myMiss.Segments[0]
    myMiss.Segments['Take-Off']
    
    return myMiss

def main():

    myAV = create_av()
    myMiss = create_miss()
    
    print myAV
    print myMiss
    
    return

# call main
if __name__ == '__main__':
    main()
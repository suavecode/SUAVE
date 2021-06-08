## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
# generate_vortex_distribution.py
# 
# Created:  May 2018, M. Clarke
#           Apr 2020, M. Clarke
# Modified: Jun 2021, A. Blaufox

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# package imports 
import numpy as np
from copy import deepcopy
from math import isclose

import SUAVE
from SUAVE.Core import  Data
from SUAVE.Components.Wings.Control_Surfaces import Aileron , Elevator , Slat , Flap , Rudder 
from SUAVE.Methods.Geometry.Two_Dimensional.Planform import populate_control_sections
from SUAVE.Methods.Flight_Dynamics.Static_Stability.Approximations.Supporting_Functions import convert_sweep_segments
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.import_airfoil_geometry\
     import import_airfoil_geometry

## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
def generate_vortex_distribution(geometry,settings):
    ''' Compute the coordinates of panels, vortices , control points
    and geometry used to build the influence coefficient matrix. All
    major sections (wings and fuslages) have the same n_sw and n_cw in
    order to allow the usage of vectorized calculations.    

    Assumptions: 
    Below is a schematic of the coordinates of an arbitrary panel  
    
    XA1 ____________________________ XB1    
       |                            |
       |        bound vortex        |
    XAH|  ________________________  |XBH
       | |           XCH          | |
       | |                        | |
       | |                        | |     
       | |                        | |
       | |                        | |
       | |           0 <--control | |       
       | |          XC     point  | |  
       | |                        | |
   XA2 |_|________________________|_|XB2
         |                        |     
         |       trailing         |  
         |   <--  vortex   -->    |  
         |         legs           | 
             
    
    In addition, all control surfaces should be appended directly
       to the wing, not the wing segments    
    
    Source:  
    None

    Inputs:
    geometry.wings                                [Unitless]  
       
    Outputs:                                   
    VD - vehicle vortex distribution              [Unitless] 

    Properties Used:
    N/A 
         
    '''
    # ---------------------------------------------------------------------------------------
    # STEP 1: Define empty vectors for coordinates of panes, control points and bound vortices
    # ---------------------------------------------------------------------------------------
    VD = Data()

    VD.XAH    = np.empty(shape=[0,1])
    VD.YAH    = np.empty(shape=[0,1])
    VD.ZAH    = np.empty(shape=[0,1])
    VD.XBH    = np.empty(shape=[0,1])
    VD.YBH    = np.empty(shape=[0,1])
    VD.ZBH    = np.empty(shape=[0,1])
    VD.XCH    = np.empty(shape=[0,1])
    VD.YCH    = np.empty(shape=[0,1])
    VD.ZCH    = np.empty(shape=[0,1])     
    VD.XA1    = np.empty(shape=[0,1])
    VD.YA1    = np.empty(shape=[0,1])  
    VD.ZA1    = np.empty(shape=[0,1])
    VD.XA2    = np.empty(shape=[0,1])
    VD.YA2    = np.empty(shape=[0,1])    
    VD.ZA2    = np.empty(shape=[0,1])    
    VD.XB1    = np.empty(shape=[0,1])
    VD.YB1    = np.empty(shape=[0,1])  
    VD.ZB1    = np.empty(shape=[0,1])
    VD.XB2    = np.empty(shape=[0,1])
    VD.YB2    = np.empty(shape=[0,1])    
    VD.ZB2    = np.empty(shape=[0,1])     
    VD.XAC    = np.empty(shape=[0,1])
    VD.YAC    = np.empty(shape=[0,1])
    VD.ZAC    = np.empty(shape=[0,1]) 
    VD.XBC    = np.empty(shape=[0,1])
    VD.YBC    = np.empty(shape=[0,1])
    VD.ZBC    = np.empty(shape=[0,1]) 
    VD.XC_TE  = np.empty(shape=[0,1])
    VD.YC_TE  = np.empty(shape=[0,1])
    VD.ZC_TE  = np.empty(shape=[0,1])     
    VD.XA_TE  = np.empty(shape=[0,1])
    VD.YA_TE  = np.empty(shape=[0,1])
    VD.ZA_TE  = np.empty(shape=[0,1]) 
    VD.XB_TE  = np.empty(shape=[0,1])
    VD.YB_TE  = np.empty(shape=[0,1])
    VD.ZB_TE  = np.empty(shape=[0,1])  
    VD.XC     = np.empty(shape=[0,1])
    VD.YC     = np.empty(shape=[0,1])
    VD.ZC     = np.empty(shape=[0,1])    
    VD.FUS_XC = np.empty(shape=[0,1])
    VD.FUS_YC = np.empty(shape=[0,1])
    VD.FUS_ZC = np.empty(shape=[0,1])      
    VD.CS     = np.empty(shape=[0,1]) 
    VD.X      = np.empty(shape=[0,1])
    VD.Y      = np.empty(shape=[0,1])
    VD.Z      = np.empty(shape=[0,1])
    VD.Y_SW   = np.empty(shape=[0,1])
    VD.DY     = np.empty(shape=[0,1]) 
    n_sw      = settings.number_spanwise_vortices 
    n_cw      = settings.number_chordwise_vortices     
    spc       = settings.spanwise_cosine_spacing
    model_fuselage = settings.model_fuselage

    # ---------------------------------------------------------------------------------------
    # STEP 2: Unpack aircraft wing geometry 
    # ---------------------------------------------------------------------------------------    
    VD.n_w         = 0  # instantiate the number of wings counter  
    VD.n_cp        = 0  # instantiate number of bound vortices counter     
    VD.wing_areas  = [] # instantiate wing areas
    VD.vortex_lift = []
    
    #reformat wings and control surfaces for VLM panelization
    VLM_wings = make_VLM_wings(geometry)
    geometry.VLM_wings = VLM_wings
    
    #generate panelization for each wing (and control surface)
    for wing in geometry.VLM_wings:
        VD = generate_wing_vortex_distribution(VD,wing,n_cw,n_sw,spc)
        
    # ---------------------------------------------------------------------------------------
    # STEP 8.1: Unpack aircraft fuselage geometry
    # --------------------------------------------------------------------------------------- 
    VD.Stot       = sum(VD.wing_areas)        
    VD.wing_areas = np.array(VD.wing_areas)   
    VD.n_fus = 0
    for fus in geometry.fuselages:   
        VD = generate_fuselage_vortex_distribution(VD,fus,n_cw,n_sw,model_fuselage) 
         
    VD.n_sw       = n_sw
    VD.n_cw       = n_cw       

    geometry.vortex_distribution = VD

    # Compute Panel Areas 
    VD.panel_areas = compute_panel_area(VD)
    
    # Compute Panel Normals
    VD.normals = compute_unit_normal(VD)
    
    return VD 


## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
def make_VLM_wings(geometry):
    """ This forces all wings to be segmented, appends all control surfaces
        as full wings to geometry, and contructs helper variables (most 
        notably span_breaks[]) for later

    Assumptions: 
    All control surfaces are appended directly to the wing, not wing segments.
    If a given wing has no segments, it must have either .taper or .chords.root 
        and .chords.tip defined

    Source:   
    None
    
    Inputs:   
    VD                   - vortex distribution 
    For geometry with one or more non-segmented wings:
    geometry.
        wings.wing.
            twists.root
            twists.tip
            dihedral
            sweeps.quarter_chord OR sweeps.leading_edge
            thickness_to_chord
            taper
            chord.root
            chords.tip
            
            control_surface.
                tag
                span_fraction_start
                span_fraction_end
                deflection
                chord_fraction
    
    Properties Used:
    N/A
    """ 
    wings = deepcopy(geometry.wings)
    
    # ------------------------------------------------------------------
    # Reformat original wings to have at least 2 segments, check that no control_surfaces are on segments yet
    # ------------------------------------------------------------------    
    for wing in wings:
        wing.is_a_control_surface = False
        n_segments           = len(wing.Segments.keys())
        if n_segments==0:
            wing       = convert_to_segmented_wing(wing)
            n_segments = 2
        else:
            for segment in wing.Segments:
                if len(segment.control_surfaces) > 0:
                    raise ValueError('Input, control surfaces should be appended to the wing, not its segments. ' + 
                                     'This function will move the control surfaces to wing segments itself.')  
        #move wing control surfaces to from wing to its segments
        wing = populate_control_sections(wing)
        
        #ensure wing has attributes that will be needed later
        for i in range(n_segments):   
            (ia, ib)    = (0, 0) if i==0 else (i-1, i)
            seg_a       = wing.Segments[ia]
            seg_b       = wing.Segments[ib]            
            seg_b.chord = seg_b.root_chord_percent *wing.chords.root
            
            #guarantee that all segments have leading edge sweep
            if (i != 0) and (seg_a.sweeps.leading_edge is None):
                old_sweep                 = seg_a.sweeps.quarter_chord
                new_sweep                 = convert_sweep_segments(old_sweep, seg_a, seg_b, wing, old_ref_chord_fraction=0.25, new_ref_chord_fraction=0.0)
                seg_a.sweeps.leading_edge = new_sweep 
        wing.Segments[-1].sweeps.leading_edge = 1e-8
    
    # each control_surface-turned-wing will have its own unique ID number
    cs_ID = 0
    
    # ------------------------------------------------------------------
    # Build wing() objects and wing.span_breaks[] from control surfaces on segments
    # ------------------------------------------------------------------    
    for wing in wings:
        if wing.is_a_control_surface == True: #skip if this wing is actually a control surface
            continue
        
        #prepare to iterate across all segments and control surfaces
        seg_breaks  = []
        LE_breaks   = []
        TE_breaks   = []
        n_segments  = len(wing.Segments.keys())

        #process all control surfaces in each segment-------------------------------------
        for i in range(n_segments):   
            (ia, ib)    = (0, 0) if i==0 else (i-1, i)
            seg_a       = wing.Segments[ia]
            seg_b       = wing.Segments[ib]            
            
            for cs in seg_b.control_surfaces: #should be no control surfaces on root segment
                # create and append a wing object from the control_surface object and relevant segments
                cs_wing = make_cs_wing_from_cs(cs, seg_a, seg_b, wing, cs_ID)
                wings.append(cs_wing)
                                
                # register cs start and end span breaks
                cs_span_breaks = make_span_breaks_from_cs(cs, seg_a, seg_b, cs_wing, cs_ID)
                if type(cs)==Slat:
                    LE_breaks.append(cs_span_breaks[0])
                    LE_breaks.append(cs_span_breaks[1])
                else:
                    TE_breaks.append(cs_span_breaks[0])
                    TE_breaks.append(cs_span_breaks[1])                    
                cs_ID += 1
            
            # register segment span break
            span_break = make_span_break_from_segment(seg_b)
            seg_breaks.append(span_break)

        #merge _breaks arrays into one span_breaks array----------------------------------
        #   1.  sort all span_breaks by their span_fraction
        #   2.  adjust outboard attributes for LE and TE _breaks coincident with a segment span break.
        #   3.  combine LE and TE breaks with the same span_fraction values (LE cuts from slats and TE cuts from others)
        #   4.  scan LE and TE to pick up cs cuts that cross over one or more span breaks
        
        # 1: 
        LE_breaks  = sorted(LE_breaks,  key=lambda span_break: span_break.span_fraction)
        TE_breaks  = sorted(TE_breaks,  key=lambda span_break: span_break.span_fraction)
        seg_breaks = sorted(seg_breaks, key=lambda span_break: span_break.span_fraction)
        
        # 2:
        for seg_break in seg_breaks:
            for LE_break in LE_breaks:
                diff = seg_break.span_fraction - LE_break.span_fraction
                if isclose(diff, 0, abs_tol=1e-6):
                    LE_break.dihdral_outboard     = seg_break.dihdral_outboard 
                    LE_break.sweep_outboard_QC    = seg_break.sweep_outboard_QC
                    LE_break.sweep_outboard_LE    = seg_break.sweep_outboard_LE
                    seg_break.is_redundant        = True
                elif diff < 0:
                    break
            for TE_break in TE_breaks:
                diff = seg_break.span_fraction - LE_break.span_fraction
                if isclose(diff, 0, abs_tol=1e-6):
                    TE_break.dihdral_outboard     = seg_break.dihdral_outboard 
                    TE_break.sweep_outboard_QC    = seg_break.sweep_outboard_QC
                    TE_break.sweep_outboard_LE    = seg_break.sweep_outboard_LE
                    seg_break.is_redundant        = True
                elif diff < 0:
                    break 
        
        # 3: similar to a 3-way merge sort
        span_breaks = []        
        n_LE        = len(LE_breaks)
        n_TE        = len(TE_breaks)
        n_seg       = len(seg_breaks)
        i, j, k     = 0,0,0
        big_num     = float('inf')
        while True:
            LE_span  = LE_breaks[i].span_fraction  if (i < n_LE)  else big_num
            TE_span  = TE_breaks[j].span_fraction  if (j < n_TE)  else big_num
            seg_span = seg_breaks[k].span_fraction if (k < n_seg) else big_num
            
            if (LE_span==big_num) and (TE_span==big_num) and (seg_span==big_num):
                break
            
            if (LE_span <= TE_span) and (LE_span <= seg_span):
                add_span_break(LE_breaks[i], span_breaks)
                i += 1
            elif (TE_span <= LE_span) and (TE_span <= seg_span):
                add_span_break(TE_breaks[j], span_breaks)
                j += 1  
            elif (seg_span <= LE_span) and (seg_span <= TE_span):
                add_span_break(seg_breaks[k], span_breaks)
                k += 1   
            else:
                raise ValueError("No suitable span break") #should never occur
                
        # 4:
        ib, ob = 0, 1 #inboard, outboard indices
        for edge, edge_str in enumerate(['LE','TE']):
            for i in range(len(span_breaks)-1):
                ID_i = span_breaks[i].cs_IDs[edge,ob]
                if ID_i == -1:
                    continue
                for j in range(i+1,len(span_breaks)):
                    i += 1
                    ID_j = span_breaks[j].cs_IDs[edge,ib]
                    if ID_j == ID_i:
                        break
                    elif ID_j == -1:
                        span_breaks[j].cs_IDs[edge,:] = [ID_i, ID_i]
                    else:
                        raise ValueError('VLM does not support multiple control surfaces on the same edge at this time')
                
        # pack span_breaks
        wing.span_breaks = span_breaks
        
    # ------------------------------------------------------------------
    # Give cs_wings simple span_breaks arrays
    # ------------------------------------------------------------------   
    for cs_wing in wings:
        if cs_wing.is_a_control_surface == False: #skip if this wing isn't actually a control surface
            continue  
        span_breaks = []
        span_break  = make_span_break_from_segment(cs_wing.Segments[0])
        span_breaks.append(span_break)
        span_break  = make_span_break_from_segment(cs_wing.Segments[1])
        span_breaks.append(span_break) 
        cs_wing.span_breaks = span_breaks
    
    return wings
  
def add_span_break(span_break, span_breaks):
    if len(span_breaks) == 0:
        span_breaks.append(span_break)
    else:
        # if non-coincident, the space between the breaks is nominal wing
        if span_breaks[-1].span_fraction < span_break.span_fraction: 
            span_breaks.append(span_break)
        
        # else coincident: need to superimpose cs_IDs and cuts, don't need to append
        else:
            boolean = span_breaks[-1].cs_IDs==-1
            span_breaks[-1].cs_IDs[boolean] = span_break.cs_IDs[boolean]
            span_breaks[-1].cuts[boolean]   = span_break.cuts[boolean]
                
    return
    

    
def make_cs_wing_from_cs(cs, seg_a, seg_b, wing, cs_ID):
    #standard wing attributes
    cs_wing = SUAVE.Components.Wings.Wing()
    cs_wing.tag                   = wing.tag + '|' + seg_b.tag + '|' + cs.tag + '|cs_ID_{}'.format(cs_ID)
    span_a                        = seg_a.percent_span_location
    span_b                        = seg_b.percent_span_location
    twist_a                       = seg_a.twist
    twist_b                       = seg_b.twist
    cs_wing.twists.root           = np.interp(cs.span_fraction_start, [span_a, span_b], [twist_a, twist_b])
    cs_wing.twists.tip            = np.interp(cs.span_fraction_end,   [span_a, span_b], [twist_a, twist_b])
    cs_wing.dihedral              = seg_a.dihedral_outboard
    cs_wing.thickness_to_chord    = (seg_a.thickness_to_chord + seg_b.thickness_to_chord)/2
    cs_wing.taper                 = seg_b.root_chord_percent / seg_a.root_chord_percent
    
    ## may have to recompute when shifting y_coords to match span breaks in generate_wing_vortex_distribution
    span_fraction_tot             = cs.span_fraction_end - cs.span_fraction_start
    cs_wing.spans.projected       = wing.spans.projected * span_fraction_tot #includes 2x lenth if cs is on a symmetric wing 
    
    ### may have to recompute when computing owning wing's modificaions in generate_wing_vortex_distribution
    wing_chord_local_at_cs_root   = np.interp(cs.span_fraction_start, [span_a, span_b], [seg_a.chord, seg_b.chord])
    wing_chord_local_at_cs_tip    = np.interp(cs.span_fraction_end,   [span_a, span_b], [seg_a.chord, seg_b.chord])
    cs_wing.chords.root           = wing_chord_local_at_cs_root * cs.chord_fraction  
    cs_wing.chords.tip            = wing_chord_local_at_cs_tip  * cs.chord_fraction             
    cs_wing.sweeps.quarter_chord  = 0  # leave at 0. VLM will use leading edge

    cs_wing.symmetric             = wing.symmetric
    cs_wing.vertical              = wing.vertical
    cs_wing.vortex_lift           = wing.vortex_lift

    #non-standard wing attributes, mostly to do with cs_wing's identity as a control surface
    cs_wing.is_a_control_surface  = True
    cs_wing.cs_ID                 = cs_ID
    cs_wing.chord_fraction        = cs.chord_fraction
    cs_wing.is_slat               = (type(cs)==Slat)
    cs_wing.pivot_edge            = 'TE' if cs_wing.is_slat else 'LE'
    cs_wing.deflection            = cs.deflection
    cs_wing.span_break_fractions  = np.array([cs.span_fraction_start, cs.span_fraction_end]) #to be multiplied by span once span is found 
    
    use_le_sweep                  = (seg_a.sweeps.leading_edge is not None)
    new_cf                        = 0 if cs_wing.is_slat else 1
    old_cf                        = 0 if use_le_sweep else 0.25
    old_sweep                     = seg_a.sweeps.leading_edge if use_le_sweep else seg_a.sweeps.quarter_chord
    new_sweep                     = convert_sweep_segments(old_sweep, seg_a, seg_b, wing, old_ref_chord_fraction=old_cf, new_ref_chord_fraction=new_cf)
    cs_wing.outside_sweep         = new_sweep
    
    #find leading edge sweep
    if cs_wing.is_slat:
        cs_wing.sweeps.leading_edge = new_sweep
    else:
        new_cf                      = 1 - cs_wing.chord_fraction
        new_sweep                   = convert_sweep_segments(old_sweep, seg_a, seg_b, wing, old_ref_chord_fraction=old_cf, new_ref_chord_fraction=new_cf)
        cs_wing.sweeps.leading_edge = new_sweep
    
    #convert to segmented wing and return
    cs_wing = convert_to_segmented_wing(cs_wing)
    return cs_wing

def make_span_break_from_segment(seg):
    span_frac       = seg.percent_span_location
    dihedral_ob     = seg.dihedral_outboard
    sweep_ob_QC     = seg.sweeps.quarter_chord
    sweep_ob_LE     = seg.sweeps.leading_edge
    twist           = seg.twist    
    local_chord     = seg.chord    
    span_break = make_span_break(-1, 0, 0, span_frac, 0., 
                                 dihedral_ob, sweep_ob_QC, sweep_ob_LE, twist, local_chord)  
    span_break.cuts = np.array([[0.,0.],  
                                [1.,1.]])
    return span_break

def make_span_breaks_from_cs(cs, seg_a, seg_b, cs_wing, cs_ID):
    is_slat        = (type(cs)==Slat)
    LE_TE          = 0 if type(cs)==Slat else 1
    
    ib_ob          = 1 #the inboard break of the cs is the outboard part of the span_break
    span_frac      = cs.span_fraction_start    
    ob_cut         = cs.chord_fraction if is_slat else 1 - cs.chord_fraction
    dihedral_ob    = seg_a.dihedral_outboard
    sweep_ob_QC    = seg_a.sweeps.quarter_chord
    sweep_ob_LE    = seg_a.sweeps.leading_edge
    twist          = cs_wing.twists.root    
    local_chord    = cs_wing.chords.root / cs.chord_fraction
    inboard_span_break  = make_span_break(cs_ID, LE_TE, ib_ob, span_frac, ob_cut, 
                                          dihedral_ob, sweep_ob_QC, sweep_ob_LE, twist, local_chord)
    
    ib_ob          = 0 #the outboard break of the cs is the inboard part of the span_break
    span_frac      = cs.span_fraction_end
    ib_cut         = cs.chord_fraction if is_slat else 1 - cs.chord_fraction
    dihedral_ob    = seg_a.dihedral_outboard     #will have to changed in a later function if the outboard edge is conicident with seg_b 
    sweep_ob_QC    = seg_a.sweeps.quarter_chord  #will have to changed in a later function if the outboard edge is conicident with seg_b
    sweep_ob_LE    = seg_a.sweeps.leading_edge   #will have to changed in a later function if the outboard edge is conicident with seg_b
    twist          = cs_wing.twists.tip    
    local_chord    = cs_wing.chords.tip  / cs.chord_fraction
    outboard_span_break = make_span_break(cs_ID, LE_TE, ib_ob, span_frac, ib_cut, 
                                          dihedral_ob, sweep_ob_QC, sweep_ob_LE, twist, local_chord)    
    return inboard_span_break, outboard_span_break

def make_span_break(cs_ID, LE_TE, ib_ob, span_frac, chord_cut, 
                    dihedral_ob, sweep_ob_QC, sweep_ob_LE, twist, local_chord):
    """ This gathers information related to a span break into one Data() object.
    A span break is the spanwise location of a discontinuity in the discretization
    of the panels. These can be caused by segments and by the inboard and outboard 
    edges of a control surface. The inboard and outboard sides of a span break can
    have different chords due to cuts made by control surfaces. A diagram is given 
    below


                                            nominal local chord
    fuselage                inboard LE  |           |           .  outboard LE
    <---                                |           |           . 
                                        |           |           .
                                        |           |           .   <-- cut from a slat
                                        |           |           |
                                        |           |           |
                                        |           |           |
                                        |           |           |
                                        |           |           |
                                        |           |           |
                                        |           |           |
                                        |           |           .   <-- cut from a non-slat control surface with a different chord 
cut from a non-slat control surface     |           |           .       fraction than the control surface on the inboard side
                                  -->   .           |           .
                                        .           |           .
                            inboard TE  .           |           .  outboard TE
                                        
                                        
                                        
                                        |_______________________|
                                                    |
                                            there is 0 spanwise 
                                            distance between inboard 
                                            and outboard sides

    Outputs:
    span_break
    
    Properties Used:
    N/A
    """     
    span_break = Data()
    span_break.cs_IDs               = np.array([[-1,-1],  #  [[inboard LE cs, outboard LE cs],
                                                [-1,-1]]) #   [inboard TE cs, outboard TE cs]]
    span_break.cs_IDs[LE_TE,ib_ob]  = cs_ID
    span_break.span_fraction        = span_frac
    # The following 'cut' attributes are in terms of the local total chord and represent positions. 
    #    (an aileron with chord fraction 0.2 would have a cut value of 0.8)
    # For inboard_cut, -1 takes value of previous outboard cut value in a later function
    # For outboard_cut, -1 takes value of next inboard cut value.
    # If no break directly touching this one, cut becomes 0 (LE) or 1 (TE).
    span_break.cuts                 = np.array([[0.,0.],   #  [[inboard LE cut, outboard LE cut],
                                                [1.,1.]])  #   [inboard TE cut, outboard TE cut]]
    span_break.cuts[LE_TE,ib_ob]    = chord_cut
    span_break.dihdral_outboard     = dihedral_ob
    span_break.sweep_outboard_QC    = sweep_ob_QC
    span_break.sweep_outboard_LE    = sweep_ob_LE
    span_break.twist                = twist
    span_break.local_chord          = local_chord #this is the local chord BEFORE cuts are made
    span_break.is_redundant         = False
    return span_break


def convert_to_segmented_wing(wing):
    """ This turns a non-segmented wing into a segmented wing

    Assumptions: 
    If a given wing has no segments, it must have either .taper or .chords.tip defined

    Source:   
    None
    
    Inputs:   
    VD                   - vortex distribution 
    geometry.
        wings.wing.
            twists.root
            twists.tip
            dihedral
            sweeps.quarter_chord
            thickness_to_chord
            taper
            chord.root
            chords.tip
    
    Properties Used:
    N/A
    """     
    if len(wing.Segments.keys()) > 0:
        return wing   
    # root segment 
    segment                               = SUAVE.Components.Wings.Segment()
    segment.tag                           = 'root_segment'
    segment.percent_span_location         = 0.0
    segment.twist                         = wing.twists.root
    segment.root_chord_percent            = 1.
    segment.chord                         = wing.chords.root #non-standard attribute!
    segment.dihedral_outboard             = wing.dihedral
    segment.sweeps.quarter_chord          = wing.sweeps.quarter_chord
    segment.sweeps.leading_edge           = wing.sweeps.leading_edge
    segment.thickness_to_chord            = wing.thickness_to_chord
    if wing.Airfoil: 
        segment.append_airfoil(wing.Airfoil.airfoil)              
    wing.append_segment(segment) 
    
    # tip segment 
    if wing.taper==0:
        wing.taper = wing.chords.tip / wing.chords.root
    elif wing.chords.tip==0:
        wing.chords.tip = wing.chords.root * wing.taper
        
    segment                               = SUAVE.Components.Wings.Segment()
    segment.tag                           = 'tip_segment'
    segment.percent_span_location         = 1.
    segment.twist                         = wing.twists.tip
    segment.root_chord_percent            = wing.taper
    segment.chord                         = wing.chords.tip #non-standard attribute!
    segment.dihedral_outboard             = 0.
    segment.sweeps.quarter_chord          = 0.
    segment.sweeps.leading_edge           = 1e-8
    segment.thickness_to_chord            = wing.thickness_to_chord
    if wing.Airfoil: 
        segment.append_airfoil(wing.Airfoil.airfoil)             
    wing.append_segment(segment) 
    
    return wing

## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
def generate_wing_vortex_distribution(VD,wing,n_cw,n_sw,spc):
    """ This generates the vortex distribution points on the wing 

    Assumptions: 
    The wing is segmented

    Source:   
    None
    
    Inputs:   
    VD                   - vortex distribution    
    
    Properties Used:
    N/A
    """       
    # get geometry of wing  
    span          = wing.spans.projected
    root_chord    = wing.chords.root
    tip_chord     = wing.chords.tip
    sweep_qc      = wing.sweeps.quarter_chord
    sweep_le      = wing.sweeps.leading_edge 
    twist_rc      = wing.twists.root
    twist_tc      = wing.twists.tip
    dihedral      = wing.dihedral
    sym_para      = wing.symmetric 
    vertical_wing = wing.vertical
    wing_origin   = wing.origin[0]
    VD.vortex_lift.append(wing.vortex_lift)

    # determine if vehicle has symmetry 
    if sym_para is True :
        span = span/2
        VD.vortex_lift.append(wing.vortex_lift)

    if spc == True:

        # discretize wing using cosine spacing
        n               = np.linspace(n_sw+1,0,n_sw+1)         # vectorize
        thetan          = n*(np.pi/2)/(n_sw+1)                 # angular stations
        y_coordinates   = span*np.cos(thetan)                  # y locations based on the angular spacing
    else:

        # discretize wing using linear spacing
        y_coordinates   = np.linspace(0,span,n_sw+1) 

    # create empty vectors for coordinates 
    xah   = np.zeros(n_cw*n_sw)
    yah   = np.zeros(n_cw*n_sw)
    zah   = np.zeros(n_cw*n_sw)
    xbh   = np.zeros(n_cw*n_sw)
    ybh   = np.zeros(n_cw*n_sw)
    zbh   = np.zeros(n_cw*n_sw)    
    xch   = np.zeros(n_cw*n_sw)
    ych   = np.zeros(n_cw*n_sw)
    zch   = np.zeros(n_cw*n_sw)    
    xa1   = np.zeros(n_cw*n_sw)
    ya1   = np.zeros(n_cw*n_sw)
    za1   = np.zeros(n_cw*n_sw)
    xa2   = np.zeros(n_cw*n_sw)
    ya2   = np.zeros(n_cw*n_sw)
    za2   = np.zeros(n_cw*n_sw)    
    xb1   = np.zeros(n_cw*n_sw)
    yb1   = np.zeros(n_cw*n_sw)
    zb1   = np.zeros(n_cw*n_sw)
    xb2   = np.zeros(n_cw*n_sw) 
    yb2   = np.zeros(n_cw*n_sw) 
    zb2   = np.zeros(n_cw*n_sw)    
    xac   = np.zeros(n_cw*n_sw)
    yac   = np.zeros(n_cw*n_sw)
    zac   = np.zeros(n_cw*n_sw)    
    xbc   = np.zeros(n_cw*n_sw)
    ybc   = np.zeros(n_cw*n_sw)
    zbc   = np.zeros(n_cw*n_sw)    
    xa_te = np.zeros(n_cw*n_sw)
    ya_te = np.zeros(n_cw*n_sw)
    za_te = np.zeros(n_cw*n_sw)    
    xb_te = np.zeros(n_cw*n_sw)
    yb_te = np.zeros(n_cw*n_sw)
    zb_te = np.zeros(n_cw*n_sw)  
    xc    = np.zeros(n_cw*n_sw) 
    yc    = np.zeros(n_cw*n_sw) 
    zc    = np.zeros(n_cw*n_sw) 
    x     = np.zeros((n_cw+1)*(n_sw+1)) 
    y     = np.zeros((n_cw+1)*(n_sw+1)) 
    z     = np.zeros((n_cw+1)*(n_sw+1))         
    cs_w  = np.zeros(n_sw)

    # ---------------------------------------------------------------------------------------
    # STEP 3: Determine if wing segments are defined  
    # ---------------------------------------------------------------------------------------
    n_segments           = len(wing.Segments.keys())
    if n_segments>0:            
        # ---------------------------------------------------------------------------------------
        # STEP 4A: Discretizing the wing sections into panels
        # ---------------------------------------------------------------------------------------
        segment_chord          = np.zeros(n_segments)
        segment_twist          = np.zeros(n_segments)
        segment_sweep          = np.zeros(n_segments)
        segment_span           = np.zeros(n_segments)
        segment_area           = np.zeros(n_segments)
        segment_dihedral       = np.zeros(n_segments)
        segment_x_coord        = [] 
        segment_camber         = []
        segment_chord_x_offset = np.zeros(n_segments)
        segment_chord_z_offset = np.zeros(n_segments)
        section_stations       = np.zeros(n_segments) 

        # ---------------------------------------------------------------------------------------
        # STEP 5A: Obtain sweep, chord, dihedral and twist at the beginning/end of each segment.
        #          If applicable, append airfoil section VD and flap/aileron deflection angles.
        # --------------------------------------------------------------------------------------- 
        for i_seg in range(n_segments):   
            segment_chord[i_seg]    = wing.Segments[i_seg].root_chord_percent*root_chord
            segment_twist[i_seg]    = wing.Segments[i_seg].twist
            section_stations[i_seg] = wing.Segments[i_seg].percent_span_location*span  
            segment_dihedral[i_seg] = wing.Segments[i_seg].dihedral_outboard                    

            # change to leading edge sweep, if quarter chord sweep givent, convert to leading edge sweep 
            if (i_seg == n_segments-1):
                segment_sweep[i_seg] = 0                                  
            else: 
                if wing.Segments[i_seg].sweeps.leading_edge != None:
                    segment_sweep[i_seg] = wing.Segments[i_seg].sweeps.leading_edge
                else:                                                                 
                    sweep_quarter_chord  = wing.Segments[i_seg].sweeps.quarter_chord
                    cf       = 0.25                          
                    seg_root_chord       = root_chord*wing.Segments[i_seg].root_chord_percent
                    seg_tip_chord        = root_chord*wing.Segments[i_seg+1].root_chord_percent
                    seg_span             = span*(wing.Segments[i_seg+1].percent_span_location - wing.Segments[i_seg].percent_span_location )
                    segment_sweep[i_seg] = np.arctan(((seg_root_chord*cf) + (np.tan(sweep_quarter_chord)*seg_span - cf*seg_tip_chord)) /seg_span)  

            if i_seg == 0:
                segment_span[i_seg]           = 0.0
                segment_chord_x_offset[i_seg] = 0.0  
                segment_chord_z_offset[i_seg] = 0.0       
            else:
                segment_span[i_seg]           = wing.Segments[i_seg].percent_span_location*span - wing.Segments[i_seg-1].percent_span_location*span
                segment_chord_x_offset[i_seg] = segment_chord_x_offset[i_seg-1] + segment_span[i_seg]*np.tan(segment_sweep[i_seg-1])
                segment_chord_z_offset[i_seg] = segment_chord_z_offset[i_seg-1] + segment_span[i_seg]*np.tan(segment_dihedral[i_seg-1])
                segment_area[i_seg]           = 0.5*(root_chord*wing.Segments[i_seg-1].root_chord_percent + root_chord*wing.Segments[i_seg].root_chord_percent)*segment_span[i_seg]

            # Get airfoil section VD  
            if wing.Segments[i_seg].Airfoil: 
                airfoil_data = import_airfoil_geometry([wing.Segments[i_seg].Airfoil.airfoil.coordinate_file])    
                segment_camber.append(airfoil_data.camber_coordinates[0])
                segment_x_coord.append(airfoil_data.x_lower_surface[0]) 
            else:
                segment_camber.append(np.zeros(30))              
                segment_x_coord.append(np.linspace(0,1,30)) 

            # ** TO DO ** Get flap/aileron locations and deflection

        VD.wing_areas.append(np.sum(segment_area[:]))
        if sym_para is True :
            VD.wing_areas.append(np.sum(segment_area[:]))            

        #Shift spanwise vortices onto section breaks  
        if len(y_coordinates) < n_segments:
            raise ValueError('Not enough spanwise VLM stations for segment breaks')

        last_idx = None            
        for i_seg in range(n_segments):
            idx =  (np.abs(y_coordinates-section_stations[i_seg])).argmin()
            if last_idx is not None and idx <= last_idx:
                idx = last_idx + 1
            y_coordinates[idx] = section_stations[i_seg]   
            last_idx = idx


        for i_seg in range(n_segments):
            if section_stations[i_seg] not in y_coordinates:
                raise ValueError('VLM did not capture all section breaks')

        # ---------------------------------------------------------------------------------------
        # STEP 6A: Define coordinates of panels horseshoe vortices and control points 
        # --------------------------------------------------------------------------------------- 
        y_a   = y_coordinates[:-1] 
        y_b   = y_coordinates[1:]             
        del_y = y_coordinates[1:] - y_coordinates[:-1]           
        i_seg = 0           
        for idx_y in range(n_sw):
            # define coordinates of horseshoe vortices and control points
            idx_x = np.arange(n_cw) 
            eta_a = (y_a[idx_y] - section_stations[i_seg])  
            eta_b = (y_b[idx_y] - section_stations[i_seg]) 
            eta   = (y_b[idx_y] - del_y[idx_y]/2 - section_stations[i_seg]) 

            segment_chord_ratio = (segment_chord[i_seg+1] - segment_chord[i_seg])/segment_span[i_seg+1]
            segment_twist_ratio = (segment_twist[i_seg+1] - segment_twist[i_seg])/segment_span[i_seg+1]

            wing_chord_section_a  = segment_chord[i_seg] + (eta_a*segment_chord_ratio) 
            wing_chord_section_b  = segment_chord[i_seg] + (eta_b*segment_chord_ratio)
            wing_chord_section    = segment_chord[i_seg] + (eta*segment_chord_ratio)

            delta_x_a = wing_chord_section_a/n_cw  
            delta_x_b = wing_chord_section_b/n_cw      
            delta_x   = wing_chord_section/n_cw                                       

            xi_a1 = segment_chord_x_offset[i_seg] + eta_a*np.tan(segment_sweep[i_seg]) + delta_x_a*idx_x                  # x coordinate of top left corner of panel
            xi_ah = segment_chord_x_offset[i_seg] + eta_a*np.tan(segment_sweep[i_seg]) + delta_x_a*idx_x + delta_x_a*0.25 # x coordinate of left corner of panel
            xi_a2 = segment_chord_x_offset[i_seg] + eta_a*np.tan(segment_sweep[i_seg]) + delta_x_a*idx_x + delta_x_a      # x coordinate of bottom left corner of bound vortex 
            xi_ac = segment_chord_x_offset[i_seg] + eta_a*np.tan(segment_sweep[i_seg]) + delta_x_a*idx_x + delta_x_a*0.75 # x coordinate of bottom left corner of control point vortex  
            xi_b1 = segment_chord_x_offset[i_seg] + eta_b*np.tan(segment_sweep[i_seg]) + delta_x_b*idx_x                  # x coordinate of top right corner of panel      
            xi_bh = segment_chord_x_offset[i_seg] + eta_b*np.tan(segment_sweep[i_seg]) + delta_x_b*idx_x + delta_x_b*0.25 # x coordinate of right corner of bound vortex         
            xi_b2 = segment_chord_x_offset[i_seg] + eta_b*np.tan(segment_sweep[i_seg]) + delta_x_b*idx_x + delta_x_b      # x coordinate of bottom right corner of panel
            xi_bc = segment_chord_x_offset[i_seg] + eta_b*np.tan(segment_sweep[i_seg]) + delta_x_b*idx_x + delta_x_b*0.75 # x coordinate of bottom right corner of control point vortex         
            xi_c  = segment_chord_x_offset[i_seg] + eta *np.tan(segment_sweep[i_seg])  + delta_x  *idx_x + delta_x*0.75   # x coordinate three-quarter chord control point for each panel
            xi_ch = segment_chord_x_offset[i_seg] + eta *np.tan(segment_sweep[i_seg])  + delta_x  *idx_x + delta_x*0.25   # x coordinate center of bound vortex of each panel 

            # adjustment of coordinates for camber
            section_camber_a  = segment_camber[i_seg]*wing_chord_section_a  
            section_camber_b  = segment_camber[i_seg]*wing_chord_section_b  
            section_camber_c    = segment_camber[i_seg]*wing_chord_section                
            section_x_coord_a = segment_x_coord[i_seg]*wing_chord_section_a
            section_x_coord_b = segment_x_coord[i_seg]*wing_chord_section_b
            section_x_coord   = segment_x_coord[i_seg]*wing_chord_section

            z_c_a1 = np.interp((idx_x    *delta_x_a)                  ,section_x_coord_a,section_camber_a) 
            z_c_ah = np.interp((idx_x    *delta_x_a + delta_x_a*0.25) ,section_x_coord_a,section_camber_a)
            z_c_a2 = np.interp(((idx_x+1)*delta_x_a)                  ,section_x_coord_a,section_camber_a) 
            z_c_ac = np.interp((idx_x    *delta_x_a + delta_x_a*0.75) ,section_x_coord_a,section_camber_a) 
            z_c_b1 = np.interp((idx_x    *delta_x_b)                  ,section_x_coord_b,section_camber_b)   
            z_c_bh = np.interp((idx_x    *delta_x_b + delta_x_b*0.25) ,section_x_coord_b,section_camber_b) 
            z_c_b2 = np.interp(((idx_x+1)*delta_x_b)                  ,section_x_coord_b,section_camber_b) 
            z_c_bc = np.interp((idx_x    *delta_x_b + delta_x_b*0.75) ,section_x_coord_b,section_camber_b) 
            z_c    = np.interp((idx_x    *delta_x   + delta_x  *0.75) ,section_x_coord,section_camber_c) 
            z_c_ch = np.interp((idx_x    *delta_x   + delta_x  *0.25) ,section_x_coord,section_camber_c) 

            zeta_a1 = segment_chord_z_offset[i_seg] + eta_a*np.tan(segment_dihedral[i_seg])  + z_c_a1  # z coordinate of top left corner of panel
            zeta_ah = segment_chord_z_offset[i_seg] + eta_a*np.tan(segment_dihedral[i_seg])  + z_c_ah  # z coordinate of left corner of bound vortex  
            zeta_a2 = segment_chord_z_offset[i_seg] + eta_a*np.tan(segment_dihedral[i_seg])  + z_c_a2  # z coordinate of bottom left corner of panel
            zeta_ac = segment_chord_z_offset[i_seg] + eta_a*np.tan(segment_dihedral[i_seg])  + z_c_ac  # z coordinate of bottom left corner of panel of control point
            zeta_bc = segment_chord_z_offset[i_seg] + eta_b*np.tan(segment_dihedral[i_seg])  + z_c_bc  # z coordinate of top right corner of panel of control point                          
            zeta_b1 = segment_chord_z_offset[i_seg] + eta_b*np.tan(segment_dihedral[i_seg])  + z_c_b1  # z coordinate of top right corner of panel  
            zeta_bh = segment_chord_z_offset[i_seg] + eta_b*np.tan(segment_dihedral[i_seg])  + z_c_bh  # z coordinate of right corner of bound vortex        
            zeta_b2 = segment_chord_z_offset[i_seg] + eta_b*np.tan(segment_dihedral[i_seg])  + z_c_b2  # z coordinate of bottom right corner of panel                 
            zeta    = segment_chord_z_offset[i_seg] + eta*np.tan(segment_dihedral[i_seg])    + z_c     # z coordinate three-quarter chord control point for each panel
            zeta_ch = segment_chord_z_offset[i_seg] + eta*np.tan(segment_dihedral[i_seg])    + z_c_ch  # z coordinate center of bound vortex on each panel

            # adjustment of coordinates for twist  
            xi_LE_a = segment_chord_x_offset[i_seg] + eta_a*np.tan(segment_sweep[i_seg])               # x location of leading edge left corner of wing
            xi_LE_b = segment_chord_x_offset[i_seg] + eta_b*np.tan(segment_sweep[i_seg])               # x location of leading edge right of wing
            xi_LE   = segment_chord_x_offset[i_seg] + eta*np.tan(segment_sweep[i_seg])                 # x location of leading edge center of wing

            zeta_LE_a = segment_chord_z_offset[i_seg] + eta_a*np.tan(segment_dihedral[i_seg])          # z location of leading edge left corner of wing
            zeta_LE_b = segment_chord_z_offset[i_seg] + eta_b*np.tan(segment_dihedral[i_seg])          # z location of leading edge right of wing
            zeta_LE   = segment_chord_z_offset[i_seg] + eta*np.tan(segment_dihedral[i_seg])            # z location of leading edge center of wing

            # determine section twist
            section_twist_a = segment_twist[i_seg] + (eta_a * segment_twist_ratio)                     # twist at left side of panel
            section_twist_b = segment_twist[i_seg] + (eta_b * segment_twist_ratio)                     # twist at right side of panel
            section_twist   = segment_twist[i_seg] + (eta* segment_twist_ratio)                        # twist at center local chord 

            xi_prime_a1  = xi_LE_a + np.cos(section_twist_a)*(xi_a1-xi_LE_a) + np.sin(section_twist_a)*(zeta_a1-zeta_LE_a)   # x coordinate transformation of top left corner
            xi_prime_ah  = xi_LE_a + np.cos(section_twist_a)*(xi_ah-xi_LE_a) + np.sin(section_twist_a)*(zeta_ah-zeta_LE_a)   # x coordinate transformation of bottom left corner
            xi_prime_a2  = xi_LE_a + np.cos(section_twist_a)*(xi_a2-xi_LE_a) + np.sin(section_twist_a)*(zeta_a2-zeta_LE_a)   # x coordinate transformation of bottom left corner
            xi_prime_ac  = xi_LE_a + np.cos(section_twist_a)*(xi_ac-xi_LE_a) + np.sin(section_twist_a)*(zeta_a2-zeta_LE_a)   # x coordinate transformation of bottom left corner of control point
            xi_prime_bc  = xi_LE_b + np.cos(section_twist_b)*(xi_bc-xi_LE_b) + np.sin(section_twist_b)*(zeta_b1-zeta_LE_b)   # x coordinate transformation of top right corner of control point                         
            xi_prime_b1  = xi_LE_b + np.cos(section_twist_b)*(xi_b1-xi_LE_b) + np.sin(section_twist_b)*(zeta_b1-zeta_LE_b)   # x coordinate transformation of top right corner 
            xi_prime_bh  = xi_LE_b + np.cos(section_twist_b)*(xi_bh-xi_LE_b) + np.sin(section_twist_b)*(zeta_bh-zeta_LE_b)   # x coordinate transformation of top right corner 
            xi_prime_b2  = xi_LE_b + np.cos(section_twist_b)*(xi_b2-xi_LE_b) + np.sin(section_twist_b)*(zeta_b2-zeta_LE_b)   # x coordinate transformation of botton right corner 
            xi_prime     = xi_LE   + np.cos(section_twist)  *(xi_c-xi_LE)    + np.sin(section_twist)*(zeta-zeta_LE)          # x coordinate transformation of control point
            xi_prime_ch  = xi_LE   + np.cos(section_twist)  *(xi_ch-xi_LE)   + np.sin(section_twist)*(zeta_ch-zeta_LE)       # x coordinate transformation of center of horeshoe vortex 

            zeta_prime_a1  = zeta_LE_a - np.sin(section_twist_a)*(xi_a1-xi_LE_a) + np.cos(section_twist_a)*(zeta_a1-zeta_LE_a) # z coordinate transformation of top left corner
            zeta_prime_ah  = zeta_LE_a - np.sin(section_twist_a)*(xi_ah-xi_LE_a) + np.cos(section_twist_a)*(zeta_ah-zeta_LE_a) # z coordinate transformation of bottom left corner
            zeta_prime_a2  = zeta_LE_a - np.sin(section_twist_a)*(xi_a2-xi_LE_a) + np.cos(section_twist_a)*(zeta_a2-zeta_LE_a) # z coordinate transformation of bottom left corner
            zeta_prime_ac  = zeta_LE_a - np.sin(section_twist_a)*(xi_ac-xi_LE_a) + np.cos(section_twist_a)*(zeta_ac-zeta_LE_a) # z coordinate transformation of bottom left corner
            zeta_prime_bc  = zeta_LE_b - np.sin(section_twist_b)*(xi_bc-xi_LE_b) + np.cos(section_twist_b)*(zeta_bc-zeta_LE_b) # z coordinate transformation of top right corner                         
            zeta_prime_b1  = zeta_LE_b - np.sin(section_twist_b)*(xi_b1-xi_LE_b) + np.cos(section_twist_b)*(zeta_b1-zeta_LE_b) # z coordinate transformation of top right corner 
            zeta_prime_bh  = zeta_LE_b - np.sin(section_twist_b)*(xi_bh-xi_LE_b) + np.cos(section_twist_b)*(zeta_bh-zeta_LE_b) # z coordinate transformation of top right corner 
            zeta_prime_b2  = zeta_LE_b - np.sin(section_twist_b)*(xi_b2-xi_LE_b) + np.cos(section_twist_b)*(zeta_b2-zeta_LE_b) # z coordinate transformation of botton right corner 
            zeta_prime     = zeta_LE   - np.sin(section_twist)*(xi_c-xi_LE)      + np.cos(-section_twist)*(zeta-zeta_LE)            # z coordinate transformation of control point
            zeta_prime_ch  = zeta_LE   - np.sin(section_twist)*(xi_ch-xi_LE)     + np.cos(-section_twist)*(zeta_ch-zeta_LE)            # z coordinate transformation of center of horseshoe

            # ** TO DO ** Get flap/aileron locations and deflection
            # store coordinates of panels, horseshoeces vortices and control points relative to wing root 
            if vertical_wing:
                xa1[idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime_a1 
                za1[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*y_a[idx_y]
                ya1[idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime_a1
                xa2[idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime_a2
                za2[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*y_a[idx_y]
                ya2[idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime_a2

                xb1[idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime_b1 
                zb1[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*y_b[idx_y]
                yb1[idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime_b1
                xb2[idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime_b2 
                zb2[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*y_b[idx_y]                        
                yb2[idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime_b2 

                xah[idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime_ah
                zah[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*y_a[idx_y]
                yah[idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime_ah                    
                xbh[idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime_bh 
                zbh[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*y_b[idx_y]
                ybh[idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime_bh

                xch[idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime_ch
                zch[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*(y_b[idx_y] - del_y[idx_y]/2)                   
                ych[idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime_ch

                xc [idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime 
                zc [idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*(y_b[idx_y] - del_y[idx_y]/2) 
                yc [idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime 

                xac[idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime_ac 
                zac[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*y_a[idx_y]
                yac[idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime_ac
                xbc[idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime_bc
                zbc[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*y_b[idx_y]                            
                ybc[idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime_bc
                x[idx_y*(n_cw+1):(idx_y+1)*(n_cw+1)] = np.concatenate([xi_prime_a1,np.array([xi_prime_a2[-1]])])
                z[idx_y*(n_cw+1):(idx_y+1)*(n_cw+1)] = np.ones(n_cw+1)*y_a[idx_y] 
                y[idx_y*(n_cw+1):(idx_y+1)*(n_cw+1)] = np.concatenate([zeta_prime_a1,np.array([zeta_prime_a2[-1]])])

            else:     
                xa1[idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime_a1 
                ya1[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*y_a[idx_y]
                za1[idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime_a1
                xa2[idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime_a2
                ya2[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*y_a[idx_y]
                za2[idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime_a2

                xb1[idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime_b1 
                yb1[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*y_b[idx_y]
                zb1[idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime_b1
                yb2[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*y_b[idx_y]
                xb2[idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime_b2 
                zb2[idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime_b2 

                xah[idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime_ah
                yah[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*y_a[idx_y]
                zah[idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime_ah                    
                xbh[idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime_bh 
                ybh[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*y_b[idx_y]
                zbh[idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime_bh

                xch[idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime_ch
                ych[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*(y_b[idx_y] - del_y[idx_y]/2)                    
                zch[idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime_ch

                xc [idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime 
                yc [idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*(y_b[idx_y] - del_y[idx_y]/2)
                zc [idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime 

                xac[idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime_ac 
                yac[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*y_a[idx_y]
                zac[idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime_ac
                xbc[idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime_bc
                ybc[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*y_b[idx_y]                            
                zbc[idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime_bc   
                x[idx_y*(n_cw+1):(idx_y+1)*(n_cw+1)] = np.concatenate([xi_prime_a1,np.array([xi_prime_a2[-1]])])
                y[idx_y*(n_cw+1):(idx_y+1)*(n_cw+1)] = np.ones(n_cw+1)*y_a[idx_y] 
                z[idx_y*(n_cw+1):(idx_y+1)*(n_cw+1)] = np.concatenate([zeta_prime_a1,np.array([zeta_prime_a2[-1]])])                   

            idx += 1

            cs_w[idx_y] = wing_chord_section       

            if y_b[idx_y] == section_stations[i_seg+1]: 
                i_seg += 1      

        if vertical_wing:    
            x[-(n_cw+1):] = np.concatenate([xi_prime_b1,np.array([xi_prime_b2[-1]])])
            z[-(n_cw+1):] = np.ones(n_cw+1)*y_b[idx_y] 
            y[-(n_cw+1):] = np.concatenate([zeta_prime_b1,np.array([zeta_prime_b2[-1]])])
        else:    
            x[-(n_cw+1):] = np.concatenate([xi_prime_b1,np.array([xi_prime_b2[-1]])])
            y[-(n_cw+1):] = np.ones(n_cw+1)*y_b[idx_y] 
            z[-(n_cw+1):] = np.concatenate([zeta_prime_b1,np.array([zeta_prime_b2[-1]])])                

    else:   # when no segments are defined on wing  
        # ---------------------------------------------------------------------------------------
        # STEP 6B: Define coordinates of panels horseshoe vortices and control points 
        # ---------------------------------------------------------------------------------------
        y_a   = y_coordinates[:-1] 
        y_b   = y_coordinates[1:] 

        if sweep_le != None:
            sweep = sweep_le
        else:                                                                
            cf    = 0.25                          
            sweep = np.arctan(((root_chord*cf) + (np.tan(sweep_qc)*span - cf*tip_chord)) /span)  

        wing_chord_ratio = (tip_chord-root_chord)/span
        wing_twist_ratio = (twist_tc-twist_rc)/span                    
        VD.wing_areas.append(0.5*(root_chord+tip_chord)*span) 
        if sym_para is True :
            VD.wing_areas.append(0.5*(root_chord+tip_chord)*span)   

        # Get airfoil section VD  
        if wing.Airfoil: 
            airfoil_data = import_airfoil_geometry([wing.Airfoil.airfoil.coordinate_file])    
            wing_camber  = airfoil_data.camber_coordinates[0]
            wing_x_coord = airfoil_data.x_lower_surface[0]
        else:
            wing_camber  = np.zeros(30) # dimension of Selig airfoil VD file
            wing_x_coord = np.linspace(0,1,30)

        del_y = y_b - y_a
        for idx_y in range(n_sw):  
            idx_x = np.arange(n_cw) 
            eta_a = (y_a[idx_y])  
            eta_b = (y_b[idx_y]) 
            eta   = (y_b[idx_y] - del_y[idx_y]/2) 

            # get spanwise discretization points
            wing_chord_section_a  = root_chord + (eta_a*wing_chord_ratio) 
            wing_chord_section_b  = root_chord + (eta_b*wing_chord_ratio)
            wing_chord_section    = root_chord + (eta*wing_chord_ratio)

            # get chordwise discretization points
            delta_x_a = wing_chord_section_a/n_cw   
            delta_x_b = wing_chord_section_b/n_cw   
            delta_x   = wing_chord_section/n_cw                                  

            xi_a1 = eta_a*np.tan(sweep) + delta_x_a*idx_x                  # x coordinate of top left corner of panel
            xi_ah = eta_a*np.tan(sweep) + delta_x_a*idx_x + delta_x_a*0.25 # x coordinate of left corner of panel
            xi_a2 = eta_a*np.tan(sweep) + delta_x_a*idx_x + delta_x_a      # x coordinate of bottom left corner of bound vortex 
            xi_ac = eta_a*np.tan(sweep) + delta_x_a*idx_x + delta_x_a*0.75 # x coordinate of bottom left corner of control point vortex  
            xi_b1 = eta_b*np.tan(sweep) + delta_x_b*idx_x                  # x coordinate of top right corner of panel      
            xi_bh = eta_b*np.tan(sweep) + delta_x_b*idx_x + delta_x_b*0.25 # x coordinate of right corner of bound vortex         
            xi_b2 = eta_b*np.tan(sweep) + delta_x_b*idx_x + delta_x_b      # x coordinate of bottom right corner of panel
            xi_bc = eta_b*np.tan(sweep) + delta_x_b*idx_x + delta_x_b*0.75 # x coordinate of bottom right corner of control point vortex         
            xi_c  =  eta *np.tan(sweep)  + delta_x  *idx_x + delta_x*0.75   # x coordinate three-quarter chord control point for each panel
            xi_ch =  eta *np.tan(sweep)  + delta_x  *idx_x + delta_x*0.25   # x coordinate center of bound vortex of each panel 

            # adjustment of coordinates for camber
            section_camber_a  = wing_camber*wing_chord_section_a
            section_camber_b  = wing_camber*wing_chord_section_b  
            section_camber_c  = wing_camber*wing_chord_section

            section_x_coord_a = wing_x_coord*wing_chord_section_a
            section_x_coord_b = wing_x_coord*wing_chord_section_b
            section_x_coord   = wing_x_coord*wing_chord_section

            z_c_a1 = np.interp((idx_x    *delta_x_a)                  ,section_x_coord_a,section_camber_a) 
            z_c_ah = np.interp((idx_x    *delta_x_a + delta_x_a*0.25) ,section_x_coord_a,section_camber_a)
            z_c_a2 = np.interp(((idx_x+1)*delta_x_a)                  ,section_x_coord_a,section_camber_a) 
            z_c_ac = np.interp((idx_x    *delta_x_a + delta_x_a*0.75) ,section_x_coord_a,section_camber_a) 
            z_c_b1 = np.interp((idx_x    *delta_x_b)                  ,section_x_coord_b,section_camber_b)   
            z_c_bh = np.interp((idx_x    *delta_x_b + delta_x_b*0.25) ,section_x_coord_b,section_camber_b) 
            z_c_b2 = np.interp(((idx_x+1)*delta_x_b)                  ,section_x_coord_b,section_camber_b) 
            z_c_bc = np.interp((idx_x    *delta_x_b + delta_x_b*0.75) ,section_x_coord_b,section_camber_b) 
            z_c    = np.interp((idx_x    *delta_x   + delta_x  *0.75) ,section_x_coord  ,section_camber_c) 
            z_c_ch = np.interp((idx_x    *delta_x   + delta_x  *0.25) ,section_x_coord  ,section_camber_c) 

            zeta_a1 = eta_a*np.tan(dihedral)  + z_c_a1  # z coordinate of top left corner of panel
            zeta_ah = eta_a*np.tan(dihedral)  + z_c_ah  # z coordinate of left corner of bound vortex  
            zeta_a2 = eta_a*np.tan(dihedral)  + z_c_a2  # z coordinate of bottom left corner of panel
            zeta_ac = eta_a*np.tan(dihedral)  + z_c_ac  # z coordinate of bottom left corner of panel of control point
            zeta_bc = eta_b*np.tan(dihedral)  + z_c_bc  # z coordinate of top right corner of panel of control point                          
            zeta_b1 = eta_b*np.tan(dihedral)  + z_c_b1  # z coordinate of top right corner of panel  
            zeta_bh = eta_b*np.tan(dihedral)  + z_c_bh  # z coordinate of right corner of bound vortex        
            zeta_b2 = eta_b*np.tan(dihedral)  + z_c_b2  # z coordinate of bottom right corner of panel                 
            zeta    =   eta*np.tan(dihedral)    + z_c     # z coordinate three-quarter chord control point for each panel
            zeta_ch =   eta*np.tan(dihedral)    + z_c_ch  # z coordinate center of bound vortex on each panel

            # adjustment of coordinates for twist  
            xi_LE_a = eta_a*np.tan(sweep)               # x location of leading edge left corner of wing
            xi_LE_b = eta_b*np.tan(sweep)               # x location of leading edge right of wing
            xi_LE   = eta  *np.tan(sweep)               # x location of leading edge center of wing

            zeta_LE_a = eta_a*np.tan(dihedral)          # z location of leading edge left corner of wing
            zeta_LE_b = eta_b*np.tan(dihedral)          # z location of leading edge right of wing
            zeta_LE   = eta  *np.tan(dihedral)          # z location of leading edge center of wing

            # determine section twist
            section_twist_a = twist_rc + (eta_a * wing_twist_ratio)                     # twist at left side of panel
            section_twist_b = twist_rc + (eta_b * wing_twist_ratio)                     # twist at right side of panel
            section_twist   = twist_rc + (eta   * wing_twist_ratio)                     # twist at center local chord 

            xi_prime_a1  = xi_LE_a + np.cos(section_twist_a)*(xi_a1-xi_LE_a) + np.sin(section_twist_a)*(zeta_a1-zeta_LE_a)   # x coordinate transformation of top left corner
            xi_prime_ah  = xi_LE_a + np.cos(section_twist_a)*(xi_ah-xi_LE_a) + np.sin(section_twist_a)*(zeta_ah-zeta_LE_a)   # x coordinate transformation of bottom left corner
            xi_prime_a2  = xi_LE_a + np.cos(section_twist_a)*(xi_a2-xi_LE_a) + np.sin(section_twist_a)*(zeta_a2-zeta_LE_a)   # x coordinate transformation of bottom left corner
            xi_prime_ac  = xi_LE_a + np.cos(section_twist_a)*(xi_ac-xi_LE_a) + np.sin(section_twist_a)*(zeta_a2-zeta_LE_a)   # x coordinate transformation of bottom left corner of control point
            xi_prime_bc  = xi_LE_b + np.cos(section_twist_b)*(xi_bc-xi_LE_b) + np.sin(section_twist_b)*(zeta_b1-zeta_LE_b)   # x coordinate transformation of top right corner of control point                         
            xi_prime_b1  = xi_LE_b + np.cos(section_twist_b)*(xi_b1-xi_LE_b) + np.sin(section_twist_b)*(zeta_b1-zeta_LE_b)   # x coordinate transformation of top right corner 
            xi_prime_bh  = xi_LE_b + np.cos(section_twist_b)*(xi_bh-xi_LE_b) + np.sin(section_twist_b)*(zeta_bh-zeta_LE_b)   # x coordinate transformation of top right corner 
            xi_prime_b2  = xi_LE_b + np.cos(section_twist_b)*(xi_b2-xi_LE_b) + np.sin(section_twist_b)*(zeta_b2-zeta_LE_b)   # x coordinate transformation of botton right corner 
            xi_prime     = xi_LE   + np.cos(section_twist)  *(xi_c-xi_LE)    + np.sin(section_twist)*(zeta-zeta_LE)          # x coordinate transformation of control point
            xi_prime_ch  = xi_LE   + np.cos(section_twist)  *(xi_ch-xi_LE)   + np.sin(section_twist)*(zeta_ch-zeta_LE)       # x coordinate transformation of center of horeshoe vortex 

            zeta_prime_a1  = zeta_LE_a - np.sin(section_twist_a)*(xi_a1-xi_LE_a) + np.cos(section_twist_a)*(zeta_a1-zeta_LE_a) # z coordinate transformation of top left corner
            zeta_prime_ah  = zeta_LE_a - np.sin(section_twist_a)*(xi_ah-xi_LE_a) + np.cos(section_twist_a)*(zeta_ah-zeta_LE_a) # z coordinate transformation of bottom left corner
            zeta_prime_a2  = zeta_LE_a - np.sin(section_twist_a)*(xi_a2-xi_LE_a) + np.cos(section_twist_a)*(zeta_a2-zeta_LE_a) # z coordinate transformation of bottom left corner
            zeta_prime_ac  = zeta_LE_a - np.sin(section_twist_a)*(xi_ac-xi_LE_a) + np.cos(section_twist_a)*(zeta_ac-zeta_LE_a) # z coordinate transformation of bottom left corner
            zeta_prime_bc  = zeta_LE_b - np.sin(section_twist_b)*(xi_bc-xi_LE_b) + np.cos(section_twist_b)*(zeta_bc-zeta_LE_b) # z coordinate transformation of top right corner                         
            zeta_prime_b1  = zeta_LE_b - np.sin(section_twist_b)*(xi_b1-xi_LE_b) + np.cos(section_twist_b)*(zeta_b1-zeta_LE_b) # z coordinate transformation of top right corner 
            zeta_prime_bh  = zeta_LE_b - np.sin(section_twist_b)*(xi_bh-xi_LE_b) + np.cos(section_twist_b)*(zeta_bh-zeta_LE_b) # z coordinate transformation of top right corner 
            zeta_prime_b2  = zeta_LE_b - np.sin(section_twist_b)*(xi_b2-xi_LE_b) + np.cos(section_twist_b)*(zeta_b2-zeta_LE_b) # z coordinate transformation of botton right corner 
            zeta_prime     = zeta_LE   - np.sin(section_twist)  *(xi_c-xi_LE)    + np.cos(-section_twist) *(zeta-zeta_LE)      # z coordinate transformation of control point
            zeta_prime_ch  = zeta_LE   - np.sin(section_twist)  *(xi_ch-xi_LE)   + np.cos(-section_twist) *(zeta_ch-zeta_LE)   # z coordinate transformation of center of horseshoe

            # ** TO DO ** Get flap/aileron locations and deflection

            # store coordinates of panels, horseshoe vortices and control points relative to wing root 
            if vertical_wing:
                xa1[idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime_a1 
                za1[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*y_a[idx_y]
                ya1[idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime_a1
                xa2[idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime_a2
                za2[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*y_a[idx_y]
                ya2[idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime_a2

                xb1[idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime_b1 
                zb1[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*y_b[idx_y]
                yb1[idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime_b1
                xb2[idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime_b2 
                zb2[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*y_b[idx_y]                        
                yb2[idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime_b2 

                xah[idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime_ah
                zah[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*y_a[idx_y]
                yah[idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime_ah                    
                xbh[idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime_bh 
                zbh[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*y_b[idx_y]
                ybh[idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime_bh

                xch[idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime_ch
                zch[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*(y_b[idx_y] - del_y[idx_y]/2)                   
                ych[idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime_ch

                xc [idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime 
                zc [idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*(y_b[idx_y] - del_y[idx_y]/2) 
                yc [idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime 

                xac[idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime_ac 
                zac[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*y_a[idx_y]
                yac[idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime_ac
                xbc[idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime_bc
                zbc[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*y_b[idx_y]                            
                ybc[idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime_bc        
                x[idx_y*(n_cw+1):(idx_y+1)*(n_cw+1)] = np.concatenate([xi_prime_a1,np.array([xi_prime_a2[-1]])])
                z[idx_y*(n_cw+1):(idx_y+1)*(n_cw+1)] = np.ones(n_cw+1)*y_a[idx_y] 
                y[idx_y*(n_cw+1):(idx_y+1)*(n_cw+1)] = np.concatenate([zeta_prime_a1,np.array([zeta_prime_a2[-1]])])                    

            else: 
                xa1[idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime_a1 
                ya1[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*y_a[idx_y]
                za1[idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime_a1
                xa2[idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime_a2
                ya2[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*y_a[idx_y]
                za2[idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime_a2

                xb1[idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime_b1 
                yb1[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*y_b[idx_y]
                zb1[idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime_b1
                yb2[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*y_b[idx_y]
                xb2[idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime_b2 
                zb2[idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime_b2 

                xah[idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime_ah
                yah[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*y_a[idx_y]
                zah[idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime_ah                    
                xbh[idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime_bh 
                ybh[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*y_b[idx_y]
                zbh[idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime_bh

                xch[idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime_ch
                ych[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*(y_b[idx_y] - del_y[idx_y]/2)                   
                zch[idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime_ch

                xc [idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime 
                yc [idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*(y_b[idx_y] - del_y[idx_y]/2) 
                zc [idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime 

                xac[idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime_ac 
                yac[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*y_a[idx_y]
                zac[idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime_ac
                xbc[idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime_bc
                ybc[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*y_b[idx_y]                            
                zbc[idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime_bc       
                x[idx_y*(n_cw+1):(idx_y+1)*(n_cw+1)] = np.concatenate([xi_prime_a1,np.array([xi_prime_a2[-1]])])
                y[idx_y*(n_cw+1):(idx_y+1)*(n_cw+1)] = np.ones(n_cw+1)*y_a[idx_y] 
                z[idx_y*(n_cw+1):(idx_y+1)*(n_cw+1)] = np.concatenate([zeta_prime_a1,np.array([zeta_prime_a2[-1]])])              

            cs_w[idx_y] = wing_chord_section

        if vertical_wing:    
            x[-(n_cw+1):] = np.concatenate([xi_prime_b1,np.array([xi_prime_b2[-1]])])
            z[-(n_cw+1):] = np.ones(n_cw+1)*y_b[idx_y] 
            y[-(n_cw+1):] = np.concatenate([zeta_prime_b1,np.array([zeta_prime_b2[-1]])])
        else:           
            x[-(n_cw+1):] = np.concatenate([xi_prime_b1,np.array([xi_prime_b2[-1]])])
            y[-(n_cw+1):] = np.ones(n_cw+1)*y_b[idx_y] 
            z[-(n_cw+1):] = np.concatenate([zeta_prime_b1,np.array([zeta_prime_b2[-1]])])   

    # adjusting coordinate axis so reference point is at the nose of the aircraft
    xah = xah + wing_origin[0] # x coordinate of left corner of bound vortex 
    yah = yah + wing_origin[1] # y coordinate of left corner of bound vortex 
    zah = zah + wing_origin[2] # z coordinate of left corner of bound vortex 
    xbh = xbh + wing_origin[0] # x coordinate of right corner of bound vortex 
    ybh = ybh + wing_origin[1] # y coordinate of right corner of bound vortex 
    zbh = zbh + wing_origin[2] # z coordinate of right corner of bound vortex 
    xch = xch + wing_origin[0] # x coordinate of center of bound vortex on panel
    ych = ych + wing_origin[1] # y coordinate of center of bound vortex on panel
    zch = zch + wing_origin[2] # z coordinate of center of bound vortex on panel  

    xa1 = xa1 + wing_origin[0] # x coordinate of top left corner of panel
    ya1 = ya1 + wing_origin[1] # y coordinate of bottom left corner of panel
    za1 = za1 + wing_origin[2] # z coordinate of top left corner of panel
    xa2 = xa2 + wing_origin[0] # x coordinate of bottom left corner of panel
    ya2 = ya2 + wing_origin[1] # x coordinate of bottom left corner of panel
    za2 = za2 + wing_origin[2] # z coordinate of bottom left corner of panel  

    xb1 = xb1 + wing_origin[0] # x coordinate of top right corner of panel  
    yb1 = yb1 + wing_origin[1] # y coordinate of top right corner of panel 
    zb1 = zb1 + wing_origin[2] # z coordinate of top right corner of panel 
    xb2 = xb2 + wing_origin[0] # x coordinate of bottom rightcorner of panel 
    yb2 = yb2 + wing_origin[1] # y coordinate of bottom rightcorner of panel 
    zb2 = zb2 + wing_origin[2] # z coordinate of bottom right corner of panel                   

    xac = xac + wing_origin[0]  # x coordinate of control points on panel
    yac = yac + wing_origin[1]  # y coordinate of control points on panel
    zac = zac + wing_origin[2]  # z coordinate of control points on panel
    xbc = xbc + wing_origin[0]  # x coordinate of control points on panel
    ybc = ybc + wing_origin[1]  # y coordinate of control points on panel
    zbc = zbc + wing_origin[2]  # z coordinate of control points on panel

    xc  = xc + wing_origin[0]  # x coordinate of control points on panel
    yc  = yc + wing_origin[1]  # y coordinate of control points on panel
    zc  = zc + wing_origin[2]  # y coordinate of control points on panel
    x   = x + wing_origin[0]   # x coordinate of control points on panel
    y   = y + wing_origin[1]   # y coordinate of control points on panel
    z   = z + wing_origin[2]   # y coordinate of control points on panel

    # find the location of the trailing edge panels of each wing
    locations = ((np.linspace(1,n_sw,n_sw, endpoint = True) * n_cw) - 1).astype(int)
    xc_te1 = np.repeat(np.atleast_2d(xc[locations]), n_cw , axis = 0)
    yc_te1 = np.repeat(np.atleast_2d(yc[locations]), n_cw , axis = 0)
    zc_te1 = np.repeat(np.atleast_2d(zc[locations]), n_cw , axis = 0)        
    xa_te1 = np.repeat(np.atleast_2d(xa2[locations]), n_cw , axis = 0)
    ya_te1 = np.repeat(np.atleast_2d(ya2[locations]), n_cw , axis = 0)
    za_te1 = np.repeat(np.atleast_2d(za2[locations]), n_cw , axis = 0)
    xb_te1 = np.repeat(np.atleast_2d(xb2[locations]), n_cw , axis = 0)
    yb_te1 = np.repeat(np.atleast_2d(yb2[locations]), n_cw , axis = 0)
    zb_te1 = np.repeat(np.atleast_2d(zb2[locations]), n_cw , axis = 0)     

    xc_te = np.hstack(xc_te1.T)
    yc_te = np.hstack(yc_te1.T)
    zc_te = np.hstack(zc_te1.T)        
    xa_te = np.hstack(xa_te1.T)
    ya_te = np.hstack(ya_te1.T)
    za_te = np.hstack(za_te1.T)
    xb_te = np.hstack(xb_te1.T)
    yb_te = np.hstack(yb_te1.T)
    zb_te = np.hstack(zb_te1.T) 

    # find spanwise locations 
    y_sw = yc[locations]        

    # if symmetry, store points of mirrored wing 
    VD.n_w += 1  
    if sym_para is True :
        VD.n_w += 1 
        # append wing spans          
        if vertical_wing:
            del_y = np.concatenate([del_y,del_y]) 
            cs_w  = np.concatenate([cs_w,cs_w])
            xah   = np.concatenate([xah,xah])
            yah   = np.concatenate([yah,yah])
            zah   = np.concatenate([zah,-zah])
            xbh   = np.concatenate([xbh,xbh])
            ybh   = np.concatenate([ybh,ybh])
            zbh   = np.concatenate([zbh,-zbh])
            xch   = np.concatenate([xch,xch])
            ych   = np.concatenate([ych,ych])
            zch   = np.concatenate([zch,-zch])

            xa1   = np.concatenate([xa1,xa1])
            ya1   = np.concatenate([ya1,ya1])
            za1   = np.concatenate([za1,-za1])
            xa2   = np.concatenate([xa2,xa2])
            ya2   = np.concatenate([ya2,ya2])
            za2   = np.concatenate([za2,-za2])

            xb1   = np.concatenate([xb1,xb1])
            yb1   = np.concatenate([yb1,yb1])    
            zb1   = np.concatenate([zb1,-zb1])
            xb2   = np.concatenate([xb2,xb2])
            yb2   = np.concatenate([yb2,yb2])            
            zb2   = np.concatenate([zb2,-zb2])

            xac   = np.concatenate([xac ,xac ])
            yac   = np.concatenate([yac ,yac ])
            zac   = np.concatenate([zac ,-zac ])            
            xbc   = np.concatenate([xbc ,xbc ])
            ybc   = np.concatenate([ybc ,ybc ])
            zbc   = np.concatenate([zbc ,-zbc ]) 
            xc_te = np.concatenate([xc_te , xc_te ])
            yc_te = np.concatenate([yc_te , yc_te ])
            zc_te = np.concatenate([zc_te ,-zc_te ])                 
            xa_te = np.concatenate([xa_te , xa_te ])
            ya_te = np.concatenate([ya_te , ya_te ])
            za_te = np.concatenate([za_te ,-za_te ])            
            xb_te = np.concatenate([xb_te , xb_te ])
            yb_te = np.concatenate([yb_te , yb_te ])
            zb_te = np.concatenate([zb_te ,-zb_te ])

            y_sw  = np.concatenate([y_sw,-y_sw ])
            xc    = np.concatenate([xc ,xc ])
            yc    = np.concatenate([yc ,yc]) 
            zc    = np.concatenate([zc ,-zc ])
            x     = np.concatenate([x , x ])
            y     = np.concatenate([y ,y])
            z     = np.concatenate([z ,-z ])                  

        else:
            del_y = np.concatenate([del_y,del_y]) 
            cs_w  = np.concatenate([cs_w,cs_w])
            xah   = np.concatenate([xah,xah])
            yah   = np.concatenate([yah,-yah])
            zah   = np.concatenate([zah,zah])
            xbh   = np.concatenate([xbh,xbh])
            ybh   = np.concatenate([ybh,-ybh])
            zbh   = np.concatenate([zbh,zbh])
            xch   = np.concatenate([xch,xch])
            ych   = np.concatenate([ych,-ych])
            zch   = np.concatenate([zch,zch])

            xa1   = np.concatenate([xa1,xa1])
            ya1   = np.concatenate([ya1,-ya1])
            za1   = np.concatenate([za1,za1])
            xa2   = np.concatenate([xa2,xa2])
            ya2   = np.concatenate([ya2,-ya2])
            za2   = np.concatenate([za2,za2])

            xb1   = np.concatenate([xb1,xb1])
            yb1   = np.concatenate([yb1,-yb1])    
            zb1   = np.concatenate([zb1,zb1])
            xb2   = np.concatenate([xb2,xb2])
            yb2   = np.concatenate([yb2,-yb2])            
            zb2   = np.concatenate([zb2,zb2])

            xac   = np.concatenate([xac ,xac ])
            yac   = np.concatenate([yac ,-yac ])
            zac   = np.concatenate([zac ,zac ])            
            xbc   = np.concatenate([xbc ,xbc ])
            ybc   = np.concatenate([ybc ,-ybc ])
            zbc   = np.concatenate([zbc ,zbc ]) 
            xc_te = np.concatenate([xc_te , xc_te ])
            yc_te = np.concatenate([yc_te ,-yc_te ])
            zc_te = np.concatenate([zc_te , zc_te ])                   
            xa_te = np.concatenate([xa_te , xa_te ])
            ya_te = np.concatenate([ya_te ,-ya_te ])
            za_te = np.concatenate([za_te , za_te ])            
            xb_te = np.concatenate([xb_te , xb_te ])
            yb_te = np.concatenate([yb_te ,-yb_te ])
            zb_te = np.concatenate([zb_te , zb_te ]) 

            y_sw  = np.concatenate([y_sw,-y_sw ])
            xc    = np.concatenate([xc ,xc ])
            yc    = np.concatenate([yc ,-yc]) 
            zc    = np.concatenate([zc ,zc ])
            x     = np.concatenate([x , x ])
            y     = np.concatenate([y ,-y])
            z     = np.concatenate([z , z ])            

    VD.n_cp += len(xch)        

    # ---------------------------------------------------------------------------------------
    # STEP 7: Store wing in vehicle vector
    # ---------------------------------------------------------------------------------------       
    VD.XAH    = np.append(VD.XAH,xah)
    VD.YAH    = np.append(VD.YAH,yah)
    VD.ZAH    = np.append(VD.ZAH,zah)
    VD.XBH    = np.append(VD.XBH,xbh)
    VD.YBH    = np.append(VD.YBH,ybh)
    VD.ZBH    = np.append(VD.ZBH,zbh)
    VD.XCH    = np.append(VD.XCH,xch)
    VD.YCH    = np.append(VD.YCH,ych)
    VD.ZCH    = np.append(VD.ZCH,zch)            
    VD.XA1    = np.append(VD.XA1,xa1)
    VD.YA1    = np.append(VD.YA1,ya1)
    VD.ZA1    = np.append(VD.ZA1,za1)
    VD.XA2    = np.append(VD.XA2,xa2)
    VD.YA2    = np.append(VD.YA2,ya2)
    VD.ZA2    = np.append(VD.ZA2,za2)        
    VD.XB1    = np.append(VD.XB1,xb1)
    VD.YB1    = np.append(VD.YB1,yb1)
    VD.ZB1    = np.append(VD.ZB1,zb1)
    VD.XB2    = np.append(VD.XB2,xb2)                
    VD.YB2    = np.append(VD.YB2,yb2)        
    VD.ZB2    = np.append(VD.ZB2,zb2)    
    VD.XC_TE  = np.append(VD.XC_TE,xc_te)
    VD.YC_TE  = np.append(VD.YC_TE,yc_te) 
    VD.ZC_TE  = np.append(VD.ZC_TE,zc_te)          
    VD.XA_TE  = np.append(VD.XA_TE,xa_te)
    VD.YA_TE  = np.append(VD.YA_TE,ya_te) 
    VD.ZA_TE  = np.append(VD.ZA_TE,za_te) 
    VD.XB_TE  = np.append(VD.XB_TE,xb_te)
    VD.YB_TE  = np.append(VD.YB_TE,yb_te) 
    VD.ZB_TE  = np.append(VD.ZB_TE,zb_te)  
    VD.XAC    = np.append(VD.XAC,xac)
    VD.YAC    = np.append(VD.YAC,yac) 
    VD.ZAC    = np.append(VD.ZAC,zac) 
    VD.XBC    = np.append(VD.XBC,xbc)
    VD.YBC    = np.append(VD.YBC,ybc) 
    VD.ZBC    = np.append(VD.ZBC,zbc)  
    VD.XC     = np.append(VD.XC ,xc)
    VD.YC     = np.append(VD.YC ,yc)
    VD.ZC     = np.append(VD.ZC ,zc)  
    VD.X      = np.append(VD.X ,x)
    VD.Y_SW   = np.append(VD.Y_SW ,y_sw)
    VD.Y      = np.append(VD.Y ,y)
    VD.Z      = np.append(VD.Z ,z)         
    VD.CS     = np.append(VD.CS,cs_w) 
    VD.DY     = np.append(VD.DY ,del_y)

    return VD


## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
def generate_fuselage_vortex_distribution(VD,fus,n_cw,n_sw,model_fuselage=False):
    """ This generates the vortex distribution points on the fuselage 

    Assumptions: 
    None

    Source:   
    None
    
    Inputs:   
    VD                   - vortex distribution    
    
    Properties Used:
    N/A
    """    
    
    fhs_xa1 = np.zeros(n_cw*n_sw)
    fhs_ya1 = np.zeros(n_cw*n_sw)
    fhs_za1 = np.zeros(n_cw*n_sw)
    fhs_xa2 = np.zeros(n_cw*n_sw)
    fhs_ya2 = np.zeros(n_cw*n_sw)
    fhs_za2 = np.zeros(n_cw*n_sw)
    fhs_xb1 = np.zeros(n_cw*n_sw)
    fhs_yb1 = np.zeros(n_cw*n_sw)
    fhs_zb1 = np.zeros(n_cw*n_sw)
    fhs_yb2 = np.zeros(n_cw*n_sw)
    fhs_xb2 = np.zeros(n_cw*n_sw)
    fhs_zb2 = np.zeros(n_cw*n_sw)
    fhs_xah = np.zeros(n_cw*n_sw)
    fhs_yah = np.zeros(n_cw*n_sw)
    fhs_zah = np.zeros(n_cw*n_sw)
    fhs_xbh = np.zeros(n_cw*n_sw)
    fhs_ybh = np.zeros(n_cw*n_sw)
    fhs_zbh = np.zeros(n_cw*n_sw)
    fhs_xch = np.zeros(n_cw*n_sw)
    fhs_ych = np.zeros(n_cw*n_sw)
    fhs_zch = np.zeros(n_cw*n_sw)
    fhs_xc  = np.zeros(n_cw*n_sw)
    fhs_yc  = np.zeros(n_cw*n_sw)
    fhs_zc  = np.zeros(n_cw*n_sw)
    fhs_xac = np.zeros(n_cw*n_sw)
    fhs_yac = np.zeros(n_cw*n_sw)
    fhs_zac = np.zeros(n_cw*n_sw)
    fhs_xbc = np.zeros(n_cw*n_sw)
    fhs_ybc = np.zeros(n_cw*n_sw)
    fhs_zbc = np.zeros(n_cw*n_sw)
    fhs_x   = np.zeros((n_cw+1)*(n_sw+1))
    fhs_y   = np.zeros((n_cw+1)*(n_sw+1))
    fhs_z   = np.zeros((n_cw+1)*(n_sw+1))      

    fvs_xc    = np.zeros(n_cw*n_sw)
    fvs_zc    = np.zeros(n_cw*n_sw)
    fvs_yc    = np.zeros(n_cw*n_sw)   
    fvs_x     = np.zeros((n_cw+1)*(n_sw+1))
    fvs_y     = np.zeros((n_cw+1)*(n_sw+1))
    fvs_z     = np.zeros((n_cw+1)*(n_sw+1))   
    fus_v_cs  = np.zeros(n_sw)     

    semispan_h = fus.width * 0.5  
    semispan_v = fus.heights.maximum * 0.5
    origin     = fus.origin[0]

    # Compute the curvature of the nose/tail given fineness ratio. Curvature is derived from general quadratic equation
    # This method relates the fineness ratio to the quadratic curve formula via a spline fit interpolation
    vec1               = [2 , 1.5, 1.2 , 1]
    vec2               = [1  ,1.57 , 3.2,  8]
    x                  = np.linspace(0,1,4)
    fus_nose_curvature =  np.interp(np.interp(fus.fineness.nose,vec2,x), x , vec1)
    fus_tail_curvature =  np.interp(np.interp(fus.fineness.tail,vec2,x), x , vec1) 

    # Horizontal Sections of fuselage
    fhs        = Data()        
    fhs.origin = np.zeros((n_sw+1,3))        
    fhs.chord  = np.zeros((n_sw+1))         
    fhs.sweep  = np.zeros((n_sw+1))     
                 
    fvs        = Data() 
    fvs.origin = np.zeros((n_sw+1,3))
    fvs.chord  = np.zeros((n_sw+1)) 
    fvs.sweep  = np.zeros((n_sw+1)) 

    si         = np.arange(1,((n_sw*2)+2))
    spacing    = np.cos((2*si - 1)/(2*len(si))*np.pi)     
    h_array    = semispan_h*spacing[0:int((len(si)+1)/2)][::-1]  
    v_array    = semispan_v*spacing[0:int((len(si)+1)/2)][::-1]  

    for i in range(n_sw+1): 
        fhs_cabin_length  = fus.lengths.total - (fus.lengths.nose + fus.lengths.tail)
        fhs.nose_length   = ((1 - ((abs(h_array[i]/semispan_h))**fus_nose_curvature ))**(1/fus_nose_curvature))*fus.lengths.nose
        fhs.tail_length   = ((1 - ((abs(h_array[i]/semispan_h))**fus_tail_curvature ))**(1/fus_tail_curvature))*fus.lengths.tail
        fhs.nose_origin   = fus.lengths.nose - fhs.nose_length 
        fhs.origin[i][:]  = np.array([origin[0] + fhs.nose_origin , origin[1] + h_array[i], origin[2]])
        fhs.chord[i]      = fhs_cabin_length + fhs.nose_length + fhs.tail_length          

        fvs_cabin_length  = fus.lengths.total - (fus.lengths.nose + fus.lengths.tail)
        fvs.nose_length   = ((1 - ((abs(v_array[i]/semispan_v))**fus_nose_curvature ))**(1/fus_nose_curvature))*fus.lengths.nose
        fvs.tail_length   = ((1 - ((abs(v_array[i]/semispan_v))**fus_tail_curvature ))**(1/fus_tail_curvature))*fus.lengths.tail
        fvs.nose_origin   = fus.lengths.nose - fvs.nose_length 
        fvs.origin[i][:]  = np.array([origin[0] + fvs.nose_origin , origin[1] , origin[2]+  v_array[i]])
        fvs.chord[i]      = fvs_cabin_length + fvs.nose_length + fvs.tail_length

    fhs.sweep[:] = np.concatenate([np.arctan((fhs.origin[:,0][1:] - fhs.origin[:,0][:-1])/(fhs.origin[:,1][1:]  - fhs.origin[:,1][:-1])) ,np.zeros(1)])
    fvs.sweep[:] = np.concatenate([np.arctan((fvs.origin[:,0][1:] - fvs.origin[:,0][:-1])/(fvs.origin[:,2][1:]  - fvs.origin[:,2][:-1])) ,np.zeros(1)])

    # ---------------------------------------------------------------------------------------
    # STEP 9: Define coordinates of panels horseshoe vortices and control points  
    # ---------------------------------------------------------------------------------------        
    fhs_eta_a = h_array[:-1] 
    fhs_eta_b = h_array[1:]            
    fhs_del_y = h_array[1:] - h_array[:-1]
    fhs_eta   = h_array[1:] - fhs_del_y/2

    fvs_eta_a = v_array[:-1] 
    fvs_eta_b = v_array[1:]                  
    fvs_del_y = v_array[1:] - v_array[:-1]
    fvs_eta   = v_array[1:] - fvs_del_y/2 

    fhs_cs = np.concatenate([fhs.chord,fhs.chord])
    fvs_cs = np.concatenate([fvs.chord,fvs.chord])
    
    fus_h_area = 0
    fus_v_area = 0    

    # define coordinates of horseshoe vortices and control points       
    for idx_y in range(n_sw):  
        idx_x = np.arange(n_cw)

        # fuselage horizontal section 
        delta_x_a = fhs.chord[idx_y]/n_cw      
        delta_x_b = fhs.chord[idx_y + 1]/n_cw    
        delta_x   = (fhs.chord[idx_y]+fhs.chord[idx_y + 1])/(2*n_cw)

        fhs_xi_a1 = fhs.origin[idx_y][0] + delta_x_a*idx_x                    # x coordinate of top left corner of panel
        fhs_xi_ah = fhs.origin[idx_y][0] + delta_x_a*idx_x + delta_x_a*0.25   # x coordinate of left corner of panel
        fhs_xi_a2 = fhs.origin[idx_y][0] + delta_x_a*idx_x + delta_x_a        # x coordinate of bottom left corner of bound vortex 
        fhs_xi_ac = fhs.origin[idx_y][0] + delta_x_a*idx_x + delta_x_a*0.75   # x coordinate of bottom left corner of control point vortex  
        fhs_xi_b1 = fhs.origin[idx_y+1][0] + delta_x_b*idx_x                  # x coordinate of top right corner of panel      
        fhs_xi_bh = fhs.origin[idx_y+1][0] + delta_x_b*idx_x + delta_x_b*0.25 # x coordinate of right corner of bound vortex         
        fhs_xi_b2 = fhs.origin[idx_y+1][0] + delta_x_b*idx_x + delta_x_b      # x coordinate of bottom right corner of panel
        fhs_xi_bc = fhs.origin[idx_y+1][0] + delta_x_b*idx_x + delta_x_b*0.75 # x coordinate of bottom right corner of control point vortex         
        fhs_xi_c  = (fhs.origin[idx_y][0] + fhs.origin[idx_y+1][0])/2  + delta_x*idx_x + delta_x*0.75   # x coordinate three-quarter chord control point for each panel
        fhs_xi_ch = (fhs.origin[idx_y][0] + fhs.origin[idx_y+1][0])/2  + delta_x*idx_x + delta_x*0.25   # x coordinate center of bound vortex of each panel 


        fhs_xa1[idx_y*n_cw:(idx_y+1)*n_cw] = fhs_xi_a1                       + fus.origin[0][0]  
        fhs_ya1[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*fhs_eta_a[idx_y]  + fus.origin[0][1]  
        fhs_za1[idx_y*n_cw:(idx_y+1)*n_cw] = np.zeros(n_cw)                  + fus.origin[0][2]
        fhs_xa2[idx_y*n_cw:(idx_y+1)*n_cw] = fhs_xi_a2                       + fus.origin[0][0]  
        fhs_ya2[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*fhs_eta_a[idx_y]  + fus.origin[0][1] 
        fhs_za2[idx_y*n_cw:(idx_y+1)*n_cw] = np.zeros(n_cw)                  + fus.origin[0][2]      
        fhs_xb1[idx_y*n_cw:(idx_y+1)*n_cw] = fhs_xi_b1                       + fus.origin[0][0]  
        fhs_yb1[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*fhs_eta_b[idx_y]  + fus.origin[0][1] 
        fhs_zb1[idx_y*n_cw:(idx_y+1)*n_cw] = np.zeros(n_cw)                  + fus.origin[0][2]
        fhs_xb2[idx_y*n_cw:(idx_y+1)*n_cw] = fhs_xi_b2                       + fus.origin[0][0] 
        fhs_yb2[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*fhs_eta_b[idx_y]  + fus.origin[0][1] 
        fhs_zb2[idx_y*n_cw:(idx_y+1)*n_cw] = np.zeros(n_cw)                  + fus.origin[0][2]       
        fhs_xah[idx_y*n_cw:(idx_y+1)*n_cw] = fhs_xi_ah                       + fus.origin[0][0]   
        fhs_yah[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*fhs_eta_a[idx_y]  + fus.origin[0][1]  
        fhs_zah[idx_y*n_cw:(idx_y+1)*n_cw] = np.zeros(n_cw)                  + fus.origin[0][2]             
        fhs_xbh[idx_y*n_cw:(idx_y+1)*n_cw] = fhs_xi_bh                       + fus.origin[0][0] 
        fhs_ybh[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*fhs_eta_b[idx_y]  + fus.origin[0][1]  
        fhs_zbh[idx_y*n_cw:(idx_y+1)*n_cw] = np.zeros(n_cw)                  + fus.origin[0][2]    
        fhs_xch[idx_y*n_cw:(idx_y+1)*n_cw] = fhs_xi_ch                       + fus.origin[0][0]  
        fhs_ych[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*fhs_eta[idx_y]    + fus.origin[0][1]                
        fhs_zch[idx_y*n_cw:(idx_y+1)*n_cw] = np.zeros(n_cw)                  + fus.origin[0][2]     
        fhs_xc [idx_y*n_cw:(idx_y+1)*n_cw] = fhs_xi_c                        + fus.origin[0][0]  
        fhs_yc [idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*fhs_eta[idx_y]    + fus.origin[0][1]  
        fhs_zc [idx_y*n_cw:(idx_y+1)*n_cw] = np.zeros(n_cw)                  + fus.origin[0][2]       
        fhs_xac[idx_y*n_cw:(idx_y+1)*n_cw] = fhs_xi_ac                       + fus.origin[0][0]  
        fhs_yac[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*fhs_eta_a[idx_y]  + fus.origin[0][1]
        fhs_zac[idx_y*n_cw:(idx_y+1)*n_cw] = np.zeros(n_cw)                  + fus.origin[0][2]
        fhs_xbc[idx_y*n_cw:(idx_y+1)*n_cw] = fhs_xi_bc                       + fus.origin[0][0]  
        fhs_ybc[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*fhs_eta_b[idx_y]  + fus.origin[0][1]                             
        fhs_zbc[idx_y*n_cw:(idx_y+1)*n_cw] = np.zeros(n_cw)                  + fus.origin[0][2]              
        fhs_x[idx_y*(n_cw+1):(idx_y+1)*(n_cw+1)] = np.concatenate([fhs_xi_a1,np.array([fhs_xi_a2[-1]])])+ fus.origin[0][0]  
        fhs_y[idx_y*(n_cw+1):(idx_y+1)*(n_cw+1)] = np.ones(n_cw+1)*fhs_eta_a[idx_y]  + fus.origin[0][1]                             
        fhs_z[idx_y*(n_cw+1):(idx_y+1)*(n_cw+1)] = np.zeros(n_cw+1)                  + fus.origin[0][2]

        # fuselage vertical section                      
        delta_x_a = fvs.chord[idx_y]/n_cw      
        delta_x_b = fvs.chord[idx_y + 1]/n_cw    
        delta_x   = (fvs.chord[idx_y]+fvs.chord[idx_y + 1])/(2*n_cw)   

        fvs_xi_a1 = fvs.origin[idx_y][0] + delta_x_a*idx_x                    # z coordinate of top left corner of panel
        fvs_xi_ah = fvs.origin[idx_y][0] + delta_x_a*idx_x + delta_x_a*0.25   # z coordinate of left corner of panel
        fvs_xi_a2 = fvs.origin[idx_y][0] + delta_x_a*idx_x + delta_x_a        # z coordinate of bottom left corner of bound vortex 
        fvs_xi_ac = fvs.origin[idx_y][0] + delta_x_a*idx_x + delta_x_a*0.75   # z coordinate of bottom left corner of control point vortex  
        fvs_xi_b1 = fvs.origin[idx_y+1][0] + delta_x_b*idx_x                    # z coordinate of top right corner of panel      
        fvs_xi_bh = fvs.origin[idx_y+1][0] + delta_x_b*idx_x + delta_x_b*0.25   # z coordinate of right corner of bound vortex         
        fvs_xi_b2 = fvs.origin[idx_y+1][0] + delta_x_b*idx_x + delta_x_b        # z coordinate of bottom right corner of panel
        fvs_xi_bc = fvs.origin[idx_y+1][0] + delta_x_b*idx_x + delta_x_b*0.75   # z coordinate of bottom right corner of control point vortex         
        fvs_xi_c  = (fvs.origin[idx_y][0] + fvs.origin[idx_y+1][0])/2 + delta_x *idx_x + delta_x*0.75     # z coordinate three-quarter chord control point for each panel
        fvs_xi_ch = (fvs.origin[idx_y][0] + fvs.origin[idx_y+1][0])/2 + delta_x *idx_x + delta_x*0.25     # z coordinate center of bound vortex of each panel 

        fvs_xc [idx_y*n_cw:(idx_y+1)*n_cw] = fvs_xi_c                       + fus.origin[0][0]  
        fvs_zc [idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*fvs_eta[idx_y]   + fus.origin[0][2]  
        fvs_yc [idx_y*n_cw:(idx_y+1)*n_cw] = np.zeros(n_cw)                 + fus.origin[0][1]  
        fvs_x[idx_y*(n_cw+1):(idx_y+1)*(n_cw+1)] = np.concatenate([fvs_xi_a1,np.array([fvs_xi_a2[-1]])]) + fus.origin[0][0]  
        fvs_z[idx_y*(n_cw+1):(idx_y+1)*(n_cw+1)] = np.ones(n_cw+1)*fvs_eta_a[idx_y] + fus.origin[0][2]               
        fvs_y[idx_y*(n_cw+1):(idx_y+1)*(n_cw+1)] = np.zeros(n_cw+1)                 + fus.origin[0][1]
        
        fus_h_area += ((fhs.chord[idx_y]+fhs.chord[idx_y + 1])/2)*(fhs_eta_b[idx_y] - fhs_eta_a[idx_y])
        fus_v_area += ((fvs.chord[idx_y]+fvs.chord[idx_y + 1])/2)*(fvs_eta_b[idx_y] - fvs_eta_a[idx_y])            

    fhs_x[-(n_cw+1):] = np.concatenate([fhs_xi_b1,np.array([fhs_xi_b2[-1]])])+ fus.origin[0][0]  
    fhs_y[-(n_cw+1):] = np.ones(n_cw+1)*fhs_eta_b[idx_y]  + fus.origin[0][1]                             
    fhs_z[-(n_cw+1):] = np.zeros(n_cw+1)                  + fus.origin[0][2]        
    fvs_x[-(n_cw+1):] = np.concatenate([fvs_xi_a1,np.array([fvs_xi_a2[-1]])]) + fus.origin[0][0]  
    fvs_z[-(n_cw+1):] = np.ones(n_cw+1)*fvs_eta_a[idx_y] + fus.origin[0][2]               
    fvs_y[-(n_cw+1):] = np.zeros(n_cw+1)                 + fus.origin[0][1]   
    fhs_cs =  (fhs.chord[:-1]+fhs.chord[1:])/2
    fvs_cs =  (fvs.chord[:-1]+fvs.chord[1:])/2  
    
    # find the location of the trailing edge panels of each wing
    locations = ((np.linspace(1,n_sw,n_sw, endpoint = True) * n_cw) - 1).astype(int)
    fhs_xc_te1 = np.repeat(np.atleast_2d(fhs_xc[locations]), n_cw , axis = 0)
    fhs_yc_te1 = np.repeat(np.atleast_2d(fhs_yc[locations]), n_cw , axis = 0)
    fhs_zc_te1 = np.repeat(np.atleast_2d(fhs_zc[locations]), n_cw , axis = 0)        
    fhs_xa_te1 = np.repeat(np.atleast_2d(fhs_xa2[locations]), n_cw , axis = 0)
    fhs_ya_te1 = np.repeat(np.atleast_2d(fhs_ya2[locations]), n_cw , axis = 0)
    fhs_za_te1 = np.repeat(np.atleast_2d(fhs_za2[locations]), n_cw , axis = 0)
    fhs_xb_te1 = np.repeat(np.atleast_2d(fhs_xb2[locations]), n_cw , axis = 0)
    fhs_yb_te1 = np.repeat(np.atleast_2d(fhs_yb2[locations]), n_cw , axis = 0)
    fhs_zb_te1 = np.repeat(np.atleast_2d(fhs_zb2[locations]), n_cw , axis = 0)     
    
    fhs_xc_te = np.hstack(fhs_xc_te1.T)
    fhs_yc_te = np.hstack(fhs_yc_te1.T)
    fhs_zc_te = np.hstack(fhs_zc_te1.T)        
    fhs_xa_te = np.hstack(fhs_xa_te1.T)
    fhs_ya_te = np.hstack(fhs_ya_te1.T)
    fhs_za_te = np.hstack(fhs_za_te1.T)
    fhs_xb_te = np.hstack(fhs_xb_te1.T)
    fhs_yb_te = np.hstack(fhs_yb_te1.T)
    fhs_zb_te = np.hstack(fhs_zb_te1.T)     
    
    fhs_xc_te = np.concatenate([fhs_xc_te , fhs_xc_te ])
    fhs_yc_te = np.concatenate([fhs_yc_te , fhs_yc_te ])
    fhs_zc_te = np.concatenate([fhs_zc_te ,-fhs_zc_te ])                 
    fhs_xa_te = np.concatenate([fhs_xa_te , fhs_xa_te ])
    fhs_ya_te = np.concatenate([fhs_ya_te , fhs_ya_te ])
    fhs_za_te = np.concatenate([fhs_za_te ,-fhs_za_te ])            
    fhs_xb_te = np.concatenate([fhs_xb_te , fhs_xb_te ])
    fhs_yb_te = np.concatenate([fhs_yb_te , fhs_yb_te ])
    fhs_zb_te = np.concatenate([fhs_zb_te ,-fhs_zb_te ])    

    # Horizontal Fuselage Sections 
    wing_areas = []
    wing_areas.append(fus_h_area)
    wing_areas.append(fus_h_area)  
    
    # store points of horizontal section of fuselage 
    fhs_cs  = np.concatenate([fhs_cs, fhs_cs])
    fhs_xah = np.concatenate([fhs_xah, fhs_xah])
    fhs_yah = np.concatenate([fhs_yah,-fhs_yah])
    fhs_zah = np.concatenate([fhs_zah, fhs_zah])
    fhs_xbh = np.concatenate([fhs_xbh, fhs_xbh])
    fhs_ybh = np.concatenate([fhs_ybh,-fhs_ybh])
    fhs_zbh = np.concatenate([fhs_zbh, fhs_zbh])
    fhs_xch = np.concatenate([fhs_xch, fhs_xch])
    fhs_ych = np.concatenate([fhs_ych,-fhs_ych])
    fhs_zch = np.concatenate([fhs_zch, fhs_zch])
    fhs_xa1 = np.concatenate([fhs_xa1, fhs_xa1])
    fhs_ya1 = np.concatenate([fhs_ya1,-fhs_ya1])
    fhs_za1 = np.concatenate([fhs_za1, fhs_za1])
    fhs_xa2 = np.concatenate([fhs_xa2, fhs_xa2])
    fhs_ya2 = np.concatenate([fhs_ya2,-fhs_ya2])
    fhs_za2 = np.concatenate([fhs_za2, fhs_za2])
    fhs_xb1 = np.concatenate([fhs_xb1, fhs_xb1])
    fhs_yb1 = np.concatenate([fhs_yb1,-fhs_yb1])    
    fhs_zb1 = np.concatenate([fhs_zb1, fhs_zb1])
    fhs_xb2 = np.concatenate([fhs_xb2, fhs_xb2])
    fhs_yb2 = np.concatenate([fhs_yb2,-fhs_yb2])            
    fhs_zb2 = np.concatenate([fhs_zb2, fhs_zb2])
    fhs_xac = np.concatenate([fhs_xac, fhs_xac])
    fhs_yac = np.concatenate([fhs_yac,-fhs_yac])
    fhs_zac = np.concatenate([fhs_zac, fhs_zac])            
    fhs_xbc = np.concatenate([fhs_xbc, fhs_xbc])
    fhs_ybc = np.concatenate([fhs_ybc,-fhs_ybc])
    fhs_zbc = np.concatenate([fhs_zbc, fhs_zbc])
    fhs_xc  = np.concatenate([fhs_xc , fhs_xc ])
    fhs_yc  = np.concatenate([fhs_yc ,-fhs_yc])
    fhs_zc  = np.concatenate([fhs_zc , fhs_zc ])     
    fhs_x   = np.concatenate([fhs_x  , fhs_x  ])
    fhs_y   = np.concatenate([fhs_y  ,-fhs_y ])
    fhs_z   = np.concatenate([fhs_z  , fhs_z  ])      
    
    if model_fuselage == True:
        
        # increment fuslage lifting surface sections  
        VD.n_fus  += 2    
        VD.n_cp += len(fhs_xch)
        VD.n_w  += 2          
    
        # Store fus in vehicle vector  
        VD.XAH  = np.append(VD.XAH,fhs_xah)
        VD.YAH  = np.append(VD.YAH,fhs_yah)
        VD.ZAH  = np.append(VD.ZAH,fhs_zah)
        VD.XBH  = np.append(VD.XBH,fhs_xbh)
        VD.YBH  = np.append(VD.YBH,fhs_ybh)
        VD.ZBH  = np.append(VD.ZBH,fhs_zbh)
        VD.XCH  = np.append(VD.XCH,fhs_xch)
        VD.YCH  = np.append(VD.YCH,fhs_ych)
        VD.ZCH  = np.append(VD.ZCH,fhs_zch)     
        VD.XA1  = np.append(VD.XA1,fhs_xa1)
        VD.YA1  = np.append(VD.YA1,fhs_ya1)
        VD.ZA1  = np.append(VD.ZA1,fhs_za1)
        VD.XA2  = np.append(VD.XA2,fhs_xa2)
        VD.YA2  = np.append(VD.YA2,fhs_ya2)
        VD.ZA2  = np.append(VD.ZA2,fhs_za2)    
        VD.XB1  = np.append(VD.XB1,fhs_xb1)
        VD.YB1  = np.append(VD.YB1,fhs_yb1)
        VD.ZB1  = np.append(VD.ZB1,fhs_zb1)
        VD.XB2  = np.append(VD.XB2,fhs_xb2)                
        VD.YB2  = np.append(VD.YB2,fhs_yb2)        
        VD.ZB2  = np.append(VD.ZB2,fhs_zb2)  
        VD.XC_TE  = np.append(VD.XC_TE,fhs_xc_te)
        VD.YC_TE  = np.append(VD.YC_TE,fhs_yc_te) 
        VD.ZC_TE  = np.append(VD.ZC_TE,fhs_zc_te)          
        VD.XA_TE  = np.append(VD.XA_TE,fhs_xa_te)
        VD.YA_TE  = np.append(VD.YA_TE,fhs_ya_te) 
        VD.ZA_TE  = np.append(VD.ZA_TE,fhs_za_te) 
        VD.XB_TE  = np.append(VD.XB_TE,fhs_xb_te)
        VD.YB_TE  = np.append(VD.YB_TE,fhs_yb_te) 
        VD.ZB_TE  = np.append(VD.ZB_TE,fhs_zb_te)      
        VD.XAC  = np.append(VD.XAC,fhs_xac)
        VD.YAC  = np.append(VD.YAC,fhs_yac) 
        VD.ZAC  = np.append(VD.ZAC,fhs_zac) 
        VD.XBC  = np.append(VD.XBC,fhs_xbc)
        VD.YBC  = np.append(VD.YBC,fhs_ybc) 
        VD.ZBC  = np.append(VD.ZBC,fhs_zbc)  
        VD.XC   = np.append(VD.XC ,fhs_xc)
        VD.YC   = np.append(VD.YC ,fhs_yc)
        VD.ZC   = np.append(VD.ZC ,fhs_zc)  
        VD.CS   = np.append(VD.CS ,fhs_cs) 
        VD.X    = np.append(VD.X  ,fhs_x )  
        VD.Y    = np.append(VD.Y  ,fhs_y )  
        VD.Z    = np.append(VD.Z  ,fhs_z )
        
        VD.wing_areas = np.append(VD.wing_areas, wing_areas)
        VD.Stot = VD.Stot + np.sum(wing_areas)
        
        VL = VD.vortex_lift
        VL.append(False)
        VL.append(False)
    
    
    return VD

## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
def compute_panel_area(VD):
    """ This computed the area of the panels on the lifting surface of the vehicle 

    Assumptions: 
    None

    Source:   
    None
    
    Inputs:   
    VD                   - vortex distribution    
    
    Properties Used:
    N/A
    """     
    
    # create vectors for panel corders
    P1P2 = np.array([VD.XB1 - VD.XA1,VD.YB1 - VD.YA1,VD.ZB1 - VD.ZA1]).T
    P1P3 = np.array([VD.XA2 - VD.XA1,VD.YA2 - VD.YA1,VD.ZA2 - VD.ZA1]).T
    P2P3 = np.array([VD.XA2 - VD.XB1,VD.YA2 - VD.YB1,VD.ZA2 - VD.ZB1]).T
    P2P4 = np.array([VD.XB2 - VD.XB1,VD.YB2 - VD.YB1,VD.ZB2 - VD.ZB1]).T   
    
    # compute area of quadrilateral panel
    A_panel = 0.5*(np.linalg.norm(np.cross(P1P2,P1P3),axis=1) + np.linalg.norm(np.cross(P2P3, P2P4),axis=1))
    
    return A_panel


## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
def compute_unit_normal(VD):
    """ This computed the unit normal vector of each panel


    Assumptions: 
    None

    Source:
    None
    
    Inputs:   
    VD                   - vortex distribution    
    
    Properties Used:
    N/A
    """     

     # create vectors for panel
    P1P2 = np.array([VD.XB1 - VD.XA1,VD.YB1 - VD.YA1,VD.ZB1 - VD.ZA1]).T
    P1P3 = np.array([VD.XA2 - VD.XA1,VD.YA2 - VD.YA1,VD.ZA2 - VD.ZA1]).T

    cross = np.cross(P1P2,P1P3) 

    unit_normal = (cross.T / np.linalg.norm(cross,axis=1)).T

     # adjust Z values, no values should point down, flip vectors if so
    unit_normal[unit_normal[:,2]<0,:] = -unit_normal[unit_normal[:,2]<0,:]

    return unit_normal

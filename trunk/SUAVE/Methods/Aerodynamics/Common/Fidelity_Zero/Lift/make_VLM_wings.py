## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
# make_VLM_wings.py

# Created:  Jun 2021, A. Blaufox

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

# ------------------------------------------------------------------
# make_VLM_wings()
# ------------------------------------------------------------------  
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
  

    
# ------------------------------------------------------------------
# wing helper functions
# ------------------------------------------------------------------     
def make_cs_wing_from_cs(cs, seg_a, seg_b, wing, cs_ID):
    """ This uses a control surface and the segment it lies between to create
    an equilvalent wing object. The wing has a couple of non-standard attributes
    that contain information about the control surface it came from

    Assumptions: 
    None

    Source:   
    None
    
    Inputs:   
    cs    - a control surface object
    seg_a - the segment object inboard of the cs
    seg_b - the segment object outboard of the cs. The cs is also attached to this
    wing  - the wing object which owns seg_a and seg_b
    cs_ID - a unique identifier for the cs_wing
    
    Properties Used:
    N/A
    """      
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
    segment.chord                         = wing.chords.root #non-standard attribute, needed for VLM
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
    segment.chord                         = wing.chords.tip #non-standard attribute, needed for VLM
    segment.dihedral_outboard             = 0.
    segment.sweeps.quarter_chord          = 0.
    segment.sweeps.leading_edge           = 1e-8
    segment.thickness_to_chord            = wing.thickness_to_chord
    if wing.Airfoil: 
        segment.append_airfoil(wing.Airfoil.airfoil)             
    wing.append_segment(segment) 
    
    return wing



# ------------------------------------------------------------------
# span_break helper functions
# ------------------------------------------------------------------  
def add_span_break(span_break, span_breaks):
    """ This is a helper function that appends or superimposes a span_break 
    into span_breaks[]

    Assumptions: 
    None

    Source:   
    None
    
    Inputs:   
    span_break
    span_breaks
    
    Properties Used:
    N/A
    """     
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

def make_span_break_from_segment(seg):
    """ This creates a span_break Data() object from a segment

    Assumptions: 
    None

    Source:   
    None
    
    Inputs:   
    seg    - a segment object
    
    Properties Used:
    N/A
    """       
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
    """ This creates span_break Data() objects from a control surface, its
    owning segments, and their owning cs_wing

    Assumptions: 
    None

    Source:   
    None
    
    Inputs:   
    cs      - a control surface object
    seg_a   - the segment object inboard of the cs
    seg_b   - the segment object outboard of the cs. The cs is also attached to this
    cs_wing - the wing object which owns seg_a and seg_b
    cs_ID   - a unique identifier for the cs_wing
    
    Properties Used:
    N/A
    """     
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



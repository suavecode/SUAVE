## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
# make_VLM_wings.py

# Created:  Jun 2021, A. Blaufox

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# package imports 
import numpy as np
from copy import deepcopy

import SUAVE
from SUAVE.Core import  Data
from SUAVE.Components.Wings import All_Moving_Surface 
from SUAVE.Components.Wings.Control_Surfaces import Aileron , Elevator , Slat , Flap , Rudder 
from SUAVE.Methods.Geometry.Two_Dimensional.Planform import populate_control_sections
from SUAVE.Methods.Flight_Dynamics.Static_Stability.Approximations.Supporting_Functions import convert_sweep_segments

# ------------------------------------------------------------------
# make_VLM_wings()
# ------------------------------------------------------------------  
## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
def make_VLM_wings(geometry, settings):
    """ This parses through geometry.wings to create a Container of Data objects.
        Relevant VLM attributes are copied from geometry.wings to the Container.
        After, the wing data objects are reformatted. All control surfaces are 
        also added to the Container as Data objects representing full wings. 
        Helper variables are then computed (most notably span_breaks) for later. 
        
        see make_span_break() for further details

    Assumptions: 
    All control surfaces are appended directly to the wing, not wing segments.
    If a given wing has no segments, it must have either .taper or .chords.root 
        and .chords.tip defined

    Source:   
    None
    
    Inputs:
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
                
    settings.discretize_control_surfaces  --> set to True to generate control surface panels
    
    Properties Used:
    N/A
    """ 
    # unpack inputs
    discretize_cs = settings.discretize_control_surfaces
    wings         = copy_wings(geometry.wings)
    
    # ------------------------------------------------------------------
    # Reformat original wings to have at least 2 segments and additional values for processing later
    # ------------------------------------------------------------------    
    for wing in wings:
        wing.is_a_control_surface = False
        n_segments           = len(wing.Segments.keys())
        if n_segments==0:
            # convert to preferred format for the panelization loop
            wing       = convert_to_segmented_wing(wing)
            n_segments = 2
        else:
            # check for invalid/unsupported/conflicting geometry input            
            if issubclass(wing.wing_type, All_Moving_Surface): # these cases unsupported due to the way the panelization loop is structured at the moment
                if not (wing.hinge_vector == np.array([0.,0.,0.])).all() and wing.use_constant_hinge_fraction:
                    raise ValueError("A hinge_vector is specified, but the surface is set to use a constant hinge fraction")
                if len(wing.control_surfaces) > 0:
                    raise ValueError('Input: control surfaces are not supported on all-moving surfaces at this time')
            for segment in wing.Segments: #unsupported by convention
                if 'control_surfaces' in segment.keys() and len(segment.control_surfaces) > 0:
                    raise ValueError('Input: control surfaces should be appended to the wing, not its segments. ' + 
                                     'This function will move the control surfaces to wing segments itself.')  
        
        #move wing control surfaces to from wing to its segments
        wing = populate_control_sections(wing) if discretize_cs else wing
        
        #ensure wing has attributes that will be needed later
        wing_halfspan = wing.spans.projected * 0.5 if wing.symmetric else wing.spans.projected
        for i in range(n_segments):   
            (ia, ib)       = (0, 0) if i==0 else (i-1, i)
            seg_a          = wing.Segments[ia]
            seg_b          = wing.Segments[ib]            
            seg_b.chord    = seg_b.root_chord_percent *wing.chords.root  ##may be worth implementing a self-calculating .chord attribute    
            
            #guarantee that all segments have leading edge sweep
            if (i != 0) and (seg_a.sweeps.leading_edge is None):
                old_sweep                 = seg_a.sweeps.quarter_chord
                new_sweep                 = convert_sweep_segments(old_sweep, seg_a, seg_b, wing, old_ref_chord_fraction=0.25, new_ref_chord_fraction=0.0)
                seg_a.sweeps.leading_edge = new_sweep 
                
            #give segments offsets for giving cs_wings an origin later
            section_span     = (seg_b.percent_span_location - seg_a.percent_span_location) * wing_halfspan
            seg_b.x_offset   = 0. if i==0 else seg_a.x_offset   + section_span*np.tan(seg_a.sweeps.leading_edge)
            seg_b.dih_offset = 0. if i==0 else seg_a.dih_offset + section_span*np.tan(seg_a.dihedral_outboard)
        wing.Segments[-1].sweeps.leading_edge = 1e-8
    
    # each control_surface-turned-wing will have its own unique ID number
    cs_ID = 0
    
    # ------------------------------------------------------------------
    # Build wing Data() objects and wing.span_breaks from control surfaces on segments
    # ------------------------------------------------------------------    
    for wing in wings:
        if wing.is_a_control_surface == True: #skip if this wing is actually a control surface
            continue
        
        #prepare to iterate across all segments and control surfaces
        seg_breaks  = SUAVE.Core.ContainerOrdered()
        LE_breaks   = SUAVE.Core.ContainerOrdered()
        TE_breaks   = SUAVE.Core.ContainerOrdered()
        n_segments  = len(wing.Segments.keys())

        #process all control surfaces in each segment-------------------------------------
        for i in range(n_segments):   
            (ia, ib)    = (0, 0) if i==0 else (i-1, i)
            seg_a       = wing.Segments[ia]
            seg_b       = wing.Segments[ib]            
            
            control_surfaces = seg_b.control_surfaces if 'control_surfaces' in seg_b.keys() else Data()
            for cs in control_surfaces: #should be no control surfaces on root segment
                # create and append a wing object from the control_surface object and relevant segments
                cs_wing = make_cs_wing_from_cs(cs, seg_a, seg_b, wing, cs_ID)
                wings.append(cs_wing)
                                
                # register cs start and end span breaks
                cs_span_breaks = make_span_breaks_from_cs(cs, seg_a, seg_b, cs_wing, cs_ID)
                if cs.cs_type==Slat:
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
        #   2.  combine LE and TE breaks with the same span_fraction values (LE cuts from slats and TE cuts from others)
        #   3.  scan LE and TE to pick up cs cuts that cross over one or more span breaks
        
        # 1: 
        LE_breaks  = sorted(LE_breaks,  key=lambda span_break: span_break.span_fraction)
        TE_breaks  = sorted(TE_breaks,  key=lambda span_break: span_break.span_fraction)
        seg_breaks = sorted(seg_breaks, key=lambda span_break: span_break.span_fraction)
        
        # 2: similar to a 3-way merge sort
        span_breaks = SUAVE.Core.ContainerOrdered()        
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
                
        # 3:
        ib, ob = 0, 1 #inboard, outboard indices
        for edge, edge_str in enumerate(['LE','TE']):
            for i in range(len(span_breaks)-1):
                ID_i = span_breaks[i].cs_IDs[edge,ob]
                cut  = span_breaks[i].cuts[edge,ob]
                if ID_i == -1:
                    continue
                #copy the cs ID and its cut until the end of the control surface is found
                for j in range(i+1,len(span_breaks)):
                    i    += 1
                    ID_j = span_breaks[j].cs_IDs[edge,ib]                    
                    if ID_j == ID_i: #found control surface end
                        break
                    elif ID_j == -1: #found a span_break within control surface. copy values
                        span_breaks[j].cs_IDs[edge,:] = [ID_i, ID_i]
                        span_breaks[j].cuts[edge,:]   = [cut, cut]
                    else:
                        raise ValueError('VLM does not support multiple control surfaces on the same edge at this time')
                
        # pack span_breaks
        wing.span_breaks = reprocess_span_breaks(span_breaks)
        
    # ------------------------------------------------------------------
    # Give cs_wings span_breaks arrays
    # ------------------------------------------------------------------   
    for cs_wing in wings:
        if cs_wing.is_a_control_surface == False: #skip if this wing isn't actually a control surface
            continue  
        span_breaks = SUAVE.Core.ContainerOrdered()
        span_break  = make_span_break_from_segment(cs_wing.Segments[0])
        span_breaks.append(span_break)
        span_break  = make_span_break_from_segment(cs_wing.Segments[1])
        span_breaks.append(span_break) 
        cs_wing.span_breaks = span_breaks
    
    return wings
  

# ------------------------------------------------------------------
# custom deepcopy(wings)
# --TO DO-- This is a stand-in for a more fleshed-out VLM_surface class
# ------------------------------------------------------------------      
def copy_wings(original_wings):
    """ This copies VLM attributes for every wing object in original_wings into 
    a new wings container with new Data objects
    
    Inputs:   
    original_wings - the original wings container
    """       
    return copy_large_container(original_wings, "wings")

def copy_large_container(large_container, type_str):
    """ This function helps avoid copying a container of large objects directly,
    especially if those objects are Physical_Components
    
    Inputs:
    objects  -  a Container of large objects
    """    
    container = SUAVE.Core.Container()  if type_str != "Segments" else SUAVE.Core.ContainerOrdered()
    paths = get_paths(type_str)
    
    for obj in large_container: 
        #copy from paths
        data = copy_data_from_paths(obj, paths)     
        
        #special case new attributes
        if type_str == 'control_surfaces':
            data.cs_type                     = type(obj) # needed to identify the class of a control surface
        elif type_str == 'wings':
            data.wing_type = type(obj)
            if issubclass(data.wing_type, All_Moving_Surface):
                data.sign_duplicate              = obj.sign_duplicate
                data.hinge_fraction              = obj.hinge_fraction 
                data.deflection                  = obj.deflection  
                data.is_slat                     = False
                data.use_constant_hinge_fraction = obj.use_constant_hinge_fraction
                data.hinge_vector                = obj.hinge_vector
        container.append(data)
        
    return container

def copy_data_from_paths(old_object, paths):
    """ This copies the attributes specified by 'paths' from old_object  
    into a new Data() object

    Inputs:   
    old_object - an object to copy
    """       
    new_object = Data()   
    for path in paths:
        val = old_object.deep_get(path)
        recursive_set(new_object, path, val)
    return new_object

def recursive_set(data_obj, path, val):
    """ This is similar to the deep_set function, but also creates
    intermediate Data() objects for keys that do not yet exist. Special
    copy cases are made for paths that lead to large class objects
    """
    special_case_keys = ['control_surfaces', 'Segments']
    keys = path.split('.')
    key  = keys[0]
    if len(keys) == 1:
        if key in special_case_keys:
            data_obj[key] = copy_large_container(val, key) # will eventually recurse back to this function
        else:
            data_obj[key] = deepcopy(val) # at this point, should only be copying primitive types or very small Data objects
        return
    
    has_key = key in data_obj.keys()
    if not has_key:
        data_obj[key] = Data()
        
    new_path = '.'.join(keys[1:])
    recursive_set(data_obj[key], new_path, val)

def get_paths(type_str):
    """ This returns a list of the paths to the attributes needed in VLM
    for a given type of object.
    
    Note that if any element in the paths array is the same as the array's correponding type_str, 
    this will cause copy_large_container() to recurse infinitely. It will also recurse infinitely 
    if any element in the current array is the same as a type_str that corresponds to a different array
    which itself has an element that is the same as the current type_str. 
    
    Inputs:
    type_str  -  "wings", "control_surfaces" or "Segments"
    """       
    if type_str == 'wings':
        paths = ['tag',
                'origin',
                'symmetric',
                'vertical',
                'taper',
                'dihedral',
                'thickness_to_chord',
                'spans.projected',
                'chords.root',
                'chords.tip',
                'sweeps.quarter_chord',
                'sweeps.leading_edge',
                'twists.root',
                'twists.tip',
                'vortex_lift',
                'Airfoil',
                'Segments',
                'control_surfaces',
                ]
    elif type_str == 'control_surfaces':
        paths = ['tag',               
                 'span',               
                 'span_fraction_start',
                 'span_fraction_end',  
                 'hinge_fraction',     
                 'chord_fraction',     
                 'sign_duplicate',
                 'deflection',         
                 'configuration_type',      
                 'gain',      
                 ]
    elif type_str == 'Segments':
        paths = ['tag',               
                 'percent_span_location',               
                 'twist',
                 'root_chord_percent',  
                 'dihedral_outboard',     
                 'thickness_to_chord',     
                 'sweeps.quarter_chord',         
                 'sweeps.leading_edge', 
                 'Airfoil', 
                 ]        
        
    return paths

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
    
    Outputs:
    cs_wing - a Data object with relevant Wing and Control_Surface attributes
    
    Properties Used:
    N/A
    """      
    hspan = wing.spans.projected*0.5 if wing.symmetric else wing.spans.projected
    
    cs_wing                       = copy_data_from_paths(SUAVE.Components.Wings.Wing(), get_paths("wings"))
    
    #standard wing attributes--------------------------------------------------------------------------------------
    cs_wing.tag                   = wing.tag + '__cs_id_{}'.format(cs_ID)
    span_a                        = seg_a.percent_span_location
    span_b                        = seg_b.percent_span_location
    twist_a                       = seg_a.twist
    twist_b                       = seg_b.twist
    cs_wing.twists.root           = np.interp(cs.span_fraction_start, [span_a, span_b], [twist_a, twist_b])
    cs_wing.twists.tip            = np.interp(cs.span_fraction_end,   [span_a, span_b], [twist_a, twist_b])
    cs_wing.dihedral              = seg_a.dihedral_outboard
    cs_wing.thickness_to_chord    = (seg_a.thickness_to_chord + seg_b.thickness_to_chord)/2
    cs_wing.origin                = np.array(wing.origin) *1.
    
    span_fraction_tot             = cs.span_fraction_end - cs.span_fraction_start
    cs_wing.spans.projected       = wing.spans.projected * span_fraction_tot #includes 2x length if cs is on a symmetric wing 
    
    wing_chord_local_at_cs_root   = np.interp(cs.span_fraction_start, [span_a, span_b], [seg_a.chord, seg_b.chord])
    wing_chord_local_at_cs_tip    = np.interp(cs.span_fraction_end,   [span_a, span_b], [seg_a.chord, seg_b.chord])
    cs_wing.chords.root           = wing_chord_local_at_cs_root * cs.chord_fraction  
    cs_wing.chords.tip            = wing_chord_local_at_cs_tip  * cs.chord_fraction             
    cs_wing.taper                 = cs_wing.chords.tip / cs_wing.chords.root
    cs_wing.sweeps.quarter_chord  = 0.  # leave at 0. VLM will use leading edge

    cs_wing.symmetric             = wing.symmetric
    cs_wing.vertical              = wing.vertical
    cs_wing.vortex_lift           = wing.vortex_lift

    #non-standard wing attributes, mostly to do with cs_wing's identity as a control surface-----------------------
    #metadata
    cs_wing.is_a_control_surface  = True
    cs_wing.cs_ID                 = cs_ID
    cs_wing.name                  = wing.tag + '__' + seg_b.tag + '__' + cs.tag + '__cs_ID_{}'.format(cs_ID)
    cs_wing.is_slat               = (cs.cs_type==Slat)
    cs_wing.is_aileron            = (cs.cs_type==Aileron)
    cs_wing.pivot_edge            = 'TE' if cs_wing.is_slat else 'LE'
    
    #control surface attributes
    cs_wing.chord_fraction        = cs.chord_fraction
    cs_wing.hinge_fraction        = cs.hinge_fraction
    cs_wing.sign_duplicate        = cs.sign_duplicate
    cs_wing.deflection            = cs.deflection
    
    #adjustments---------------------------------------------------------------------------------------------------
    #adjust origin - may need to be adjusted later
    wing_halfspan                 = wing.spans.projected * 0.5 if wing.symmetric else wing.spans.projected
    LE_TE_cs_offset               = 0. if cs_wing.is_slat else (1 - cs.chord_fraction)*wing_chord_local_at_cs_root
    cs_wing.origin[0,0]          += np.interp(cs.span_fraction_start, [span_a, span_b], [seg_a.x_offset, seg_b.x_offset]) + LE_TE_cs_offset
    cs_wing.origin[0,1]          += cs.span_fraction_start * wing_halfspan
    cs_wing.origin[0,2]          += np.interp(cs.span_fraction_start, [span_a, span_b], [seg_a.dih_offset, seg_b.dih_offset])
    if wing.vertical:
        cs_wing[0,1], cs_wing[0,2] = cs_wing[0,2], cs_wing[0,1]
    
    # holds all required y-coords. Will be added to during discretization to ensure y-coords match up between wing and control surface.
    rel_offset                    = cs_wing.origin[0,1] if not cs_wing.vertical else cs_wing.origin[0,2]
    cs_wing.y_coords_required     = [cs.span_fraction_end*hspan - rel_offset] #initialize with the tip y-coord. Other coords to be added in VLM

    #find sweep of the 'outside' edge (LE for slats, TE for everything else)
    use_le_sweep                  = not (seg_a.sweeps.leading_edge is None)
    new_cf                        = 0. if cs_wing.is_slat else 1
    old_cf                        = 0. if use_le_sweep else 0.25
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
    
    #convert to segmented wing-------------------------------------------------------------------------------------
    cs_wing = convert_to_segmented_wing(cs_wing)
    
    # give segments offsets (in coordinates relative to the cs_wing)
    cs_wing.Segments[0].x_offset   = 0.
    cs_wing.Segments[0].dih_offset = 0.  
    cs_wing.Segments[1].x_offset   = wing_halfspan * span_fraction_tot *np.tan(cs_wing.Segments[0].sweeps.leading_edge)
    cs_wing.Segments[1].dih_offset = wing_halfspan * span_fraction_tot *np.tan(cs_wing.Segments[0].dihedral_outboard)    
    
    #add airfoil
    cs_wing.Segments[0].Airfoil     = seg_a.Airfoil
    cs_wing.Segments[1].Airfoil     = seg_b.Airfoil if cs.span_fraction_end==span_b else seg_a.Airfoil
    
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
    wing.Segments.append(segment) 
    
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
    wing.Segments.append(segment) 
    
    return wing

# ------------------------------------------------------------------
# span_break processing helper functions
# ------------------------------------------------------------------  
def add_span_break(span_break, span_breaks):
    """ This is a helper function that appends or superimposes a span_break 
    into span_breaks

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
        # if non-coincident, the space between the breaks is nominal wing: append the new span_break
        if span_breaks[-1].span_fraction < span_break.span_fraction: 
            span_breaks.append(span_break)
        
        # else coincident: need to superimpose cs_IDs and cuts, not append
        else:
            boolean = span_breaks[-1].cs_IDs==-1
            span_breaks[-1].cs_IDs[boolean] = span_break.cs_IDs[boolean]
            span_breaks[-1].cuts[boolean]   = span_break.cuts[boolean]
                
    return


def reprocess_span_breaks(span_breaks):
    """ This reprocesses the tags in a newly superimposed set of
    span_breaks and creates a new object so that the new keys match 
    the new tags
    
    Inputs:
    span_breaks
    """     
    sbs = SUAVE.Core.ContainerOrdered()
    for i,span_break in enumerate(span_breaks):
        span_break.tag = make_span_break_tag(span_break)
        sbs.append(span_break)
    return sbs

# ------------------------------------------------------------------
# span_break creation helper functions
# ------------------------------------------------------------------ 
def make_span_break_from_segment(seg):
    """ This creates a span_break Data() object from a segment

    Assumptions: 
    None

    Source:   
    None
    
    Inputs:   
    seg    - a segment object with standard attributes except for:
        .chord
    
    Properties Used:
    N/A
    """       
    span_frac       = seg.percent_span_location
    Airfoil         = seg.Airfoil
    dihedral_ob     = seg.dihedral_outboard
    sweep_ob_QC     = seg.sweeps.quarter_chord
    sweep_ob_LE     = seg.sweeps.leading_edge
    twist           = seg.twist    
    local_chord     = seg.chord       #non-standard attribute
    x_offset        = seg.x_offset
    dih_offset      = seg.dih_offset   
    span_break = make_span_break(-1, 0, 0, span_frac, 0., Airfoil,
                                 dihedral_ob, sweep_ob_QC, sweep_ob_LE, twist, local_chord,
                                 x_offset, dih_offset)  
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
    is_slat        = (cs.cs_type==Slat)
    LE_TE          = 0 if is_slat else 1
    span_a         = seg_a.percent_span_location
    span_b         = seg_b.percent_span_location    
    
    #inboard span break
    ib_ob          = 1 #the inboard break of the cs is the outboard part of the span_break
    span_frac      = cs.span_fraction_start    
    ob_cut         = cs.chord_fraction if is_slat else 1 - cs.chord_fraction
    Airfoil        = seg_a.Airfoil
    dihedral_ob    = seg_a.dihedral_outboard
    sweep_ob_QC    = seg_a.sweeps.quarter_chord
    sweep_ob_LE    = seg_a.sweeps.leading_edge
    twist          = cs_wing.twists.root    
    local_chord    = cs_wing.chords.root / cs.chord_fraction
    x_offset       = np.interp(cs.span_fraction_start, [span_a, span_b], [seg_a.x_offset, seg_b.x_offset])
    dih_offset     = np.interp(cs.span_fraction_start, [span_a, span_b], [seg_a.dih_offset, seg_b.dih_offset])
    inboard_span_break  = make_span_break(cs_ID, LE_TE, ib_ob, span_frac, ob_cut, Airfoil,
                                          dihedral_ob, sweep_ob_QC, sweep_ob_LE, twist, local_chord,
                                          x_offset, dih_offset)
    
    #outboard span break
    is_coincident  = (cs.span_fraction_end==seg_b.percent_span_location)
    ib_ob          = 0 #the outboard break of the cs is the inboard part of the span_break
    span_frac      = cs.span_fraction_end
    ib_cut         = cs.chord_fraction if is_slat else 1 - cs.chord_fraction
    Airfoil        = seg_b.Airfoil               if is_coincident else seg_a.Airfoil #take seg_b value if this outboard break is conicident with seg_b 
    dihedral_ob    = seg_b.dihedral_outboard     if is_coincident else seg_a.dihedral_outboard
    sweep_ob_QC    = seg_b.sweeps.quarter_chord  if is_coincident else seg_a.sweeps.quarter_chord
    sweep_ob_LE    = seg_b.sweeps.leading_edge   if is_coincident else seg_a.sweeps.leading_edge
    twist          = cs_wing.twists.tip    
    local_chord    = cs_wing.chords.tip  / cs.chord_fraction
    x_offset       = np.interp(cs.span_fraction_end, [span_a, span_b], [seg_a.x_offset, seg_b.x_offset])
    dih_offset     = np.interp(cs.span_fraction_end, [span_a, span_b], [seg_a.dih_offset, seg_b.dih_offset])    
    outboard_span_break = make_span_break(cs_ID, LE_TE, ib_ob, span_frac, ib_cut, Airfoil,
                                          dihedral_ob, sweep_ob_QC, sweep_ob_LE, twist, local_chord,
                                          x_offset, dih_offset)    
    return inboard_span_break, outboard_span_break

def make_span_break(cs_ID, LE_TE, ib_ob, span_frac, chord_cut, Airfoil,
                    dihedral_ob, sweep_ob_QC, sweep_ob_LE, twist, local_chord,
                    x_offset, dih_offset):
    """ This gathers information related to a span break into one Data() object.
    A span break is the spanwise location of a discontinuity in the discretization
    of the panels. These can be caused by segments and by the inboard and outboard 
    edges of a control surface. The inboard and outboard sides of a span break can
    have different chords due to cuts made by control surfaces. Ultimately, the
    attributes of the span_breaks of the wing will provide the discretization function 
    generate_wing_vortex_distribution() with the necessary values to make VLM panels
    as well as reshape those panels to make the control surface cuts dipicted below.
    
    A diagram is given below:


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
    span_break.Airfoil              = Airfoil
    span_break.dihedral_outboard    = dihedral_ob
    span_break.sweep_outboard_QC    = sweep_ob_QC
    span_break.sweep_outboard_LE    = sweep_ob_LE
    span_break.twist                = twist
    span_break.local_chord          = local_chord #this is the local chord BEFORE cuts are made
    span_break.x_offset             = x_offset
    span_break.dih_offset           = dih_offset  #dih_offset is the y or z accumulated offset from dihedral
    span_break.tag                  = make_span_break_tag(span_break)
    return span_break

def make_span_break_tag(span_break):
    location   = round(span_break.span_fraction, 3)
    cs_IDs_arr = span_break.cs_IDs.flatten()
    cs_IDs_str = '{}'.format(cs_IDs_arr).replace('[','').replace(']','').replace('-1', 'na').replace('  ', '_')
    
    return "{}___{}".format(location, cs_IDs_str)

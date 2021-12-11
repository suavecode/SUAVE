## @ingroup Methods-Aerodynamics-AVL
#create_avl_datastructure.py
# 
# Created:  Oct 2014, T. Momose
# Modified: Jan 2016, E. Botero
#           Apr 2017, M. Clarke
#           Jul 2017, T. MacDonald
#           Aug 2019, M. Clarke
#           Mar 2020, M. Clarke
#           Dec 2021, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ---------------------------------------------------------------------- 
import numpy as np 

# SUAVE Imports
from SUAVE.Core import  Units

# SUAVE-AVL Imports 
from .Data.Wing                                                          import Wing, Section, Control_Surface
from .Data.Body                                                          import Body 
from SUAVE.Components.Wings.Control_Surfaces                             import Aileron , Elevator , Slat , Flap , Rudder 
from SUAVE.Methods.Aerodynamics.AVL.write_avl_airfoil_file               import write_avl_airfoil_file   
## @ingroup Methods-Aerodynamics-AVL

def translate_avl_wing(suave_wing):
    """ Translates wing geometry from the vehicle setup to AVL format

    Assumptions:
        None

    Source:
        None

    Inputs:
        suave_wing.tag                                                          [-]
        suave_wing.symmetric                                                    [boolean]
        suave_wing.verical                                                      [boolean]
        suave_wing - passed into the populate_wing_sections function            [data stucture]

    Outputs:
        w - aircraft wing in AVL format                                         [data stucture] 

    Properties Used:
        N/A
    """         
    w                 = Wing()
    w.tag             = suave_wing.tag
    w.symmetric       = suave_wing.symmetric
    w.vertical        = suave_wing.vertical
    w                 = populate_wing_sections(w,suave_wing)

    return w

def translate_avl_body(suave_body):
    """ Translates body geometry from the vehicle setup to AVL format

    Assumptions:
        None

    Source:
        None

    Inputs:
        body.tag                                                       [-]
        suave_wing.lengths.total                                       [meters]    
        suave_body.lengths.nose                                        [meters]
        suave_body.lengths.tail                                        [meters]
        suave_wing.verical                                             [meters]
        suave_body.width                                               [meters]
        suave_body.heights.maximum                                     [meters]
        suave_wing - passed into the populate_body_sections function   [data stucture]

    Outputs:
        b - aircraft body in AVL format                                [data stucture] 

    Properties Used:
        N/A
    """  
    b                 = Body()
    b.tag             = suave_body.tag
    b.symmetric       = True
    b.lengths.total   = suave_body.lengths.total
    b.lengths.nose    = suave_body.lengths.nose
    b.lengths.tail    = suave_body.lengths.tail
    b.widths.maximum  = suave_body.width
    b.heights.maximum = suave_body.heights.maximum
    b                 = populate_body_sections(b,suave_body)

    return b

def populate_wing_sections(avl_wing,suave_wing): 
    """ Creates sections of wing geometry and populates the AVL wing data structure

    Assumptions:
        None

    Source:
        None

    Inputs:
        avl_wing.symmetric                         [boolean]
        suave_wing.spans.projected                 [meters]
        suave_wing.origin                          [meters]
        suave_wing.dihedral                        [radians]
        suave_wing.Segments.sweeps.leading_edge    [radians]
        suave_wing.Segments.root_chord_percent     [-]
        suave_wing.Segments.percent_span_location  [-]
        suave_wing.Segments.sweeps.quarter_chord   [radians]
        suave_wing.Segment.twist                   [radians]

    Outputs:
        avl_wing - aircraft wing in AVL format     [data stucture] 

    Properties Used:
        N/A
    """           
        
    # obtain the geometry for each segment in a loop                                            
    symm                 = avl_wing.symmetric
    semispan             = suave_wing.spans.projected*0.5 * (2 - symm)
    avl_wing.semispan    = semispan   
    root_chord           = suave_wing.chords.root
    segments             = suave_wing.Segments
    n_segments           = len(segments.keys()) 
    origin               = suave_wing.origin  
        
    if len(suave_wing.Segments.keys())>0: 
        for i_segs in range(n_segments):
            if (i_segs == n_segments-1):
                sweep = 0                                 
            else: # This converts all sweeps defined by the quarter chord to leading edge sweep since AVL needs the start of each wing section
                # from the leading edge coordinate and not the quarter chord coordinate
                if segments[i_segs].sweeps.leading_edge is not None: 
                    # If leading edge sweep is defined 
                    sweep       = segments[i_segs].sweeps.leading_edge  
                else:   
                    # If quarter chord sweep is defined, convert it to leading edge sweep
                    sweep_quarter_chord = segments[i_segs].sweeps.quarter_chord 
                    chord_fraction      = 0.25                          
                    segment_root_chord  = root_chord*segments[i_segs].root_chord_percent
                    segment_tip_chord   = root_chord*segments[i_segs+1].root_chord_percent
                    segment_span        = semispan*(segments[i_segs+1].percent_span_location - segments[i_segs].percent_span_location )
                    sweep               = np.arctan(((segment_root_chord*chord_fraction) + (np.tan(sweep_quarter_chord )*segment_span - chord_fraction*segment_tip_chord)) /segment_span) 
            dihedral       = segments[i_segs].dihedral_outboard   
    
            # append section 
            section        = Section() 
            section.tag    = segments[i_segs].tag
            section.chord  = root_chord*segments[i_segs].root_chord_percent 
            section.twist  = segments[i_segs].twist/Units.degrees    
            section.origin = origin # first origin in wing root, overwritten by section origin 
            if segments[i_segs].Airfoil:
                if segments[i_segs].Airfoil.airfoil.coordinate_file is not None:
                    section.airfoil_coord_file   = write_avl_airfoil_file(segments[i_segs].Airfoil.airfoil.coordinate_file)
                elif segments[i_segs].Airfoil.airfoil.naca_airfoil is not None:
                    section.naca_airfoil         = segments[i_segs].Airfoil.airfoil.naca_airfoil     
                    
            # append section to wing
            avl_wing.append_section(section)   
    
            if (i_segs == n_segments-1):
                return avl_wing 
            else:
                # condition for the presence of control surfaces in segment 
                if getattr(suave_wing,'control_surfaces',False):   
                    root_chord_percent = segments[i_segs].root_chord_percent
                    tip_chord_percent  = segments[i_segs+1].root_chord_percent
                    tip_percent_span   = segments[i_segs+1].percent_span_location
                    root_percent_span  = segments[i_segs].percent_span_location
                    root_twist         = segments[i_segs].twist
                    tip_twist          = segments[i_segs+1].twist
                    tip_airfoil        = segments[i_segs+1].Airfoil 
                    seg_tag            = segments[i_segs+1].tag 
                    
                    # append control surfaces
                    append_avl_wing_control_surfaces(suave_wing,avl_wing,semispan,root_chord_percent,tip_chord_percent,tip_percent_span,
                                                     root_percent_span,root_twist,tip_twist,tip_airfoil,seg_tag,dihedral,origin,sweep) 
    
            # update origin for next segment 
            segment_percent_span =    segments[i_segs+1].percent_span_location - segments[i_segs].percent_span_location     
            if avl_wing.vertical:
                inverted_wing = -np.sign(abs(dihedral) - np.pi/2)
                if inverted_wing  == 0:
                    inverted_wing  = 1
                dz = inverted_wing*semispan*segment_percent_span
                dy = dz*np.tan(dihedral)
                l  = dz/np.cos(dihedral)
                dx = l*np.tan(sweep)
            else:
                inverted_wing = np.sign(dihedral)
                if inverted_wing  == 0:
                    inverted_wing  = 1
                dy = inverted_wing*semispan*segment_percent_span
                dz = dy*np.tan(dihedral)
                l  = dy/np.cos(dihedral)
                dx = l*np.tan(sweep)
            origin= [[origin[0][0] + dx , origin[0][1] + dy, origin[0][2] + dz]]  
        
    else:    
        dihedral              = suave_wing.dihedral
        if suave_wing.sweeps.leading_edge  is not None: 
            sweep      = suave_wing.sweeps.leading_edge
        else:  
            sweep_quarter_chord = suave_wing.sweeps.quarter_chord 
            chord_fraction      = 0.25                          
            segment_root_chord  = suave_wing.chords.root
            segment_tip_chord   = suave_wing.chords.tip
            segment_span        = semispan 
            sweep       = np.arctan(((segment_root_chord*chord_fraction) + (np.tan(sweep_quarter_chord )*segment_span - chord_fraction*segment_tip_chord)) /segment_span)  
        avl_wing.semispan     = semispan 
        
        # define root section 
        root_section          = Section()
        root_section.tag      = 'root_section'
        root_section.origin   = origin
        root_section.chord    = suave_wing.chords.root 
        root_section.twist    = suave_wing.twists.root/Units.degrees
        
        # append control surfaces
        if suave_wing.Airfoil:
            tip_airfoil  = suave_wing.Airfoil.airfoil.coordinate_file     
        else:
            tip_airfoil = None
        seg_tag            = 'section'   
        append_avl_wing_control_surfaces(suave_wing,avl_wing,semispan,1.0,suave_wing.taper,1.0,
                                         0.0,suave_wing.twists.root,suave_wing.twists.tip,tip_airfoil,seg_tag,dihedral,origin,sweep)  
        
        # define tip section
        tip_section           = Section()
        tip_section.tag       = 'tip_section'
        tip_section.chord     = suave_wing.chords.tip 
        tip_section.twist     = suave_wing.twists.tip/Units.degrees  

        # assign location of wing tip         
        if avl_wing.vertical:
            tip_section.origin    = [[origin[0][0]+semispan*np.tan(sweep), origin[0][1]+semispan*np.tan(dihedral), origin[0][2]+semispan]]
        else: 
            tip_section.origin    = [[origin[0][0]+semispan*np.tan(sweep), origin[0][1]+semispan,origin[0][2]+semispan*np.tan(dihedral)]]

        # assign wing airfoil
        if suave_wing.Airfoil:
            root_section.airfoil_coord_file  = suave_wing.Airfoil.airfoil.coordinate_file          
            tip_section.airfoil_coord_file   = suave_wing.Airfoil.airfoil.coordinate_file     

        avl_wing.append_section(root_section)
        avl_wing.append_section(tip_section)
        
    return avl_wing

def append_avl_wing_control_surfaces(suave_wing,avl_wing,semispan,root_chord_percent,tip_chord_percent,tip_percent_span,root_percent_span,root_twist,tip_twist,tip_airfoil,seg_tag,dihedral,origin,sweep):

    """ Converts control surfaces on a suave wing to sections in avl wing

    Assumptions:
        None

    Source:
        None

    Inputs: 
        suave_wing           [-]
        avl_wing             [-]
        semispan             [meters]
        root_chord_percent   [unitless]
        tip_chord_percent    [unitless]
        tip_percent_span     [unitless]
        root_percent_span    [unitless]
        root_twist           [radians]
        tip_twist            [radians]
        tip_airfoil          [unitless]
        seg_tag              [unitless]
        dihedral             [radians]
        origin               [meters]
        sweep                [radians]
        
    Outputs: 
        None

    Properties Used:
        N/A
    """         

    root_chord    = suave_wing.chords.root                    
    section_spans = []
    for cs in suave_wing.control_surfaces:     
        # Create a vectorof all the section breaks from control surfaces on wings.
        # Section breaks include beginning and end of control surfaces as well as the end of segment       
        control_surface_start = semispan*cs.span_fraction_start
        control_surface_end   = semispan*cs.span_fraction_end
        if (control_surface_start < semispan*tip_percent_span) and (control_surface_start >= semispan*root_percent_span) : 
            section_spans.append(control_surface_start) 
        if (control_surface_end  < semispan*tip_percent_span) and (control_surface_end  >= semispan*root_percent_span):                         
            section_spans.append(control_surface_end)                                
    ordered_section_spans = sorted(list(set(section_spans)))     # sort the section_spans in order to create sections in spanwise order
    num_sections = len(ordered_section_spans)                    # count the number of sections breaks that the segment will contain    \

    for section_count in range(num_sections):        
        # create and append sections onto avl wing structure  
        if ordered_section_spans[section_count] == semispan*root_percent_span:  
            # if control surface begins at beginning of segment, redundant section is removed
            section_tags = list(avl_wing.sections.keys())
            del avl_wing.sections[section_tags[-1]]

        # create section for each break in the wing        
        section                   = Section()              
        section.tag               = seg_tag + '_section_'+ str(ordered_section_spans[section_count]) + 'm'
        root_section_chord        = root_chord*root_chord_percent
        tip_section_chord         = root_chord*tip_chord_percent
        semispan_section_fraction = (ordered_section_spans[section_count] - semispan*root_percent_span)/(semispan*(tip_percent_span - root_percent_span ))   
        section.chord             = np.interp(semispan_section_fraction,[0.,1.],[root_section_chord,tip_section_chord])
        root_section_twist        = root_twist/Units.degrees 
        tip_section_twist         = root_chord*tip_twist/Units.degrees  
        section.twist             = np.interp(semispan_section_fraction,[0.,1.],[root_section_twist,tip_section_twist]) 

        # if wing is a vertical wing, the y and z coordinates are swapped 
        if avl_wing.vertical:
            inverted_wing = -np.sign(abs(dihedral) - np.pi/2)
            if inverted_wing  == 0: inverted_wing  = 1
            dz = ordered_section_spans[section_count] -  inverted_wing*semispan*root_percent_span
            dy = dz*np.tan(dihedral)
            l  = dz/np.cos(dihedral)
            dx = l*np.tan(sweep)                                                            
        else:
            inverted_wing = np.sign(dihedral)
            if inverted_wing  == 0: inverted_wing  = 1
            dy = ordered_section_spans[section_count] - inverted_wing*semispan*root_percent_span
            dz = dy*np.tan(dihedral)
            l  = dy/np.cos(dihedral)
            dx = l*np.tan(sweep)
        section.origin = [[origin[0][0] + dx , origin[0][1] + dy, origin[0][2] + dz]]              

        # this loop appends all the control surfaces within a particular wing section
        for index  , ctrl_surf in enumerate(suave_wing.control_surfaces):
            if  (semispan*ctrl_surf.span_fraction_start == ordered_section_spans[section_count]) or \
                                        (ordered_section_spans[section_count] == semispan*ctrl_surf.span_fraction_end):
                c                     = Control_Surface()
                c.tag                 = ctrl_surf.tag                # name of control surface   
                c.sign_duplicate      = '+1'                         # this float indicates control surface deflection symmetry
                c.x_hinge             = 1 - ctrl_surf.chord_fraction # this float is the % location of the control surface hinge on the wing 
                c.deflection          = ctrl_surf.deflection / Units.degrees 
                c.order               = index

                # if control surface is an aileron, the deflection is asymmetric. This is standard convention from AVL
                if (type(ctrl_surf) ==  Aileron):
                    c.sign_duplicate = '-1'
                    c.function       = 'aileron'
                    c.gain           = -1.0
                # if control surface is a slat, the hinge is taken from the leading edge        
                elif (type(ctrl_surf) ==  Slat):
                    c.x_hinge   =  -ctrl_surf.chord_fraction
                    c.function  = 'slat'
                    c.gain      = -1.0
                elif (type(ctrl_surf) ==  Flap):
                    c.function  = 'flap'    
                    c.gain      = 1.0
                elif (type(ctrl_surf) ==  Elevator):
                    c.function  = 'elevator'
                    c.gain      = 1.0
                elif (type(ctrl_surf) ==  Rudder):
                    c.function  = 'rudder'
                    c.gain      = 1.0
                else:
                    raise AttributeError("Define control surface function as 'slat', 'flap', 'elevator' , 'aileron' or 'rudder'")
                section.append_control_surface(c) 

            elif  (semispan*ctrl_surf.span_fraction_start < ordered_section_spans[section_count]) and \
                                          (ordered_section_spans[section_count] < semispan*ctrl_surf.span_fraction_end):
                c                     = Control_Surface()
                c.tag                 = ctrl_surf.tag                # name of control surface   
                c.sign_duplicate      = '+1'                         # this float indicates control surface deflection symmetry
                c.x_hinge             = 1 - ctrl_surf.chord_fraction # this float is the % location of the control surface hinge on the wing 
                c.deflection          = ctrl_surf.deflection / Units.degrees 
                c.order               = index

                # if control surface is an aileron, the deflection is asymmetric. This is standard convention from AVL
                if (type(ctrl_surf) ==  Aileron):
                    c.sign_duplicate = '-1'
                    c.function       = 'aileron'
                    c.gain           = -1.0
                # if control surface is a slat, the hinge is taken from the leading edge        
                elif (type(ctrl_surf) ==  Slat):
                    c.x_hinge   =  -ctrl_surf.chord_fraction
                    c.function  = 'slat'
                    c.gain      = -1.0
                elif (type(ctrl_surf) ==  Flap):
                    c.function  = 'flap'    
                    c.gain      = 1.0
                elif (type(ctrl_surf) ==  Elevator):
                    c.function  = 'elevator'
                    c.gain      = 1.0
                elif (type(ctrl_surf) ==  Rudder):
                    c.function  = 'rudder'
                    c.gain      = 1.0
                else:
                    raise AttributeError("Define control surface function as 'slat', 'flap', 'elevator' , 'aileron' or 'rudder'")
                section.append_control_surface(c)                                                  

        if tip_airfoil:
            if tip_airfoil.airfoil.coordinate_file is not None:
                section.airfoil_coord_file   = write_avl_airfoil_file(tip_airfoil.airfoil.coordinate_file)
            elif tip_airfoil.airfoil.naca_airfoil is not None:
                section.naca_airfoil         = tip_airfoil.airfoil.naca_airfoil

        avl_wing.append_section(section)  
                        
    return 
def populate_body_sections(avl_body,suave_body):
    """ Creates sections of body geometry and populates the AVL body data structure

    Assumptions:
        None

    Source:
        None

    Inputs:
        avl_wing.symmetric                       [boolean]
        avl_body.widths.maximum                  [meters]
        avl_body.heights.maximum                 [meters]
        suave_body.fineness.nose                 [meters]
        suave_body.fineness.tail                 [meters]
        avl_body.lengths.total                   [meters]
        avl_body.lengths.nose                    [meters] 
        avl_body.lengths.tail                    [meters]  

    Outputs:
        avl_body - aircraft body in AVL format   [data stucture] 

    Properties Used:
        N/A
    """  

    symm = avl_body.symmetric   
    semispan_h = avl_body.widths.maximum * 0.5 * (2 - symm)
    semispan_v = avl_body.heights.maximum * 0.5
    origin = suave_body.origin[0]

    # Compute the curvature of the nose/tail given fineness ratio. Curvature is derived from general quadratic equation
    # This method relates the fineness ratio to the quadratic curve formula via a spline fit interpolation
    vec1 = [2 , 1.5, 1.2 , 1]
    vec2 = [1  ,1.57 , 3.2,  8]
    x = np.linspace(0,1,4)
    fuselage_nose_curvature =  np.interp(np.interp(suave_body.fineness.nose,vec2,x), x , vec1)
    fuselage_tail_curvature =  np.interp(np.interp(suave_body.fineness.tail,vec2,x), x , vec1) 


    # Horizontal Sections of Fuselage
    if semispan_h != 0.0:                
        width_array = np.linspace(-semispan_h, semispan_h, num=11,endpoint=True)
        for section_width in width_array:
            fuselage_h_section               = Section()
            fuselage_h_section_cabin_length  = avl_body.lengths.total - (avl_body.lengths.nose + avl_body.lengths.tail)
            fuselage_h_section_nose_length   = ((1 - ((abs(section_width/semispan_h))**fuselage_nose_curvature ))**(1/fuselage_nose_curvature))*avl_body.lengths.nose
            fuselage_h_section_tail_length   = ((1 - ((abs(section_width/semispan_h))**fuselage_tail_curvature ))**(1/fuselage_tail_curvature))*avl_body.lengths.tail
            fuselage_h_section_nose_origin   = avl_body.lengths.nose - fuselage_h_section_nose_length
            fuselage_h_section.tag           =  'fuselage_horizontal_section_at_' +  str(section_width) + '_m'
            fuselage_h_section.origin        = [ origin[0] + fuselage_h_section_nose_origin , origin[1] + section_width, origin[2]]
            fuselage_h_section.chord         = fuselage_h_section_cabin_length + fuselage_h_section_nose_length + fuselage_h_section_tail_length
            avl_body.append_section(fuselage_h_section,'horizontal')

    # Vertical Sections of Fuselage 
    if semispan_v != 0:               
        height_array = np.linspace(-semispan_v, semispan_v, num=11,endpoint=True)
        for section_height in height_array :
            fuselage_v_section               = Section()
            fuselage_v_section_cabin_length  = avl_body.lengths.total - (avl_body.lengths.nose + avl_body.lengths.tail)
            fuselage_v_section_nose_length   = ((1 - ((abs(section_height/semispan_v))**fuselage_nose_curvature ))**(1/fuselage_nose_curvature))*avl_body.lengths.nose
            fuselage_v_section_tail_length   = ((1 - ((abs(section_height/semispan_v))**fuselage_tail_curvature ))**(1/fuselage_tail_curvature))*avl_body.lengths.tail
            fuselage_v_section_nose_origin   = avl_body.lengths.nose - fuselage_v_section_nose_length
            fuselage_v_section.tag           = 'fuselage_vertical_top_section_at_' +  str(section_height) + '_m'        
            fuselage_v_section.origin        = [ origin[0] + fuselage_v_section_nose_origin,  origin[1],  origin[2] + section_height ]
            fuselage_v_section.chord         = fuselage_v_section_cabin_length + fuselage_v_section_nose_length + fuselage_v_section_tail_length
            avl_body.append_section(fuselage_v_section,'vertical')

    return avl_body


## @ingroup Methods-Aerodynamics-AVL
#create_avl_datastructure.py
# 
# Created:  Oct 2014, T. Momose
# Modified: Jan 2016, E. Botero
#           Apr 2017, M. Clarke
#           Jul 2017, T. MacDonald
#           Aug 2019, M. Clarke
#           Mar 2020, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import scipy
import numpy as np

from copy import deepcopy

# SUAVE Imports
from SUAVE.Core import Data , Units

# SUAVE-AVL Imports
from .Data.Inputs                                                  import Inputs
from .Data.Wing                                                    import Wing, Section, Control_Surface
from .Data.Body                                                    import Body
from .Data.Aircraft                                                import Aircraft
from .Data.Cases                                                   import Run_Case
from .Data.Configuration                                           import Configuration
from SUAVE.Components.Wings.Control_Surfaces                       import Aileron , Elevator , Slat , Flap , Rudder 
from SUAVE.Methods.Aerodynamics.AVL.write_avl_airfoil_file         import write_avl_airfoil_file  
from SUAVE.Methods.Geometry.Two_Dimensional.Planform.wing_planform import wing_planform

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

    if len(suave_wing.Segments.keys())>0:
        # obtain the geometry for each segment in a loop                                            
        symm                 = avl_wing.symmetric
        semispan             = suave_wing.spans.projected*0.5 * (2 - symm)
        avl_wing.semispan    = semispan   
        root_chord           = suave_wing.chords.root
        segment_percent_span = 0    
        segments             = suave_wing.Segments
        n_segments           = len(segments.keys())
        segment_sweeps       = []
        origin               = []

        origin.append(suave_wing.origin)

        for i_segs in range(n_segments):
            if (i_segs == n_segments-1):
                segment_sweeps.append(0)                                  
            else: # this converts all sweeps defined by the quarter chord to leading edge sweep since AVL needs the start of each wing section
                #from the leading edge coordinate and not the quarter chord coordinate
                if segments[i_segs].sweeps.leading_edge is not None: 
                    # if leading edge sweep is defined 
                    segment_sweep       = segments[i_segs].sweeps.leading_edge  
                else:   
                    # if quarter chord sweep is defined, convert it to leading edge sweep
                    sweep_quarter_chord = segments[i_segs].sweeps.quarter_chord 
                    chord_fraction      = 0.25                          
                    segment_root_chord  = root_chord*segments[i_segs].root_chord_percent
                    segment_tip_chord   = root_chord*segments[i_segs+1].root_chord_percent
                    segment_span        = semispan*(segments[i_segs+1].percent_span_location - segments[i_segs].percent_span_location )
                    segment_sweep       = np.arctan(((segment_root_chord*chord_fraction) + (np.tan(sweep_quarter_chord )*segment_span - chord_fraction*segment_tip_chord)) /segment_span)
                segment_sweeps.append(segment_sweep)
            dihedral       = segments[i_segs].dihedral_outboard  
            ctrl_surf_at_seg = False 

            # condition for the presence of control surfaces in segment 
            if getattr(segments[i_segs],'control_surfaces',False):    
                dihedral_ob   = segments[i_segs-1].dihedral_outboard 
                section_spans = []
                for cs in segments[i_segs].control_surfaces:     
                    # create a vector if all the section breaks in a segment. sections include beginning and end of control surfaces and end of segment      
                    control_surface_start = semispan*cs.span_fraction_start
                    control_surface_end   = semispan*cs.span_fraction_end
                    section_spans.append(control_surface_start)
                    section_spans.append(control_surface_end)                                
                ordered_section_spans = sorted(list(set(section_spans)))     # sort the section_spans in order to create sections in spanwise order
                num_sections = len(ordered_section_spans)                    # count the number of sections breaks that the segment will contain    \

                for section_count in range(num_sections):        
                    # create and append sections onto avl wing structure  
                    if ordered_section_spans[section_count] == semispan*segments[i_segs-1].percent_span_location:  
                        # if control surface begins at beginning of segment, redundant section is removed
                        section_tags = list(avl_wing.sections.keys())
                        del avl_wing.sections[section_tags[-1]]

                    # create section for each break in the wing        
                    section                   = Section()              
                    section.tag               = segments[i_segs].tag + '_section_'+ str(ordered_section_spans[section_count]) + 'm'
                    root_section_chord        = root_chord*segments[i_segs-1].root_chord_percent
                    tip_section_chord         = root_chord*segments[i_segs].root_chord_percent
                    semispan_section_fraction = (ordered_section_spans[section_count] - semispan*segments[i_segs-1].percent_span_location)/(semispan*(segments[i_segs].percent_span_location - segments[i_segs-1].percent_span_location ))   
                    section.chord             = np.interp(semispan_section_fraction,[0.,1.],[root_section_chord,tip_section_chord])
                    root_section_twist        = segments[i_segs-1].twist/Units.degrees 
                    tip_section_twist         = root_chord*segments[i_segs].twist/Units.degrees  
                    section.twist             = np.interp(semispan_section_fraction,[0.,1.],[root_section_twist,tip_section_twist]) 

                    # if wing is a vertical wing, the y and z coordinates are swapped 
                    if avl_wing.vertical:
                        dz = ordered_section_spans[section_count] -  semispan*segments[i_segs-1].percent_span_location 
                        dy = dz*np.tan(dihedral_ob)
                        l  = dz/np.cos(dihedral_ob)
                        dx = l*np.tan(segment_sweeps[i_segs-1])                                                            
                    else:
                        dy = ordered_section_spans[section_count] - semispan*segments[i_segs-1].percent_span_location 
                        dz = dy*np.tan(dihedral_ob)
                        l  = dy/np.cos(dihedral_ob)
                        dx = l*np.tan(segment_sweeps[i_segs-1])
                    section.origin = [[origin[i_segs-1][0][0] + dx , origin[i_segs-1][0][1] + dy, origin[i_segs-1][0][2] + dz]]              

                    # this loop appends all the control surfaces within a particular wing section
                    for index  , ctrl_surf in enumerate(segments[i_segs].control_surfaces):
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

                    if segments[i_segs].Airfoil:
                        if segments[i_segs].Airfoil.airfoil.coordinate_file is not None:
                            section.airfoil_coord_file   = write_avl_airfoil_file(segments[i_segs].Airfoil.airfoil.coordinate_file)
                        elif segments[i_segs].Airfoil.airfoil.naca_airfoil is not None:
                            section.naca_airfoil         = segments[i_segs].Airfoil.airfoil.naca_airfoil 

                    avl_wing.append_section(section)   

                # check if control surface ends at end of segment         
                if ordered_section_spans[section_count] == semispan*segments[i_segs].percent_span_location:  
                    ctrl_surf_at_seg = True

            if ctrl_surf_at_seg:  # if a control surface ends at the end of the segment, there is not need to append another segment
                pass
            else: # if there is no control surface break at the end of the segment, this block appends a segment
                section        = Section() 
                section.tag    = segments[i_segs].tag
                section.chord  = root_chord*segments[i_segs].root_chord_percent 
                section.twist  = segments[i_segs].twist/Units.degrees    
                section.origin = origin[i_segs]
                if segments[i_segs].Airfoil:
                    if segments[i_segs].Airfoil.airfoil.coordinate_file is not None:
                        section.airfoil_coord_file   = write_avl_airfoil_file(segments[i_segs].Airfoil.airfoil.coordinate_file)
                    elif segments[i_segs].Airfoil.airfoil.naca_airfoil is not None:
                        section.naca_airfoil         = segments[i_segs].Airfoil.airfoil.naca_airfoil     
                # append section to wing
                avl_wing.append_section(section)                               

            # update origin for next segment
            if (i_segs == n_segments-1):                                          
                return avl_wing

            segment_percent_span =    segments[i_segs+1].percent_span_location - segments[i_segs].percent_span_location     
            if avl_wing.vertical:
                dz = semispan*segment_percent_span
                dy = dz*np.tan(dihedral)
                l  = dz/np.cos(dihedral)
                dx = l*np.tan(segment_sweep)
            else:
                dy = semispan*segment_percent_span
                dz = dy*np.tan(dihedral)
                l  = dy/np.cos(dihedral)
                dx = l*np.tan(segment_sweep)
            origin.append( [[origin[i_segs][0][0] + dx , origin[i_segs][0][1] + dy, origin[i_segs][0][2] + dz]])               

    else:    
        symm                  = avl_wing.symmetric  
        dihedral              = suave_wing.dihedral
        span                  = suave_wing.spans.projected
        semispan              = suave_wing.spans.projected * 0.5 * (2 - symm) 
        if suave_wing.sweeps.leading_edge  is not None: 
            sweep      = suave_wing.sweeps.leading_edge
        else: 
            suave_wing = wing_planform(suave_wing)
            sweep      = suave_wing.sweeps.leading_edge
        avl_wing.semispan     = semispan
        origin                = suave_wing.origin[0]  
        
        # define root section 
        root_section          = Section()
        root_section.tag      = 'root_section'
        root_section.origin   = [origin]
        root_section.chord    = suave_wing.chords.root 
        root_section.twist    = suave_wing.twists.root/Units.degrees 
        root_section.semispan  = semispan

        # define tip section
        tip_section           = Section()
        tip_section.tag       = 'tip_section'
        tip_section.chord     = suave_wing.chords.tip 
        tip_section.twist     = suave_wing.twists.tip/Units.degrees 
        tip_section.semispan  = 0

        # assign location of wing tip         
        if avl_wing.vertical:
            tip_section.origin    = [[origin[0]+semispan*np.tan(sweep),origin[1]+semispan*np.tan(dihedral),origin[2]+semispan]]
        else: 
            tip_section.origin    = [[origin[0]+semispan*np.tan(sweep),origin[1]+semispan,origin[2]+semispan*np.tan(dihedral)]]

        # assign wing airfoil
        if suave_wing.Airfoil:
            root_section.airfoil_coord_file  = suave_wing.Airfoil.airfoil.coordinate_file          
            tip_section.airfoil_coord_file   = suave_wing.Airfoil.airfoil.coordinate_file    


        avl_wing.append_section(root_section)
        avl_wing.append_section(tip_section)

    return avl_wing

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


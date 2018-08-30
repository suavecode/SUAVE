## @ingroup Methods-Aerodynamics-AVL
#create_avl_datastructure.py
# 
# Created:  Oct 2014, T. Momose
# Modified: Jan 2016, E. Botero
#           Apr 2017, M. Clarke
#           Jul 2017, T. MacDonald

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import scipy
import numpy as np

from copy import deepcopy

# SUAVE Imports
from SUAVE.Core import Data , Units

# SUAVE-AVL Imports
from .Data.Inputs   import Inputs
from .Data.Wing     import Wing, Section, Control_Surface
from .Data.Body     import Body
from .Data.Aircraft import Aircraft
from .Data.Cases    import Run_Case
from .Data.Configuration import Configuration

## @ingroup Methods-Aerodynamics-AVL
def create_avl_datastructure(geometry,conditions):
        """ This translates the aircraft geometry into the format used in the AVL run file

        Assumptions:
            None
    
        Source:
            Drela, M. and Youngren, H., AVL, http://web.mit.edu/drela/Public/web/avle
    
        Inputs:
            geometry    
    
        Outputs:
            avl_inputs
    
        Properties Used:
            N/A
        """    
        avl_aircraft             = translate_avl_geometry(geometry)
        avl_configuration        = translate_avl_configuration(geometry,conditions)

        # pack results in a new AVL inputs structure
        avl_inputs               = Inputs()
        avl_inputs.aircraft      = avl_aircraft
        avl_inputs.configuration = avl_configuration
        return avl_inputs


def translate_avl_geometry(geometry):
        """ Translates geometry from the vehicle setup to AVL format

        Assumptions:
            None

        Source:
            None

        Inputs:
            geometry
                geometry.wing - passed into the translate_avl_wing function      [data stucture] 
                geometry.fuselage - passed into the translate_avl_body function  [data stucture]

        Outputs:
            aircraft - aircraft geometry in AVL format                           [data stucture] 

        Properties Used:
            N/A
        """ 
        aircraft                 = Aircraft()
        aircraft.tag             = geometry.tag

        for wing in geometry.wings:
                w  = translate_avl_wing(wing)
                aircraft.append_wing(w)
                
        for body in geometry.fuselages:
                if body.tag == 'fuselage':
                        b = translate_avl_body(body)
                        aircraft.append_body(b)

        return aircraft


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
                origin               = []
                origin.append(suave_wing.origin)
                root_chord           =  suave_wing.chords.root
                segment_percent_span = 0;   
                n_segments           = len(suave_wing.Segments.keys())

                segment_sweeps = []
                for i_segs in range(n_segments):
                        if (i_segs == n_segments-1):
                                segment_sweeps.append(0)                                  
                        else: 
                                if suave_wing.Segments[i_segs].sweeps.leading_edge > 0:
                                        segment_sweep       = suave_wing.Segments[i_segs].sweeps.leading_edge
                                else:   
                                        # convert quarter chord sweep to leading edge sweep       
                                        sweep_quarter_chord = suave_wing.Segments[i_segs].sweeps.quarter_chord
                                        chord_fraction      = 0.25                          
                                        segment_root_chord  = root_chord*suave_wing.Segments[i_segs].root_chord_percent
                                        segment_tip_chord   = root_chord*suave_wing.Segments[i_segs+1].root_chord_percent
                                        segment_span        = semispan*(suave_wing.Segments[i_segs+1].percent_span_location - suave_wing.Segments[i_segs].percent_span_location )
                                        segment_sweep       = np.arctan(((segment_root_chord*chord_fraction) + (np.tan(sweep_quarter_chord )*segment_span - chord_fraction*segment_tip_chord)) /segment_span)
                        segment_sweeps.append(segment_sweep)
                        dihedral       = suave_wing.Segments[i_segs].dihedral_outboard  
                        ctrl_surf_at_seg = False 
                        
                        # condition for the presence of control surfaces in segment 
                        if suave_wing.Segments[i_segs].control_surfaces:                     
                                section_spans = []
                                for cs in suave_wing.Segments[i_segs].control_surfaces:     
                                        # create a vector if all the section breaks in a segment. sections include beginning and end of control surfaces and end of segment      
                                        control_surface_start = semispan*cs.span_fraction[0]
                                        control_surface_end   = semispan*cs.span_fraction[1]
                                        section_spans.append(control_surface_start)
                                        section_spans.append(control_surface_end)                                
                                ordered_section_spans = sorted(list(set(section_spans)))     # sort the section_spans in order to create sections in chronological order
                                num_sections = len(ordered_section_spans)                    # count the number of sections breaks that the segment will contain    \
                                
                                for section_count in range(num_sections):        
                                        # create and append sections onto avl wing structure  
                                        if ordered_section_spans[section_count] == semispan*suave_wing.Segments[i_segs-1].percent_span_location:  
                                                # if control surface begins at beginning of segment, redundant section is removed
                                                avl_wing.remove(section) 
                                                
                                        # creat section for each break in the wing        
                                        section                   = Section ()              
                                        section.tag               = suave_wing.Segments[i_segs].tag + '_section_'+ str(ordered_section_spans[section_count]) + 'm'
                                        root_section_chord        = root_chord*suave_wing.Segments[i_segs-1].root_chord_percent
                                        tip_section_chord         = root_chord*suave_wing.Segments[i_segs].root_chord_percent
                                        semispan_section_fraction = (ordered_section_spans[section_count] - semispan*suave_wing.Segments[i_segs-1].percent_span_location)/(semispan*(suave_wing.Segments[i_segs].percent_span_location - suave_wing.Segments[i_segs-1].percent_span_location ))   
                                        section.chord             = scipy.interp(semispan_section_fraction,[0.,1.],[root_section_chord,tip_section_chord])
                                        root_section_twist        = suave_wing.Segments[i_segs-1].twist
                                        tip_section_twist         = root_chord*suave_wing.Segments[i_segs].twist
                                        section.twist             = scipy.interp(semispan_section_fraction,[0.,1.],[root_section_twist,tip_section_twist]) 

                                        if avl_wing.vertical:
                                                dz = ordered_section_spans[section_count] -  semispan*suave_wing.Segments[i_segs-1].percent_span_location 
                                                dy = dz*np.tan(dihedral)
                                                l  = dz/np.cos(dihedral)
                                                dx = l*np.tan(segment_sweeps[i_segs-1])                                                            
                                        else:
                                                dy = ordered_section_spans[section_count] - semispan*suave_wing.Segments[i_segs-1].percent_span_location 
                                                dz = dy*np.tan(dihedral)
                                                l  = dy/np.cos(dihedral)
                                                dx = l*np.tan(segment_sweeps[i_segs-1])
                                        section.origin = ( [origin[i_segs-1][0] + dx , origin[i_segs-1][1] + dy, origin[i_segs-1][2] + dz])               

                                        index = 0
                                        for crtl_surf in suave_wing.Segments[i_segs].control_surfaces:
                                                if  semispan*crtl_surf.span_fraction[0] == ordered_section_spans[section_count] or ordered_section_spans[section_count] == semispan*crtl_surf.span_fraction[1]:
                                                        c                     = Control_Surface()
                                                        c.tag                 = crtl_surf.tag
                                                        c.gain                = crtl_surf.deflection_gain
                                                        c.sign_duplicate      = crtl_surf.deflection_symmetry 
                                                        c.order               = index                                                        
                                                        if crtl_surf.tag == 'slat':            
                                                                # check if section is the beginning of slat or flap/aileron
                                                                c.x_hinge =  -1 * crtl_surf.chord_fraction
                                                        else:                                  
                                                                # if control surface is not a slat, it is a flap/aileron
                                                                c.x_hinge   = 1 - crtl_surf.chord_fraction                                                      
                                                                        
                                                        section.append_control_surface(c)
                                                        index =+ 1                                                       
                   
                                        if suave_wing.Segments[i_segs].Airfoil:
                                                section.airfoil_coord_file   = suave_wing.Segments[i_segs].Airfoil.airfoil.coordinate_file     
                                        avl_wing.append_section(section)   
                                        
                                if ordered_section_spans[section_count] == semispan*suave_wing.Segments[i_segs].percent_span_location:  
                                        # check if control surface ends at end of segment, redundant section is removed
                                        ctrl_surf_at_seg = True

                        if ctrl_surf_at_seg:   
                                pass
                        else: 
                                section        = Section() 
                                section.tag    = suave_wing.Segments[i_segs].tag
                                section.chord  = root_chord*suave_wing.Segments[i_segs].root_chord_percent 
                                section.twist  = (suave_wing.Segments[i_segs].twist)*180/np.pi
                                section.origin = origin[i_segs]
                                if suave_wing.Segments[i_segs].Airfoil:
                                        section.airfoil_coord_file   = suave_wing.Segments[i_segs].Airfoil.airfoil.coordinate_file
                
                                # append section to wing
                                avl_wing.append_section(section)                               

                        # update origin for next segment
                        if (i_segs == n_segments-1):                                          
                                return avl_wing
                                        
                        segment_percent_span =    suave_wing.Segments[i_segs+1].percent_span_location - suave_wing.Segments[i_segs].percent_span_location     
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
                        origin.append( [origin[i_segs][0] + dx , origin[i_segs][1] + dy, origin[i_segs][2] + dz])               
      
        else:    
                symm                  = avl_wing.symmetric
                sweep                 = suave_wing.sweeps.quarter_chord
                dihedral              = suave_wing.dihedral
                span                  = suave_wing.spans.projected
                semispan              = suave_wing.spans.projected * 0.5 * (2 - symm)
                avl_wing.semispan     = semispan
                origin                = suave_wing.origin
               
                # define root section 
                root_section          = Section()
                root_section.tag      = 'root_section'
                root_section.origin   = origin
                root_section.chord    = suave_wing.chords.root
                root_section.twist    = suave_wing.twists.root
                root_section.semispan  = semispan
                 
                # define tip section
                tip_section           = Section()
                tip_section.tag       = 'tip_section'
                tip_section.chord     = suave_wing.chords.tip
                tip_section.twist     = suave_wing.twists.tip
                tip_section.semispan  = 0
             
                # assign location of wing tip         
                if avl_wing.vertical:
                        tip_section.origin    = [origin[0]+semispan*np.tan(sweep),origin[1]+semispan*np.tan(dihedral),origin[1]+semispan]
                else: 
                        tip_section.origin    = [origin[0]+semispan*np.tan(sweep),origin[1]+semispan,origin[2]+semispan*np.tan(dihedral)]
                
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
        origin = [0, 0, 0]

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

def translate_avl_configuration(geometry,conditions):
        """ Translates mass properties of the aircraft configuration into AVL format

        Assumptions:
            None

        Source:
            None

        Inputs:
            geometry.reference_area                              [meters**2]
            geometry.wings['Main Wing'].spans.projected          [meters]
            geometry.wings['Main Wing'].chords.mean_aerodynamic  [meters]
            geometry.mass_properties.center_of_gravity           [meters]
            geometry.mass_properties.moments_of_inertia.tensor   [kilograms-meters**2]
                  
        Outputs:
            config                                               [-]

        Properties Used:
            N/A
        """  
        
        config                                   = Configuration()
        config.reference_values.sref             = geometry.reference_area
        config.reference_values.bref             = geometry.wings['Main Wing'].spans.projected
        config.reference_values.cref             = geometry.wings['Main Wing'].chords.mean_aerodynamic
        config.reference_values.cg_coords        = geometry.mass_properties.center_of_gravity
        config.mass_properties.mass              = 0 
        moment_tensor                            = geometry.mass_properties.moments_of_inertia.tensor
        config.mass_properties.inertial.Ixx      = moment_tensor[0][0]
        config.mass_properties.inertial.Iyy      = moment_tensor[1][1]
        config.mass_properties.inertial.Izz      = moment_tensor[2][2]
        config.mass_properties.inertial.Ixy      = moment_tensor[0][1]
        config.mass_properties.inertial.Iyz      = moment_tensor[1][2]
        config.mass_properties.inertial.Izx      = moment_tensor[2][0]

        #No Iysym, Izsym assumed for now

        return config

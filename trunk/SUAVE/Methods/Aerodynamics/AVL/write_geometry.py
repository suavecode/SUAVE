## @ingroup Methods-Aerodynamics-AVL
#write_geometry.py
# 
# Created:  Oct 2015, T. Momose
# Modified: Jan 2016, E. Botero
#           Oct 2018, M. Clarke
#           Aug 2019, M. Clarke
#           Apr 2020, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
from .purge_files import purge_files
from SUAVE.Methods.Aerodynamics.AVL.Data.Settings    import Settings
import numpy as np
import shutil
from .create_avl_datastructure import translate_avl_wing, translate_avl_body 

## @ingroup Methods-Aerodynamics-AVL
def write_geometry(avl_object,run_script_path):
    """This function writes the translated aircraft geometry into text file read 
    by AVL when it is called

    Assumptions:
        None
        
    Source:
        Drela, M. and Youngren, H., AVL, http://web.mit.edu/drela/Public/web/avl

    Inputs:
        avl_object

    Outputs:
        None

    Properties Used:
        N/A
    """    
    
    # unpack inputs
    aircraft                   = avl_object.geometry
    geometry_file              = avl_object.settings.filenames.features
    number_spanwise_vortices   = avl_object.settings.number_spanwise_vortices
    number_chordwise_vortices  = avl_object.settings.number_chordwise_vortices
    # Open the geometry file after purging if it already exists
    purge_files([geometry_file]) 
    geometry             = open(geometry_file,'w')

    with open(geometry_file,'w') as geometry:
        header_text       = make_header_text(avl_object)
        geometry.write(header_text)
        
        for w in aircraft.wings:
            avl_wing      = translate_avl_wing(w)
            wing_text     = make_surface_text(avl_wing,number_spanwise_vortices,number_chordwise_vortices)
            geometry.write(wing_text)  
            
        for b in aircraft.fuselages:
            avl_body  = translate_avl_body(b)
            body_text = make_body_text(avl_body,number_chordwise_vortices)
            geometry.write(body_text)
            
    return


def make_header_text(avl_object):  
    """This function writes the header using the template required for the AVL executable to read

    Assumptions:
        None
        
    Source:
        None

    Inputs:
        avl_object.settings.flow_symmetry.xz_plane                      [-]
        avl_object.settings.flow_symmetry.xy_parallel                   [-]
        avl_object.settings.flow_symmetry.z_symmetry_plane              [-]
        avl_object.geometry.wings['main_wing'].areas.reference          [meters**2]
        avl_object.geometry.wings['main_wing'].chords.mean_aerodynamic  [meters]
        avl_object.geometry.wings['main_wing'].spans.projected          [meters]
        avl_object.geometry.mass_properties.center_of_gravity           [meters]
        avl_object.geometry.tag                                         [-]
    
    Outputs:
        header_text                                                     [-]

    Properties Used:
        N/A
    """      
    header_base = \
'''{0}

#Mach
 {1}
 
#Iysym   IZsym   Zsym
  {2}      {3}     {4}
  
#Sref    Cref    Bref 	<meters>
{5}      {6}     {7}

#Xref    Yref    Zref   <meters>
{8}      {9}     {10}

'''

    # Unpack inputs
    Iysym = avl_object.settings.flow_symmetry.xz_plane
    Izsym = avl_object.settings.flow_symmetry.xy_parallel
    Zsym  = avl_object.settings.flow_symmetry.z_symmetry_plane
    Sref  = avl_object.geometry.wings['main_wing'].areas.reference
    Cref  = avl_object.geometry.wings['main_wing'].chords.mean_aerodynamic
    Bref  = avl_object.geometry.wings['main_wing'].spans.projected
    Xref  = avl_object.geometry.mass_properties.center_of_gravity[0][0]
    Yref  = avl_object.geometry.mass_properties.center_of_gravity[0][1]
    Zref  = avl_object.geometry.mass_properties.center_of_gravity[0][2]
    name  = avl_object.geometry.tag

    mach = 0.0

    # Insert inputs into the template
    header_text = header_base.format(name,mach,Iysym,Izsym,Zsym,Sref,Cref,Bref,Xref,Yref,Zref)

    return header_text


def make_surface_text(avl_wing,number_spanwise_vortices,number_chordwise_vortices):
    """This function writes the surface text using the template required for the AVL executable to read

    Assumptions:
        None
        
    Source:
        None

    Inputs:
       avl_wing.symmetric
       avl_wing.tag
        
    Outputs:
        surface_text                                                 

    Properties Used:
        N/A
    """       
    ordered_tags = []         
    surface_base = \
        '''

#---------------------------------------------------------
SURFACE
{0}
#Nchordwise  Cspace   Nspanwise  Sspace
{1}         {2}         {3}      {4}{5}
'''        
    # Unpack inputs
    symm = avl_wing.symmetric
    name = avl_wing.tag

    if symm:
        ydup = '\n\nYDUPLICATE\n0.0\n' # Duplication of wing about xz plane
    else:
        ydup     = ' ' 
    
    # Vertical Wings
    if avl_wing.vertical:
        # Define precision of analysis. See AVL documentation for reference 
        chordwise_vortex_spacing = 1.0
        spanwise_vortex_spacing  = -1.1                              # cosine distribution i.e. || |   |    |    |  | ||
        ordered_tags = sorted(avl_wing.sections, key = lambda x: x.origin[0][2])
        
        # Write text 
        surface_text = surface_base.format(name,number_chordwise_vortices,chordwise_vortex_spacing,number_spanwise_vortices ,spanwise_vortex_spacing,ydup)
        for i in range(len(ordered_tags)):
            section_text    = make_wing_section_text(ordered_tags[i])
            surface_text    = surface_text + section_text
            
    # Horizontal Wings        
    else:        
        # Define precision of analysis. See AVL documentation for reference
        chordwise_vortex_spacing = 1.0        
        spanwise_vortex_spacing  = 1.0                              # cosine distribution i.e. || |   |    |    |  | ||
        ordered_tags = sorted(avl_wing.sections, key = lambda x: x.origin[0][1])
    
        # Write text  
        surface_text = surface_base.format(name,number_chordwise_vortices,chordwise_vortex_spacing,number_spanwise_vortices ,spanwise_vortex_spacing,ydup)
        for i in range(len(ordered_tags)):
            section_text    = make_wing_section_text(ordered_tags[i])
            surface_text    = surface_text + section_text

    return surface_text


def make_body_text(avl_body,number_chordwise_vortices):   
    """This function writes the body text using the template required for the AVL executable to read

    Assumptions:
        None
        
    Source:
        None

    Inputs:
        avl_body.sections.horizontal
        avl_body.sections.vertical
    
    Outputs:
        body_text                                                 

    Properties Used:
        N/A
    """      
    surface_base = \
'''

#---------------------------------------------------------
SURFACE
{0}
#Nchordwise  Cspace   Nspanwise  Sspace
{1}           {2}      
'''
    # Unpack inputs
    name = avl_body.tag
    
    # Define precision of analysis. See AVL documentation for reference 
    chordwise_vortex_spacing = 1.0 
    
    # Form the horizontal part of the + shaped fuselage    
    hname           = name + '_horizontal'
    horizontal_text = surface_base.format(hname,number_chordwise_vortices,chordwise_vortex_spacing)
       
    ordered_tags = []
    ordered_tags = sorted(avl_body.sections.horizontal, key = lambda x: x.origin[1])
    for i in range(len(ordered_tags)):
        section_text    = make_body_section_text(ordered_tags[i])
        horizontal_text = horizontal_text + section_text
        
    # Form the vertical part of the + shaped fuselage
    vname         = name + '_vertical'
    vertical_text = surface_base.format(vname,number_chordwise_vortices,chordwise_vortex_spacing)   
    ordered_tags = []
    ordered_tags = sorted(avl_body.sections.vertical, key = lambda x: x.origin[2])
    for i in range(len(ordered_tags)):
        section_text    = make_body_section_text(ordered_tags[i])
        vertical_text = vertical_text + section_text
        
    body_text = horizontal_text + vertical_text
    return body_text  


def make_wing_section_text(avl_section):
    """This function writes the wing text using the template required for the AVL executable to read

    Assumptions:
        None
        
    Source:
        None

    Inputs:
       avl_section.origin             [meters]
       avl_section.chord              [meters]
       avl_section.twist              [radians]
       avl_section.airfoil_coord_file [-] 
        
    Outputs:
        wing_section_text                                                 

    Properties Used:
        N/A
    """      
    section_base = \
'''
SECTION
#Xle    Yle      Zle      Chord     Ainc  Nspanwise  Sspace
{0}  {1}    {2}    {3}    {4}     
'''
    airfoil_base = \
'''AFILE
{}
'''
    naca_airfoil_base = \
'''NACA
{}
'''
    # Unpack inputs
    x_le          = avl_section.origin[0][0]
    y_le          = avl_section.origin[0][1]
    z_le          = avl_section.origin[0][2]
    chord         = avl_section.chord
    ainc          = avl_section.twist
    airfoil_coord = avl_section.airfoil_coord_file
    naca_airfoil  = avl_section.naca_airfoil 
     
    wing_section_text = section_base.format(round(x_le,4),round(y_le,4), round(z_le,4),round(chord,4),round(ainc,4))
    if airfoil_coord:
        wing_section_text = wing_section_text + airfoil_base.format(airfoil_coord)
    if naca_airfoil:
        wing_section_text = wing_section_text + naca_airfoil_base.format(naca_airfoil)        
    
    ordered_cs = []
    ordered_cs = sorted(avl_section.control_surfaces, key = lambda x: x.order)
    for i in range(len(ordered_cs)):
        control_text = make_controls_text(ordered_cs[i])
        wing_section_text = wing_section_text + control_text

    return wing_section_text

    
def make_body_section_text(avl_body_section):
    """This function writes the body text using the template required for the AVL executable to read

    Assumptions:
        None
        
    Source:
        None

    Inputs:
       avl_section.origin             [meters]
       avl_section.chord              [meters]
       avl_section.twist              [radians]
       avl_section.airfoil_coord_file [-] 
                  
    Outputs:
        body_section_text                                                 

    Properties Used:
        N/A
    """    
    section_base = \
'''
SECTION
#Xle     Yle      Zle      Chord     Ainc  Nspanwise  Sspace
{0}    {1}     {2}     {3}     {4}      1        0
'''
    airfoil_base = \
'''AFILE
{}
'''

    # Unpack inputs
    x_le    = avl_body_section.origin[0]
    y_le    = avl_body_section.origin[1]
    z_le    = avl_body_section.origin[2]
    chord   = avl_body_section.chord
    ainc    = avl_body_section.twist
    airfoil = avl_body_section.airfoil_coord_file

    body_section_text = section_base.format(round(x_le,4),round(y_le,4), round(z_le,4),round(chord,4),round(ainc,4))
    if airfoil:
        body_section_text = body_section_text + airfoil_base.format(airfoil)
    
    return body_section_text

    
def make_controls_text(avl_control_surface):
    """This function writes the control surface text using the template required 
    for the AVL executable to read

    Assumptions:
        None
        
    Source:
        None

    Inputs:
        avl_control_surface.tag             [-]
        avl_control_surface.gain            [-]
        avl_control_surface.x_hinge         [-]
        avl_control_surface.hinge_vector    [-]
        avl_control_surface.sign_duplicate  [-]
                  
    Outputs:
        control_text                                                 

    Properties Used:
        N/A
    """    
    control_base = \
'''CONTROL
{0}    {1}   {2}   {3}  {4}
'''

    # Unpack inputs
    name     = avl_control_surface.tag
    gain     = avl_control_surface.gain
    xhinge   = avl_control_surface.x_hinge
    hv       = avl_control_surface.hinge_vector
    sign_dup = avl_control_surface.sign_duplicate

    control_text = control_base.format(name,gain,xhinge,hv,sign_dup)

    return control_text

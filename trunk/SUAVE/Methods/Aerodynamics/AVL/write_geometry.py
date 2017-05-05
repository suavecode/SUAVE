# translate_data.py
# 
# Created:  Oct 2015, T. Momose
# Modified: Jan 2016, E. Botero
# Modified: Apr 2017, M. Clarke


# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
from purge_files import purge_files
from create_avl_datastructure import translate_avl_wing, translate_avl_body 


def write_geometry(avl_object):

    # unpack inputs
    aircraft      = avl_object.geometry
    geometry_file = avl_object.settings.filenames.features

    # Open the geometry file after purging if it already exists
    purge_files([geometry_file])

    geometry = open(geometry_file,'w')

    with open(geometry_file,'w') as geometry:

        header_text = make_header_text(avl_object)
        geometry.write(header_text)
        
        for w in aircraft.wings:
            avl_wing = translate_avl_wing(w)
            wing_text = make_surface_text(avl_wing)
            geometry.write(wing_text)
        
                     
        for b in aircraft.fuselages:
            if b.tag == 'fuselage':
                avl_body = translate_avl_body(b)
                body_text = make_body_text(avl_body)
                geometry.write(body_text)
            
    return


def make_header_text(avl_object):
    # Template for header
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
    Xref  = avl_object.geometry.mass_properties.center_of_gravity[0]
    Yref  = avl_object.geometry.mass_properties.center_of_gravity[1]
    Zref  = avl_object.geometry.mass_properties.center_of_gravity[2]
    name  = avl_object.geometry.tag

    mach = 0.0

    # Insert inputs into the template
    header_text = header_base.format(name,mach,Iysym,Izsym,Zsym,Sref,Cref,Bref,Xref,Yref,Zref)

    return header_text


def make_surface_text(avl_wing):
    # Template for a surface
    ordered_tags = []         
    
    if avl_wing.vertical:
        surface_base = \
 '''

#---------------------------------------------------------
SURFACE
{0}
#Nchordwise  Cspace   Nspanwise  Sspace
20           1.0         20           1 {1}
'''        
        # Unpack inputs
        symm = avl_wing.symmetric
        name = avl_wing.tag
    
        if symm:
            ydup = '\n\nYDUPLICATE\n0.0\n'
        else:
            ydup     = ' '
        
        surface_text = surface_base.format(name,ydup)     
        ordered_tags = sorted(avl_wing.sections, key = lambda x: x.origin[2])
        for i in xrange(len(ordered_tags)):
            section_text    = make_wing_section_text(ordered_tags[i])
            surface_text = surface_text + section_text 
            
    else:
        surface_base = \
'''
        
#---------------------------------------------------------
SURFACE
{0}
#Nchordwise  Cspace   Nspanwise  Sspace
12         1.0      26      -1.0 {1}
'''      
        # Unpack inputs
        symm = avl_wing.symmetric
        name = avl_wing.tag
    
        if symm:
            ydup = '\n\nYDUPLICATE\n0.0\n'
        else:
            ydup     = ' '
        
        surface_text = surface_base.format(name,ydup)
        ordered_tags = sorted(avl_wing.sections, key = lambda x: x.origin[1])
        for i in xrange(len(ordered_tags)):
            section_text    = make_wing_section_text(ordered_tags[i])
            surface_text = surface_text + section_text

    return surface_text


def make_body_text(avl_body):    
    # Template for a surface
    surface_base = \
'''

#---------------------------------------------------------
SURFACE
{0}
#Nchordwise  Cspace   Nspanwise  Sspace
20           1.0      
'''
    # Unpack inputs
    name = avl_body.tag
    
    # Form the horizontal part of the + shaped fuselage    
    hname           = name + '_horizontal'
    horizontal_text = surface_base.format(hname,'  ')   
       
    ordered_tags = []
    ordered_tags = sorted(avl_body.sections.horizontal, key = lambda x: x.origin[1])
    for i in xrange(len(ordered_tags)):
        section_text    = make_body_section_text(ordered_tags[i])
        horizontal_text = horizontal_text + section_text
        
    # Form the vertical part of the + shaped fuselage
    vname         = name + '_vertical'
    vertical_text = surface_base.format(vname,' ')   
    ordered_tags = []
    ordered_tags = sorted(avl_body.sections.vertical, key = lambda x: x.origin[2])
    for i in xrange(len(ordered_tags)):
        section_text    = make_body_section_text(ordered_tags[i])
        vertical_text = vertical_text + section_text
        
    body_text = horizontal_text + vertical_text
    return body_text  


def make_wing_section_text(avl_section):
    # Template for a section
    section_base = \
'''
SECTION
#Xle    Yle      Zle      Chord     Ainc  Nspanwise  Sspace
{0}  {1}    {2}    {3}    {4}      
'''
    airfoil_base = \
'''
AFILE
{}
'''

    # Unpack inputs
    x_le    = avl_section.origin[0]
    y_le    = avl_section.origin[1]
    z_le    = avl_section.origin[2]
    chord   = avl_section.chord
    ainc    = avl_section.twist
    airfoil = avl_section.airfoil_coord_file

    section_text = section_base.format(x_le,y_le,z_le,chord,ainc)
    if airfoil:
        section_text = section_text + airfoil_base.format(airfoil)
    for cs in avl_section.control_surfaces:
        control_text = make_controls_text(cs)
        section_text = section_text + control_text

    return section_text

    
def make_body_section_text(avl_body_section):
    # Template for a section
    section_base = \
'''
SECTION
#Xle     Yle      Zle      Chord     Ainc  Nspanwise  Sspace
{0}    {1}     {2}     {3}     {4}      1        0
'''
    airfoil_base = \
'''
AFILE
{}
'''

    # Unpack inputs
    x_le    = avl_body_section.origin[0]
    y_le    = avl_body_section.origin[1]
    z_le    = avl_body_section.origin[2]
    chord   = avl_body_section.chord
    ainc    = avl_body_section.twist
    airfoil = avl_body_section.airfoil_coord_file

    section_text = section_base.format(x_le,y_le,z_le,chord,ainc)
    if airfoil:
        section_text = section_text + airfoil_base.format(airfoil)
    for cs in avl_body_section.control_surfaces:
        control_text = make_controls_text(cs)
        section_text = section_text + control_text

    return section_text

    
def make_controls_text(avl_control_surface):
    # Template for a control surface
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

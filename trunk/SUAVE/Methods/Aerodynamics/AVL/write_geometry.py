# translate_data.py
# 
# Created:  Oct 2015, T. Momose
# Modified: Jan 2016, E. Botero



# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
from purge_files import purge_files
from create_avl_datastructure import translate_avl_wing, translate_avl_body  #, translate_avl_engine


def write_geometry(avl_object):

    # unpack inputs
    aircraft      = avl_object.features
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
            avl_body = translate_avl_body(b)
            body_text = make_body_text(avl_body)
            geometry.write(body_text)
            
#       for e in aircraft.engines:
#           avl_engine = translate_avl_engine(e)
#           engine_text = make_engine_text(avl_engine)
#           geometry.write(engine_text)
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
    Sref  = avl_object.features.wings['main_wing'].areas.reference
    Cref  = avl_object.features.wings['main_wing'].chords.mean_aerodynamic
    Bref  = avl_object.features.wings['main_wing'].spans.projected
    Xref  = avl_object.features.mass_properties.center_of_gravity[0]
    Yref  = avl_object.features.mass_properties.center_of_gravity[1]
    Zref  = avl_object.features.mass_properties.center_of_gravity[2]
    name  = avl_object.features.tag

    mach = 0.0

    # Insert inputs into the template
    header_text = header_base.format(name,mach,Iysym,Izsym,Zsym,Sref,Cref,Bref,Xref,Yref,Zref)

    return header_text



def make_surface_text(avl_wing):
    # Template for a surface
    surface_base = \
'''#=============================================
SURFACE
{0}
#Nchordwise  Cspace   Nspanwise  Sspace
20           1.0      15         1.0 {1}
'''

    # Unpack inputs
    symm = avl_wing.symmetric
#	vert = avl_wing.vertical
    name = avl_wing.tag

    if symm:
        ydup = '\nYDUPLICATE\n0.0\n'
    else:
        ydup     = ' '
    surface_text = surface_base.format(name,ydup)
    
    ordered_tags = []
    if avl_wing.vertical:
        for s in avl_wing.sections:
            if len(ordered_tags)==0:
                ordered_tags.append(s.tag)
            else:
                for i in range(len(ordered_tags)+1):
                    if i == len(ordered_tags):
                        ordered_tags.append(s.tag)
                    elif s.origin[2] < avl_wing.sections[ordered_tags[i]].origin[2]:
                        ordered_tags.insert(i,s.tag)
                        break      
    else:
        for s in avl_wing.sections:
            if len(ordered_tags)==0:
                ordered_tags.append(s.tag)
            else:
                for i in range(len(ordered_tags)+1):
                    if i == len(ordered_tags):
                        ordered_tags.append(s.tag)
                    elif s.origin[1] < avl_wing.sections[ordered_tags[i]].origin[1]:
                        ordered_tags.insert(i,s.tag)
                        break
                
    for t in ordered_tags:
        section_text = make_wing_section_text(avl_wing.sections[t])
        surface_text = surface_text + section_text

    return surface_text



def make_body_text(avl_body):
    
    # Template for a surface
    surface_base = \
'''#----------------------------------------------
SURFACE
{0}
#Nchordwise  Cspace   Nspanwise  Sspace
20           1.0      

'''
    # Unpack inputs
    symm = avl_body.symmetric
    name = avl_body.tag

    
    # Form the horizontal part of the + shaped fuselage
    if symm:
        ydup = '\nYDUPLICATE\n0.0\n'
    else:
        ydup     = ' '
    hname           = name + '_horizontal'
    horizontal_text = surface_base.format(hname,ydup)               
    ordered_tags = []
    for s in avl_body.sections.horizontal:
        if len(ordered_tags)==0:
            ordered_tags.append(s.tag)
        else:
            for i in range(len(ordered_tags)+1):
                print ordered_tags
                if i == len(ordered_tags):
                    ordered_tags.append(s.tag)
                elif s.origin[1] < avl_body.sections.horizontal[ordered_tags[i]].origin[1]:
                    ordered_tags.insert(i,s.tag)
                    break
    for t in ordered_tags:
        section_text    = make_body_section_text(t)
        horizontal_text = horizontal_text + section_text
   
        
        
    # Form the vertical part of the + shaped fuselage
    vname         = name + '_vertical'
    vertical_text = surface_base.format(vname,' ')   
    ordered_tags = []
    for s in avl_body.sections.vertical:
        if len(ordered_tags)==0:
            ordered_tags.append(s.tag)
        else:
            for i in range(len(ordered_tags)+1):
                if i == len(ordered_tags):
                    ordered_tags.append(s.tag)
                elif s.origin[2] < avl_body.sections.vertical[ordered_tags[i]].origin[2]:
                    ordered_tags.insert(i,s.tag)
                    break
    for t in ordered_tags:
        section_text  = make_body_section_text(t)
        vertical_text = vertical_text + section_text

# ----------------------------------------------------------------------------
# Matthew: This code refers to the addition of engine geometry to the aircraft (to be added later once code is running)
# 
#def make_engine_text(avl_engine):
#    # Template for a surface
#    surface_base = \
#'''#=============================================
#SURFACE
#{0}
##Nchordwise  Cspace   Nspanwise  Sspace
#6            1.0      15           0
#
#'''
#    # Unpack inputs
#    symm = avl_engine.symmetric
#    name = avl_engine.tag
#
#    if symm:
#        ydup = '\nYDUPLICATE\n0.0\n'
#    else:
#        ydup     = ' '
#    engine_text = surface_base.format(name,ydup)
#
#    for s in avl_engine.sections:
#        section_text    = make_section_text(s)
#        engine_text = engine_text + section_text
#
#   return engine_text
# ----------------------------------------------------------------------------

def make_wing_section_text(avl_section):
    # Template for a section
    section_base = \
'''
SECTION
#Xle  Yle  Zle  Chord  Ainc  Nspanwise  Sspace
{0}   {1}  {2}  {3}    {4}     

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


    
def make_body_section_text(avl_section):
    # Template for a section
    section_base = \
'''
SECTION
#Xle  Yle  Zle  Chord  Ainc  Nspanwise  Sspace
{0}   {1}  {2}  {3}    {4}       1        0

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

    

def make_controls_text(avl_control_surface):
    # Template for a control surface
    control_base = \
'''CONTROL
#Name gain Xhinge hinge_vector SgnDup
{0}   {1}  {2}    {3} {4} {5}  {6}

'''

    # Unpack inputs
    name     = avl_control_surface.tag
    gain     = avl_control_surface.gain
    xhinge   = avl_control_surface.x_hinge
    hv       = avl_control_surface.hinge_vector
    sign_dup = avl_control_surface.sign_duplicate

    control_text = control_base.format(name,gain,xhinge,hv[0],hv[1],hv[2],sign_dup)

    return control_text

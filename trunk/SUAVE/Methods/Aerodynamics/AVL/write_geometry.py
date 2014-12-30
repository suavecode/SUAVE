# Tim Momose, October 2014


# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Structure import Data, Data_Exception, Data_Warning
from purge_files import purge_files


def write_geometry(avl_inputs):
	
	# unpack inputs
	aircraft      = avl_inputs.aircraft
	configuration = avl_inputs.configuration
	files_path    = avl_inputs.input_files.reference_path
	geometry_file = avl_inputs.input_files.geometry
	
	# Open the geometry file after purging if it already exists
	geometry_path= files_path + geometry_file
	purge_files([geometry_path])

	geometry = open(geometry_path,'w')

	try:
		header_text = make_header_text(aircraft,configuration)
		geometry.write(header_text)
		for w in aircraft.wings:
			wing_text = make_surface_text(w)
			geometry.write(wing_text)
		for b in aircraft.bodies:
			body_text = make_body_text(b)
			geometry.write(body_text)
	finally:	# don't leave the file open if something goes wrong
		geometry.close()
	
	return avl_inputs



def make_header_text(avl_aircraft,avl_configuration):
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
{11}	CDp

'''
	
	# Unpack inputs
	Iysym = avl_configuration.symmetry_settings.Iysym
	Izsym = avl_configuration.symmetry_settings.Izsym
	Zsym  = avl_configuration.symmetry_settings.Zsym
	Sref  = avl_configuration.reference_values.sref
	Cref  = avl_configuration.reference_values.cref
	Bref  = avl_configuration.reference_values.bref
	Xref  = avl_configuration.reference_values.cg_coords[0]
	Yref  = avl_configuration.reference_values.cg_coords[1]
	Zref  = avl_configuration.reference_values.cg_coords[2]
	CDp   = avl_configuration.parasite_drag
	name  = avl_aircraft.tag
	
	mach = 0.0
	
	# Insert inputs into the template
	header_text = header_base.format(name,mach,Iysym,Izsym,Zsym,Sref,Cref,Bref,Xref,Yref,Zref,CDp)
	
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
	
	print(ordered_tags)
	print(avl_wing.sections)
	
	for t in ordered_tags:
		section_text = make_section_text(avl_wing.sections[t])
		surface_text = surface_text + section_text

	return surface_text



def make_body_text(avl_body):
	# Template for a surface
	surface_base = \
'''#=============================================
SURFACE
{0}
#Nchordwise  Cspace   Nspanwise  Sspace
20           1.0      15         1.0 {1}

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

	for s in avl_body.sections.horizontal:
		section_text    = make_section_text(s)
		horizontal_text = horizontal_text + section_text
	
	# Form the vertical part of the + shaped fuselage
	vname         = name + '_vertical'
	vertical_text = surface_base.format(vname,' ')
	
	for s in avl_body.sections.vertical:
		section_text  = make_section_text(s)
		vertical_text = vertical_text + section_text
	
	# Combine and return
	body_text = horizontal_text + vertical_text
	return body_text



def make_section_text(avl_section):
	# Template for a section
	section_base = \
'''#------------------------------------------------------------
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


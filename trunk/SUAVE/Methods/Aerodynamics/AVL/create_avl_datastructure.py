# create_avl_datastructure.py
# 
# Created:  Oct 2014, T. Momose
# Modified: Jan 2016, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import scipy
import numpy as np
from copy import deepcopy

# SUAVE Imports
from SUAVE.Core import Data

# SUAVE-AVL Imports
from .Data.Inputs   import Inputs
from .Data.Wing     import Wing, Section, Control_Surface
from .Data.Body     import Body
from .Data.Aircraft import Aircraft
from .Data.Cases    import Run_Case
from .Data.Configuration import Configuration


def create_avl_datastructure(geometry,conditions):
	
	avl_aircraft      = translate_avl_geometry(geometry)
	avl_configuration = translate_avl_configuration(geometry,conditions)
	avl_cases         = translate_avl_cases(conditions,settings.run_cases)
	#avl_cases         = setup_test_cases(conditions)

	# pack results in a new AVL inputs structure
	avl_inputs = Inputs()
	avl_inputs.aircraft      = avl_aircraft
	avl_inputs.configuration = avl_configuration
	avl_inputs.cases         = avl_cases
	return avl_inputs


def translate_avl_geometry(geometry):

	aircraft = Aircraft()
	aircraft.tag = geometry.tag
	
	# FOR NOW, ASSUMING THAT CONTROL SURFACES ARE NOT ALIGNED WITH WING SECTIONS (IN THIS CASE, ROOT AND TIP SECTIONS)
	for wing in geometry.wings:
		w = translate_avl_wing(wing)
		aircraft.append_wing(w)
	
	for body in geometry.fuselages:
		b = translate_avl_body(body)
		aircraft.append_body(b)
	
	return aircraft


def translate_avl_wing(suave_wing):
	
	w = Wing()
	w.tag       = suave_wing.tag
	w.symmetric = suave_wing.symmetric
	w.vertical  = suave_wing.vertical
	w.sweep     = suave_wing.sweeps.quarter_chord
	w.dihedral  = suave_wing.dihedral
	w = populate_wing_sections(w,suave_wing)
	
	return w


def translate_avl_body(suave_body):
	
	b = Body()
	b.tag       = suave_body.tag
	b.symmetric = True #body.symmetric
	b.lengths.total = suave_body.lengths.total
	b.lengths.nose  = suave_body.lengths.nose
	b.lengths.tail  = suave_body.lengths.tail
	b.widths.maximum = suave_body.width
	b.heights.maximum = suave_body.heights.maximum
	b = populate_body_sections(b,suave_body)
	
	return b


def populate_wing_sections(avl_wing,suave_wing):
	symm     = avl_wing.symmetric
	sweep    = avl_wing.sweeps.quarter_chord
	dihedral = avl_wing.dihedral
	span     = suave_wing.spans.projected
	semispan = suave_wing.spans.projected * 0.5 * (2 - symm)
	origin   = suave_wing.origin
	root_section = Section()
	root_section.tag    = 'root_section'
	root_section.origin = origin
	root_section.chord  = suave_wing.chords.root
	root_section.twist  = suave_wing.twists.root
	
	tip_section = Section()
	tip_section.tag  = 'tip_section'
	tip_section.chord = suave_wing.chords.tip
	tip_section.twist = suave_wing.twists.tip
	tip_section.origin = [origin[0]+semispan*np.tan(sweep),origin[1]+semispan,origin[2]+semispan*np.tan(dihedral)]
	
	if avl_wing.vertical:
		temp = tip_section.origin[2]
		tip_section.origin[2] = tip_section.origin[1]
		tip_section.origin[1] = temp
	
	avl_wing.append_section(root_section)
	avl_wing.append_section(tip_section)
	
	if suave_wing.control_surfaces:
		for ctrl in suave_wing.control_surfaces:
			num = 1
			for section in ctrl.sections:
				semispan_fraction = (span/semispan) * section.origins.span_fraction
				s = Section()
				s.chord  = scipy.interp(semispan_fraction,[0.,1.],[root_section.chord,tip_section.chord])
				s.tag    = '{0}_section{1}'.format(ctrl.tag,num)
				s.origin = section.origins.dimensional
				s.origin[0] = s.origin[0] - s.chord*section.origins.chord_fraction
				s.twist  = scipy.interp(semispan_fraction,[0.,1.],[root_section.twist,tip_section.twist])
				c = Control_Surface()
				c.tag     = ctrl.tag
				c.x_hinge = 1. - section.chord_fraction
				c.sign_duplicate = ctrl.deflection_symmetry
				
				s.append_control_surface(c)
				avl_wing.append_section(s)
				num += 1
	
	return avl_wing


def populate_body_sections(avl_body,suave_body):

	symm = avl_body.symmetric
	semispan_h = avl_body.widths.maximum * 0.5 * (2 - symm)
	semispan_v = avl_body.heights.maximum * 0.5
	origin = suave_body.origin
	root_section = Section()
	root_section.tag    = 'center_horizontal_section'
	root_section.origin = origin
	root_section.chord  = avl_body.lengths.total
	
	tip_section = Section()
	tip_section.tag  = 'outer_horizontal_section'
	nl = avl_body.lengths.nose
	tl = avl_body.lengths.tail
	tip_section.origin = [origin[0]+nl,origin[1]+semispan_h,origin[2]]
	tip_section.chord = root_section.chord - nl - tl
	
	avl_body.append_section(root_section,'horizontal')
	avl_body.append_section(tip_section,'horizontal')
	tip_sectionv1 = deepcopy(tip_section)
	tip_sectionv1.origin[1] = origin[1]
	tip_sectionv1.origin[2] = origin[2] - semispan_v
	tip_sectionv1.tag       = 'lower_vertical_section'
	avl_body.append_section(tip_sectionv1,'vertical')
	avl_body.append_section(root_section,'vertical')
	tip_sectionv2 = deepcopy(tip_section)
	tip_sectionv2.origin[1] = origin[1]
	tip_sectionv2.origin[2] = origin[2] + semispan_v
	tip_sectionv2.tag       = 'upper_vertical_section'
	avl_body.append_section(tip_sectionv2,'vertical')
	
	return avl_body
	

def translate_avl_configuration(geometry,conditions):
	
	config = Configuration()
	config.reference_values.sref = geometry.reference_area
	config.reference_values.bref = geometry.wings['Main Wing'].spans.projected
	config.reference_values.cref = geometry.wings['Main Wing'].chords.mean_aerodynamic
	config.reference_values.cg_coords = geometry.mass_properties.center_of_gravity
	
	#config.parasite_drag = 0.0177#parasite_drag_aircraft(conditions,configuration,geometry)
	
	config.mass_properties.mass = geometry.mass_properties.max_takeoff ###
	moment_tensor = geometry.mass_properties.moments_of_inertia.tensor
	config.mass_properties.inertial.Ixx = moment_tensor[0][0]
	config.mass_properties.inertial.Iyy = moment_tensor[1][1]
	config.mass_properties.inertial.Izz = moment_tensor[2][2]
	config.mass_properties.inertial.Ixy = moment_tensor[0][1]
	config.mass_properties.inertial.Iyz = moment_tensor[1][2]
	config.mass_properties.inertial.Izx = moment_tensor[2][0]

	#No Iysym, Izsym assumed for now
	
	return config


def translate_avl_cases(conditions,suave_cases):
	
	runcases = Run_Case.Container()
	
	for case in suave_cases:
		kase = Run_Case()
		kase.tag = case.tag
		kase.conditions.mach  = case.conditions.freestream.mach
		kase.conditions.v_inf = case.conditions.freestream.velocity
		kase.conditions.rho   = case.conditions.freestream.density
		kase.conditions.g     = case.conditions.freestream.gravitational_acceleration
		kase.angles.alpha     = case.conditions.aerodynamics.angle_of_attack
		kase.angles.beta      = case.conditions.aerodynamics.side_slip_angle
		kase.parasite_drag    = case.conditions.aerodynamics.parasite_drag
		
		for deflect in case.conditions.stability_and_control.control_surface_deflections:
			kase.append_control_deflection(deflect.tag,deflect.magnitude)
		
		runcases.append_case(case)
	
	return runcases


def setup_test_cases(conditions):
	
	runcases = Run_Case.Container()
	
	alphas = [-10,-5,-2,0,2,5,10,20]
	mach   = conditions.freestream.mach
	v_inf  = conditions.freestream.velocity
	rho    = conditions.density
	g      = conditions.g
	for alpha in alphas:
		case = Run_Case()
		case.tag = 'Alpha={}'.format(alpha)
		case.conditions.mach  = mach
		case.conditions.v_inf = v_inf
		case.conditions.rho   = rho
		case.conditions.gravitation_acc = g
		case.angles.alpha     = alpha
		runcases.append_case(case)
	
	return runcases

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
#from .Data.Engine   import Engine
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
# ----------------------------------------------------------------------------
# This code refers to the addition of engine geometry to the aircraft 
#     
#	for engine in geometry.engines:
#            e = transpate_avl_engine(engine)
#            aircraft.append_engine(e)
# ----------------------------------------------------------------------------	
	return aircraft


def translate_avl_wing(suave_wing):
	#change to segments **
	w = Wing()
	w.tag = suave_wing.tag
	w.symmetric = suave_wing.symmetric
	w.vertical  = suave_wing.vertical
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

 # ----------------------------------------------------------------------------
# This code refers to the addition of engine geometry to the aircraft 
#   
# def translate_avl_engine(suave_engine):
#	
#	e = Engine()
#	e.tag       = suave_engine.tag
#	e.symmetric = True #engine.symmetric
#	e.engine_length = suave_body.engine_length
#	e.nacelle_diameter  = suave_body.nacelle_diameter
#	e = populate_engine_sections(e,suave_body)
#	
#	return e
# ----------------------------------------------------------------------------
def populate_wing_sections(avl_wing,suave_wing):  
    
      symm     = avl_wing.symmetric
      span     = suave_wing.spans.projected
      semispan = suave_wing.spans.projected*0.5 * (2 - symm)
      origin   = suave_wing.origin
      root_chord =  suave_wing.chords.root
      absolute_percent_chord = 0; 
      

      
      # Check to see if segments are defined. Get count
      if len(suave_wing.Segments.keys())>0:
          n_segments = len(suave_wing.Segments.keys())
      else:
          n_segments = 0      
         
      for i_segs in xrange(n_segments):  

#----------------------------------------------------------------
# this commented code relates to the presence of control surfaces 
#                       
#          Check to see if segment has control surfaces but count the number of instances of type control surface           
#          if num_control_surfaces == 0: 
#----------------------------------------------------------------          
              # TO DO: find a way to convert the instance "section" of the class Section () to a unique object using the segment tag of the wing in the loop
              section = Section()
              absolute_percent_chord =  suave_wing.Segments[i_segs].percent_span_location - absolute_percent_chord
              sweep = suave_wing.Segments[i_segs].sweeps.quarter_chord
              dihedral = suave_wing.Segments[i_segs].dihedral_outboard

              # define object properties 
              section.tag = suave_wing.Segments[i_segs].tag
              section.chord  = root_chord*suave_wing.Segments[i_segs].root_chord_percent 
              section.twist  = suave_wing.Segments[i_segs].twist
              section.origin = origin
              
              avl_wing.append_section(section)
              
              # update origin for next segment 
              if avl_wing.vertical:
                  origin = [origin[0]+ semispan*absolute_percent_chord*np.tan(sweep),\
                           origin[1] + semispan*absolute_percent_chord*np.tan(dihedral) ,\
                           origin[2] + semispan*absolute_percent_chord]
              else:
                  origin = [origin[0]+ semispan*absolute_percent_chord*np.tan(sweep) ,\
                        origin[1] + semispan*absolute_percent_chord,\
                        origin[2] + semispan*absolute_percent_chord*np.tan(dihedral)]
#----------------------------------------------------------------
# this commented code relates to the presence of control surfaces        
#  
#              else 
#              append the first section (root) of segment
#              for each control surface:
#                   # Start of control surface
#                   # find a way to convert a string describing the start of the control surface to an instance of a class 
#                   creat a name for this section using 'control_surface_%d_start' index number of the control surface
#                   find location of the start of the control surface (span distance)
#                   find the chord,twist,origin at that span distance
#                   make a section instance 
#                   append a section onto the alv_wing
#                   append a section onto control surface    
#                   # End of control surface 
#                   # find a way to convert a string describing the start of the control surface to an instance of a class 
#                   creat a name for this section using 'control_surface_%d_end' index number of the control surface
#                   find location of the start of the control surface (span distance)
#                   make a section instance 
#                   find the chord,twist,origin at that span distance 
#                   append a section onto the alv_wing
#                   append a section onto control surface               
#              append the last section (tip) of the segment
#
# Emilio's code is below 
#      if suave_wing.control_surfaces:
#		for ctrl in suave_wing.control_surfaces:
#			num = 1
#			for section in ctrl.sections:
#				semispan_fraction = (span/semispan) * section.origins.span_fraction
#				s = Section()
#				s.chord  = scipy.interp(semispan_fraction,[0.,1.],[root_section.chord,tip_section.chord])
#				s.tag    = '{0}_section{1}'.format(ctrl.tag,num)
#				s.origin = section.origins.dimensional
#				s.origin[0] = s.origin[0] - s.chord*section.origins.chord_fraction
#				s.twist  = scipy.interp(semispan_fraction,[0.,1.],[root_section.twist,tip_section.twist])
#				c = Control_Surface()
#				c.tag     = ctrl.tag
#				c.x_hinge = 1. - section.chord_fraction
#				c.sign_duplicate = ctrl.deflection_symmetry
#				
#				s.append_control_surface(c)
#				avl_wing.append_section(s)
#				num += 1
#--------------------------------------------------------------------------                            
      

      return avl_wing

def populate_body_sections(avl_body,suave_body):
      
      symm = avl_body.symmetric
      semispan_h = avl_body.widths.maximum * 0.5 * (2 - symm)
      semispan_v = avl_body.heights.maximum * 0.5
      origin = suave_body.origin
      fuselage_fineness_nose = suave_body.fineness.nose
      fuselage_fineness_tail = suave_body.fineness.tail
      
      # Vertical Sections of Fuselage       
      height_array = np.linspace(-semispan_v, semispan_v, num=10)
      for section_height in height_array :
          # find a way to change name of fuselage_v_section instance of the Class called Section to a unique object using section_index
          fuselage_v_section = Section()
          fuselage_v_section_cabin_length  = avl_body.lengths.total - (avl_body.lengths.nose + avl_body.lengths.tail)
          fuselage_v_section_nose_length= ((1 - ((abs(section_height/semispan_v))**fuselage_fineness_nose ))**(1/fuselage_fineness_nose))*avl_body.lengths.nose
          fuselage_v_section_tail_length = ((1 - ((abs(section_height/semispan_v))**fuselage_fineness_tail ))**(1/fuselage_fineness_nose))*avl_body.lengths.nose
          fuselage_v_section_nose_origin = avl_body.lengths.nose - fuselage_v_section_nose_length
          
          # define object properties 
          fuselage_v_section.tag = 'fuselage_vertical_section_at_s%m'% height_array
          fuselage_v_section.origin = [origin[0] + fuselage_v_section_nose_origin, origin[1], origin[2]+section_height ]
          fuselage_v_section.chord = fuselage_v_section_cabin_length + fuselage_v_section_nose_length + fuselage_v_section_tail_length
          avl_body.append_section(fuselage_v_section,'vertical')

      
      # Horizontal Sections of Fuselage
      width_array = np.linspace(0, semispan_h, num=5)
      for section_width in width_array:
          # find a way to change name of fuselage_v_section instance of the Class called Section using section_index
          fuselage_h_section = Section()
          fuselage_h_section_cabin_length  = avl_body.lengths.total - (avl_body.lengths.nose + avl_body.lengths.tail)
          fuselage_h_section_nose_length = ((1 - ((abs(section_width/semispan_h))**fuselage_fineness_nose ))**(1/fuselage_fineness_nose))*avl_body.lengths.nose
          fuselage_h_section_tail_length = ((1 - ((abs(section_width/semispan_h))**fuselage_fineness_tail ))**(1/fuselage_fineness_tail))*avl_body.lengths.tail
          fuselage_h_section_nose_origin  = avl_body.lengths.nose - fuselage_h_section_nose_length
          
          # define object properties 
          fuselage_h_section.tag = 'fuselage_horizontal_section_at_%sm'% width_array
          fuselage_h_section.origin = [origin[0] + fuselage_h_section_nose_origin , origin[1] + section_width, origin[2]]
          fuselage_h_section.chord = fuselage_h_section_cabin_length + fuselage_h_section_nose_length + fuselage_h_section_tail_length
          avl_body.append_section(fuselage_h_section,'horizontal')

          
      return avl_body
	
# ----------------------------------------------------------------------------
# This code refers to the addition of engine geometry to the aircraft 
#   
# def populate_engine_sections(avl_engine,suave_engine):
#
#      symm = avl_body.symmetric
#      origin = [0, 0, 0]
#      angle_array = np.linspace(0, 360 , num=30)
#      nacelle_radius = suave_engine.nacelle_diameter
#      section_index = 0
#      for section_angle in angle_array  
#          engine = Section()
#          engine.tag = 'section_index_%s'(section_index)
#          engine.chord = suave_engine.engine_length
#          engine.nacelle_section_origin = [suave_engine.origin[0],  suave_engine.origin[1] + np.sin(section_angle)*nacelle_radius , suave_engine.origin[1] + np.cos(section_angle)*nacelle_radius] 
#          avl_body.append_section(engine)
#          section_index = section_index + 1
#	return avl_engine
# ----------------------------------------------------------------------------	

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

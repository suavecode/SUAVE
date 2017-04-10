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
	w = Wing()
	w.tag = suave_wing.tag
	w.symmetric = suave_wing.symmetric
	w.vertical  = suave_wing.vertical
	w = populate_wing_sections(w,suave_wing)
	
	return w

def translate_avl_body(suave_body):
	
	b = Body()
	b.tag       = suave_body.tag
	b.symmetric = True
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
      semispan = suave_wing.spans.projected*0.5 * (2 - symm)
      origin = []
      origin.append(suave_wing.origin)

      root_chord =  suave_wing.chords.root
      segment_percent_span = 0;
      
      # Check to see if segments are defined. Get count
      if len(suave_wing.Segments.keys())>0:
          n_segments = len(suave_wing.Segments.keys())
      else:
          n_segments = 0      

      # obtain the geometry for each segment in a loop
      for i_segs in xrange(n_segments):
          sweep = suave_wing.Segments[i_segs].sweeps.quarter_chord
          dihedral = suave_wing.Segments[i_segs].dihedral_outboard
          section = Section() 
          section.tag = suave_wing.Segments[i_segs].tag
          section.chord  = root_chord*suave_wing.Segments[i_segs].root_chord_percent 
          section.twist  = (suave_wing.Segments[i_segs].twist)*180/np.pi
          section.origin =  origin[i_segs]
          avl_wing.append_section(section)
          # update origin for next segment
          if (i_segs == n_segments-1):
              return avl_wing
              break
          segment_percent_span =    suave_wing.Segments[i_segs+1].percent_span_location - suave_wing.Segments[i_segs].percent_span_location     
          if avl_wing.vertical:
              origin.append( [origin[i_segs][0] + semispan*segment_percent_span*np.tan(sweep), origin[i_segs][1] + semispan*segment_percent_span*np.tan(dihedral) , origin[i_segs][2] + semispan*segment_percent_span])
          else:
              origin.append( [origin[i_segs][0] + semispan*segment_percent_span*np.tan(sweep) , origin[i_segs][1] + semispan*segment_percent_span, origin[i_segs][2] + semispan*segment_percent_span*np.tan(dihedral)])
  
                            
#      # obtain the geometry for each segment in a loop
#      for i_segs in xrange(n_segments):
#          #condition for the absence of control surfaces in segment
#          if not suave_wing.Segments[i_segs].control_surfaces:
#              sweep = suave_wing.Segments[i_segs].sweeps.quarter_chord
#              dihedral = suave_wing.Segments[i_segs].dihedral_outboard
#              section = Section() 
#              section.tag = suave_wing.Segments[i_segs].tag
#              section.chord  = root_chord*suave_wing.Segments[i_segs].root_chord_percent 
#              section.twist  = (suave_wing.Segments[i_segs].twist)*180/np.pi
#              section.origin = origin[i_segs]
#              #append section onto avl wing
#              avl_wing.append_section(section)
#              
#              # break condition for segment tip
#              if (i_segs == n_segments-1):
#                  return avl_wing
#                  #break      
#              # update origin for next segment              
#              segment_percent_span =    suave_wing.Segments[i_segs+1].percent_span_location - suave_wing.Segments[i_segs].percent_span_location     
#              if avl_wing.vertical:
#                  origin.append( [ origin[i_segs][0] + semispan*segment_percent_span*np.tan(sweep), origin[i_segs][1] + semispan*segment_percent_span*np.tan(dihedral) , origin[i_segs][2] + semispan*segment_percent_span] )
#              else:
#                  origin.append( [ origin[i_segs][0] + semispan*segment_percent_span*np.tan(sweep) , origin[i_segs][1] + semispan*segment_percent_span, origin[i_segs][2] + semispan*segment_percent_span*np.tan(dihedral)] )
#          
#          #condition for the presence of control surfaces in segment
#          elif suave_wing.Segments[i_segs].control_surfaces:
#              sweep = suave_wing.Segments[i_segs].sweeps.quarter_chord
#              dihedral = suave_wing.Segments[i_segs].dihedral_outboard
#
#              #create a vector if all the sections to be made in each segment
#              section_spans = []
#              # span distance at beginning of segment
#              for cs in suave_wing.Segments[i_segs].control_surfaces:
#                  control_surface_start = semispan*cs.span_fraction[0]
#                  control_surface_end = semispan*cs.span_fraction[1]
#                  section_spans.append(control_surface_start)
#                  section_spans.append(control_surface_end)
#              # spand distance at end of segment           
#              segment_span_end = semispan*suave_wing.Segments[i_segs].percent_span_location  
#              section_spans.append(segment_span_end)
#              # sort the section_spans in order to create sections 
#              ordered_section_spans = sorted(list(set(section_spans))) 
#              print ordered_section_spans
#              # count the number of sections that the segment will contain
#              num_sections = len(ordered_section_spans)       
#
#              # creating and appending sections onto avl wing structure  
#              for section_count in xrange(num_sections-1):
#                  section = Section ()
#                  section.tag = suave_wing.Segments[i_segs].tag + '_section_at'+ str(ordered_section_spans[section_count]) + '_m'
#                  root_section_chord = root_chord*suave_wing.Segments[i_segs-1].root_chord_percent
#                  tip_section_chord = root_chord*suave_wing.Segments[i_segs].root_chord_percent
#                  semispan_section_fraction = (ordered_section_spans[section_count] - semispan*suave_wing.Segments[i_segs-1].percent_span_location) /(semispan*segment_percent_span)  #check!!!  
#                  section.chord  =  scipy.interp(semispan_section_fraction,[0.,1.],[root_section_chord,tip_section_chord])
#                  root_section_twist = suave_wing.Segments[i_segs-1].twist
#                  tip_section_twist = root_chord*suave_wing.Segments[i_segs].twist
#                  section.twist  = scipy.interp(semispan_section_fraction,[0.,1.],[root_section_twist,tip_section_twist]) 
#
#                  #get origin of control surface section  
#                  if avl_wing.vertical:
#                      section.origin = [origin[i_segs-1][0] + (ordered_section_spans[section_count] - origin[i_segs-1][1] )*np.tan(suave_wing.Segments[i_segs-1].sweeps.quarter_chord),\
#                                        origin[i_segs-1][1] +  (ordered_section_spans[section_count] - semispan*suave_wing.Segments[i_segs-1].percent_span_location)*np.tan(dihedral),\
#                                        ordered_section_spans[section_count]]
#                  else:
#                      section.origin = [origin[i_segs-1][0] + (ordered_section_spans[section_count] - origin[i_segs-1][1] )*np.tan(suave_wing.Segments[i_segs-1].sweeps.quarter_chord),\
#                                        ordered_section_spans[section_count],\
#                                        origin[i_segs-1][2] + (ordered_section_spans[section_count] - semispan*suave_wing.Segments[i_segs-1].percent_span_location)*np.tan(dihedral)]
#
#                  for crtl_surf in suave_wing.Segments[i_segs].control_surfaces:
#                      if (semispan*crtl_surf.span_fraction[0]  <= ordered_section_spans[section_count]) and (semispan*crtl_surf.span_fraction[1]  >= ordered_section_spans[section_count]):
#                          c                     = Control_Surface()
#                          c.tag                 = crtl_surf.tag
#                          c.gain                = crtl_surf.gain
#                          if c.tag == 'slat':
#                              hinge_index = -1
#                              c.x_hinge = hinge_index * (root_chord*crtl_surf.chord_fraction[0])/section.chord  
#                          else:
#                              hinge_index = 1
#                              c.x_hinge             = 1 - ( hinge_index * (root_chord*crtl_surf.chord_fraction[0])/section.chord )
#                          
#                          c.sign_duplicate      = crtl_surf.deflection_symmetry
#                          section.append_control_surface(c)
#                          # append control surface section onto avlwing        
#                  avl_wing.append_section(section)
#     
#              
#              # append segment section onto avl wing
#              section = Section() 
#              section.tag = suave_wing.Segments[i_segs].tag
#              section.chord  = root_chord*suave_wing.Segments[i_segs].root_chord_percent 
#              section.twist  = (suave_wing.Segments[i_segs].twist)*180/np.pi
#              section.origin = origin[i_segs]
#              avl_wing.append_section(section)
#              
#              # break condition for segment tip
#              if (i_segs == n_segments-1):
#                  return avl_wing
#                  break        
#              # update origin for next segment
#              segment_percent_span =    suave_wing.Segments[i_segs+1].percent_span_location - suave_wing.Segments[i_segs].percent_span_location     
#       
#              if avl_wing.vertical:
#                  origin.append( [origin[i_segs][0] + semispan*segment_percent_span*np.tan(sweep), origin[i_segs][1] + semispan*segment_percent_span*np.tan(dihedral) , origin[i_segs][2] + semispan*segment_percent_span])
#              else:
#                  origin.append( [origin[i_segs][0] + semispan*segment_percent_span*np.tan(sweep) , origin[i_segs][1] + semispan*segment_percent_span, origin[i_segs][2] + semispan*segment_percent_span*np.tan(dihedral)])
#              
      return avl_wing

def populate_body_sections(avl_body,suave_body):
    
      symm = avl_body.symmetric   
      semispan_h = avl_body.widths.maximum * 0.5 * (2 - symm)
      semispan_v = avl_body.heights.maximum * 0.5
      origin = [0, 0, 0]
      
      # I need to run this with Emilio: This computes the curvature of the nose/tail given 
      # the fineness ratio. The curvature is obtained using the general equation of a cicle/conic sections 
      vec1 = [2 , 1.5, 1.2 , 1]
      vec2 = [1  ,1.57 , 3.2,  8]
      x = np.linspace(0,1,4)
      fuselage_nose_curvature =  np.interp(np.interp(suave_body.fineness.nose,vec2,x), x , vec1)
      fuselage_tail_curvature =  np.interp(np.interp(suave_body.fineness.tail,vec2,x), x , vec1) 
      
      
      # Horizontal Sections of Fuselage
      width_array = np.linspace(-semispan_h, semispan_h, num=11,endpoint=True)
      for section_width in width_array:
          fuselage_h_section = Section()
          fuselage_h_section_cabin_length  = avl_body.lengths.total - (avl_body.lengths.nose + avl_body.lengths.tail)
          fuselage_h_section_nose_length = ((1 - ((abs(section_width/semispan_h))**fuselage_nose_curvature ))**(1/fuselage_nose_curvature))*avl_body.lengths.nose
          fuselage_h_section_tail_length = ((1 - ((abs(section_width/semispan_h))**fuselage_tail_curvature ))**(1/fuselage_tail_curvature))*avl_body.lengths.tail
          fuselage_h_section_nose_origin  = avl_body.lengths.nose - fuselage_h_section_nose_length
          fuselage_h_section.tag =  'fuselage_horizontal_section_at_' +  str(section_width) + '_m'
          fuselage_h_section.origin = [ origin[0] + fuselage_h_section_nose_origin , origin[1] + section_width, origin[2]]
          fuselage_h_section.chord = fuselage_h_section_cabin_length + fuselage_h_section_nose_length + fuselage_h_section_tail_length
          avl_body.append_section(fuselage_h_section,'horizontal')
              
      # Vertical Sections of Fuselage       
      height_array = np.linspace(-semispan_v, semispan_v, num=11,endpoint=True)
      for section_height in height_array :
          fuselage_v_section = Section()
          fuselage_v_section_cabin_length  = avl_body.lengths.total - (avl_body.lengths.nose + avl_body.lengths.tail)
          fuselage_v_section_nose_length = ((1 - ((abs(section_height/semispan_v))**fuselage_nose_curvature ))**(1/fuselage_nose_curvature))*avl_body.lengths.nose
          fuselage_v_section_tail_length = ((1 - ((abs(section_height/semispan_v))**fuselage_tail_curvature ))**(1/fuselage_tail_curvature))*avl_body.lengths.tail
          fuselage_v_section_nose_origin = avl_body.lengths.nose - fuselage_v_section_nose_length
          fuselage_v_section.tag = 'fuselage_vertical_top_section_at_' +  str(section_height) + '_m'        
          fuselage_v_section.origin = [ origin[0] + fuselage_v_section_nose_origin,  origin[1],  origin[2] + section_height ]
          fuselage_v_section.chord = fuselage_v_section_cabin_length + fuselage_v_section_nose_length + fuselage_v_section_tail_length
          avl_body.append_section(fuselage_v_section,'vertical')
      
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

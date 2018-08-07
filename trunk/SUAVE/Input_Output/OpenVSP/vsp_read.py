## @ingroup Input_Output-OpenVSP
# vsp_read.py

# Created:  Jun 2018, T. St Francis
# Modified: Aug 2018, T. St Francis

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Units, Data
from SUAVE.Input_Output.OpenVSP import get_vsp_areas
from SUAVE.Components.Wings.Airfoils.Airfoil import Airfoil 
from SUAVE.Components.Fuselages.Fuselage import Fuselage
import vsp_g as vsp
import numpy as np


## @ingroup Input_Output-OpenVSP
def vsp_read(tag): 	
	vsp.ClearVSPModel() 
	vsp.ReadVSPFile(PLANE)	
	
	vsp_fuselages = []
	vsp_wings = []	
	vsp_props = []
	
	vsp_geoms = vsp.FindGeoms()
	
	# Label each geom type. 
	# The API call for GETGEOMTYPE was not released as of 07/25/18

	'''
	for geom in vsp_geoms:
		if vsp.GETGEOMTYPE(str(geom)) == 'FUSELAGE':
			vsp_fuselages.append(geom)
		if vsp.GETGEOMTYPE(str(geom)) == 'WING':
			vsp_wings.append(geom)
		if vsp.GETGEOMTYPE(str(geom)) == 'PROP':
			vsp_props.append(geom)
	'''
	
	vehicle = SUAVE.Vehicle()
	vehicle.tag = tag
	
	# Read VSP geoms and store in SUAVE.
	'''
	for vsp_fuselage in vsp_fuselages:
		fuselage_id = vsp_fuselages[vsp_fuselage]
		fuselage = readFuselage( fuselage_id )
		vehicle.append_component(fuselage)
	for vsp_wing in vsp_wings:
		wing_id = vsp_wings[vsp_wing]
		wing = readWing( wing_id )
		vehicle.append_component(wing)		
	for vsp_prop in vsp_props:
		prop_id = vsp_props[vsp_prop]
		prop = readProp( prop_id )		
		vehicle.append_component(prop)
	
	'''
	
	
	return vehicle

def readWing(wing_id):
	
	wing = SUAVE.Components.Wings.Wing()
	
	if vsp.GetGeomName( wing_id ):
		wing.tag = vsp.GetGeomName( wing_id )
	else: 
		wing.tag = 'WingGeom'
	
	wing.origin[0] = vsp.GetParmVal( wing_id, 'X_Rel_Location', 'XForm')
	wing.origin[1] = vsp.GetParmVal( wing_id, 'Y_Rel_Location', 'XForm')
	wing.origin[2] = vsp.GetParmVal( wing_id, 'Z_Rel_Location', 'XForm')	
	
	#SYMMETRY    
	sym_planar = vsp.GetParmVal( wing_id, 'Sym_Planar_Flag', 'Sym')
	sym_origin = vsp.GetParmVal( wing_id, 'Sym_Ancestor', 'Sym')
	if sym_planar == 2 and sym_origin == 2: #origin at wing, not vehicle
		wing.symmetric == True	
	else:
		wing.symmetric == False


	#OTHER WING-WIDE PARMS
	wing.aspect_ratio = vsp.GetParmVal(wing_id, 'TotalAR', 'WingGeom')
	xsec_surf_id = vsp.GetXSecSurf( wing_id, 0) #get number surfaces in geom
	segment_num = vsp.GetNumXSec( xsec_surf_id)   #get # segments, is one more than in GUI	


	#WING SEGMENTS	

	#get root chord and proj_span_sum for use below
	total_chord = vsp.GetParmVal( wing_id, 'Root_Chord', 'XSec_1')	
	total_proj_span = vsp.GetParmVal( wing_id, 'TotalProjectedSpan', 'WingGeom')  
	proj_span_sum = 0.
	segment_spans = [None] * (segment_num) # these spans are non-projected
	
	span_sum = 0.
	segment_dihedral = [None] * (segment_num)
	
	# check for extra segment at wing root, then skip XSec_0 to start at exposed segment
	if vsp.GetParmVal( wing_id, 'Root_Chord', 'XSec_0') == 1.:
		start = 1
	else:
		start = 0
		
	
	# Iterate VSP XSecs into SUAVE segments. Note: Wing segments are defined by outboard sections in VSP 
	for i in xrange( start, segment_num+1):		# but inboard sections in SUAVE
		segment = SUAVE.Components.Wings.Segment()
		segment.tag                   = 'Section_' + str(i)
		thick_cord                    = vsp.GetParmVal( wing_id, 'ThickChord', 'XSecCurve_' + str(i-1))
		segment.thickness_to_chord    = thick_cord	# Also used in airfoil, below.		
		segment_root_chord            = vsp.GetParmVal( wing_id, 'Root_Chord', 'XSec_' + str(i))
		segment.root_chord_percent    = segment_root_chord / total_chord		
		segment.percent_span_location = proj_span_sum / (total_proj_span/2)
		segment.twist                 = vsp.GetParmVal( wing_id, 'Twist', 'XSec_' + str(i-1)) * Units.deg
		
		if i < segment_num:  # This excludes the tip xsec, but we need a segment in SUAVE to store airfoil.
			segment.sweeps.quarter_chord  = vsp.GetParmVal( wing_id, 'Sec_Sweep', 'XSec_' + str(i)) * Units.deg
	
			segment_dihedral[i]	      = vsp.GetParmVal( wing_id, 'Dihedral', 'XSec_' + str(i)) * Units.deg
			segment.dihedral_outboard     = segment_dihedral[i]
			
			segment_spans[i] 	      = vsp.GetParmVal( wing_id, 'Span', 'XSec_' + str(i))
			proj_span_sum += segment_spans[i] * np.cos(segment_dihedral[i])	
			span_sum += segment_spans[i]
		else:
			segment.root_chord_percent = (vsp.GetParmVal( wing_id, 'Tip_Chord', 'XSec_' + str(i-1)))/total_chord
			
		# XSec airfoil
		jj = i-1  # Airfoil index
		xsec_id = str(vsp.GetXSec(xsec_surf_id, jj))
		airfoil = Airfoil()
		if vsp.GetXSecShape( xsec_id ) == 7: # XSec shape: NACA 4-series
			camber = vsp.GetParmVal( wing_id, 'Camber', 'XSecCurve_' + str(jj)) 
			if camber == 0.:	# i-1 because vsp airfoils and sections are one index off relative to SUAVE
				camber_loc = 0.
			else:
				camber_loc = vsp.GetParmVal( wing_id, 'CamberLoc', 'XSecCurve_' + str(jj))
			airfoil.thickness_to_chord = thick_cord
			camber_round = int(np.around(camber*100))
			camber_loc_round = int(np.around(camber_loc*10))  # Camber and TC won't round up for NACA.
			thick_cord_round = int(np.around(thick_cord*100))
			airfoil.tag = 'NACA ' + str(camber_round) + str(camber_loc_round) + str(thick_cord_round)
		
		elif vsp.GetXSecShape( xsec_id ) == 8: # XSec shape: NACA 6-series
			thick_cord_round = int(np.around(thick_cord*100))
			a_value = vsp.GetParmVal( wing_id, 'A', 'XSecCurve_' + str(jj))
			ideal_CL = int(np.around(vsp.GetParmVal( wing_id, 'IdealCl', 'XSecCurve_' + str(jj))*10))
			series_vsp = int(vsp.GetParmVal( wing_id, 'Series', 'XSecCurve_' + str(jj)))
			series_dict = Data({0:'63',1:'64',2:'65',3:'66',4:'67',5:'63A',6:'64A',7:'65A'})
			series = series_dict[series_vsp]
			airfoil.tag = 'NACA ' + series + str(ideal_CL) + str(thick_cord_round) + ' a=' + str(np.around(a_value,1))
		
		elif vsp.GetXSecShape( xsec_id ) == 12:	# XSec shape: 12 is type AF_FILE
			airfoil.thickness_to_chord = thick_cord
			airfoil.points = vsp.GetAirfoilCoordinates( wing_id, jj/segment_num )
			vsp.WriteSeligAirfoil(str(wing.tag) + '_airfoil_XSec_' + str(jj) +'.txt', wing_id, .86)
			airfoil.coordinate_file = str(wing.tag) + '_airfoil_XSec_' + str(jj) +'.dat'
			airfoil.tag = 'AF_file'	
		
		segment.append_airfoil(airfoil)
		
		wing.Segments.append(segment)
	
	
	
	# Wing dihedral: exclude segments with dihedral values over 70deg (like wingtips)
	proj_span_sum_alt = 0.
	span_sum_alt = 0.
	for ii in xrange( start, segment_num):
		if segment_dihedral[ii] <= (70. * Units.deg):
			span_sum_alt += segment_spans[ii]
			proj_span_sum_alt += segment_spans[ii] * np.cos(segment_dihedral[ii])
		else:
			pass
	wing.dihedral = np.arccos(proj_span_sum_alt / span_sum_alt) / Units.deg

	# Chords
	wing.chords.root             = vsp.GetParmVal( wing_id, 'Tip_Chord', 'XSec_1')
	wing.chords.tip              = vsp.GetParmVal( wing_id, 'Tip_Chord', 'XSec_' + str(segment_num-1))	
	wing.chords.mean_geometric = vsp.GetParmVal( wing_id, 'TotalArea', 'WingGeom') / vsp.GetParmVal( wing_id, 'TotalChord', 'WingGeom')
	#wing.chords.mean_aerodynamic = ________ / vsp.GetParmVal( wing_id, 'TotalSpan', 'WingGeom')
	
	
	# Areas
	wing.areas.reference         = vsp.GetParmVal( wing_id, 'TotalArea', 'WingGeom')
	wetted_areas = get_vsp_areas(wing.tag)	
	wing.areas.wetted   = wetted_areas[wing.tag]
	wing.areas.exposed   = wetted_areas[wing.tag]
	
	#wing.sweeps.quarter_chord    = 33. * Units.degrees

	# Twists
	wing.twists.root             = vsp.GetParmVal( wing_id, 'Twist', 'XSec_0') * Units.deg
	wing.twists.tip              = vsp.GetParmVal( wing_id, 'Twist', 'XSec_' + str(segment_num-1)) * Units.deg
	


	#FINISH
	if wing.symmetric == True:
		wing.spans.projected = proj_span_sum*2
	else:
		wing.spans.projected = proj_span_sum	
		
	return wing





def readFuselage(fuselage_id):
	fuselage = SUAVE.Components.Fuselages.Fuselage()	#Create SUAVE fuselage.
	if vsp.GetGeomName(fuselage_id):
		fuselage.tag = vsp.GetGeomName(fuselage_id)
	else: 
		fuselage.tag = 'FuselageGeom'	

	fuselage.lengths.total = vsp.GetParmVal( fuselage_id, 'Length', 'Design')	
	fuselage.vsp.xsec_surf_id = vsp.GetXSecSurf( fuselage_id, 0 ) 	# There is only one XSecSurf in geom.
	fuselage.vsp.xsec_num = vsp.GetNumXSec( fuselage.vsp.xsec_surf_id ) 		# Number of xsecs in fuselage.	
	
	x_locs = []
	heights = []
	widths = []
	eff_diams = []
	lengths = []
	
	for ii in xrange(0, fuselage.vsp.xsec_num):
		segment = SUAVE.Components.Fuselages.Segment()
		segment.vsp.xsec_id	= vsp.GetXSec( fuselage.vsp.xsec_surf_id, ii )
		segment.tag = 'segment_' + str(ii)
		segment.percent_x_location = vsp.GetParmVal( fuselage_id, 'XLocPercent', 'XSec_' + str(ii))
		segment.percent_z_location = vsp.GetParmVal( fuselage_id, 'ZLocPercent', 'XSec_' + str(ii))
		segment.height             = vsp.GetXSecHeight(segment.vsp.xsec_id)
		segment.width              = vsp.GetXSecWidth(segment.vsp.xsec_id)
		segment.effective_diameter = (segment.height+segment.width)/2.
		x_locs.append(segment.percent_x_location)
		heights.append(segment.height)
		widths.append(segment.width)
		eff_diams.append((segment.height+segment.width)/2)
		
		if ii !=0: # Segment length: stored as length since previous segment. First segment will have length 0.0.
			segment.length = fuselage.lengths.total*(segment.percent_x_location-fuselage.Segments[ii-1].percent_x_location)
		else:
			segment.length = 0.0
		lengths.append(segment.length)
		
		shape		= vsp.GetXSecShape(segment.vsp.xsec_id)
		shape_dict 	= {0:'point',1:'circle',2:'ellipse',3:'super ellipse',4:'rounded rectangle',5:'general fuse',6:'fuse file'}
		segment.vsp.shape = shape_dict[shape]	
	
		fuselage.Segments.append(segment)

	fuselage.heights.at_quarter_length = get_fuselage_height(fuselage, .25)
	fuselage.heights.at_three_quarters_length = get_fuselage_height(fuselage, .75)

	fuselage.heights.maximum = max(heights)			# Max segment height.	
	fuselage.width		 = max(widths)			# Max segment width.
	fuselage.effective_diameter = max(eff_diams)		# Max segment effective diam.

	eff_diam_gradients_fwd = np.array(eff_diams[1:]) - np.array(eff_diams[:-1])
	eff_diam_gradients_fwd = np.multiply(eff_diam_gradients_fwd, np.reciprocal(lengths[1:]))
		
	# Compute nose fineness.    
	x_locs = np.array(x_locs)
	eff_diams = np.array(eff_diams)
	min_val = np.min(eff_diam_gradients_fwd[x_locs[:-1]<=0.5])
	x_loc = x_locs[eff_diam_gradients_fwd==min_val][0]
	index_seg = np.where(x_locs==x_loc)[0][0]
	fuselage.lengths.nose = (x_loc-fuselage.Segments[0].percent_x_location)*fuselage.lengths.total
	fuselage.fineness.nose = fuselage.lengths.nose/(eff_diams[x_locs==x_loc][0])
	
	# Compute tail fineness.
	x_locsg5 = x_locs>=0.5
	eff_diam_gradients_fwd_g5 = eff_diam_gradients_fwd[x_locsg5[1:]]
	min_val = np.min(-eff_diam_gradients_fwd_g5)
	x_loc = x_locs[np.hstack([False,-eff_diam_gradients_fwd==min_val])][-1]
	fuselage.lengths.tail = (fuselage.Segments[0].percent_x_location-x_loc)*fuselage.lengths.total
	fuselage.fineness.tail = fuselage.lengths.tail/(eff_diams[x_locs==x_loc][0])
	
	wetted_areas = get_vsp_areas(fuselage.tag)		# Wetted_areas array contains areas for all vehicle geometries.
	fuselage.areas.wetted = wetted_areas[fuselage.tag]	
	
	return fuselage

def readProp(prop_id):
	prop = SUAVE.Components.Energy.Converters.Propeller()
	
	if vsp.GetGeomName(prop_id): # Mostly relevant for eVTOLs with > 1 propeller.
		fuselage.tag = vsp.GetGeomName(prop_id)
	else: 
		fuselage.tag = 'PropGeom'	
	
	prop.prop_attributes.number_blades = vsp.GetParmVal(prop_id, 'NumBlade', 'Design')
	tip_radius = vsp.GetParmVal(prop_id, 'Diameter', 'Design')/2.
	prop.prop_attributes.tip_radius = tip_radius
	prop.prop_attributes.hub_radius = vsp.GetParmVal(prop_id, 'RadiusFrac', 'XSec_0') * tip_radius	
	
	prop.location[0] = vsp.GetParmVal(prop_id, 'X_Location', 'XForm')
	prop.location[1] = vsp.GetParmVal(prop_id, 'Y_Location', 'XForm')
	prop.location[2] = vsp.GetParmVal(prop_id, 'Z_Location', 'XForm')
	prop.rotation[0] = vsp.GetParmVal(prop_id, 'X_Rotation', 'XForm')
	prop.rotation[1] = vsp.GetParmVal(prop_id, 'Y_Rotation', 'XForm')
	prop.rotation[2] = vsp.GetParmVal(prop_id, 'Z_Rotation', 'XForm')
	
	prop.thrust_angle = prop.rotation[1]
	
	xsecsurf_id = vsp.GetXSecSurf(prop_id, 0)
	
	curve_type = {0:'linear',1:'spline',2:'Bezier_cubic'}
	
	# Chord
	chord_curve = curve_type[int(vsp.GetParmVal(prop_id, 'CrvType', 'Chord'))]
	chord_split_point = vsp.GetParmVal(prop_id, 'SplitPt', 'Chord')
	chords = []
	chords_rad = [] # This is r/R value.
	chords_num = 10  # Find this with API somehow 
	for ii in xrange(chord_num):
		chords.append(vsp.GetParmVal(prop_id, 'crd_' + str(ii), 'Chord'))
		chords_rad.append(vsp.GetParmVal(prop_id, 'r_' + str(ii), 'Chord'))
	
	# Twist
	twist_curve = curve_type[int(vsp.GetParmVal(prop_id, 'CrvType', 'Twist'))]
	twist_split_point = vsp.GetParmVal(prop_id, 'SplitPt', 'Twist')	
	twists = []
	twists_rad = []
	twists_num = 3
	for ii in xrange(twist_num):
		twist = vsp.GetParmVal(prop_id, 'tw_' + str(ii), 'Twist')
		twists.append(twist)
		twists_rad.append(vsp.GetParmVal(prop_id, 'r_' + str(ii), 'Twist'))		
	
	# Skew
	skew_curve = curve_type[int(vsp.GetParmVal(prop_id, 'CrvType', 'Skew'))]
	skew_split_point = vsp.GetParmVal(prop_id, 'SplitPt', 'Skew')
	skews = []
	skews_rad = []
	skews_num = 10
	for ii in xrange(skews_num):
		skews.append(vsp.GetParmVal(prop_id, 'skw_' + str(ii), 'Skew'))
		skews_rad.append(vsp.GetParmVal(prop_id, 'r_' + str(ii), 'Skew'))	
	
	# Rake
	rake_curve = curve_type[int(vsp.GetParmVal(prop_id, 'CrvType', 'Rake'))]
	rake_split_point = vsp.GetParmVal(prop_id, 'SplitPt', 'Rake')
	rakes = []
	rakes_rad = []
	rakes_num = 3
	for ii in xrange(rakes_num):
		skews.append(vsp.GetParmVal(prop_id, 'rak_' + str(ii), 'Rake'))
		skews_rad.append(vsp.GetParmVal(prop_id, 'r_' + str(ii), 'Rake'))
		
	# Sweep
	sweep_curve = curve_type[int(vsp.GetParmVal(prop_id, 'CrvType', 'Sweep'))]
	sweep_split_point = vsp.GetParmVal(prop_id, 'SplitPt', 'Sweep')
	sweeps = []
	sweeps_rad = []
	sweeps_num = 3
	for ii in xrange(sweeps_num):
		skews.append(vsp.GetParmVal(prop_id, 'sw_' + str(ii), 'Sweep'))
		skews_rad.append(vsp.GetParmVal(prop_id, 'r_' + str(ii), 'Sweep'))	

	return prop



def get_fuselage_height(fuselage, location):	# Linearly estimate the height of the fuselage at *any* point.
	for jj in xrange(1, fuselage.vsp.xsec_num):
		if fuselage.Segments[jj].percent_x_location>=location and fuselage.Segments[jj-1].percent_x_location<location:
			a = fuselage.Segments[jj].percent_x_location
			b = fuselage.Segments[jj-1].percent_x_location
			a_height = fuselage.Segments[jj].height
			b_height = fuselage.Segments[jj-1].height
			slope = (a_height - b_height)/(a-b)
			height = ((location-b)*(slope)) + (b_height)	
	return height



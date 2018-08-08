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
def vsp_read(tag, units): 	
	"""This reads an OpenVSP vehicle geometry and writes it into a SUAVE vehicle format.
	Includes wings, fuselages, and propellers.

	Assumptions:
	OpenVSP vehicle is composed of conventionally shaped fuselages, wings, and propellers. 
	
	Source:
	N/A

	Inputs:
	An XML file in format .vsp3.

	Outputs:
	Writes SUAVE vehicle with these geometries from VSP:
		Wings.Wing.    (* is all keys)
			origin                                  [m] in all three dimensions
			spans.projected                         [m]
			chords.root                             [m]
			chords.tip                              [m]
			aspect_ratio                            [-]
			sweeps.quarter_chord                    [radians]
			twists.root                             [radians]
			twists.tip                              [radians]
			thickness_to_chord                      [-]
			dihedral                                [radians]
			symmetric                               <boolean>
			tag                                     <string>
			areas.exposed                           [m^2]
			areas.reference                         [m^2]
			areas.wetted                            [m^2]
			Segments.
			  tag                                   <string>
			  twist                                 [radians]
			  percent_span_location                 [-]  .1 is 10%
			  root_chord_percent                    [-]  .1 is 10%
			  dihedral_outboard                     [radians]
			  sweeps.quarter_chord                  [radians]
			  thickness_to_chord                    [-]
			  airfoil                               <NACA 4-series, 6 series, or airfoil file>
			
		Fuselages.Fuselage.			
			origin                                  [m] in all three dimensions
			width                                   [m]
			lengths.
			  total                                 [m]
			  nose                                  [m]
			  tail                                  [m]
			heights.
			  maximum                               [m]
			  at_quarter_length                     [m]
			  at_three_quarters_length              [m]
			effective_diameter                      [m]
			fineness.nose                           [-] ratio of nose section length to fuselage effective diameter
			fineness.tail                           [-] ratio of tail section length to fuselage effective diameter
			areas.wetted                            [m^2]
			tag                                     <string>
			segment[].   (segments are in ordered container and callable by number)
			  vsp.shape                               [point,circle,round_rect,general_fuse,fuse_file]
			  vsp.xsec_id                             <10 digit string>
			  percent_x_location
			  percent_z_location
			  height
			  width
			  length
			  effective_diameter
			  tag
			vsp.xsec_num                              <integer of fuselage segment quantity>
			vsp.xsec_surf_id                          <10 digit string>
	
		Propellers.Propeller.
		
	
	Properties Used:
	N/A
	"""  	
	
	vsp.ClearVSPModel() 
	vsp.ReadVSPFile(tag)	
	
	vsp_fuselages = []
	vsp_wings = []	
	vsp_props = []
	
	vsp_geoms  = vsp.FindGeoms()
	geom_names = []

	'''
	print "VSP geometry IDs: " 	# Until OpenVSP is released with a call for GetGeomType, each geom must be manually processed.
	
	for geom in vsp_geoms:
		geom_name = vsp.GetGeomName(geom)
		geom_names.append(geom_name)
		print str(geom_name) + ': ' + geom
		
	
	# Label each geom type by storing its VSP geom ID. (The API call for GETGEOMTYPE was not released as of 08/06/18, v 3.16.1)
	
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
	
	# Read VSP geoms and store in SUAVE components.
	'''
	for vsp_fuselage in vsp_fuselages:
		fuselage_id = vsp_fuselages[vsp_fuselage]
		fuselage = read_vsp_fuselage(fuselage_id, units)
		vehicle.append_component(fuselage)
	
	for vsp_wing in vsp_wings:
		wing_id = vsp_wings[vsp_wing]
		wing = read_vsp_wing(wing_id, units)
		vehicle.append_component(wing)		
	
	for vsp_prop in vsp_props:
		prop_id = vsp_props[vsp_prop]
		prop = read_vsp_prop(prop_id, units)		
		vehicle.append_component(prop)
	
	'''
	
	return vehicle


def vsp_read_fuselage(fuselage_id, fineness=True):

	fuselage = SUAVE.Components.Fuselages.Fuselage()	
	
	if vsp.GetGeomName(fuselage_id):
		fuselage.tag = vsp.GetGeomName(fuselage_id)
	else: 
		fuselage.tag = 'FuselageGeom'	

	fuselage.origin[0] = vsp.GetParmVal(fuselage_id, 'X_Rel_Location', 'XForm')
	fuselage.origin[1] = vsp.GetParmVal(fuselage_id, 'Y_Rel_Location', 'XForm')
	fuselage.origin[2] = vsp.GetParmVal(fuselage_id, 'Z_Rel_Location', 'XForm')

	fuselage.lengths.total    = vsp.GetParmVal(fuselage_id, 'Length', 'Design')	
	fuselage.vsp.xsec_surf_id = vsp.GetXSecSurf(fuselage_id, 0) 			# There is only one XSecSurf in geom.
	fuselage.vsp.xsec_num     = vsp.GetNumXSec(fuselage.vsp.xsec_surf_id) 		# Number of xsecs in fuselage.	
	
	x_locs    = []
	heights   = []
	widths    = []
	eff_diams = []
	lengths   = []
	
	# -------------
	# Fuselage segments
	# -------------	
	
	for ii in xrange(0, fuselage.vsp.xsec_num):
		segment = SUAVE.Components.Fuselages.Segment()
		segment.vsp.xsec_id	   = vsp.GetXSec(fuselage.vsp.xsec_surf_id, ii)	# VSP XSec ID.
		segment.tag                = 'segment_' + str(ii)
		segment.percent_x_location = vsp.GetParmVal(fuselage_id, 'XLocPercent', 'XSec_' + str(ii)) # Along fuselage length.
		segment.percent_z_location = vsp.GetParmVal(fuselage_id, 'ZLocPercent', 'XSec_' + str(ii)) # Vertical deviation of fuselage center.
		segment.height             = vsp.GetXSecHeight(segment.vsp.xsec_id)
		segment.width              = vsp.GetXSecWidth(segment.vsp.xsec_id)
		segment.effective_diameter = (segment.height+segment.width)/2.
		
		x_locs.append(segment.percent_x_location)	 # Save into arrays for later computation.
		heights.append(segment.height)
		widths.append(segment.width)
		eff_diams.append(segment.effective_diameter)
		
		if ii !=0: # Segment length: stored as length since previous segment. (First segment will have length 0.0.)
			segment.length = fuselage.lengths.total*(segment.percent_x_location-fuselage.Segments[ii-1].percent_x_location)
		else:
			segment.length = 0.0
		lengths.append(segment.length)
		
		shape	 	  = vsp.GetXSecShape(segment.vsp.xsec_id)
		shape_dict 	  = {0:'point',1:'circle',2:'ellipse',3:'super ellipse',4:'rounded rectangle',5:'general fuse',6:'fuse file'}
		segment.vsp.shape = shape_dict[shape]	
	
		fuselage.Segments.append(segment)

	fuselage.heights.at_quarter_length        = get_fuselage_height(fuselage, .25)	# Calls get_fuselage_height function.
	fuselage.heights.at_three_quarters_length = get_fuselage_height(fuselage, .75)

	fuselage.heights.maximum    = max(heights)		# Max segment height.	
	fuselage.width		    = max(widths)		# Max segment width.
	fuselage.effective_diameter = max(eff_diams)		# Max segment effective diam.

	eff_diam_gradients_fwd = np.array(eff_diams[1:]) - np.array(eff_diams[:-1])		# Compute gradients of segment effective diameters.
	eff_diam_gradients_fwd = np.multiply(eff_diam_gradients_fwd, np.reciprocal(lengths[1:]))
		
	fuselage = compute_fuselage_fineness(fuselage, x_locs, eff_diams, eff_diam_gradients_fwd)

	return fuselage
	
def compute_fuselage_fineness(fuselage, x_locs, eff_diams, eff_diam_gradients_fwd):
	# Compute nose fineness.    
	x_locs    = np.array(x_locs)					# Make numpy arrays.
	eff_diams = np.array(eff_diams)
	min_val   = np.min(eff_diam_gradients_fwd[x_locs[:-1]<=0.5])	# Computes smallest eff_diam gradient value in front 50% of fuselage.
	x_loc     = x_locs[eff_diam_gradients_fwd==min_val][0]		# Determines x-location of the first instance of that value (if gradient=0, gets frontmost x-loc).
	fuselage.lengths.nose = (x_loc-fuselage.Segments[0].percent_x_location)*fuselage.lengths.total	# Subtracts first segment x-loc in case not at global origin.
	fuselage.fineness.nose = fuselage.lengths.nose/(eff_diams[x_locs==x_loc][0])
	
	# Compute tail fineness.
	x_locs_tail		    = x_locs>=0.5				# Searches aft 50% of fuselage.
	eff_diam_gradients_fwd_tail = eff_diam_gradients_fwd[x_locs_tail[1:]]	# Smaller array of tail gradients.
	min_val 		    = np.min(-eff_diam_gradients_fwd_tail)	# Computes min gradient, where fuselage tapers (minus sign makes positive).
	x_loc = x_locs[np.hstack([False,-eff_diam_gradients_fwd==min_val])][-1] # Saves aft-most value (useful for straight fuselage with multiple zero gradients.) 
	fuselage.lengths.tail       = (x_loc-fuselage.Segments[0].percent_x_location)*fuselage.lengths.total
	fuselage.fineness.tail      = -fuselage.lengths.tail/(eff_diams[x_locs==x_loc][0])	# Minus sign converts tail fineness to positive value.
	
	wetted_areas = get_vsp_areas(fuselage.tag)				# Wetted_areas array contains areas for all vehicle geometries.
	fuselage.areas.wetted = wetted_areas[fuselage.tag]	
	
	return fuselage

def vsp_read_prop(prop_id):
	prop = SUAVE.Components.Energy.Converters.Propeller()
	
	if vsp.GetGeomName(prop_id): # Mostly relevant for eVTOLs with > 1 propeller.
		fuselage.tag = vsp.GetGeomName(prop_id)
	else: 
		fuselage.tag = 'PropGeom'	
	
	prop.prop_attributes.number_blades = vsp.GetParmVal(prop_id, 'NumBlade', 'Design')
	tip_radius = vsp.GetParmVal(prop_id, 'Diameter', 'Design')/2.
	prop.prop_attributes.tip_radius = tip_radius
	prop.prop_attributes.hub_radius = vsp.GetParmVal(prop_id, 'RadiusFrac', 'XSec_0') * tip_radius	
	
	prop.location[0] = vsp.GetParmVal(prop_id, 'X_Rel_Location', 'XForm')
	prop.location[1] = vsp.GetParmVal(prop_id, 'Y_Rel_Location', 'XForm')
	prop.location[2] = vsp.GetParmVal(prop_id, 'Z_Rel_Location', 'XForm')
	prop.rotation[0] = vsp.GetParmVal(prop_id, 'X_Rel_Rotation', 'XForm')
	prop.rotation[1] = vsp.GetParmVal(prop_id, 'Y_Rel_Rotation', 'XForm')
	prop.rotation[2] = vsp.GetParmVal(prop_id, 'Z_Rel_Rotation', 'XForm')
	
	prop.thrust_angle = prop.rotation[1]
	
	xsecsurf_id = vsp.GetXSecSurf(prop_id, 0)
	
	curve_type = {0:'linear',1:'spline',2:'Bezier_cubic'}
	
	# -------------
	# Blade geometry
	# -------------	
	
	# Chord
	chord_curve = curve_type[int(vsp.GetParmVal(prop_id, 'CrvType', 'Chord'))]
	chord_split_point = vsp.GetParmVal(prop_id, 'SplitPt', 'Chord')
	chords = []
	chords_rad = []  # This is r/R value.
	chords_num = 10  # Find this with API somehow.  HARDCODED
	for ii in xrange(chord_num):
		chords.append(vsp.GetParmVal(prop_id, 'crd_' + str(ii), 'Chord'))
		chords_rad.append(vsp.GetParmVal(prop_id, 'r_' + str(ii), 'Chord'))
	
	# Twist
	twist_curve = curve_type[int(vsp.GetParmVal(prop_id, 'CrvType', 'Twist'))]
	twist_split_point = vsp.GetParmVal(prop_id, 'SplitPt', 'Twist')	
	twists = []
	twists_rad = []
	twists_num = 3	# HARDCODED
	for ii in xrange(twist_num):
		twist = vsp.GetParmVal(prop_id, 'tw_' + str(ii), 'Twist')
		twists.append(twist)
		twists_rad.append(vsp.GetParmVal(prop_id, 'r_' + str(ii), 'Twist'))		
	
	# Skew
	skew_curve = curve_type[int(vsp.GetParmVal(prop_id, 'CrvType', 'Skew'))]
	skew_split_point = vsp.GetParmVal(prop_id, 'SplitPt', 'Skew')
	skews = []
	skews_rad = []
	skews_num = 10	#HARDCODED
	for ii in xrange(skews_num):
		skews.append(vsp.GetParmVal(prop_id, 'skw_' + str(ii), 'Skew'))
		skews_rad.append(vsp.GetParmVal(prop_id, 'r_' + str(ii), 'Skew'))	
	
	# Rake
	rake_curve = curve_type[int(vsp.GetParmVal(prop_id, 'CrvType', 'Rake'))]
	rake_split_point = vsp.GetParmVal(prop_id, 'SplitPt', 'Rake')
	rakes = []
	rakes_rad = []
	rakes_num = 3	#HARDCODED
	for ii in xrange(rakes_num):
		skews.append(vsp.GetParmVal(prop_id, 'rak_' + str(ii), 'Rake'))
		skews_rad.append(vsp.GetParmVal(prop_id, 'r_' + str(ii), 'Rake'))
		
	# Sweep
	sweep_curve = curve_type[int(vsp.GetParmVal(prop_id, 'CrvType', 'Sweep'))]
	sweep_split_point = vsp.GetParmVal(prop_id, 'SplitPt', 'Sweep')
	sweeps = []
	sweeps_rad = []
	sweeps_num = 3	#HARDCODED
	for ii in xrange(sweeps_num):
		skews.append(vsp.GetParmVal(prop_id, 'sw_' + str(ii), 'Sweep'))
		skews_rad.append(vsp.GetParmVal(prop_id, 'r_' + str(ii), 'Sweep'))	

	return prop


vsp.PCurveGetTVec
vsp.PCurveGetValVec


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

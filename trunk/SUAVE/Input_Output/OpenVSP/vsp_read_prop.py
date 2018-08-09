## @ingroup Input_Output-OpenVSP
# vsp_read_prop.py

# Created:  Jun 2018, T. St Francis
# Modified: Aug 2018, T. St Francis

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Units, Data
from SUAVE.Input_Output.OpenVSP import get_vsp_areas
from SUAVE.Components.Wings.Airfoils.Airfoil import Airfoil 
import vsp_g as vsp
import numpy as np


## @ingroup Input_Output-OpenVSP
def vsp_read(tag, units='SI'): 	
	"""This reads an OpenVSP propeller geometry and writes it to a SUAVE propeller format.

	Assumptions:
	Written for OpenVSP 3.16.1

	Source:
	N/A

	Inputs:
	1. A tag for an XML file in format .vsp3.
	2. Units set to 'SI' (default) or 'Imperial'

	Outputs:
	Writes SUAVE vehicle with these geometries from VSP:    (all defaults are SI, but user may specify Imperial)
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
			location[X,Y,Z]                            [radians]
			rotation[X,Y,Z]                            [radians]
			prop_attributes.tip_radius                 [m]
		        prop_attributes.hub_radius                 [m]
			thrust_angle                               [radians]

	Properties Used:
	N/A
	"""  	

prop = SUAVE.Components.Energy.Converters.Propeller()
	
	if units == 'SI':
		units = Units.meter 
	else units == 'Imperial':
		units = Units.foot	
	
	if vsp.GetGeomName(prop_id): # Mostly relevant for eVTOLs with > 1 propeller.
		prop.tag = vsp.GetGeomName(prop_id)
	else: 
		prop.tag = 'PropGeom'	
	
	prop.prop_attributes.number_blades = vsp.GetParmVal(prop_id, 'NumBlade', 'Design')
	tip_radius = (vsp.GetParmVal(prop_id, 'Diameter', 'Design')/2.) * units
	prop.prop_attributes.tip_radius = tip_radius
	prop.prop_attributes.hub_radius = vsp.GetParmVal(prop_id, 'RadiusFrac', 'XSec_0') * tip_radius	
	
	prop.location[0] = vsp.GetParmVal(prop_id, 'X_Rel_Location', 'XForm') * units
	prop.location[1] = vsp.GetParmVal(prop_id, 'Y_Rel_Location', 'XForm') * units
	prop.location[2] = vsp.GetParmVal(prop_id, 'Z_Rel_Location', 'XForm') * units
	prop.rotation[0] = vsp.GetParmVal(prop_id, 'X_Rel_Rotation', 'XForm') * Units.deg
	prop.rotation[1] = vsp.GetParmVal(prop_id, 'Y_Rel_Rotation', 'XForm') * Units.deg
	prop.rotation[2] = vsp.GetParmVal(prop_id, 'Z_Rel_Rotation', 'XForm') * Units.deg
	
	prop.thrust_angle = prop.rotation[1]			# Y-rotation
	
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
	for ii in xrange(chords_num):
		chords.append(vsp.GetParmVal(prop_id, 'crd_' + str(ii), 'Chord'))
		chords_rad.append(vsp.GetParmVal(prop_id, 'r_' + str(ii), 'Chord'))
	
	# Twist
	twist_curve = curve_type[int(vsp.GetParmVal(prop_id, 'CrvType', 'Twist'))]
	twist_split_point = vsp.GetParmVal(prop_id, 'SplitPt', 'Twist')	
	twists = []
	twists_rad = []
	twists_num = 3	# HARDCODED
	for ii in xrange(twists_num):
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
		rakes.append(vsp.GetParmVal(prop_id, 'rak_' + str(ii), 'Rake'))
		rakes_rad.append(vsp.GetParmVal(prop_id, 'r_' + str(ii), 'Rake'))
		
	# Sweep
	sweep_curve = curve_type[int(vsp.GetParmVal(prop_id, 'CrvType', 'Sweep'))]
	sweep_split_point = vsp.GetParmVal(prop_id, 'SplitPt', 'Sweep')
	sweeps = []
	sweeps_rad = []
	sweeps_num = 3	#HARDCODED
	for ii in xrange(sweeps_num):
		sweeps.append(vsp.GetParmVal(prop_id, 'sw_' + str(ii), 'Sweep'))
		sweeps_rad.append(vsp.GetParmVal(prop_id, 'r_' + str(ii), 'Sweep'))	

	return prop


	#vsp.PCurveGetTVec
	#vsp.PCurveGetValVec



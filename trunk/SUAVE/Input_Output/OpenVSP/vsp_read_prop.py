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
def vsp_read_prop(prop_id, units_type='SI'): 	
	"""This reads an OpenVSP propeller geometry and writes it to a SUAVE propeller format.

	Assumptions:
	Written for OpenVSP 3.16.1

	Source:
	N/A

	Inputs:
	1. A tag for an XML file in format .vsp3.
	2. units_type set to 'SI' (default) or 'Imperial'

	Outputs:
	Writes SUAVE propeller with these geometries from VSP:    (all defaults are SI, but user may specify Imperial)
		Propellers.Propeller.
			location[X,Y,Z]                            [radians]
			rotation[X,Y,Z]                            [radians]
			prop_attributes.tip_radius                 [m]
		        prop_attributes.hub_radius                 [m]
			thrust_angle                               [radians]
	
	Note: fills arrays with parametric curve points for:
	        twists
		chords
		skews
		rakes
		sweeps
		
	Properties Used:
	N/A
	"""  	

	prop = SUAVE.Components.Energy.Converters.Propeller()
	
	if units_type == 'SI':
		units_factor = Units.meter 
	else:
		units_factor = Units.foot	
	
	if vsp.GetGeomName(prop_id): # Mostly relevant for eVTOLs with > 1 propeller.
		prop.tag = vsp.GetGeomName(prop_id)
	else: 
		prop.tag = 'PropGeom'	
	
	prop.prop_attributes.number_blades = vsp.GetParmVal(prop_id, 'NumBlade', 'Design')
	tip_radius = (vsp.GetParmVal(prop_id, 'Diameter', 'Design')/2.) * units_factor
	prop.prop_attributes.tip_radius = tip_radius
	prop.prop_attributes.hub_radius = vsp.GetParmVal(prop_id, 'RadiusFrac', 'XSec_0') * tip_radius	
	
	prop.origin[0] = vsp.GetParmVal(prop_id, 'X_Rel_Location', 'XForm') * units_factor
	prop.origin[1] = vsp.GetParmVal(prop_id, 'Y_Rel_Location', 'XForm') * units_factor
	prop.origin[2] = vsp.GetParmVal(prop_id, 'Z_Rel_Location', 'XForm') * units_factor
	prop.rotation[0] = vsp.GetParmVal(prop_id, 'X_Rel_Rotation', 'XForm') * Units.deg
	prop.rotation[1] = vsp.GetParmVal(prop_id, 'Y_Rel_Rotation', 'XForm') * Units.deg
	prop.rotation[2] = vsp.GetParmVal(prop_id, 'Z_Rel_Rotation', 'XForm') * Units.deg
	
	prop.thrust_angle = prop.rotation[1]			# Y-rotation for thrust angle.
	
	xsecsurf_id = vsp.GetXSecSurf(prop_id, 0)
	
	curve_type = {0:'linear',1:'spline',2:'Bezier_cubic'}
	
	# -------------
	# Blade geometry
	# -------------	
	
	# Chord
	chord_curve       = curve_type[int(vsp.GetParmVal(prop_id, 'CrvType', 'Chord'))]
	chord_split_point = vsp.GetParmVal(prop_id, 'SplitPt', 'Chord')
	chords            = []
	chords_rad        = []  						# This is r/R value.
	chords_num        = 50  						# HARDCODED, see break below.
	for ii in xrange(chords_num):						# Future API call goes for Pcurve chord number goes here.
		chords.append(vsp.GetParmVal(prop_id, 'crd_' + str(ii), 'Chord')) * units_factor
		chords_rad.append(vsp.GetParmVal(prop_id, 'r_' + str(ii), 'Chord'))
		if ii!=0 and chords[ii] == 0.0 and chords[ii-1] == 0.0:		# Allows for two zero conditions before breaking, then resizes array.
			chords     = chords[:-2]
			chords_rad = chords_rad[:-2]
			break
	
	# Twist
	twist_curve       = curve_type[int(vsp.GetParmVal(prop_id, 'CrvType', 'Twist'))]
	twist_split_point = vsp.GetParmVal(prop_id, 'SplitPt', 'Twist')	
	twists            = []
	twists_rad        = []							# This is r/R value.
	twists_num        = 50							# HARDCODED
	for ii in xrange(twists_num):
		twists.append(vsp.GetParmVal(prop_id, 'tw_' + str(ii), 'Twist') * Units.deg) 
		twists_rad.append(vsp.GetParmVal(prop_id, 'r_' + str(ii), 'Twist'))		
		if ii!=0 and twists[ii] == 0.0 and twists[ii-1] == 0.0:
			twists     = twists[:-2]
			twists_rad = twists_rad[:-2]
			break
	
	# Skew
	skew_curve       = curve_type[int(vsp.GetParmVal(prop_id, 'CrvType', 'Skew'))]
	skew_split_point = vsp.GetParmVal(prop_id, 'SplitPt', 'Skew')
	skews            = []
	skews_rad        = []
	skews_num        = 50							#HARDCODED
	for ii in xrange(skews_num):
		skews.append(vsp.GetParmVal(prop_id, 'skw_' + str(ii), 'Skew') * Units.deg)
		skews_rad.append(vsp.GetParmVal(prop_id, 'r_' + str(ii), 'Skew'))	
		if ii!=0 and skews[ii] == 0.0 and skews[ii-1] == 0.0:
			skews     = skews[:-2]
			skews_rad = skews_rad[:-2]
			break
	
	# Rake
	rake_curve       = curve_type[int(vsp.GetParmVal(prop_id, 'CrvType', 'Rake'))]
	rake_split_point = vsp.GetParmVal(prop_id, 'SplitPt', 'Rake')
	rakes            = []
	rakes_rad        = []	
	rakes_num        = 50							#HARDCODED
	for ii in xrange(rakes_num):
		rakes.append(vsp.GetParmVal(prop_id, 'rak_' + str(ii), 'Rake') * Units.deg)
		rakes_rad.append(vsp.GetParmVal(prop_id, 'r_' + str(ii), 'Rake'))
		if ii!=0 and rakes[ii] == 0.0 and rakes[ii-1] == 0.0:
			rakes     = rakes[:-2]
			rakes_rad = rakes_rad[:-2]
			break
		
	# Sweep
	sweep_curve       = curve_type[int(vsp.GetParmVal(prop_id, 'CrvType', 'Sweep'))]
	sweep_split_point = vsp.GetParmVal(prop_id, 'SplitPt', 'Sweep')
	sweeps            = []
	sweeps_rad        = []
	sweeps_num        = 50							#HARDCODED
	for ii in xrange(sweeps_num):
		sweeps.append(vsp.GetParmVal(prop_id, 'sw_' + str(ii), 'Sweep') * Units.deg)
		sweeps_rad.append(vsp.GetParmVal(prop_id, 'r_' + str(ii), 'Sweep'))	
		if ii!=0 and sweeps[ii] == 0.0 and sweeps[ii-1] == 0.0:
			sweeps     = sweeps[:-2]
			sweeps_rad = sweeps_rad[:-2]			
			break
		
	return prop






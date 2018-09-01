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
			origin[X,Y,Z]                              [radians]
			rotation[X,Y,Z]                            [radians]
			prop_attributes.tip_radius                 [m]
		        prop_attributes.hub_radius                 [m]
			thrust_angle                               [radians]
	
	Not outputted: fills 10 total arrays with parametric curve points for:
	        twists & twists_rad (twist angle & radial distance from hub of each point)
		chords & chords_rad (chord length & radial distance from hub of each point)
		skews & skews_rad (skew angle & radial distance from hub of each point)
		rakes & rakes_rad (rake angle & radial distance from hub of each point)
		sweeps & sweeps_rad (sweep angle & radial distance from hub of each point)
		
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
	
	prop.prop_attributes.number_blades = int(vsp.GetParmVal(prop_id, 'NumBlade', 'Design'))
	tip_radius                         = (vsp.GetParmVal(prop_id, 'Diameter', 'Design')/2.) * units_factor
	prop.prop_attributes.tip_radius    = tip_radius
	prop.prop_attributes.hub_radius    = vsp.GetParmVal(prop_id, 'RadiusFrac', 'XSec_0') * tip_radius	
	
	prop.origin[0]   = vsp.GetParmVal(prop_id, 'X_Rel_Location', 'XForm') * units_factor
	prop.origin[1]   = vsp.GetParmVal(prop_id, 'Y_Rel_Location', 'XForm') * units_factor
	prop.origin[2]   = vsp.GetParmVal(prop_id, 'Z_Rel_Location', 'XForm') * units_factor
	prop.rotation[0] = vsp.GetParmVal(prop_id, 'X_Rel_Rotation', 'XForm') * Units.deg
	prop.rotation[1] = vsp.GetParmVal(prop_id, 'Y_Rel_Rotation', 'XForm') * Units.deg
	prop.rotation[2] = vsp.GetParmVal(prop_id, 'Z_Rel_Rotation', 'XForm') * Units.deg
	
	prop.thrust_angle = prop.rotation[1]		# Y-rotation indicates thrust angle relative to horizontal.
	
	xsecsurf_id = vsp.GetXSecSurf(prop_id, 0)
	
	curve_type = {0:'linear',1:'spline',2:'Bezier_cubic'}
	
	# -------------
	# Blade geometry
	# -------------	
		
	# Chord
	chord_curve       = curve_type[int(vsp.GetParmVal(prop_id, 'CrvType', 'Chord'))]
	chord_split_point = vsp.GetParmVal(prop_id, 'SplitPt', 'Chord')
	chords            = [vsp.GetParmVal(prop_id, 'crd_0', 'Chord') * units_factor]
	chords_rad        = [vsp.GetParmVal(prop_id, 'r_0', 'Chord')]  			# This is r/R value.
	ii                = 1
	while np.round(chords_rad[ii-1], 3)!=1.:					# np.round for safety. Sometimes 1.0 returns .999 repeating.
		chords.append(vsp.GetParmVal(prop_id, 'crd_' + str(ii), 'Chord')) * units_factor
		chords_rad.append(vsp.GetParmVal(prop_id, 'r_' + str(ii), 'Chord'))
		ii += 1
		
	# Twist
	twist_curve       = curve_type[int(vsp.GetParmVal(prop_id, 'CrvType', 'Twist'))]
	twist_split_point = vsp.GetParmVal(prop_id, 'SplitPt', 'Twist')	
	twists            = [vsp.GetParmVal(prop_id, 'tw_0', 'Twist') * Units.deg]
	twists_rad        = [vsp.GetParmVal(prop_id, 'r_0', 'Twist')]			# This is r/R value.
	ii                = 1
	while np.round(twists_rad[ii-1], 3)!=1.:
		twists.append(vsp.GetParmVal(prop_id, 'tw_' + str(ii), 'Twist') * Units.deg) 
		twists_rad.append(vsp.GetParmVal(prop_id, 'r_' + str(ii), 'Twist'))		
		ii += 1
		
	# Skew
	skew_curve       = curve_type[int(vsp.GetParmVal(prop_id, 'CrvType', 'Skew'))]
	skew_split_point = vsp.GetParmVal(prop_id, 'SplitPt', 'Skew')
	skews            = [vsp.GetParmVal(prop_id, 'skw_0', 'Skew') * Units.deg]
	skews_rad        = [vsp.GetParmVal(prop_id, 'r_0', 'Skew')]
	ii		 = 1
	while np.round(skews_rad[ii-1], 3)!=1.:
		skews.append(vsp.GetParmVal(prop_id, 'skw_' + str(ii), 'Skew') * Units.deg)
		skews_rad.append(vsp.GetParmVal(prop_id, 'r_' + str(ii), 'Skew'))	
		ii += 1
	
	# Rake
	rake_curve       = curve_type[int(vsp.GetParmVal(prop_id, 'CrvType', 'Rake'))]
	rake_split_point = vsp.GetParmVal(prop_id, 'SplitPt', 'Rake')
	rakes            = [vsp.GetParmVal(prop_id, 'rak_0', 'Rake') * Units.deg]
	rakes_rad        = [vsp.GetParmVal(prop_id, 'r_0', 'Rake')]	
	ii 		 = 1
	while np.round(rakes_rad[ii-1], 3)!=1.:
		rakes.append(vsp.GetParmVal(prop_id, 'rak_' + str(ii), 'Rake') * Units.deg)
		rakes_rad.append(vsp.GetParmVal(prop_id, 'r_' + str(ii), 'Rake'))
		ii += 1
		
	# Sweep
	sweep_curve       = curve_type[int(vsp.GetParmVal(prop_id, 'CrvType', 'Sweep'))]
	sweep_split_point = vsp.GetParmVal(prop_id, 'SplitPt', 'Sweep')
	sweeps            = [vsp.GetParmVal(prop_id, 'sw_0', 'Sweep') * Units.deg]
	sweeps_rad        = [vsp.GetParmVal(prop_id, 'r_0', 'Sweep')]
	ii                = 1
	while np.round(sweeps_rad[ii-1], 3)!=1.:
		sweeps.append(vsp.GetParmVal(prop_id, 'sw_' + str(ii), 'Sweep') * Units.deg)
		sweeps_rad.append(vsp.GetParmVal(prop_id, 'r_' + str(ii), 'Sweep'))	
		ii += 1

	# -------------
	# All 10 arrays above are now ready for processing, which should be included here. 
	# Note: They are not included in the prop object.
	# -------------	
		
	return prop
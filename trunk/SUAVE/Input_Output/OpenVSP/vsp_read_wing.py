## @ingroup Input_Output-OpenVSP
# vsp_read_wing.py

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
def vsp_read_wing(wing_id, units='SI'): 	
	"""This reads an OpenVSP wing vehicle geometry and writes it into a SUAVE wing format.

	Assumptions:
	1. OpenVSP wing is divided into segments ("XSecs" in VSP).
	2. Written for OpenVSP 3.16.1

	Source:
	N/A

	Inputs:
	0. Pre-loaded VSP vehicle in memory, via vsp_read.
	1. VSP 10-digit geom ID for wing.
	2. Units set to 'SI' (default) or 'Imperial'.

	Outputs:
	Writes SUAVE wing object, with these geometries, from VSP:
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

	Properties Used:
	N/A
	"""  
	if units == 'SI':
		units = Units.meter 
	else:
		units = Units.foot 
	
	wing = SUAVE.Components.Wings.Wing()
	
	if vsp.GetGeomName(wing_id):
		wing.tag = vsp.GetGeomName(wing_id)
	else: 
		wing.tag = 'WingGeom'
	
	wing.origin[0] = vsp.GetParmVal(wing_id, 'X_Rel_Location', 'XForm')
	wing.origin[1] = vsp.GetParmVal(wing_id, 'Y_Rel_Location', 'XForm')
	wing.origin[2] = vsp.GetParmVal(wing_id, 'Z_Rel_Location', 'XForm')	
	
	sym_planar = vsp.GetParmVal(wing_id, 'Sym_Planar_Flag', 'Sym')
	sym_origin = vsp.GetParmVal(wing_id, 'Sym_Ancestor', 'Sym')
	
	if sym_planar == 2. and sym_origin == 2.: #origin at wing, not vehicle
		wing.symmetric == True	
	else:
		wing.symmetric == False
	
	wing.aspect_ratio = vsp.GetParmVal(wing_id, 'TotalAR', 'WingGeom')
	xsec_surf_id      = vsp.GetXSecSurf(wing_id, 0)			# This is how VSP stores surfaces.
	segment_num       = vsp.GetNumXSec(xsec_surf_id)		# Get number of wing segments (is one more than the VSP GUI shows).
	
	total_chord      = vsp.GetParmVal(wing_id, 'Root_Chord', 'XSec_1')	
	total_proj_span  = vsp.GetParmVal(wing_id, 'TotalProjectedSpan', 'WingGeom')  
	span_sum         = 0.				# Non-projected.
	proj_span_sum    = 0.				# Projected.
	segment_spans    = [None] * (segment_num) 	# Non-projected.
	segment_dihedral = [None] * (segment_num)
	segment_sweeps_quarter_chord = [None] * (segment_num)
	
	# Check for wing segment *inside* fuselage, then skip XSec_0 to start at first exposed segment.
	if vsp.GetParmVal(wing_id, 'Root_Chord', 'XSec_0') == 1.:
		start = 1
	else:
		start = 0
	
	# -------------
	# Wing segments
	# -------------		
	
	# Convert VSP XSecs to SUAVE segments. (Wing segments are defined by outboard sections in VSP, but inboard sections in SUAVE.) 
	for i in xrange(start, segment_num+1):		
		segment = SUAVE.Components.Wings.Segment()
		segment.tag                   = 'Section_' + str(i)
		thick_cord                    = vsp.GetParmVal(wing_id, 'ThickChord', 'XSecCurve_' + str(i-1))
		segment.thickness_to_chord    = thick_cord	# Thick_cord stored for use in airfoil, below.		
		segment_root_chord            = vsp.GetParmVal(wing_id, 'Root_Chord', 'XSec_' + str(i)) * units
		segment.root_chord_percent    = segment_root_chord / total_chord		
		segment.percent_span_location = proj_span_sum / (total_proj_span/2)
		segment.twist                 = vsp.GetParmVal(wing_id, 'Twist', 'XSec_' + str(i-1)) * Units.deg
	
		if i < segment_num:      # This excludes the tip xsec, but we need a segment in SUAVE to store airfoil.
			segment_sweeps_quarter_chord[i]   = vsp.GetParmVal(wing_id, 'Sec_Sweep', 'XSec_' + str(i)) * Units.deg
			segment.sweeps.quarter_chord      = -segment_sweeps_quarter_chord[i]  # Used again, below
	
			segment_dihedral[i]	      = vsp.GetParmVal(wing_id, 'Dihedral', 'XSec_' + str(i)) * Units.deg # Used for dihedral computation, below.
			segment.dihedral_outboard     = segment_dihedral[i]
	
			segment_spans[i] 	      = vsp.GetParmVal(wing_id, 'Span', 'XSec_' + str(i))
			proj_span_sum += segment_spans[i] * np.cos(segment_dihedral[i])	
			span_sum      += segment_spans[i]
		else:
			segment.root_chord_percent    = (vsp.GetParmVal(wing_id, 'Tip_Chord', 'XSec_' + str(i-1)))/total_chord
	
		# XSec airfoil
		jj = i-1  # Airfoil index i-1 because VSP airfoils and sections are one index off relative to SUAVE.
		xsec_id = str(vsp.GetXSec(xsec_surf_id, jj))
		airfoil = Airfoil()
		if vsp.GetXSecShape(xsec_id) == 7: 	# XSec shape: NACA 4-series
			camber = vsp.GetParmVal(wing_id, 'Camber', 'XSecCurve_' + str(jj)) 
			
			if camber == 0.:
				camber_loc = 0.
			else:
				camber_loc = vsp.GetParmVal(wing_id, 'CamberLoc', 'XSecCurve_' + str(jj))
			
			airfoil.thickness_to_chord = thick_cord
			camber_round               = int(np.around(camber*100))
			camber_loc_round           = int(np.around(camber_loc*10)) 
			thick_cord_round           = int(np.around(thick_cord*100))
			airfoil.tag                = 'NACA ' + str(camber_round) + str(camber_loc_round) + str(thick_cord_round)
	
		elif vsp.GetXSecShape(xsec_id) == 8: 	# XSec shape: NACA 6-series
			thick_cord_round = int(np.around(thick_cord*100))
			a_value          = vsp.GetParmVal(wing_id, 'A', 'XSecCurve_' + str(jj))
			ideal_CL         = int(np.around(vsp.GetParmVal(wing_id, 'IdealCl', 'XSecCurve_' + str(jj))*10))
			series_vsp       = int(vsp.GetParmVal(wing_id, 'Series', 'XSecCurve_' + str(jj)))
			series_dict      = {0:'63',1:'64',2:'65',3:'66',4:'67',5:'63A',6:'64A',7:'65A'} # VSP series values.
			series           = series_dict[series_vsp]
			airfoil.tag      = 'NACA ' + series + str(ideal_CL) + str(thick_cord_round) + ' a=' + str(np.around(a_value,1))
	
		elif vsp.GetXSecShape(xsec_id) == 12:	# XSec shape: 12 is type AF_FILE
			airfoil.thickness_to_chord = thick_cord
			airfoil.points             = vsp.GetAirfoilCoordinates(wing_id, float(jj/segment_num))
			# VSP airfoil API calls get coordinates and write files with the final argument being the fraction of segment position, regardless of relative spans. 
			# (Write the root airfoil with final arg = 0. Write 4th airfoil of 5 segments with final arg = .8)
			vsp.WriteSeligAirfoil(str(wing.tag) + '_airfoil_XSec_' + str(jj) +'.dat', wing_id, float(jj/segment_num))
			airfoil.coordinate_file    = str(wing.tag) + '_airfoil_XSec_' + str(jj) +'.dat'
			airfoil.tag                = 'AF_file'	
	
		segment.append_airfoil(airfoil)
	
		wing.Segments.append(segment)
	
	# Wing dihedral 
	proj_span_sum_alt = 0.
	span_sum_alt      = 0.
	sweeps_sum        = 0.
	
	for ii in xrange(start, segment_num):
		if segment_dihedral[ii] <= (70. * Units.deg): # Stop at segment with dihedral value over 70deg (wingtips).
			span_sum_alt += segment_spans[ii]
			proj_span_sum_alt += segment_spans[ii] * np.cos(segment_dihedral[ii])  # Use projected span to find total wing dihedral.
			sweeps_sum += segment_spans[ii] * np.tan(segment_sweeps_quarter_chord[ii])
		else:
			break  
	
	wing.dihedral              = np.arccos(proj_span_sum_alt / span_sum_alt) / Units.deg
	wing.sweeps.quarter_chord  = -np.arctan(sweeps_sum / span_sum_alt) / Units.deg  # Minus sign makes it positive sweep.
	
	# Chords
	wing.chords.root              = vsp.GetParmVal(wing_id, 'Tip_Chord', 'XSec_1')
	wing.chords.tip               = vsp.GetParmVal(wing_id, 'Tip_Chord', 'XSec_' + str(segment_num-1))	
	wing.chords.mean_geometric    = vsp.GetParmVal(wing_id, 'TotalArea', 'WingGeom') / vsp.GetParmVal(wing_id, 'TotalChord', 'WingGeom')
	
	# Areas
	wing.areas.reference  = vsp.GetParmVal(wing_id, 'TotalArea', 'WingGeom')
	wetted_areas          = get_vsp_areas(wing.tag)	
	wing.areas.wetted     = wetted_areas[wing.tag]	# Meters
	wing.areas.exposed    = wetted_areas[wing.tag]	# Meters
		
	# Twists
	wing.twists.root      = vsp.GetParmVal(wing_id, 'Twist', 'XSec_0') * Units.deg
	wing.twists.tip       = vsp.GetParmVal(wing_id, 'Twist', 'XSec_' + str(segment_num-1)) * Units.deg
	
	if wing.symmetric == True:
		wing.spans.projected = proj_span_sum*2
	else:
		wing.spans.projected = proj_span_sum	
	
	return wing
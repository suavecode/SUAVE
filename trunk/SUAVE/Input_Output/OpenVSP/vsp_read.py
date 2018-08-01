# Theo St Francis 6/28/18

import SUAVE
from SUAVE.Core import Units, Data
from SUAVE.Input_Output.OpenVSP import get_vsp_areas
from SUAVE.Components.Wings.Airfoils.Airfoil import Airfoil 
from SUAVE.Components.Fuselages.Fuselage import Fuselage
import vsp_g as vsp
import numpy as np



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
	
	# Post-processing
	
	fuselage.heights.at_quarter_length          = 3.32 * Units.meter   
	fuselage.heights.at_wing_root_quarter_chord = 3.32 * Units.meter   
	fuselage.heights.at_three_quarters_length   = 3.32 * Units.meter 	
	
	
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
			airfoil.points = vsp.GetAirfoilCoordinates( wing_id, i/segment_num )
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





def readFuselage( fuselage_id ):
	fuselage = SUAVE.Components.Fuselages.Fuselage()	#Create SUAVE fuselage.
	if vsp.GetGeomName( fuselage_id ):
		fuselage.tag = vsp.GetGeomName( fuselage_id )
	else: 
		fuselage.tag = 'FuselageGeom'	
	
	xsec_surf_id = vsp.GetXSecSurf( fuselage_id, 0 ) 	# There is only one XSecSurf in geom.
	xsec_num = vsp.GetNumXSec( xsec_surf_id ) 		# Number of xsecs in fuselage.
	xsec_ids = []
	xsec_eff_diams = []
	xsec_rel_locations = []
	xsec_heights = []
	xsec_widths = []	
	ref_length = vsp.GetParmVal( fuselage_id, 'RefLength', 'XSec_0')
	fuselage.lengths.total = ref_length
	for ii in xrange( 0, xsec_num ): 			
		xsec_ids.append(vsp.GetXSec( xsec_surf_id, ii ))# Store fuselage xsec IDs.  
		height = vsp.GetXSecHeight(xsec_ids[ii])	# xsec height.
		xsec_heights.append(height)
		width = vsp.GetXSecWidth(xsec_ids[ii])		# xsec width.
		xsec_widths.append(width)
		xsec_eff_diams.append((height+width)/2)		# Effective diameter.
		x_loc = vsp.GetParmVal( fuselage_id, 'XLocPercent', 'XSec_' + str(ii))
		xsec_rel_locations.append(x_loc)
		if ii >= 2 and (x_loc - xsec_rel_locations[ii-1])>= (xsec_rel_locations[ii-1]-xsec_rel_locations[ii-2]) and (xsec_eff_diams[ii]-xsec_eff_diams[ii-1]) < (xsec_eff_diams[ii-1]-xsec_eff_diams[ii-2]):
			end_nose = ii-1	# This if-clause tests for which fuselage segment is longest and assumes the previous
			begin_tail = ii	# section is the end of the nose, and the current section is the beginning of the tail. 
			                # These are used in fineness calculaations, below.
	fuselage.lengths.nose = ref_length*(xsec_rel_locations[end_nose])	# Reference length by relative locations.
	fuselage.lengths.tail = ref_length*(1-xsec_rel_locations[begin_tail])
	fuselage.fineness.nose = fuselage.lengths.nose/xsec_eff_diams[end_nose]		
	fuselage.fineness.tail = fuselage.lengths.tail/xsec_eff_diams[begin_tail]		
	fuselage.heights.maximum = max(xsec_heights)		# Max section height.
	fuselage.width		 = max(xsec_widths)		# Max section width.
	fuselage.effective_diameter = max(xsec_eff_diams)	# Max section effective diam.
	
	vsp.SetSetName(5,'fuselage')
	vsp.SetSetFlag(fuselage_id, 5, True)
	vsp.SetComputationFileName( 3, str(fuselage_id) + '_wetted_area.csv')
	vsp.ComputeCompGeom( 5, True, 3 )
   
	#fuselage.areas.wetted          = 447. * Units['meter**2'] 
	#fuselage.areas.front_projected = 11.9 * Units['meter**2'] 
	
	
	return fuselage

def main():

	return None


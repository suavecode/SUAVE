# Theo St Francis 6/28/18

import SUAVE
from SUAVE.Core import Units, Data
import vsp_g as vsp
import numpy as np

def readWing(PLANE):
	
	geoms = vsp.FindGeoms()
	wing_geom_num = 0	#???how to determine this? in symmetry, its listed as ordered list
	wing_id = str(geoms[wing_geom_num])

	vehicle = SUAVE.Vehicle()
	vehicle.tag = 'BWB2'	
	wing = SUAVE.Components.Wings.Wing()
	'''
	wing.origin[0] = vsp.GetParmVal( wing_id, 'X_Rel_Location', 'XForm')
	wing.origin[1] = vsp.GetParmVal( wing_id, 'Y_Rel_Location', 'XForm')
	wing.origin[2] = vsp.GetParmVal( wing_id, 'Z_Rel_Location', 'XForm')	
	'''
	#SYMMETRY     #???no need for axial
	sym_planar = vsp.GetParmVal( wing_id, 'Sym_Planar_Flag', 'Sym')
	sym_origin = vsp.GetParmVal( wing_id, 'Sym_Ancestor', 'Sym')
	if sym_planar == 2 and sym_origin == wing_geom_num+1: #origin at wing, not vehicle
		wing.symmetric == True	#???assuming wing always symmetric across XZ axis...
	else:
		wing.symmetric == False


	#OTHER WING-WIDE PARMS
	wing.aspect_ratio = vsp.GetParmVal(wing_id, 'TotalAR', 'WingGeom')
	xsec_surf_id = vsp.GetXSecSurf( wing_id, 0) #get number surfaces in geom
	segment_num = vsp.GetNumXSec( xsec_surf_id)   #get number segments in surface, one more than in GUI	

	wing.chords.root             = vsp.GetParmVal( wing_id, 'Tip_Chord', 'XSec_1')
	wing.chords.tip              = vsp.GetParmVal( wing_id, 'Tip_Chord', 'XSec_' + str(segment_num))

	#wing.areas.reference         = 15680. * Units.feet**2 #14.57
	#wing.sweeps.quarter_chord    = 33. * Units.degrees

	#wing.twists.root             = 0.0 * Units.degrees
	#wing.twists.tip              = 0.0 * Units.degrees
	#wing.dihedral


	#WING SEGMENTS	


	#get root chord and span for use below
	total_chord = vsp.GetParmVal( wing_id, 'Root_Chord', 'XSec_1')	
	total_proj_span = vsp.GetParmVal( wing_id, 'TotalProjectedSpan', 'WingGeom')  
	proj_span_sum = 0.
	mean_aero_chords = [segment_num+1]
	segment_spans = [segment_num+1]
	mean_aero_by_span = 0.

	#iterate getting parms through xsecs of wing
	for i in range( 1, segment_num+1):
		segment = SUAVE.Components.Wings.Segment()
		segment.tag                   = 'section_' + str(i)
		segment.twist                 = vsp.GetParmVal( wing_id, 'Twist', 'XSec_' + str(i)) * Units.deg
		segment.sweeps.quarter_chord  = vsp.GetParmVal( wing_id, 'Sec_Sweep', 'XSec_' + str(i)) * Units.deg
		segment.thickness_to_chord    = vsp.GetParmVal( wing_id, 'ThickChord', 'XSecCurve_' + str(i))	

		segment_tip_chord 	      = vsp.GetParmVal( wing_id, 'Tip_Chord', 'XSec_' + str(i))
		segment_root_chord            = vsp.GetParmVal( wing_id, 'Root_Chord', 'XSec' + str(i))
		segment.root_chord_percent    = segment_root_chord / total_chord

		segment_dihedral	      = vsp.GetParmVal( wing_id, 'Dihedral', 'Xsec_' + str(i)) * Units.deg
		segment.dihedral_outboard     = segment_dihedral

		segment.percent_span_location = proj_span_sum / total_proj_span
		segment_spans[i] 	      = vsp.GetParmVal( wing_id, 'Span', 'XSec_' + str(i))
		proj_span_sum += segment_spans[i] * np.cos(segment_dihedral * Units.deg)	

		# compute segment M.A.C.
		mean_aero_chords[i] = segment_root_chord - (2/3)(((segment_root_chord-segment_tip_chord)((segment_root_chord/2)+segment_tip_chord))/(segment_root_chord+segment_tip_chord))

		# to compute wing M.A.C., defined for wing below
		mean_aero_by_span += mean_aero_chords[i] * segment_spans[i]

		wing.Segments.append(segment)



	wing.chords.mean_aerodynamic = mean_aero_by_span / vsp.GetParmVal( wing_id, 'TotalSpan', 'WingGeom')





	#FINISH SYMMETRY/SPAN
	if wing.symmetric == True:
		wing.spans.projected = span*2
	else:
		wing.spans.projected = span	
		
		
		
	return wing


def printGeoms():
		
	i = 0
	
	for parm in parm_ids:
		#print 'parm_id: ', parm_ids[i]
		value = vsp.GetParmVal(parm_ids[i])
		#print 'value:', value
		valuebool = vsp.GetBoolParmVal(parm_ids[i])
		#print valuebool
		name = vsp.GetParmName(parm_ids[i])
		print name, ',', value, ',', valuebool
		i += 1
		
		#print "\n"
	
	print "# of parms in geom #", wing_geom, ": ", len(parm_ids)
	
	geomCount = 0
	for geom in geoms:
		print geomCount, vsp.GetGeomName(geom)
		geomCount += 1
	#for parm in parm_ids:
		#print parm_ids[parm]
	
	print '\ngeoms: \n', geoms	
	
	return None

def main():
	print 'test complete'
	return None

def vsp_read(tag):
	
	
	return vehicle
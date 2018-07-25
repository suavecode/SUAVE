# Theo St Francis 6/28/18

import SUAVE
from SUAVE.Core import Units, Data
from SUAVE.Input_Output.OpenVSP import get_vsp_areas
from SUAVE.Components.Wings.Airfoils.Airfoil import Airfoil 
import vsp_g as vsp
import numpy as np



def vsp_read(tag):
	
	vsp.ClearVSPModel() 
	vsp.ReadVSPFile(PLANE)	
	
	vsp_fuselages = []
	vsp_wings = []	
	vsp_props = []
	
	vsp_geoms = vsp.FindGeoms()
	
	''' The API call for GETGEOMTYPE was not released as of 07/25/18
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
		wing.tag = 'wing'
		vsp.SetGeomName( wing_id, 'wing')
	
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
	if vsp.GetParmVal( wing_id, 'Root_Chord', 'XSec_0')==1.:
		start = 1
	else:
		start = 0
		
	
	# Iterate wing xsecs (SUAVE segments)
	for i in xrange( start, segment_num):
		segment = SUAVE.Components.Wings.Segment()
		segment.tag                   = 'Section_' + str(i)
		segment.twist                 = vsp.GetParmVal( wing_id, 'Twist', 'XSec_' + str(i)) * Units.deg
		segment.sweeps.quarter_chord  = vsp.GetParmVal( wing_id, 'Sec_Sweep', 'XSec_' + str(i)) * Units.deg
		segment.thickness_to_chord    = vsp.GetParmVal( wing_id, 'ThickChord', 'XSecCurve_' + str(i))	

		segment_tip_chord 	      = vsp.GetParmVal( wing_id, 'Tip_Chord', 'XSec_' + str(i))
		segment_root_chord            = vsp.GetParmVal( wing_id, 'Root_Chord', 'XSec_' + str(i))
		segment.root_chord_percent    = segment_root_chord / total_chord

		segment_dihedral[i]	      = vsp.GetParmVal( wing_id, 'Dihedral', 'XSec_' + str(i)) * Units.deg
		segment.dihedral_outboard     = segment_dihedral[i]

		segment.percent_span_location = proj_span_sum / total_proj_span
		segment_spans[i] 	      = vsp.GetParmVal( wing_id, 'Span', 'XSec_' + str(i))
		proj_span_sum += segment_spans[i] * np.cos(segment_dihedral[i])	
		span_sum += segment_spans[i]

		# XSec airfoil			
		xsec_id = str(vsp.GetXSec(xsec_surf_id, i))		
		if vsp.GetXSecShape( xsec_id ) == 7:
			camber = vsp.GetParmVal( wing_id, 'Camber', 'XSecCurve_' + str(i))
		elif vsp.GetXSecShape( xsec_id ) == 12:	# XSec shape: 12 is type AF_FILE
			airfoil = Airfoil()
			lower = vsp.GetAirfoilLowerPnts( xsec_id )  
			upper = vsp.GetAirfoilUpperPnts( xsec_id )
			
			# Lednicer airfoil: TE underneath to LE, over top to TE
			# Airfoil lower points from *TE*
			if lower[0].x() <= lower[-1].x():  # test for airfoil format
				for jj in xrange(len(lower)-1, 0, -1):
					airfoil.points.append( [lower[jj].x(), lower[jj].y()])
			else: 
				for jj in xrange(0, len(lower)-1):
					airfoil.points.append( [lower[jj].x(), lower[jj].y()])
			
			# Airfoil upper points from *LE*
			if upper[0].x() <= upper[-1].x():  # test for airfoil format
				for jj in xrange(0, len(upper)-1):
					airfoil.points.append( [upper[jj].x(), upper[jj].y()])
			else: 
				for jj in xrange(len(upper)-1, 0, -1):
					airfoil.points.append( [upper[jj].x(), upper[jj].y()])			
			segment.append_airfoil(airfoil)
		
		wing.Segments.append(segment)
	
	# Wing dihedral: exclude segments with dihedral values over 70deg
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
	wing.twists.root             = vsp.GetParmVal( wing_id, 'Twist', 'XSec_1') * Units.deg
	wing.twists.tip              = vsp.GetParmVal( wing_id, 'Twist', 'XSec_' + str(segment_num-1)) * Units.deg
	


	#FINISH SYMMETRY/SPAN
	if wing.symmetric == True:
		wing.spans.projected = proj_span_sum*2
	else:
		wing.spans.projected = proj_span_sum	
		
		
		
	return wing





def readFuselage():
	
	geoms = vsp.FindGeomsWithName('')
	print geoms
	fuselage = SUAVE.Components.Fuselages.Fuselage()
	
	#fuselage.width = vsp.GetParmVal( fuselage_id, )
	
	
	
	
	
	
	return fuselage

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

	return None


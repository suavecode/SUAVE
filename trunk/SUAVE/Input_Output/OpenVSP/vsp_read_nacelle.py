## @ingroup Input_Output-OpenVSP
# vsp_read_nacelle.py

# Created:  Sep 2021, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Units, Data
from SUAVE.Components.Nacelles.Nacelle import Nacelle
import vsp as vsp
import numpy as np

# ----------------------------------------------------------------------
#  vsp read nacelle
# ----------------------------------------------------------------------

## @ingroup Input_Output-OpenVSP
def vsp_read_nacelle(nacelle_id, units_type='SI'):
    """This reads an OpenVSP stack geometry or body of revolution and writes it to a SUAVE nacelle format.

    Assumptions: 

    Source:
    N/A

    Inputs:
    0. Pre-loaded VSP vehicle in memory, via vsp_read.
    1. VSP 10-digit geom ID for nacelle.
    2. Units_type set to 'SI' (default) or 'Imperial'. 

    Outputs:
    Writes SUAVE nacelle, with these geometries:           (all defaults are SI, but user may specify Imperial)

        Nacelles.Nacelle.			
            origin                                  [m] in all three dimensions
            width                                   [m]
            lengths. 
            heights. 

    Properties Used:
    N/A
    """  	
    nacelle = SUAVE.Components.Nacelles.Nacelle()	

    if units_type == 'SI':
        units_factor = Units.meter * 1.
    elif units_type == 'imperial':
        units_factor = Units.foot * 1.
    elif units_type == 'inches':
        units_factor = Units.inch * 1.	

    if vsp.GetGeomName(nacelle_id):
        nacelle.tag = vsp.GetGeomName(nacelle_id)
    else: 
        nacelle.tag = 'NacelleGeom'	

    nacelle.origin[0][0] = vsp.GetParmVal(nacelle_id, 'X_Location', 'XForm') * units_factor
    nacelle.origin[0][1] = vsp.GetParmVal(nacelle_id, 'Y_Location', 'XForm') * units_factor
    nacelle.origin[0][2] = vsp.GetParmVal(nacelle_id, 'Z_Location', 'XForm') * units_factor

    nacelle.lengths.total         = vsp.GetParmVal(nacelle_id, 'Length', 'Design') * units_factor
    nacelle.vsp_data.xsec_surf_id = vsp.GetXSecSurf(nacelle_id, 0) 			# There is only one XSecSurf in geom.
    nacelle.vsp_data.xsec_num     = vsp.GetNumXSec(nacelle.vsp_data.xsec_surf_id) 		# Number of xsecs in nacelle.	

    x_locs    = []
    heights   = []
    widths    = []
    eff_diams = []
    lengths   = []

    # -----------------
    # Nacelle segments
    # -----------------
    if stack_geometry:
        pass
    else: 
        for ii in range(0, nacelle.vsp_data.xsec_num):
    
            # Create the segment
            x_sec                     = vsp.GetXSec(nacelle.vsp_data.xsec_surf_id, ii) # VSP XSec ID.
            segment                   = SUAVE.Components.Nacelles.Segment()
            segment.vsp_data.xsec_id  = x_sec 
            segment.tag               = 'segment_' + str(ii)
    
            # Pull out Parms that will be needed
            X_Loc_P = vsp.GetXSecParm(x_sec, 'XLocPercent')
            Z_Loc_P = vsp.GetXSecParm(x_sec, 'ZLocPercent')
    
            segment.percent_x_location = vsp.GetParmVal(X_Loc_P) # Along nacelle length.
            segment.percent_z_location = vsp.GetParmVal(Z_Loc_P ) # Vertical deviation of nacelle center.
            segment.height             = vsp.GetXSecHeight(segment.vsp_data.xsec_id) * units_factor
            segment.width              = vsp.GetXSecWidth(segment.vsp_data.xsec_id) * units_factor
            segment.effective_diameter = (segment.height+segment.width)/2. 
    
            x_locs.append(segment.percent_x_location)	 # Save into arrays for later computation.
            heights.append(segment.height)
            widths.append(segment.width)
            eff_diams.append(segment.effective_diameter)
    
            if ii != (nacelle.vsp_data.xsec_num-1): # Segment length: stored as length since previous segment. (last segment will have length 0.0.)
                next_xsec = vsp.GetXSec(nacelle.vsp_data.xsec_surf_id, ii+1)
                X_Loc_P_p = vsp.GetXSecParm(next_xsec, 'XLocPercent')
                percent_x_loc_p1 = vsp.GetParmVal(X_Loc_P_p) 
                segment.length = nacelle.lengths.total*(percent_x_loc_p1 - segment.percent_x_location) * units_factor
            else:
                segment.length = 0.0
            lengths.append(segment.length)
    
            shape	   = vsp.GetXSecShape(segment.vsp_data.xsec_id)
            shape_dict = {0:'point',1:'circle',2:'ellipse',3:'super ellipse',4:'rounded rectangle',5:'general fuse',6:'fuse file'}
            segment.vsp_data.shape = shape_dict[shape]	
    
            nacelle.Segments.append(segment) 
    
        nacelle.heights.maximum    = max(heights) 		# Max segment height.	
        nacelle.width		    = max(widths) 		# Max segment width.
        nacelle.effective_diameter = max(eff_diams)		# Max segment effective diam.
    
        nacelle.areas.front_projected  = np.pi*((nacelle.effective_diameter)/2)**2
    
        eff_diam_gradients_fwd = np.array(eff_diams[1:]) - np.array(eff_diams[:-1])		# Compute gradients of segment effective diameters.
        eff_diam_gradients_fwd = np.multiply(eff_diam_gradients_fwd, lengths[:-1])
     

    return nacelle

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
def vsp_read_nacelle(nacelle_id,vsp_nacelle_type, units_type='SI'):
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
            origin                  [m] in all three dimensions
            width                   [m]
            lengths                 [m]
            heights                 [m]
            tag                     <string>
            segment[].   (segments are in ordered container and callable by number) 
              percent_x_location    [unitless]
              percent_z_location    [unitless]
              height                [m]
              width                 [m]

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
    nacelle.x_rotation   = vsp.GetParmVal(nacelle_id, 'X_Rotation', 'XForm') * units_factor
    nacelle.y_rotation   = vsp.GetParmVal(nacelle_id, 'Y_Rotation', 'XForm') * units_factor
    nacelle.z_rotation   = vsp.GetParmVal(nacelle_id, 'Z_Rotation', 'XForm') * units_factor  
    
    if vsp_nacelle_type == 'Stack': 
        
        xsec_surf_id = vsp.GetXSecSurf(nacelle_id, 0) 			# There is only one XSecSurf in geom.
        num_segs     = vsp.GetNumXSec(xsec_surf_id)   # Number of xsecs in nacelle.	
        abs_x_location = 0 
        abs_y_location = 0
        abs_z_location = 0
        abs_x_location_vec = []
        abs_y_location_vec = []
        abs_z_location_vec = []
        
        for i in range(num_segs): 
            # Create the segment
            xsec_id      = vsp.GetXSec(xsec_surf_id, i) # VSP XSec ID.
            segment      = SUAVE.Components.Nacelles.Segment() 
            segment.tag  = 'segment_' + str(i)
    
            # Pull out Parms that will be needed
            X_Loc_P = vsp.GetXSecParm(xsec_id, 'XDelta')
            Y_Loc_P = vsp.GetXSecParm(xsec_id, 'YDelta')
            Z_Loc_P = vsp.GetXSecParm(xsec_id, 'XDelta') 
            
            del_x = vsp.GetParmVal(X_Loc_P)
            del_y = vsp.GetParmVal(Y_Loc_P)
            del_z = vsp.GetParmVal(Z_Loc_P)
            
            abs_x_location = abs_x_location + del_x
            abs_y_location = abs_y_location + del_y
            abs_z_location = abs_z_location + del_z
            
            abs_x_location_vec.append(abs_x_location)
            abs_y_location_vec.append(abs_y_location)
            abs_z_location_vec.append(abs_z_location) 
  
            shape      = vsp.GetXSecShape(xsec_id)
            shape_dict = {0:'point',1:'circle',2:'ellipse',3:'super ellipse',4:'rounded rectangle',5:'general fuse',6:'fuse file'} 
                         
            if shape_dict[shape] == 'point':
                segment.height = 0.0
                segment.width  = 0.0
                if i == 0:
                    nacelle.flow_through = False 
            else:
                segment.height = vsp.GetXSecHeight(xsec_id) * units_factor
                segment.width  = vsp.GetXSecWidth(xsec_id) * units_factor  
                if i == 0:
                    nacelle.flow_through = True 
                
            nacelle.Segments.append(segment)
            
        nacelle.length = abs_x_location_vec[-1]  
        segs = nacelle.Segments
        for seg in range(num_segs):    
            segs[seg].percent_x_location = np.array(abs_x_location_vec)/abs_x_location_vec[-1]
            segs[seg].percent_y_location = np.array(abs_y_location_vec)/abs_x_location_vec[-1]
            segs[seg].percent_z_location = np.array(abs_z_location_vec)/abs_x_location_vec[-1] 
          
 
    elif vsp_nacelle_type =='BodyOfRevolution':  
        diameter  = vsp.GetParmVal(nacelle_id, "Diameter","Design") * units_factor
        angle     = vsp.GetParmVal(nacelle_id, "Diameter","Design") * Units.degrees 
        ft_flag_idx   = vsp.GetParmVal(nacelle_id,"Mode","Design")	 
        if ft_flag_idx == 0.0:
            ft_flag = True 
        else:
            ft_flag = False         
        nacelle.flow_through = ft_flag 

        shape      = vsp.GetBORXSecShape(nacelle_id)
        shape_dict = {0:'point',1:'circle',2:'ellipse',3:'super ellipse',4:'rounded rectangle',5:'general fuse',6:'fuse file',\
                      7:'four series',8:'six series',9:'biconvex'}  
        if shape_dict[shape] == 'four series':     
            length     = vsp.GetParmVal(nacelle_id, "Chord", "XSecCurve")
            thickness  = int(round(vsp.GetParmVal(nacelle_id, "ThickChord", "XSecCurve")*10,0))
            camber     = int(round(vsp.GetParmVal(nacelle_id, "Camber", "XSecCurve")*100,0))
            camber_loc = int(round( vsp.GetParmVal(nacelle_id, "CamberLoc", "XSecCurve" )*10,0)) 
            
            airfoil = str(camber) +  str(camber_loc) +  str(thickness)
            nacelle.naca_4_series_airfoil  = str(airfoil) 
            height  =  thickness  
            
        elif shape_dict[shape] == 'super ellipse':  
            if ft_flag:
                height   = vsp.GetParmVal(nacelle_id, "Super_Height", "XSecCurve") 
                diamater = vsp.GetParmVal(nacelle_id,"Diameter","Design")
                length   = vsp.GetParmVal(nacelle_id, "Super_Width", "XSecCurve")  
            else:
                diamater = vsp.GetParmVal(nacelle_id, "Super_Height", "XSecCurve") 
                length   = vsp.GetParmVal(nacelle_id, "Super_Width", "XSecCurve")  
                height   = diamater/2
            
        nacelle.length                = length  
        nacelle.diameter              = diameter + height/2    
        nacelle.inlet_diameter        = nacelle.diameter - height 
        nacelle.cowling_airfoil_angle = angle   
        
    return nacelle

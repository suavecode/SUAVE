## @ingroup Input_Output-OpenVSP
# vsp_nacelle.py

# Created:  Sep 2021, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Units, Data 
import numpy as np
try:
    import vsp as vsp
except ImportError:
    # This allows SUAVE to build without OpenVSP
    pass 
# ----------------------------------------------------------------------
#  vsp_nacelle
# ----------------------------------------------------------------------
## @ingroup Input_Output-OpenVSP
def write_vsp_nacelle(nacelle, OML_set_ind):
    """This converts nacelles into OpenVSP format.
    
    Assumptions: 
    1. If nacelle segments are defined, geometry written to OpenVSP is of type "StackGeom". 
       1.a  This type of nacelle can be either set as flow through or not flow through.
       1.b  Segments are defined in a similar manner to fuselage segments. See geometric 
            documentation in SUAVE-Components-Nacelles-Nacelle
    
    2. If nacelle segments are not defined, geometry written to OpenVSP is of type "BodyofRevolution".
       2.a This type of nacelle can be either set as flow through or not flow through.
       2.b BodyofRevolution can be either be a 4 digit airfoil (type string) or super ellipse (default)
    Source:
    N/A
    Inputs: 
      nacelle.
      origin                              [m] in all three dimension, should have as many origins as engines 
      length                              [m]
      diameter                            [m]
      flow_through                        <boolean> if True create a flow through nacelle, if False create a cylinder
      segment(optional).
         width                            [m]
         height                           [m]
         lenght                           [m]
         percent_x_location               [m]     
         percent_y_location               [m]        
         percent_z_location               [m] 
       
    Outputs:
    Operates on the active OpenVSP model, no direct output
    Properties Used:
    N/A
    """    
    # default tesselation 
    radial_tesselation = 21
    axial_tesselation  = 25
    
    # True will create a flow-through subsonic nacelle (which may have dimensional errors)
    # False will create a cylindrical stack (essentially a cylinder)       
    ft_flag        = nacelle.flow_through            
    length         = nacelle.length  
    height         = nacelle.diameter - nacelle.inlet_diameter  
    diameter       = nacelle.diameter  - height/2 
    nac_tag        = nacelle.tag 
    nac_x          = nacelle.origin[0][0]
    nac_y          = nacelle.origin[0][1]
    nac_z          = nacelle.origin[0][2]
    nac_x_rotation = nacelle.orientation_euler_angles[0]/Units.degrees    
    nac_y_rotation = nacelle.orientation_euler_angles[1]/Units.degrees    
    nac_z_rotation = nacelle.orientation_euler_angles[2]/Units.degrees      
    num_segs       = len(nacelle.Segments)
    
    if num_segs > 0: 
        if nacelle.Airfoil.naca_4_series_airfoil != None:
            raise AssertionError('Nacelle segments defined. Airfoil section will not be used.')
        nac_id = vsp.AddGeom( "STACK")
        vsp.SetGeomName(nac_id,nac_tag)  
        
        # set nacelle relative location and rotation
        vsp.SetParmVal( nac_id,'Abs_Or_Relitive_flag','XForm',vsp.ABS)
        vsp.SetParmVal( nac_id,'X_Rotation','XForm',nac_x_rotation)
        vsp.SetParmVal( nac_id,'Y_Rotation','XForm',nac_y_rotation)
        vsp.SetParmVal( nac_id,'Z_Rotation','XForm',nac_z_rotation) 
        vsp.SetParmVal( nac_id,'X_Location','XForm',nac_x)
        vsp.SetParmVal( nac_id,'Y_Location','XForm',nac_y)
        vsp.SetParmVal( nac_id,'Z_Location','XForm',nac_z)     
        vsp.SetParmVal( nac_id,'Tess_U','Shape',radial_tesselation)
        vsp.SetParmVal( nac_id,'Tess_W','Shape',axial_tesselation)
        
        widths  = []
        heights = []
        x_delta = []
        x_poses = []
        z_delta = []
        
        segs = nacelle.Segments
        for seg in range(num_segs):   
            widths.append(segs[seg].width)
            heights.append(segs[seg].height) 
            x_poses.append(segs[seg].percent_x_location)
            if seg == 0: 
                x_delta.append(0)
                z_delta.append(0) 
            else:
                x_delta.append(length*(segs[seg].percent_x_location - segs[seg-1].percent_x_location))
                z_delta.append(length*(segs[seg].percent_z_location - segs[seg-1].percent_z_location))  
               
        vsp.CutXSec(nac_id,4) # remove point section at end  
        vsp.CutXSec(nac_id,0) # remove point section at beginning 
        vsp.CutXSec(nac_id,1) # remove point section at beginning 
        for _ in range(num_segs-2): # add back the required number of sections
            vsp.InsertXSec(nac_id, 1, vsp.XS_ELLIPSE)          
            vsp.Update() 
        xsec_surf = vsp.GetXSecSurf(nac_id, 0 )  
        for i3 in reversed(range(num_segs)): 
            xsec = vsp.GetXSec( xsec_surf, i3 ) 
            if i3 == 0:
                pass
            else:
                vsp.SetParmVal(nac_id, "XDelta", "XSec_"+str(i3),x_delta[i3])
                vsp.SetParmVal(nac_id, "ZDelta", "XSec_"+str(i3),z_delta[i3])  
            vsp.SetXSecWidthHeight( xsec, widths[i3], heights[i3])
            vsp.SetXSecTanAngles(xsec,vsp.XSEC_BOTH_SIDES,0,0,0,0)
            vsp.SetXSecTanSlews(xsec,vsp.XSEC_BOTH_SIDES,0,0,0,0)
            vsp.SetXSecTanStrengths( xsec, vsp.XSEC_BOTH_SIDES,0,0,0,0)     
            vsp.Update()          

        if ft_flag: 
            pass
        else:   
            # append front point  
            xsecsurf = vsp.GetXSecSurf(nac_id,0)
            vsp.ChangeXSecShape(xsecsurf,0,vsp.XS_POINT)
            vsp.Update()          
            xsecsurf = vsp.GetXSecSurf(nac_id,0)
            vsp.ChangeXSecShape(xsecsurf,num_segs-1,vsp.XS_POINT)
            vsp.Update()      
            
    else: 
        nac_id = vsp.AddGeom( "BODYOFREVOLUTION")  
        vsp.SetGeomName(nac_id, nac_tag)

        # Origin 
        vsp.SetParmVal( nac_id,'Abs_Or_Relitive_flag','XForm',vsp.ABS)
        vsp.SetParmVal( nac_id,'X_Rotation','XForm',nac_x_rotation)
        vsp.SetParmVal( nac_id,'Y_Rotation','XForm',nac_y_rotation)
        vsp.SetParmVal( nac_id,'Z_Rotation','XForm',nac_z_rotation) 
        vsp.SetParmVal( nac_id,'X_Location','XForm',nac_x)
        vsp.SetParmVal( nac_id,'Y_Location','XForm',nac_y)
        vsp.SetParmVal( nac_id,'Z_Location','XForm',nac_z)  
        vsp.SetParmVal( nac_id,'Tess_U','Shape',radial_tesselation)
        vsp.SetParmVal( nac_id,'Tess_W','Shape',axial_tesselation)      

        # Length and overall diameter
        vsp.SetParmVal(nac_id,"Diameter","Design",diameter)
        if ft_flag:
            vsp.SetParmVal(nac_id,"Mode","Design",0.0)
        else:
            vsp.SetParmVal(nac_id,"Mode","Design",1.0) 
         
        if nacelle.Airfoil.naca_4_series_airfoil != None:
            if isinstance(nacelle.Airfoil.naca_4_series_airfoil, str) and len(nacelle.Airfoil.naca_4_series_airfoil) != 4:
                raise AssertionError('Nacelle cowling airfoil must be of type < string > and length < 4 >')
            else: 
                angle        = nacelle.cowling_airfoil_angle/Units.degrees 
                camber       = float(nacelle.Airfoil.naca_4_series_airfoil[0])/100
                camber_loc   = float(nacelle.Airfoil.naca_4_series_airfoil[1])/10
                thickness    = float(nacelle.Airfoil.naca_4_series_airfoil[2:])/100
                
                vsp.ChangeBORXSecShape(nac_id ,vsp.XS_FOUR_SERIES)
                vsp.Update()
                vsp.SetParmVal(nac_id,"Diameter","Design",diameter)
                vsp.SetParmVal(nac_id,"Angle","Design",angle)
                vsp.SetParmVal(nac_id, "Chord", "XSecCurve", length)
                vsp.SetParmVal(nac_id, "ThickChord", "XSecCurve", thickness)
                vsp.SetParmVal(nac_id, "Camber", "XSecCurve", camber )
                vsp.SetParmVal(nac_id, "CamberLoc", "XSecCurve",camber_loc)  
                vsp.Update()
        else:
            vsp.ChangeBORXSecShape(nac_id ,vsp.XS_SUPER_ELLIPSE)
            vsp.Update()
            if ft_flag:
                vsp.SetParmVal(nac_id, "Super_Height", "XSecCurve", height) 
                vsp.SetParmVal(nac_id,"Diameter","Design",diameter)
            else:
                vsp.SetParmVal(nac_id, "Super_Height", "XSecCurve", diameter) 
            vsp.SetParmVal(nac_id, "Super_Width", "XSecCurve", length)
            vsp.SetParmVal(nac_id, "Super_MaxWidthLoc", "XSecCurve", 0.)
            vsp.SetParmVal(nac_id, "Super_M", "XSecCurve", 2.)
            vsp.SetParmVal(nac_id, "Super_N", "XSecCurve", 1.)  

    vsp.SetSetFlag(nac_id, OML_set_ind, True)

    vsp.Update()  
    return 
    

## @ingroup Input_Output-OpenVSP
def read_vsp_nacelle(nacelle_id,vsp_nacelle_type, units_type='SI'):
    """This reads an OpenVSP stack geometry or body of revolution and writes it to a SUAVE nacelle format.
    If an airfoil is defined in body-of-revolution, its coordinates are not read in due to absence of
    API functions in VSP.

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
        
        xsec_surf_id = vsp.GetXSecSurf(nacelle_id, 0) # There is only one XSecSurf in geom.
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
            segment      = SUAVE.Components.Lofted_Body_Segment.Segment() 
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
                      7:'four series',8:'six series',9:'biconvex',10:'wedge',11:'editcurve',12:'file airfoil'}  
        if shape_dict[shape] == 'four series': 
            naf        = SUAVE.Components.Airfoils.Airfoil()
            length     = vsp.GetParmVal(nacelle_id, "Chord", "XSecCurve") * units_factor 
            thickness  = int(round(vsp.GetParmVal(nacelle_id, "ThickChord", "XSecCurve")*10,0))
            camber     = int(round(vsp.GetParmVal(nacelle_id, "Camber", "XSecCurve")*100,0))
            camber_loc = int(round( vsp.GetParmVal(nacelle_id, "CamberLoc", "XSecCurve" )*10,0)) 
            
            airfoil = str(camber) +  str(camber_loc) +  str(thickness)
            height  =  thickness  
            naf.naca_4_series_airfoil  = str(airfoil)  
            naf.thickness_to_chord     = thickness 
            nacelle.append_airfoil(naf)
            
        elif shape_dict[shape] == 'super ellipse':  
            if ft_flag:
                height   = vsp.GetParmVal(nacelle_id, "Super_Height", "XSecCurve") * units_factor 
                diameter = vsp.GetParmVal(nacelle_id, "Diameter","Design")         * units_factor 
                length   = vsp.GetParmVal(nacelle_id, "Super_Width", "XSecCurve")  * units_factor 
            else:
                diameter = vsp.GetParmVal(nacelle_id, "Super_Height", "XSecCurve") * units_factor 
                length   = vsp.GetParmVal(nacelle_id, "Super_Width", "XSecCurve")  * units_factor 
                height   = diameter/2
        
        elif shape_dict[shape] == 'file airfoil': 
            naf                = SUAVE.Components.Airfoils.Airfoil()
            thickness_to_chord = vsp.GetParmVal(nacelle_id, "ThickChord", "XSecCurve")   * units_factor
            length             = vsp.GetParmVal(nacelle_id, "Chord", "XSecCurve")   * units_factor 
            height             = thickness_to_chord*length  * units_factor            
            if ft_flag: 
                diameter= vsp.GetParmVal(nacelle_id,  "Diameter","Design") * units_factor
            else: 
                diameter= 0   
            naf.thickness_to_chord     = thickness_to_chord 
            nacelle.append_airfoil(naf)
            
        nacelle.length                = length  
        nacelle.diameter              = diameter + height/2    
        nacelle.inlet_diameter        = nacelle.diameter - height 
        nacelle.cowling_airfoil_angle = angle   
        
    return nacelle

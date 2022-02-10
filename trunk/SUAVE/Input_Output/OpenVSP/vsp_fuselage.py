## @ingroup Input_Output-OpenVSP
# vsp_fuselage.py

# Created:  Jun 2018, T. St Francis
# Modified: Aug 2018, T. St Francis
#           Jan 2020, T. MacDonald
#           Jul 2020, E. Botero

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
#  vsp read fuselage
# ----------------------------------------------------------------------

## @ingroup Input_Output-OpenVSP
def read_vsp_fuselage(fuselage_id,fux_idx,sym_flag, units_type='SI', fineness=True):
    """This reads an OpenVSP fuselage geometry and writes it to a SUAVE fuselage format.

    Assumptions:
    1. OpenVSP fuselage is "conventionally shaped" (generally narrow at nose and tail, wider in center). 
    2. Fuselage is designed in VSP as it appears in real life. That is, the VSP model does not rely on
       superficial elements such as canopies, stacks, or additional fuselages to cover up internal lofting oddities.
    3. This program will NOT account for multiple geometries comprising the fuselage. For example: a wingbox mounted beneath
       is a separate geometry and will NOT be processed.
    4. Fuselage origin is located at nose. VSP file origin can be located anywhere, preferably at the forward tip
       of the vehicle or in front (to make all X-coordinates of vehicle positive).
    5. Written for OpenVSP 3.21.1

    Source:
    N/A

    Inputs:
    0. Pre-loaded VSP vehicle in memory, via vsp_read.
    1. VSP 10-digit geom ID for fuselage.
    2. Units_type set to 'SI' (default) or 'Imperial'.
    3. Boolean for whether or not to compute fuselage finenesses (default = True).

    Outputs:
    Writes SUAVE fuselage, with these geometries:           (all defaults are SI, but user may specify Imperial)

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

    Properties Used:
    N/A
    """  	
    fuselage = SUAVE.Components.Fuselages.Fuselage()	

    if units_type == 'SI':
        units_factor = Units.meter * 1.
    elif units_type == 'imperial':
        units_factor = Units.foot * 1.
    elif units_type == 'inches':
        units_factor = Units.inch * 1.	 
     
    if vsp.GetGeomName(fuselage_id):
        fuselage.tag = vsp.GetGeomName(fuselage_id) + '_' + str(fux_idx+1)
    else: 
        fuselage.tag = 'FuselageGeom' + '_' + str(fux_idx+1)	
    
    scaling           = vsp.GetParmVal(fuselage_id, 'Scale', 'XForm')  
    units_factor      = units_factor*scaling

    fuselage.origin[0][0] = vsp.GetParmVal(fuselage_id, 'X_Location', 'XForm') * units_factor
    fuselage.origin[0][1] = vsp.GetParmVal(fuselage_id, 'Y_Location', 'XForm') * units_factor*sym_flag
    fuselage.origin[0][2] = vsp.GetParmVal(fuselage_id, 'Z_Location', 'XForm') * units_factor

    fuselage.lengths.total         = vsp.GetParmVal(fuselage_id, 'Length', 'Design') * units_factor
    fuselage.vsp_data.xsec_surf_id = vsp.GetXSecSurf(fuselage_id, 0) 			        # There is only one XSecSurf in geom.
    fuselage.vsp_data.xsec_num     = vsp.GetNumXSec(fuselage.vsp_data.xsec_surf_id) 		# Number of xsecs in fuselage.	 

        
    x_locs    = []
    heights   = []
    widths    = []
    eff_diams = []
    lengths   = []

    # -----------------
    # Fuselage segments
    # -----------------

    for ii in range(0, fuselage.vsp_data.xsec_num): 
        # Create the segment
        x_sec                     = vsp.GetXSec(fuselage.vsp_data.xsec_surf_id, ii) # VSP XSec ID.
        segment                   = SUAVE.Components.Lofted_Body_Segment.Segment()
        segment.vsp_data.xsec_id  = x_sec 
        segment.tag               = 'segment_' + str(ii)

        # Pull out Parms that will be needed
        X_Loc_P = vsp.GetXSecParm(x_sec, 'XLocPercent')
        Z_Loc_P = vsp.GetXSecParm(x_sec, 'ZLocPercent')

        segment.percent_x_location = vsp.GetParmVal(X_Loc_P) # Along fuselage length.
        segment.percent_z_location = vsp.GetParmVal(Z_Loc_P ) # Vertical deviation of fuselage center.
        segment.height             = vsp.GetXSecHeight(segment.vsp_data.xsec_id) * units_factor
        segment.width              = vsp.GetXSecWidth(segment.vsp_data.xsec_id) * units_factor
        segment.effective_diameter = (segment.height+segment.width)/2. 

        x_locs.append(segment.percent_x_location)	 # Save into arrays for later computation.
        heights.append(segment.height)
        widths.append(segment.width)
        eff_diams.append(segment.effective_diameter)

        if ii != (fuselage.vsp_data.xsec_num-1): # Segment length: stored as length since previous segment. (last segment will have length 0.0.)
            next_xsec = vsp.GetXSec(fuselage.vsp_data.xsec_surf_id, ii+1)
            X_Loc_P_p = vsp.GetXSecParm(next_xsec, 'XLocPercent')
            percent_x_loc_p1 = vsp.GetParmVal(X_Loc_P_p) 
            segment.length = fuselage.lengths.total*(percent_x_loc_p1 - segment.percent_x_location) * units_factor
        else:
            segment.length = 0.0
        lengths.append(segment.length)

        shape	   = vsp.GetXSecShape(segment.vsp_data.xsec_id)
        shape_dict = {0:'point',1:'circle',2:'ellipse',3:'super ellipse',4:'rounded rectangle',5:'general fuse',6:'fuse file'}
        segment.vsp_data.shape = shape_dict[shape]	

        fuselage.Segments.append(segment)

    fuselage.heights.at_quarter_length          = get_fuselage_height(fuselage, .25)  # Calls get_fuselage_height function (below).
    fuselage.heights.at_three_quarters_length   = get_fuselage_height(fuselage, .75) 
    fuselage.heights.at_wing_root_quarter_chord = get_fuselage_height(fuselage, .4) 

    fuselage.heights.maximum    = max(heights)          # Max segment height.	
    fuselage.width              = max(widths)           # Max segment width.
    fuselage.effective_diameter = max(eff_diams)        # Max segment effective diam.

    fuselage.areas.front_projected  = np.pi*((fuselage.effective_diameter)/2)**2

    eff_diam_gradients_fwd = np.array(eff_diams[1:]) - np.array(eff_diams[:-1])		# Compute gradients of segment effective diameters.
    eff_diam_gradients_fwd = np.multiply(eff_diam_gradients_fwd, lengths[:-1])

    fuselage = compute_fuselage_fineness(fuselage, x_locs, eff_diams, eff_diam_gradients_fwd)	

    return fuselage


## @ingroup Input_Output-OpenVSP
def write_vsp_fuselage(fuselage,area_tags, main_wing, fuel_tank_set_ind, OML_set_ind):
    """This writes a fuselage into OpenVSP format.

    Assumptions:
    None

    Source:
    N/A

    Inputs:
    fuselage
      width                                   [m]
      lengths.total                           [m]
      heights.
        maximum                               [m]
        at_quarter_length                     [m]
        at_wing_root_quarter_chord            [m]
        at_three_quarters_length              [m]
      effective_diameter                      [m]
      fineness.nose                           [-] ratio of nose section length to fuselage width
      fineness.tail                           [-] ratio of tail section length to fuselage width
      tag                                     <string>
      OpenVSP_values.  (optional)
        nose.top.angle                        [degrees]
        nose.top.strength                     [-] this determines how much the specified angle influences that shape
        nose.side.angle                       [degrees]
        nose.side.strength                    [-]
        nose.TB_Sym                           <boolean> determines if top angle is mirrored on bottom
        nose.z_pos                            [-] z position of the nose as a percentage of fuselage length (.1 is 10%)
        tail.top.angle                        [degrees]
        tail.top.strength                     [-]
        tail.z_pos (optional, 0.02 default)   [-] z position of the tail as a percentage of fuselage length (.1 is 10%)
      Segments. (optional)
        width                                 [m]
        height                                [m]
        percent_x_location                    [-] .1 is 10% length
        percent_z_location                    [-] .1 is 10% length
    area_tags                                 <dict> used to keep track of all tags needed in wetted area computation           
    main_wing.origin                          [m]
    main_wing.chords.root                     [m]
    fuel_tank_set_index                       <int> OpenVSP object set containing the fuel tanks    

    Outputs:
    Operates on the active OpenVSP model, no direct output

    Properties Used:
    N/A
    """     

    num_segs           = len(fuselage.Segments)
    length             = fuselage.lengths.total
    fuse_x             = fuselage.origin[0][0]    
    fuse_y             = fuselage.origin[0][1]
    fuse_z             = fuselage.origin[0][2]
    fuse_x_rotation    = fuselage.x_rotation   
    fuse_y_rotation    = fuselage.y_rotation
    fuse_z_rotation    = fuselage.z_rotation    
    if num_segs==0: # SUAVE default fuselage shaping

        width    = fuselage.width
        hmax     = fuselage.heights.maximum
        height1  = fuselage.heights.at_quarter_length
        height2  = fuselage.heights.at_wing_root_quarter_chord 
        height3  = fuselage.heights.at_three_quarters_length
        effdia   = fuselage.effective_diameter
        n_fine   = fuselage.fineness.nose 
        t_fine   = fuselage.fineness.tail  

        try:
            if main_wing != None:                
                w_origin = main_wing.origin
                w_c_4    = main_wing.chords.root/4.
            else:
                w_origin = 0.5*length
                w_c_4    = 0.5*length
        except AttributeError:
            raise AttributeError('Main wing not detected. Fuselage must have specified sections in this configuration.')

        # Figure out the location x location of each section, 3 sections, end of nose, wing origin, and start of tail

        x1 = n_fine*width/length
        x2 = (w_origin[0][0]+w_c_4)/length
        x3 = 1-t_fine*width/length

        end_ind = 4

    else: # Fuselage shaping based on sections
        widths  = []
        heights = []
        x_poses = []
        z_poses = []
        segs = fuselage.Segments
        for seg in segs:
            widths.append(seg.width)
            heights.append(seg.height)
            x_poses.append(seg.percent_x_location)
            z_poses.append(seg.percent_z_location)

        end_ind = num_segs-1

    fuse_id = vsp.AddGeom("FUSELAGE") 
    vsp.SetGeomName(fuse_id, fuselage.tag)
    area_tags[fuselage.tag] = ['fuselages',fuselage.tag]

    tail_z_pos = 0.02 # default value

    # set fuselage relative location and rotation
    vsp.SetParmVal( fuse_id,'X_Rel_Rotation','XForm',fuse_x_rotation)
    vsp.SetParmVal( fuse_id,'Y_Rel_Rotation','XForm',fuse_y_rotation)
    vsp.SetParmVal( fuse_id,'Z_Rel_Rotation','XForm',fuse_z_rotation)

    vsp.SetParmVal( fuse_id,'X_Rel_Location','XForm',fuse_x)
    vsp.SetParmVal( fuse_id,'Y_Rel_Location','XForm',fuse_y)
    vsp.SetParmVal( fuse_id,'Z_Rel_Location','XForm',fuse_z)


    if 'OpenVSP_values' in fuselage:        
        vals = fuselage.OpenVSP_values

        # for wave drag testing
        fuselage.OpenVSP_ID = fuse_id

        # Nose
        vsp.SetParmVal(fuse_id,"TopLAngle","XSec_0",vals.nose.top.angle)
        vsp.SetParmVal(fuse_id,"TopLStrength","XSec_0",vals.nose.top.strength)
        vsp.SetParmVal(fuse_id,"RightLAngle","XSec_0",vals.nose.side.angle)
        vsp.SetParmVal(fuse_id,"RightLStrength","XSec_0",vals.nose.side.strength)
        vsp.SetParmVal(fuse_id,"TBSym","XSec_0",vals.nose.TB_Sym)
        vsp.SetParmVal(fuse_id,"ZLocPercent","XSec_0",vals.nose.z_pos)
        if not vals.nose.TB_Sym:
            vsp.SetParmVal(fuse_id,"BottomLAngle","XSec_0",vals.nose.bottom.angle)
            vsp.SetParmVal(fuse_id,"BottomLStrength","XSec_0",vals.nose.bottom.strength)           

        # Tail
        # Below can be enabled if AllSym (below) is removed
        #vsp.SetParmVal(fuse_id,"RightLAngle","XSec_4",vals.tail.side.angle)
        #vsp.SetParmVal(fuse_id,"RightLStrength","XSec_4",vals.tail.side.strength)
        #vsp.SetParmVal(fuse_id,"TBSym","XSec_4",vals.tail.TB_Sym)
        #vsp.SetParmVal(fuse_id,"BottomLAngle","XSec_4",vals.tail.bottom.angle)
        #vsp.SetParmVal(fuse_id,"BottomLStrength","XSec_4",vals.tail.bottom.strength)
        if 'z_pos' in vals.tail:
            tail_z_pos = vals.tail.z_pos
        else:
            pass # use above default


    if num_segs == 0:
        vsp.SetParmVal(fuse_id,"Length","Design",length)
        vsp.SetParmVal(fuse_id,"Diameter","Design",width)
        vsp.SetParmVal(fuse_id,"XLocPercent","XSec_1",x1)
        vsp.SetParmVal(fuse_id,"XLocPercent","XSec_2",x2)
        vsp.SetParmVal(fuse_id,"XLocPercent","XSec_3",x3)
        vsp.SetParmVal(fuse_id,"ZLocPercent","XSec_4",tail_z_pos)
        vsp.SetParmVal(fuse_id, "Ellipse_Width", "XSecCurve_1", width)
        vsp.SetParmVal(fuse_id, "Ellipse_Width", "XSecCurve_2", width)
        vsp.SetParmVal(fuse_id, "Ellipse_Width", "XSecCurve_3", width)
        vsp.SetParmVal(fuse_id, "Ellipse_Height", "XSecCurve_1", height1);
        vsp.SetParmVal(fuse_id, "Ellipse_Height", "XSecCurve_2", height2);
        vsp.SetParmVal(fuse_id, "Ellipse_Height", "XSecCurve_3", height3);  
    else:
        # OpenVSP vals do not exist:
        vals                   = Data()
        vals.nose              = Data()
        vals.tail              = Data()
        vals.tail.top          = Data()

        vals.nose.z_pos        = 0.0
        vals.tail.top.angle    = 0.0
        vals.tail.top.strength = 0.0

        if len(np.unique(x_poses)) != len(x_poses):
            raise ValueError('Duplicate fuselage section positions detected.')
        vsp.SetParmVal(fuse_id,"Length","Design",length)
        if num_segs != 5: # reduce to only nose and tail
            vsp.CutXSec(fuse_id,1) # remove extra default section
            vsp.CutXSec(fuse_id,1) # remove extra default section
            vsp.CutXSec(fuse_id,1) # remove extra default section
            for i in range(num_segs-2): # add back the required number of sections
                vsp.InsertXSec(fuse_id, 0, vsp.XS_ELLIPSE)           
                vsp.Update()
        for i in range(num_segs-2):
            # Bunch sections to allow proper length settings in the next step
            # This is necessary because OpenVSP will not move a section past an adjacent section
            vsp.SetParmVal(fuse_id, "XLocPercent", "XSec_"+str(i+1),1e-6*(i+1))
            vsp.Update()
        if x_poses[1] < (num_segs-2)*1e-6:
            print('Warning: Second fuselage section is too close to the nose. OpenVSP model may not be accurate.')
        for i in reversed(range(num_segs-2)):
            # order is reversed because sections are initially bunched in the front and cannot be extended passed the next
            vsp.SetParmVal(fuse_id, "XLocPercent", "XSec_"+str(i+1),x_poses[i+1])
            vsp.SetParmVal(fuse_id, "ZLocPercent", "XSec_"+str(i+1),z_poses[i+1])
            vsp.SetParmVal(fuse_id, "Ellipse_Width", "XSecCurve_"+str(i+1), widths[i+1])
            vsp.SetParmVal(fuse_id, "Ellipse_Height", "XSecCurve_"+str(i+1), heights[i+1])   
            vsp.Update()             
            set_section_angles(i, vals.nose.z_pos, tail_z_pos, x_poses, z_poses, heights, widths,length,end_ind,fuse_id)            

        vsp.SetParmVal(fuse_id, "XLocPercent", "XSec_"+str(0),x_poses[0])
        vsp.SetParmVal(fuse_id, "ZLocPercent", "XSec_"+str(0),z_poses[0])
        vsp.SetParmVal(fuse_id, "XLocPercent", "XSec_"+str(end_ind),x_poses[-1])
        vsp.SetParmVal(fuse_id, "ZLocPercent", "XSec_"+str(end_ind),z_poses[-1])    

        # Tail
        if heights[-1] > 0.:
            stdout = vsp.cvar.cstdout
            errorMgr = vsp.ErrorMgrSingleton_getInstance()
            errorMgr.PopErrorAndPrint(stdout)

            pos = len(heights)-1
            vsp.InsertXSec(fuse_id, pos-1, vsp.XS_ELLIPSE)
            vsp.Update()
            vsp.SetParmVal(fuse_id, "Ellipse_Width", "XSecCurve_"+str(pos), widths[-1])
            vsp.SetParmVal(fuse_id, "Ellipse_Height", "XSecCurve_"+str(pos), heights[-1])
            vsp.SetParmVal(fuse_id, "XLocPercent", "XSec_"+str(pos),x_poses[-1])
            vsp.SetParmVal(fuse_id, "ZLocPercent", "XSec_"+str(pos),z_poses[-1])              

            xsecsurf = vsp.GetXSecSurf(fuse_id,0)
            vsp.ChangeXSecShape(xsecsurf,pos+1,vsp.XS_POINT)
            vsp.Update()           
            vsp.SetParmVal(fuse_id, "XLocPercent", "XSec_"+str(pos+1),x_poses[-1])
            vsp.SetParmVal(fuse_id, "ZLocPercent", "XSec_"+str(pos+1),z_poses[-1])     

            # update strengths to make end flat
            vsp.SetParmVal(fuse_id,"TopRStrength","XSec_"+str(pos), 0.)
            vsp.SetParmVal(fuse_id,"RightRStrength","XSec_"+str(pos), 0.)
            vsp.SetParmVal(fuse_id,"BottomRStrength","XSec_"+str(pos), 0.)
            vsp.SetParmVal(fuse_id,"TopLStrength","XSec_"+str(pos+1), 0.)
            vsp.SetParmVal(fuse_id,"RightLStrength","XSec_"+str(pos+1), 0.)            

        else:
            vsp.SetParmVal(fuse_id,"TopLAngle","XSec_"+str(end_ind),vals.tail.top.angle)
            vsp.SetParmVal(fuse_id,"TopLStrength","XSec_"+str(end_ind),vals.tail.top.strength)
            vsp.SetParmVal(fuse_id,"AllSym","XSec_"+str(end_ind),1)
            vsp.Update()


        if 'z_pos' in vals.tail:
            tail_z_pos = vals.tail.z_pos
        else:
            pass # use above default         

    if 'Fuel_Tanks' in fuselage:
        for tank in fuselage.Fuel_Tanks:
            write_fuselage_conformal_fuel_tank(fuse_id, tank, fuel_tank_set_ind)    

    vsp.SetSetFlag(fuse_id, OML_set_ind, True)

    return area_tags

## ingroup Input_Output-OpenVSP
def set_section_angles(i,nose_z,tail_z,x_poses,z_poses,heights,widths,length,end_ind,fuse_id):
    """Set fuselage section angles to create a smooth (in the non-technical sense) fuselage shape.
    Note that i of 0 corresponds to the first section that is not the end point.

    Assumptions:
    May fail to give reasonable angles for very irregularly shaped fuselages
    Does not work on the nose and tail sections.

    Source:
    N/A

    Inputs:  
    nose_z   [-] # 0.1 is 10% of the fuselage length
    widths   np.array of [m]
    heights  np.array of [m]
    tail_z   [-] # 0.1 is 10% of the fuselage length

    Outputs:
    Operates on the active OpenVSP model, no direct output

    Properties Used:
    N/A
    """    
    w0 = widths[i]
    h0 = heights[i]
    x0 = x_poses[i]
    z0 = z_poses[i]   
    w2 = widths[i+2]
    h2 = heights[i+2]
    x2 = x_poses[i+2]
    z2 = z_poses[i+2]

    x0 = x0*length
    x2 = x2*length
    z0 = z0*length
    z2 = z2*length

    top_z_diff = (h2/2+z2)-(h0/2+z0)
    bot_z_diff = (z2-h2/2)-(z0-h0/2)
    y_diff     = w2/2-w0/2
    x_diff     = x2-x0

    top_angle  = np.tan(top_z_diff/x_diff)/Units.deg
    bot_angle  = np.tan(-bot_z_diff/x_diff)/Units.deg
    side_angle = np.tan(y_diff/x_diff)/Units.deg

    vsp.SetParmVal(fuse_id,"TBSym","XSec_"+str(i+1),0)
    vsp.SetParmVal(fuse_id,"TopLAngle","XSec_"+str(i+1),top_angle)
    vsp.SetParmVal(fuse_id,"TopLStrength","XSec_"+str(i+1),0.75)
    vsp.SetParmVal(fuse_id,"BottomLAngle","XSec_"+str(i+1),bot_angle)
    vsp.SetParmVal(fuse_id,"BottomLStrength","XSec_"+str(i+1),0.75)   
    vsp.SetParmVal(fuse_id,"RightLAngle","XSec_"+str(i+1),side_angle)
    vsp.SetParmVal(fuse_id,"RightLStrength","XSec_"+str(i+1),0.75)   

    return  

def compute_fuselage_fineness(fuselage, x_locs, eff_diams, eff_diam_gradients_fwd):
    """This computes fuselage finenesses for nose and tail.

    Assumptions:
    Written for OpenVSP 3.16.1

    Source:
    N/A

    Inputs:
    0. Pre-loaded VSP vehicle in memory, via vsp_read.
    1. Suave fuselage [object].
    2. Array of x_locations of fuselage segments. (length = L)
    3. Array of effective diameters of fuselage segments. (length = L)
    4. Array of effective diameter gradients from nose to tail. (length = L-1)

    Outputs:
    Writes fineness values to SUAVE fuselage, returns fuselage.

    Properties Used:
    N/A
    """ 	
    # Compute nose fineness.    
    x_locs    = np.array(x_locs)					# Make numpy arrays.
    eff_diams = np.array(eff_diams)
    min_val   = np.min(eff_diam_gradients_fwd[x_locs[:-1]<=0.5])	# Computes smallest eff_diam gradient value in front 50% of fuselage.
    x_loc     = x_locs[:-1][eff_diam_gradients_fwd==min_val][0]		# Determines x-location of the first instance of that value (if gradient=0, gets frontmost x-loc).
    fuselage.lengths.nose  = (x_loc-fuselage.Segments[0].percent_x_location)*fuselage.lengths.total	# Subtracts first segment x-loc in case not at global origin.
    fuselage.fineness.nose = fuselage.lengths.nose/(eff_diams[x_locs==x_loc][0])

    # Compute tail fineness.
    x_locs_tail		    = x_locs>=0.5						# Searches aft 50% of fuselage.
    eff_diam_gradients_fwd_tail = eff_diam_gradients_fwd[x_locs_tail[1:]]			# Smaller array of tail gradients.
    min_val 		    = np.min(-eff_diam_gradients_fwd_tail)			# Computes min gradient, where fuselage tapers (minus sign makes positive).
    x_loc = x_locs[np.hstack([False,-eff_diam_gradients_fwd==min_val])][-1]			# Saves aft-most value (useful for straight fuselage with multiple zero gradients.)
    fuselage.lengths.tail       = (1.-x_loc)*fuselage.lengths.total
    fuselage.fineness.tail      = fuselage.lengths.tail/(eff_diams[x_locs==x_loc][0])	# Minus sign converts tail fineness to positive value.

    return fuselage

def get_fuselage_height(fuselage, location):
    """This linearly estimates fuselage height at any percentage point (0,100) along fuselage length.

    Assumptions:
    Written for OpenVSP 3.16.1

    Source:
    N/A

    Inputs:
    0. Pre-loaded VSP vehicle in memory, via vsp_read.
    1. Suave fuselage [object], containing fuselage.vsp_data.xsec_num in its data structure.
    2. Fuselage percentage point [float].

    Outputs:
    height [m]

    Properties Used:
    N/A
    """
    for jj in range(1, fuselage.vsp_data.xsec_num):		# Begin at second section, working toward tail.
        if fuselage.Segments[jj].percent_x_location>=location and fuselage.Segments[jj-1].percent_x_location<location:  
            # Find two sections on either side (or including) the desired fuselage length percentage.
            a        = fuselage.Segments[jj].percent_x_location							
            b        = fuselage.Segments[jj-1].percent_x_location
            a_height = fuselage.Segments[jj].height		# Linear approximation.
            b_height = fuselage.Segments[jj-1].height
            slope    = (a_height - b_height)/(a-b)
            height   = ((location-b)*(slope)) + (b_height)	
            break
    return height

## @ingroup Input_Output-OpenVSP
def find_fuse_u_coordinate(x_target,fuse_id,fuel_tank_tag):
    """Determines the u coordinate of an OpenVSP fuselage that matches an x coordinate

    Assumptions:
    Fuselage is aligned with the x axis

    Source:
    N/A

    Inputs:
    x_target      [m]
    fuse_id       <str>
    fuel_tank_tag <str>

    Outputs:
    u_current     [-] u coordinate for the requests x position

    Properties Used:
    N/A
    """     
    tol   = 1e-3
    diff  = 1000    
    u_min = 0
    u_max = 1    
    while np.abs(diff) > tol:
        u_current = (u_max+u_min)/2
        probe_id = vsp.AddProbe(fuse_id,0,u_current,0,fuel_tank_tag+'_probe')
        vsp.Update()
        x_id  = vsp.FindParm(probe_id,'X','Measure')
        x_pos = vsp.GetParmVal(x_id) 
        diff = x_target-x_pos
        if diff > 0:
            u_min = u_current
        else:
            u_max = u_current
        vsp.DelProbe(probe_id)
    return u_current


## @ingroup Input_Output-OpenVSP
def write_fuselage_conformal_fuel_tank(fuse_id,fuel_tank,fuel_tank_set_ind):
    """This writes a conformal fuel tank in a fuselage.

    Assumptions:
    Fuselage is aligned with the x axis

    Source:
    N/A

    Inputs:
    fuse_id                                     <str>
    fuel_tank.
      inward_offset                             [m]
      start_length_percent                      [-] .1 is 10%
      end_length_percent                        [-]
      fuel_type.density                         [kg/m^3]
    fuel_tank_set_ind                           <int>

    Outputs:
    Operates on the active OpenVSP model, no direct output

    Properties Used:
    N/A
    """        


    #stdout = vsp.cvar.cstdout
    #errorMgr = vsp.ErrorMgrSingleton_getInstance()
    #errorMgr.PopErrorAndPrint(stdout)

    # Unpack
    try:
        offset         = fuel_tank.inward_offset
        len_trim_max   = fuel_tank.end_length_percent
        len_trim_min   = fuel_tank.start_length_percent  
        density        = fuel_tank.fuel_type.density
    except:
        print('Fuel tank does not contain parameters needed for OpenVSP geometry. Tag: '+fuel_tank.tag)
        return        

    tank_id = vsp.AddGeom('CONFORMAL',fuse_id)
    vsp.SetGeomName(tank_id, fuel_tank.tag)    

    # Search for proper x position
    # Get min x
    probe_id = vsp.AddProbe(fuse_id,0,0,0,fuel_tank.tag+'_probe')
    vsp.Update()
    x_id  = vsp.FindParm(probe_id,'X','Measure')
    x_pos = vsp.GetParmVal(x_id)    
    fuse_x_min = x_pos
    vsp.DelProbe(probe_id)
    # Get min x
    probe_id = vsp.AddProbe(fuse_id,0,1,0,fuel_tank.tag+'_probe')
    vsp.Update()
    x_id  = vsp.FindParm(probe_id,'X','Measure')
    x_pos = vsp.GetParmVal(x_id)    
    fuse_x_max = x_pos 
    vsp.DelProbe(probe_id)
    # Search for u values
    x_target_start  = (fuse_x_max-fuse_x_min)*fuel_tank.start_length_percent
    x_target_end    = (fuse_x_max-fuse_x_min)*fuel_tank.end_length_percent
    u_start = find_fuse_u_coordinate(x_target_start, fuse_id, fuel_tank.tag)
    u_end   = find_fuse_u_coordinate(x_target_end, fuse_id, fuel_tank.tag)
    # Offset
    vsp.SetParmVal(tank_id,'Offset','Design',offset)      

    # Fuel tank length bounds
    vsp.SetParmVal(tank_id,'UTrimFlag','Design',1.)
    vsp.SetParmVal(tank_id,'UTrimMax','Design',u_end)
    vsp.SetParmVal(tank_id,'UTrimMin','Design',u_start)  

    # Set density
    vsp.SetParmVal(tank_id,'Density','Mass_Props',density)  

    # Add to the full fuel tank set
    vsp.SetSetFlag(tank_id, fuel_tank_set_ind, True)

    return
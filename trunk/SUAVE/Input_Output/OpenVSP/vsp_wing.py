## @ingroup Input_Output-OpenVSP
# vsp_wing.py

# Created:  Jun 2018, T. St Francis
# Modified: Aug 2018, T. St Francis
#           Jan 2020, T. MacDonald
#           Jul 2020, E. Botero
#           May 2021, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Units
from SUAVE.Components.Airfoils.Airfoil import Airfoil 
from SUAVE.Methods.Geometry.Two_Dimensional.Planform import wing_planform, wing_segmented_planform 
import numpy as np
import string
try:
    import vsp as vsp
except ImportError:
    # This allows SUAVE to build without OpenVSP
    pass 
# This enforces lowercase names
chars = string.punctuation + string.whitespace
t_table = str.maketrans( chars          + string.ascii_uppercase , 
                         '_'*len(chars) + string.ascii_lowercase )

# ----------------------------------------------------------------------
#  vsp read wing
# ----------------------------------------------------------------------

## @ingroup Input_Output-OpenVSP
def read_vsp_wing(wing_id, units_type='SI',write_airfoil_file=True): 	
    """This reads an OpenVSP wing vehicle geometry and writes it into a SUAVE wing format.

    Assumptions:
    1. OpenVSP wing is divided into segments ("XSecs" in VSP).
    2. Written for OpenVSP 3.21.1

    Source:
    N/A

    Inputs:
    1. VSP 10-digit geom ID for wing.
    2. units_type set to 'SI' (default) or 'Imperial'.

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

    # Check if this is vertical tail, this seems like a weird first step but it's necessary
    # Get the initial rotation to get the dihedral angles
    x_rot = vsp.GetParmVal( wing_id,'X_Rotation','XForm')
    y_rot = vsp.GetParmVal( wing_id,'Y_Rotation','XForm')
    if  abs(x_rot) >=70:
        wing = SUAVE.Components.Wings.Vertical_Tail()
        wing.vertical = True
        sign = (np.sign(x_rot))
        x_rot = (sign*90 - sign*x_rot) * Units.deg
    else:
        # Instantiate a wing
        wing = SUAVE.Components.Wings.Wing()
        x_rot =  x_rot  * Units.deg

    y_rot =  y_rot  * Units.deg

    # Set the units
    if units_type == 'SI':
        units_factor = Units.meter * 1.
    elif units_type == 'imperial':
        units_factor = Units.foot * 1.
    elif units_type == 'inches':
        units_factor = Units.inch * 1.		

    # Apply a tag to the wing
    if vsp.GetGeomName(wing_id):
        tag = vsp.GetGeomName(wing_id)
        tag = tag.translate(t_table)
        wing.tag = tag
    else: 
        wing.tag = 'winggeom'
    
    scaling           = vsp.GetParmVal(wing_id, 'Scale', 'XForm')  
    units_factor      = units_factor*scaling
        
    # Top level wing parameters
    # Wing origin
    wing.origin[0][0] = vsp.GetParmVal(wing_id, 'X_Location', 'XForm') * units_factor 
    wing.origin[0][1] = vsp.GetParmVal(wing_id, 'Y_Location', 'XForm') * units_factor 
    wing.origin[0][2] = vsp.GetParmVal(wing_id, 'Z_Location', 'XForm') * units_factor 

    # Wing Symmetry
    sym_planar = vsp.GetParmVal(wing_id, 'Sym_Planar_Flag', 'Sym')
    sym_origin = vsp.GetParmVal(wing_id, 'Sym_Ancestor_Origin_Flag', 'Sym')

    # Check for symmetry
    if sym_planar == 2. and sym_origin == 1.: #origin at wing, not vehicle
        wing.symmetric = True	
    else:
        wing.symmetric = False 

    #More top level parameters
    total_proj_span      = vsp.GetParmVal(wing_id, 'TotalProjectedSpan', 'WingGeom') * units_factor
    wing.aspect_ratio    = vsp.GetParmVal(wing_id, 'TotalAR', 'WingGeom')
    wing.areas.reference = vsp.GetParmVal(wing_id, 'TotalArea', 'WingGeom') * units_factor**2 
    wing.spans.projected = total_proj_span 

    # Check if this is a single segment wing
    xsec_surf_id      = vsp.GetXSecSurf(wing_id, 0)   # This is how VSP stores surfaces.
    x_sec_1           = vsp.GetXSec(xsec_surf_id, 1) 

    if vsp.GetNumXSec(xsec_surf_id) == 2:
        single_seg = True
    else:
        single_seg = False
    
    segment_num = vsp.GetNumXSec(xsec_surf_id) # Get number of segments

    span_sum         = 0.				# Non-projected.
    proj_span_sum    = 0.				# Projected.
    segment_spans    = [None] * (segment_num) 	        # Non-projected.
    segment_dihedral = [None] * (segment_num)
    segment_sweeps_quarter_chord = [None] * (segment_num) 

    # Necessary wing segment definitions start at XSec_1 (XSec_0 exists mainly to hold the root airfoil)
    xsec_surf_id = vsp.GetXSecSurf(wing_id, 0)
    x_sec = vsp.GetXSec(xsec_surf_id, 1)
    chord_parm = vsp.GetXSecParm(x_sec,'Root_Chord')
    root_chord = vsp.GetParmVal(chord_parm) * units_factor

    # -------------
    # Wing segments
    # -------------

    if single_seg == False:

        # Convert VSP XSecs to SUAVE segments. (Wing segments are defined by outboard sections in VSP, but inboard sections in SUAVE.) 
        for i in range(1, segment_num+1):	
            # XSec airfoil
            jj = i-1  # Airfoil index i-1 because VSP airfoils and sections are one index off relative to SUAVE.
		
            segment = SUAVE.Components.Wings.Segment()
            segment.tag                   = 'Section_' + str(i)
            thick_cord                    = vsp.GetParmVal(wing_id, 'ThickChord', 'XSecCurve_' + str(jj))
            segment.thickness_to_chord    = thick_cord	# Thick_cord stored for use in airfoil, below.		
            if i!=segment_num:
                segment_root_chord    = vsp.GetParmVal(wing_id, 'Root_Chord', 'XSec_' + str(i)) * units_factor
            else:
                segment_root_chord    = 0.0
            segment.root_chord_percent    = segment_root_chord / root_chord		
            segment.percent_span_location = proj_span_sum / (total_proj_span/(1+wing.symmetric))
            segment.twist                 = vsp.GetParmVal(wing_id, 'Twist', 'XSec_' + str(jj)) * Units.deg +  y_rot

            if i==1:
                wing.thickness_to_chord = thick_cord

            if i < segment_num:      # This excludes the tip xsec, but we need a segment in SUAVE to store airfoil.
                sweep     = vsp.GetParmVal(wing_id, 'Sweep', 'XSec_' + str(i)) * Units.deg
                sweep_loc = vsp.GetParmVal(wing_id, 'Sweep_Location', 'XSec_' + str(i))
                AR        = 2*vsp.GetParmVal(wing_id, 'Aspect', 'XSec_' + str(i))
                taper     = vsp.GetParmVal(wing_id, 'Taper', 'XSec_' + str(i))

                segment_sweeps_quarter_chord[i] = convert_sweep(sweep,sweep_loc,0.25,AR,taper)
                segment.sweeps.quarter_chord    = segment_sweeps_quarter_chord[i]  # Used again, below

                # Used for dihedral computation, below.
                segment_dihedral[i]	      = vsp.GetParmVal(wing_id, 'Dihedral', 'XSec_' + str(i)) * Units.deg  + x_rot
                segment.dihedral_outboard     = segment_dihedral[i]

                segment_spans[i] 	      = vsp.GetParmVal(wing_id, 'Span', 'XSec_' + str(i)) * units_factor
                proj_span_sum += segment_spans[i] * np.cos(segment_dihedral[i])	
                span_sum      += segment_spans[i]
            else:
                segment.root_chord_percent    = (vsp.GetParmVal(wing_id, 'Tip_Chord', 'XSec_' + str(i-1))) * units_factor /root_chord


            xsec_id = str(vsp.GetXSec(xsec_surf_id, jj))
            airfoil = Airfoil()
            if vsp.GetXSecShape(xsec_id) == vsp.XS_FOUR_SERIES: 	# XSec shape: NACA 4-series
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

            elif vsp.GetXSecShape(xsec_id) == vsp.XS_SIX_SERIES: 	# XSec shape: NACA 6-series
                thick_cord_round = int(np.around(thick_cord*100))
                a_value          = vsp.GetParmVal(wing_id, 'A', 'XSecCurve_' + str(jj))
                ideal_CL         = int(np.around(vsp.GetParmVal(wing_id, 'IdealCl', 'XSecCurve_' + str(jj))*10))
                series_vsp       = int(vsp.GetParmVal(wing_id, 'Series', 'XSecCurve_' + str(jj)))
                series_dict      = {0:'63',1:'64',2:'65',3:'66',4:'67',5:'63A',6:'64A',7:'65A'} # VSP series values.
                series           = series_dict[series_vsp]
                airfoil.tag      = 'NACA ' + series + str(ideal_CL) + str(thick_cord_round) + ' a=' + str(np.around(a_value,1))			


            elif vsp.GetXSecShape(xsec_id) == vsp.XS_FILE_AIRFOIL:	# XSec shape: 12 is type AF_FILE
                airfoil.thickness_to_chord = thick_cord
                # VSP airfoil API calls get coordinates and write files with the final argument being the fraction of segment position, regardless of relative spans. 
                # (Write the root airfoil with final arg = 0. Write 4th airfoil of 5 segments with final arg = .8)

            if write_airfoil_file==True:
                vsp.WriteSeligAirfoil(str(wing.tag) + '_airfoil_XSec_' + str(jj) +'.dat', wing_id, float(jj/segment_num))
                airfoil.coordinate_file    = str(wing.tag) + '_airfoil_XSec_' + str(jj) +'.dat'
                airfoil.tag                = 'airfoil'	

                segment.append_airfoil(airfoil)

            wing.Segments.append(segment)

        # Wing dihedral 
        proj_span_sum_alt = 0.
        span_sum_alt      = 0.
        sweeps_sum        = 0.			

        for ii in range(1, segment_num):
            span_sum_alt += segment_spans[ii]
            proj_span_sum_alt += segment_spans[ii] * np.cos(segment_dihedral[ii])  # Use projected span to find total wing dihedral.
            sweeps_sum += segment_spans[ii] * np.tan(segment_sweeps_quarter_chord[ii])	

        wing.dihedral              = np.arccos(proj_span_sum_alt / span_sum_alt) 
        wing.sweeps.quarter_chord  = -np.arctan(sweeps_sum / span_sum_alt)  # Minus sign makes it positive sweep.

        # Add a tip segment, all values are zero except the tip chord
        tc = vsp.GetParmVal(wing_id, 'Tip_Chord', 'XSec_' + str(segment_num-1)) * units_factor

        # Chords
        wing.chords.root              = vsp.GetParmVal(wing_id, 'Tip_Chord', 'XSec_0') * units_factor
        wing.chords.tip               = tc
        wing.chords.mean_geometric    = wing.areas.reference / wing.spans.projected

        # Just double calculate and fix things:
        wing = wing_segmented_planform(wing)


    else:
        # Single segment

        # Get ID's
        x_sec_1_dih_parm       = vsp.GetXSecParm(x_sec_1,'Dihedral')
        x_sec_1_sweep_parm     = vsp.GetXSecParm(x_sec_1,'Sweep')
        x_sec_1_sweep_loc_parm = vsp.GetXSecParm(x_sec_1,'Sweep_Location')
        x_sec_1_taper_parm     = vsp.GetXSecParm(x_sec_1,'Taper')
        x_sec_1_rc_parm        = vsp.GetXSecParm(x_sec_1,'Root_Chord')
        x_sec_1_tc_parm        = vsp.GetXSecParm(x_sec_1,'Tip_Chord')
        x_sec_1_t_parm        = vsp.GetXSecParm(x_sec_1,'ThickChord')
     
        # Calcs
        sweep     = vsp.GetParmVal(x_sec_1_sweep_parm) * Units.deg
        sweep_loc = vsp.GetParmVal(x_sec_1_sweep_loc_parm)
        taper     = vsp.GetParmVal(x_sec_1_taper_parm)
        c_4_sweep = convert_sweep(sweep,sweep_loc,0.25,wing.aspect_ratio,taper)		

        # Pull and pack
        wing.sweeps.quarter_chord  = c_4_sweep
        wing.taper                 = taper
        wing.dihedral              = vsp.GetParmVal(x_sec_1_dih_parm) * Units.deg + x_rot
        wing.chords.root           = vsp.GetParmVal(x_sec_1_rc_parm)* units_factor
        wing.chords.tip            = vsp.GetParmVal(x_sec_1_tc_parm) * units_factor	
        wing.chords.mean_geometric = wing.areas.reference / wing.spans.projected
        wing.thickness_to_chord    = vsp.GetParmVal(x_sec_1_t_parm) 

        # Just double calculate and fix things:
        wing = wing_planform(wing)		


    # Twists
    wing.twists.root      = vsp.GetParmVal(wing_id, 'Twist', 'XSec_0') * Units.deg +  y_rot
    wing.twists.tip       = vsp.GetParmVal(wing_id, 'Twist', 'XSec_' + str(segment_num-1)) * Units.deg +  y_rot

    # check if control surface (sub surfaces) are defined
    tags                 = []
    LE_flags             = []
    span_fraction_starts = []
    span_fraction_ends   = []
    chord_fractions      = []
    
    num_cs = vsp.GetNumSubSurf(wing_id)
    
    # loop through wing and get all control surface parameters 
    for cs_idx in range(num_cs):
        cs_id   = vsp.GetSubSurf(wing_id,cs_idx)
        param_names = vsp.GetSubSurfParmIDs(cs_id)
        tags.append(vsp.GetSubSurfName(cs_id))
        for p_idx in range(len(param_names)):
            if 'LE_Flag' == vsp.GetParmName(param_names[p_idx]):
                LE_flags.append(vsp.GetParmVal(param_names[p_idx]))
            if 'UStart' == vsp.GetParmName(param_names[p_idx]):
                span_fraction_starts.append(vsp.GetParmVal(param_names[p_idx]))
            if 'UEnd' == vsp.GetParmName(param_names[p_idx]):
                span_fraction_ends.append(vsp.GetParmVal(param_names[p_idx]))
            if 'Length_C_Start' == vsp.GetParmName(param_names[p_idx]):
                chord_fractions.append(vsp.GetParmVal(param_names[p_idx]))
                
    # assign control surface parameters to wings. Outer most control surface on main/horizontal wing is assigned a aileron
    for cs_idx in range(num_cs):   
        aileron_present = False
        if num_cs > 1:
            aileron_loc = np.argmax(np.array(span_fraction_starts))   
            if cs_idx == aileron_loc: 
                aileron_present = True
        if LE_flags[cs_idx] == 1.0:
            CS = SUAVE.Components.Wings.Control_Surfaces.Slat()
        else:
            if wing.vertical == True:
                CS = SUAVE.Components.Wings.Control_Surfaces.Rudder()
            else:
                if aileron_present:
                    CS = SUAVE.Components.Wings.Control_Surfaces.Aileron()
                else: 
                    CS = SUAVE.Components.Wings.Control_Surfaces.Flap()
        CS.tag                 = tags[cs_idx]
        CS.span_fraction_start = span_fraction_starts[cs_idx]*3 - 1
        CS.span_fraction_end   = span_fraction_ends[cs_idx]*3 - 1
        CS.chord_fraction      = chord_fractions[cs_idx]
        CS.span                = (CS.span_fraction_end - CS.span_fraction_start)*wing.spans.projected
        wing.append_control_surface(CS)
    
    return wing


## @ingroup Input_Output-OpenVSP
def write_vsp_wing(vehicle,wing, area_tags, fuel_tank_set_ind, OML_set_ind):
    """This write a given wing into OpenVSP format

    Assumptions:
    If wing segments are defined, they must cover the full span.
    (may work in some other cases, but functionality will not be maintained)

    Source:
    N/A

    Inputs:
    vehicle                                   [-] vehicle data structure
    wing.
      origin                                  [m] in all three dimensions
      spans.projected                         [m]
      chords.root                             [m]
      chords.tip                              [m]
      sweeps.quarter_chord                    [radians]
      twists.root                             [radians]
      twists.tip                              [radians]
      thickness_to_chord                      [-]
      dihedral                                [radians]
      tag                                     <string>
      Segments.*. (optional)
        twist                                 [radians]
        percent_span_location                 [-]  .1 is 10%
        root_chord_percent                    [-]  .1 is 10%
        dihedral_outboard                     [radians]
        sweeps.quarter_chord                  [radians]
        thickness_to_chord                    [-]
    area_tags                                 <dict> used to keep track of all tags needed in wetted area computation           
    fuel_tank_set_index                       <int> OpenVSP object set containing the fuel tanks    

    Outputs:
    area_tags                                 <dict> used to keep track of all tags needed in wetted area computation           
    wing_id                                   <str>  OpenVSP ID for given wing

    Properties Used:
    N/A
    """       
    wing_x = wing.origin[0][0]    
    wing_y = wing.origin[0][1]
    wing_z = wing.origin[0][2]
    if wing.symmetric == True:
        span   = wing.spans.projected/2. # span of one side
    else:
        span   = wing.spans.projected
    root_chord = wing.chords.root
    tip_chord  = wing.chords.tip
    sweep      = wing.sweeps.quarter_chord / Units.deg
    sweep_loc  = 0.25
    root_twist = wing.twists.root / Units.deg
    tip_twist  = wing.twists.tip  / Units.deg
    root_tc    = wing.thickness_to_chord 
    tip_tc     = wing.thickness_to_chord 
    dihedral   = wing.dihedral / Units.deg

    # Check to see if segments are defined. Get count
    if len(wing.Segments.keys())>0:
        n_segments = len(wing.Segments.keys())
    else:
        n_segments = 0

    # Create the wing
    wing_id = vsp.AddGeom( "WING" )
    vsp.SetGeomName(wing_id, wing.tag)
    area_tags[wing.tag] = ['wings',wing.tag]

    # Make names for each section and insert them into the wing if necessary
    x_secs       = []
    x_sec_curves = []
    # n_segments + 2 will create an extra segment if the root segment is 
    # included in the list of segments. This is not used and the tag is
    # removed when the segments are checked for this case.
    for i_segs in range(0,n_segments+2):
        x_secs.append('XSec_' + str(i_segs))
        x_sec_curves.append('XSecCurve_' + str(i_segs))

    # Apply the basic characteristics of the wing to root and tip
    if wing.symmetric == False:
        vsp.SetParmVal( wing_id,'Sym_Planar_Flag','Sym',0)
    if wing.vertical == True:
        vsp.SetParmVal( wing_id,'X_Rel_Rotation','XForm',90)
        dihedral = -dihedral # check for vertical tail, direction reverses from SUAVE/AVL

    vsp.SetParmVal( wing_id,'X_Rel_Location','XForm',wing_x)
    vsp.SetParmVal( wing_id,'Y_Rel_Location','XForm',wing_y)
    vsp.SetParmVal( wing_id,'Z_Rel_Location','XForm',wing_z)

    # This ensures that the other VSP parameters are driven properly
    vsp.SetDriverGroup( wing_id, 1, vsp.SPAN_WSECT_DRIVER, vsp.ROOTC_WSECT_DRIVER, vsp.TIPC_WSECT_DRIVER )

    # Root chord
    vsp.SetParmVal( wing_id,'Root_Chord',x_secs[1],root_chord)

    # Sweep of the first section
    vsp.SetParmVal( wing_id,'Sweep',x_secs[1],sweep)
    vsp.SetParmVal( wing_id,'Sweep_Location',x_secs[1],sweep_loc)

    # Twists
    if n_segments != 0:
        if np.isclose(wing.Segments[0].percent_span_location,0.):
            vsp.SetParmVal( wing_id,'Twist',x_secs[0],wing.Segments[0].twist / Units.deg) # root
        else:
            vsp.SetParmVal( wing_id,'Twist',x_secs[0],root_twist) # root
        # The tips should write themselves
    else:
        vsp.SetParmVal( wing_id,'Twist',x_secs[0],root_twist) # root
        vsp.SetParmVal( wing_id,'Twist',x_secs[1],tip_twist) # tip


    # Figure out if there is an airfoil provided

    # Airfoils should be in Lednicer format
    # i.e. :
    #
    #EXAMPLE AIRFOIL
    # 3. 3. 
    #
    # 0.0 0.0
    # 0.5 0.1
    # 1.0 0.0
    #
    # 0.0 0.0
    # 0.5 -0.1
    # 1.0 0.0

    # Note this will fail silently if airfoil is not in correct format
    # check geometry output

    airfoil_vsp_types = []
    if n_segments > 0:
        for i in range(n_segments): 
            if 'airfoil_type' in wing.Segments[i].keys():
                if wing.Segments[i].airfoil_type == 'biconvex': 
                    airfoil_vsp_types.append(vsp.XS_BICONVEX)
                else:
                    airfoil_vsp_types.append(vsp.XS_FILE_AIRFOIL)
            else:
                airfoil_vsp_types.append(vsp.XS_FILE_AIRFOIL)
    elif 'airfoil_type' in wing.keys():
        if wing.airfoil_type == 'biconvex': 
            airfoil_vsp_types.append(vsp.XS_BICONVEX)
        else:
            airfoil_vsp_types.append(vsp.XS_FILE_AIRFOIL)        
    else:
        airfoil_vsp_types = [vsp.XS_FILE_AIRFOIL]    

    if n_segments==0:
        if len(wing.Airfoil) != 0 or 'airfoil_type' in wing.keys():
            xsecsurf = vsp.GetXSecSurf(wing_id,0)
            vsp.ChangeXSecShape(xsecsurf,0,airfoil_vsp_types[0])
            vsp.ChangeXSecShape(xsecsurf,1,airfoil_vsp_types[0])
            if len(wing.Airfoil) != 0:
                xsec1 = vsp.GetXSec(xsecsurf,0)
                xsec2 = vsp.GetXSec(xsecsurf,1)
                vsp.ReadFileAirfoil(xsec1,wing.Airfoil['airfoil'].coordinate_file)
                vsp.ReadFileAirfoil(xsec2,wing.Airfoil['airfoil'].coordinate_file)
            vsp.Update()
    else:
        if len(wing.Segments[0].Airfoil) != 0 or 'airfoil_type' in wing.Segments[0].keys():
            xsecsurf = vsp.GetXSecSurf(wing_id,0)
            vsp.ChangeXSecShape(xsecsurf,0,airfoil_vsp_types[0])
            vsp.ChangeXSecShape(xsecsurf,1,airfoil_vsp_types[0])
            if len(wing.Segments[0].Airfoil) != 0:
                xsec1 = vsp.GetXSec(xsecsurf,0)
                xsec2 = vsp.GetXSec(xsecsurf,1)
                vsp.ReadFileAirfoil(xsec1,wing.Segments[0].Airfoil['airfoil'].coordinate_file)
                vsp.ReadFileAirfoil(xsec2,wing.Segments[0].Airfoil['airfoil'].coordinate_file)
            vsp.Update()              

    # Thickness to chords
    vsp.SetParmVal( wing_id,'ThickChord','XSecCurve_0',root_tc)
    vsp.SetParmVal( wing_id,'ThickChord','XSecCurve_1',tip_tc)

    # Dihedral
    vsp.SetParmVal( wing_id,'Dihedral',x_secs[1],dihedral)

    # Span and tip of the section
    if n_segments>1:
        local_span    = span*wing.Segments[0].percent_span_location  
        sec_tip_chord = root_chord*wing.Segments[0].root_chord_percent
        vsp.SetParmVal( wing_id,'Span',x_secs[1],local_span) 
        vsp.SetParmVal( wing_id,'Tip_Chord',x_secs[1],sec_tip_chord)
    else:
        vsp.SetParmVal( wing_id,'Span',x_secs[1],span/np.cos(dihedral*Units.degrees)) 

    vsp.Update()

    if n_segments>0:
        if wing.Segments[0].percent_span_location==0.:
            x_secs[-1] = [] # remove extra section tag (for clarity)
            segment_0_is_root_flag = True
            adjust = 0 # used for indexing
        else:
            segment_0_is_root_flag = False
            adjust = 1
    else:
        adjust = 1


    # Loop for the number of segments left over
    for i_segs in range(1,n_segments+1):  

        if (wing.Segments[i_segs-1] == wing.Segments[-1]) and (wing.Segments[-1].percent_span_location == 1.):
            break

        # Unpack
        dihedral_i = wing.Segments[i_segs-1].dihedral_outboard / Units.deg
        chord_i    = root_chord*wing.Segments[i_segs-1].root_chord_percent
        try:
            twist_i    = wing.Segments[i_segs].twist / Units.deg
            no_twist_flag = False
        except:
            no_twist_flag = True
        sweep_i    = wing.Segments[i_segs-1].sweeps.quarter_chord / Units.deg
        tc_i       = wing.Segments[i_segs-1].thickness_to_chord

        # Calculate the local span
        if i_segs == n_segments:
            span_i = span*(1 - wing.Segments[i_segs-1].percent_span_location)/np.cos(dihedral_i*Units.deg)
        else:
            span_i = span*(wing.Segments[i_segs].percent_span_location-wing.Segments[i_segs-1].percent_span_location)/np.cos(dihedral_i*Units.deg)                      

        # Insert the new wing section with specified airfoil if available
        if len(wing.Segments[i_segs-1].Airfoil) != 0 or 'airfoil_type' in wing.Segments[i_segs-1].keys():
            vsp.InsertXSec(wing_id,i_segs-1+adjust,airfoil_vsp_types[i_segs-1])
            if len(wing.Segments[i_segs-1].Airfoil) != 0:
                xsecsurf = vsp.GetXSecSurf(wing_id,0)
                xsec = vsp.GetXSec(xsecsurf,i_segs+adjust)
                vsp.ReadFileAirfoil(xsec, wing.Segments[i_segs-1].Airfoil['airfoil'].coordinate_file)                
        else:
            vsp.InsertXSec(wing_id,i_segs-1+adjust,vsp.XS_FOUR_SERIES)

        # Set the parms
        vsp.SetParmVal( wing_id,'Span',x_secs[i_segs+adjust],span_i)
        vsp.SetParmVal( wing_id,'Dihedral',x_secs[i_segs+adjust],dihedral_i)
        vsp.SetParmVal( wing_id,'Sweep',x_secs[i_segs+adjust],sweep_i)
        vsp.SetParmVal( wing_id,'Sweep_Location',x_secs[i_segs+adjust],sweep_loc)      
        vsp.SetParmVal( wing_id,'Root_Chord',x_secs[i_segs+adjust],chord_i)
        if not no_twist_flag:
            vsp.SetParmVal( wing_id,'Twist',x_secs[i_segs+adjust],twist_i)
        vsp.SetParmVal( wing_id,'ThickChord',x_sec_curves[i_segs+adjust],tc_i)

        if adjust and (i_segs == 1):
            vsp.Update()
            vsp.SetParmVal( wing_id,'Twist',x_secs[1],wing.Segments[i_segs-1].twist / Units.deg)

        vsp.Update()

    if (n_segments != 0) and (wing.Segments[-1].percent_span_location == 1.):
        tip_chord = root_chord*wing.Segments[-1].root_chord_percent
        vsp.SetParmVal( wing_id,'Tip_Chord',x_secs[n_segments-1+adjust],tip_chord)
        vsp.SetParmVal( wing_id,'ThickChord',x_sec_curves[n_segments-1+adjust],wing.Segments[-1].thickness_to_chord)
        # twist is set in the normal loop
    else:
        vsp.SetParmVal( wing_id,'Tip_Chord',x_secs[-1-(1-adjust)],tip_chord)
        vsp.SetParmVal( wing_id,'Twist',x_secs[-1-(1-adjust)],tip_twist)
        # a single trapezoidal wing is assumed to have constant thickness to chord
    vsp.Update()
    vsp.SetParmVal(wing_id,'CapUMaxOption','EndCap',2.)
    vsp.SetParmVal(wing_id,'CapUMaxStrength','EndCap',1.)

    vsp.Update()  

    if 'control_surfaces' in wing:
        for ctrl_surf in wing.control_surfaces:
            write_vsp_control_surface(wing_id,ctrl_surf)


    if 'Fuel_Tanks' in wing:
        for tank in wing.Fuel_Tanks:
            write_wing_conformal_fuel_tank(vehicle,wing, wing_id, tank, fuel_tank_set_ind)

    vsp.SetSetFlag(wing_id, OML_set_ind, True)

    return area_tags, wing_id 


## @ingroup Input_Output-OpenVSP
def write_vsp_control_surface(wing_id,ctrl_surf):
    """This writes a control surface in a wing.
    
    Assumptions:
    None
    
    Source:
    N/A
    
    Inputs:
    wind_id              <str>
    ctrl_surf            [-]
    
    Outputs:
    Operates on the active OpenVSP model, no direct output
    
    Properties Used:
    N/A
    """
    cs_id =  vsp.AddSubSurf( wing_id, vsp.SS_CONTROL)
    param_names = vsp.GetSubSurfParmIDs(cs_id)
    for p_idx in range(len(param_names)):
        if 'LE_Flag' == vsp.GetParmName(param_names[p_idx]):
            if type(ctrl_surf) == SUAVE.Components.Wings.Control_Surfaces.Slat:
                vsp.SetParmVal(param_names[p_idx], 1.0)
            else:
                vsp.SetParmVal( param_names[p_idx], 0.0)
        if 'UStart' == vsp.GetParmName(param_names[p_idx]):
            vsp.SetParmVal(param_names[p_idx], (ctrl_surf.span_fraction_start+1)/3)
        if 'UEnd' ==vsp.GetParmName(param_names[p_idx]):
            vsp.SetParmVal(param_names[p_idx], (ctrl_surf.span_fraction_end+1)/3)
        if 'Length_C_Start' == vsp.GetParmName(param_names[p_idx]):
            vsp.SetParmVal(param_names[p_idx], ctrl_surf.chord_fraction)
        if 'Length_C_End' == vsp.GetParmName(param_names[p_idx]):
            vsp.SetParmVal(param_names[p_idx], ctrl_surf.chord_fraction)
        if 'SE_Const_Flag' == vsp.GetParmName(param_names[p_idx]):
            vsp.SetParmVal(param_names[p_idx], 1.0)
            
    return

## @ingroup Input_Output-OpenVSP
def write_wing_conformal_fuel_tank(vehicle,wing, wing_id,fuel_tank,fuel_tank_set_ind):
    """This writes a conformal fuel tank in a wing.

    Assumptions:
    None

    Source:
    N/A

    Inputs:
    vehicle                                     [-] vehicle data structure
    wing.Segments.*.percent_span_location       [-]
    wing.spans.projected                        [m]
    wind_id                                     <str>
    fuel_tank.
      inward_offset                             [m]
      start_chord_percent                       [-] .1 is 10%
      end_chord_percent                         [-]
      start_span_percent                        [-]
      end_span_percent                          [-]
      fuel_type.density                         [kg/m^3]
    fuel_tank_set_ind                           <int>

    Outputs:
    Operates on the active OpenVSP model, no direct output

    Properties Used:
    N/A
    """        
    # Unpack
    try:
        offset            = fuel_tank.inward_offset
        chord_trim_max    = 1.-fuel_tank.start_chord_percent
        chord_trim_min    = 1.-fuel_tank.end_chord_percent
        span_trim_max     = fuel_tank.end_span_percent
        span_trim_min     = fuel_tank.start_span_percent  
        density           = fuel_tank.fuel_type.density
    except:
        print('Fuel tank does not contain parameters needed for OpenVSP geometry. Tag: '+fuel_tank.tag)
        return

    tank_id = vsp.AddGeom('CONFORMAL',wing_id)
    vsp.SetGeomName(tank_id, fuel_tank.tag)    
    n_segments        = len(wing.Segments.keys())
    if n_segments > 0.:
        seg_span_percents  = np.array([v['percent_span_location'] for (k,v)\
                                       in wing.Segments.iteritems()])
        vsp_segment_breaks = np.linspace(0.,1.,n_segments)
    else:
        seg_span_percents = np.array([0.,1.])
    span = wing.spans.projected

    # Offset
    vsp.SetParmVal(tank_id,'Offset','Design',offset)      

    for key, fuselage in vehicle.fuselages.items():
        width    = fuselage.width
        length   = fuselage.lengths.total
        hmax     = fuselage.heights.maximum
        height1  = fuselage.heights.at_quarter_length
        height2  = fuselage.heights.at_wing_root_quarter_chord 
        height3  = fuselage.heights.at_three_quarters_length
        effdia   = fuselage.effective_diameter
        n_fine   = fuselage.fineness.nose 
        t_fine   = fuselage.fineness.tail  
        w_ac     = wing.aerodynamic_center

        w_origin = vehicle.wings.main_wing.origin
        w_c_4    = vehicle.wings.main_wing.chords.root/4.

        # Figure out the location x location of each section, 3 sections, end of nose, wing origin, and start of tail

        x1 = 0.25
        x2 = (w_origin[0]+w_c_4)/length
        x3 = 0.75

        fuse_id = vsp.AddGeom("FUSELAGE") 
        vsp.SetGeomName(fuse_id, fuselage.tag)
        wing_id[fuselage.tag] = ['fuselages',fuselage.tag]

        # Set the origins:
        x = fuselage.origin[0][0]
        y = fuselage.origin[0][1]
        z = fuselage.origin[0][2]
        vsp.SetParmVal(fuse_id,'X_Location','XForm',x)
        vsp.SetParmVal(fuse_id,'Y_Location','XForm',y)
        vsp.SetParmVal(fuse_id,'Z_Location','XForm',z)
        vsp.SetParmVal(fuse_id,'Abs_Or_Relitive_flag','XForm',vsp.ABS) # misspelling from OpenVSP
        vsp.SetParmVal(fuse_id,'Origin','XForm',0.0)

    # Fuel tank chord bounds
    vsp.SetParmVal(tank_id,'ChordTrimFlag','Design',1.)
    vsp.SetParmVal(tank_id,'ChordTrimMax','Design',chord_trim_max)
    vsp.SetParmVal(tank_id,'ChordTrimMin','Design',chord_trim_min)

    # Fuel tank span bounds
    if n_segments>0:
        span_trim_max = get_vsp_trim_from_SUAVE_trim(seg_span_percents,
                                                     vsp_segment_breaks,  
                                                             span_trim_max)
        span_trim_min = get_vsp_trim_from_SUAVE_trim(seg_span_percents,
                                                     vsp_segment_breaks,
                                                             span_trim_min)
    else:
        pass # no change to span_trim

    vsp.SetParmVal(tank_id,'UTrimFlag','Design',1.)
    vsp.SetParmVal(tank_id,'UTrimMax','Design',span_trim_max)
    vsp.SetParmVal(tank_id,'UTrimMin','Design',span_trim_min)  

    # Set density
    vsp.SetParmVal(tank_id,'Density','Mass_Props',density)  

    # Add to the full fuel tank set
    vsp.SetSetFlag(tank_id, fuel_tank_set_ind, True)

    return 

## @ingroup Input_Output-OpenVSP
def get_vsp_trim_from_SUAVE_trim(seg_span_percents,vsp_segment_breaks,trim):
    """Compute OpenVSP span trim coordinates based on SUAVE coordinates

    Assumptions:
    Wing does not have end caps

    Source:
    N/A

    Inputs:
    seg_span_percents   [-] range of 0 to 1
    vsp_segment_breaks  [-] range of 0 to 1
    trim                [-] range of 0 to 1 (SUAVE value)

    Outputs:
    trim                [-] OpenVSP trim value

    Properties Used:
    N/A
    """      
    # Determine max chord trim correction
    y_seg_ind = next(i for i,per_y in enumerate(seg_span_percents) if per_y > trim)
    segment_percent_of_total_span = seg_span_percents[y_seg_ind] -\
        seg_span_percents[y_seg_ind-1]
    remaining_percent_within_segment = trim - seg_span_percents[y_seg_ind-1]
    percent_of_segment = remaining_percent_within_segment/segment_percent_of_total_span
    trim = vsp_segment_breaks[y_seg_ind-1] + \
        (vsp_segment_breaks[y_seg_ind]-vsp_segment_breaks[y_seg_ind-1])*percent_of_segment  
    return trim


## @ingroup Input_Output-OpenVSP
def convert_sweep(sweep,sweep_loc,new_sweep_loc,AR,taper): 
    """This converts arbitrary sweep into a desired sweep given 
    wing geometry.

    Assumptions:
    None

    Source:
    N/A

    Inputs: 
    sweep               [degrees]
    sweep_loc           [unitless]
    new_sweep_loc       [unitless]
    AR                  [unitless]
    taper               [unitless]

    Outputs:
    quarter chord sweep

    Properties Used:
    N/A
    """   
    sweep_LE = np.arctan(np.tan(sweep)+4*sweep_loc*
                         (1-taper)/(AR*(1+taper))) 

    new_sweep = np.arctan(np.tan(sweep_LE)-4*new_sweep_loc*
                          (1-taper)/(AR*(1+taper))) 

    return new_sweep

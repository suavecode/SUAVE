## @ingroup Input_Output-OpenVSP
# vsp_write.py
# 
# Created:  Jul 2016, T. MacDonald
# Modified: Jun 2017, T. MacDonald
#           Jul 2017, T. MacDonald
#           Oct 2018, T. MacDonald
#           Nov 2018, T. MacDonald
#           Jan 2019, T. MacDonald
#           Jan 2020, T. MacDonald 
#           Mar 2020, M. Clarke
#           May 2020, E. Botero
#           Jul 2020, E. Botero 
#           Feb 2021, T. MacDonald
#           May 2021, E. Botero 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Units, Data

try:
    import vsp as vsp
except ImportError:
    # This allows SUAVE to build without OpenVSP
    pass
import numpy as np
import os

## @ingroup Input_Output-OpenVSP
def write(vehicle, tag, fuel_tank_set_ind=3, verbose=True, write_file=True, OML_set_ind = 4, write_igs = False):
    """This writes a SUAVE vehicle to OpenVSP format. It will take wing segments into account
    if they are specified in the vehicle setup file.
    
    Assumptions:
    Vehicle is composed of conventional shape fuselages, wings, and networks. Any network
    that should be created is tagged as 'turbofan'.

    Source:
    N/A

    Inputs:
    vehicle.
      tag                                       [-]
      wings.*.    (* is all keys)
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
      networks.turbofan. (optional)
        number_of_engines                       [-]
        engine_length                           [m]
        nacelle_diameter                        [m]
        origin                                  [m] in all three dimension, should have as many origins as engines
        OpenVSP_simple (optional)               <boolean> if False (default) create a flow through nacelle, if True creates a roughly biparabolic shape
      fuselages.fuselage (optional)
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
    fuel_tank_set_index                         <int> OpenVSP object set containing the fuel tanks    

    Outputs:
    <tag>.vsp3           This is the OpenVSP representation of the aircraft

    Properties Used:
    N/A
    """    
    
    # Reset OpenVSP to avoid including a previous vehicle
    if verbose:
        print('Reseting OpenVSP Model in Memory')
    try:
        vsp.ClearVSPModel()
    except NameError:
        print('VSP import failed')
        return -1
    
    area_tags = dict() # for wetted area assignment
    
    # -------------
    # Wings
    # -------------
    
    # Default Set_0 in OpenVSP is index 3
    vsp.SetSetName(fuel_tank_set_ind, 'fuel_tanks')
    vsp.SetSetName(OML_set_ind, 'OML')
    
    for wing in vehicle.wings:       
        if verbose:
            print('Writing '+wing.tag+' to OpenVSP Model')
            area_tags, wing_id = write_vsp_wing(wing,area_tags, fuel_tank_set_ind, OML_set_ind)
        if wing.tag == 'main_wing':
            main_wing_id = wing_id    
    
    # -------------
    # Engines
    # -------------
    ## Skeleton code for props and pylons can be found in previous commits (~Dec 2016) if desired
    ## This was a place to start and may not still be functional    
    
    if 'turbofan' in vehicle.networks:
        if verbose:
            print('Writing '+vehicle.networks.turbofan.tag+' to OpenVSP Model')
        turbofan  = vehicle.networks.turbofan
        write_vsp_turbofan(turbofan, OML_set_ind)
        
    if 'turbojet' in vehicle.networks:
        turbofan  = vehicle.networks.turbojet
        write_vsp_turbofan(turbofan, OML_set_ind)    
    
    # -------------
    # Fuselage
    # -------------    
    
    for key, fuselage in vehicle.fuselages.items():
        if verbose:
            print('Writing '+fuselage.tag+' to OpenVSP Model')
        try:
            area_tags = write_vsp_fuselage(fuselage, area_tags, vehicle.wings.main_wing, 
                                           fuel_tank_set_ind, OML_set_ind)
        except AttributeError:
            area_tags = write_vsp_fuselage(fuselage, area_tags, None, fuel_tank_set_ind,
                                           OML_set_ind)
    
    vsp.Update()
    
    # Write the vehicle to the file    
    if write_file ==True:
        cwd = os.getcwd()
        filename = tag + ".vsp3"
        if verbose:
            print('Saving OpenVSP File at '+ cwd + '/' + filename)
        vsp.WriteVSPFile(filename)
    elif verbose:
        print('Not Saving OpenVSP File')
        
    if write_igs:
        if verbose:
            print('Exporting IGS File')        
        vehicle_id = vsp.FindContainersWithName('Vehicle')[0]
        parm_id = vsp.FindParm(vehicle_id,'LabelID','IGESSettings')
        vsp.SetParmVal(parm_id, 0.)
        vsp.ExportFile(tag + ".igs", OML_set_ind, vsp.EXPORT_IGES)
    
    return area_tags

## @ingroup Input_Output-OpenVSP
def write_vsp_wing(wing, area_tags, fuel_tank_set_ind, OML_set_ind):
    """This write a given wing into OpenVSP format
    
    Assumptions:
    If wing segments are defined, they must cover the full span.
    (may work in some other cases, but functionality will not be maintained)

    Source:
    N/A

    Inputs:
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

    vsp.Update() # to fix problems with chords not matching up
    
    if 'Fuel_Tanks' in wing:
        for tank in wing.Fuel_Tanks:
            write_wing_conformal_fuel_tank(wing, wing_id, tank, fuel_tank_set_ind)
            
    vsp.SetSetFlag(wing_id, OML_set_ind, True)
    
    return area_tags, wing_id

## @ingroup Input_Output-OpenVSP
def write_vsp_turbofan(turbofan, OML_set_ind):
    """This converts turbofans into OpenVSP format.
    
    Assumptions:
    None

    Source:
    N/A

    Inputs:
    turbofan.
      number_of_engines                       [-]
      engine_length                           [m]
      nacelle_diameter                        [m]
      origin                                  [m] in all three dimension, should have as many origins as engines
      OpenVSP_flow_through                    <boolean> if True create a flow through nacelle, if False create a cylinder

    Outputs:
    Operates on the active OpenVSP model, no direct output

    Properties Used:
    N/A
    """    
    n_engines   = turbofan.number_of_engines
    length      = turbofan.engine_length
    width       = turbofan.nacelle_diameter
    origins     = turbofan.origin
    inlet_width = turbofan.inlet_diameter
    tf_tag      = turbofan.tag
    
    # True will create a flow-through subsonic nacelle (which may have dimensional errors)
    # False will create a cylindrical stack (essentially a cylinder)
    ft_flag = turbofan.OpenVSP_flow_through
    
    import operator # import here since engines are not always needed
    # sort engines per left to right convention
    origins_sorted = sorted(origins, key=operator.itemgetter(1))
    
    for ii in range(0,int(n_engines)):

        origin = origins_sorted[ii]
        
        x = origin[0]
        y = origin[1]
        z = origin[2]
        
        if ft_flag == True:
            nac_id = vsp.AddGeom( "BODYOFREVOLUTION")
            vsp.SetGeomName(nac_id, tf_tag+'_'+str(ii+1))
            
            # Origin
            vsp.SetParmVal(nac_id,'X_Location','XForm',x)
            vsp.SetParmVal(nac_id,'Y_Location','XForm',y)
            vsp.SetParmVal(nac_id,'Z_Location','XForm',z)
            vsp.SetParmVal(nac_id,'Abs_Or_Relitive_flag','XForm',vsp.ABS) # misspelling from OpenVSP  
            
            # Length and overall diameter
            vsp.SetParmVal(nac_id,"Diameter","Design",inlet_width)
            
            vsp.ChangeBORXSecShape(nac_id ,vsp.XS_SUPER_ELLIPSE)
            vsp.Update()
            vsp.SetParmVal(nac_id, "Super_Height", "XSecCurve", (width-inlet_width)/2)
            vsp.SetParmVal(nac_id, "Super_Width", "XSecCurve", length)
            vsp.SetParmVal(nac_id, "Super_MaxWidthLoc", "XSecCurve", -1.)
            vsp.SetParmVal(nac_id, "Super_M", "XSecCurve", 2.)
            vsp.SetParmVal(nac_id, "Super_N", "XSecCurve", 1.)             
            
        else:
            nac_id = vsp.AddGeom("STACK")
            vsp.SetGeomName(nac_id, tf_tag+'_'+str(ii+1))
            
            # Origin
            vsp.SetParmVal(nac_id,'X_Location','XForm',x)
            vsp.SetParmVal(nac_id,'Y_Location','XForm',y)
            vsp.SetParmVal(nac_id,'Z_Location','XForm',z)
            vsp.SetParmVal(nac_id,'Abs_Or_Relitive_flag','XForm',vsp.ABS) # misspelling from OpenVSP
            vsp.SetParmVal(nac_id,'Origin','XForm',0.5)            
            
            vsp.CutXSec(nac_id,2) # remove extra default subsurface
            xsecsurf = vsp.GetXSecSurf(nac_id,0)
            vsp.ChangeXSecShape(xsecsurf,1,vsp.XS_CIRCLE)
            vsp.ChangeXSecShape(xsecsurf,2,vsp.XS_CIRCLE)
            vsp.Update()
            vsp.SetParmVal(nac_id, "Circle_Diameter", "XSecCurve_1", width)
            vsp.SetParmVal(nac_id, "Circle_Diameter", "XSecCurve_2", width)
            vsp.SetParmVal(nac_id, "XDelta", "XSec_1", 0)
            vsp.SetParmVal(nac_id, "XDelta", "XSec_2", length)
            vsp.SetParmVal(nac_id, "XDelta", "XSec_3", 0)
            
        vsp.SetSetFlag(nac_id, OML_set_ind, True)
        
        vsp.Update()
        
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

## @ingroup Input_Output-OpenVSP
def write_wing_conformal_fuel_tank(wing, wing_id,fuel_tank,fuel_tank_set_ind):
    """This writes a conformal fuel tank in a wing.
    
    Assumptions:
    None

    Source:
    N/A

    Inputs:
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
    span              = wing.spans.projected
    
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
        area_tags[fuselage.tag] = ['fuselages',fuselage.tag]
        
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

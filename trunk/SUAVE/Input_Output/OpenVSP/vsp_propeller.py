## @ingroup Input_Output-OpenVSP
# vsp_propeller.py

# Created:  Sep 2021, R. Erhard
# Modified: 

# ----------------------------------------------------------------------
#  Imports
# ---------------------------------------------------------------------- 
import SUAVE
from SUAVE.Core import Units , Data
import vsp 
import numpy as np
import string
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.import_airfoil_geometry\
     import import_airfoil_geometry


# This enforces lowercase names
chars = string.punctuation + string.whitespace
t_table = str.maketrans( chars          + string.ascii_uppercase , 
                         '_'*len(chars) + string.ascii_lowercase )

# ----------------------------------------------------------------------
#  vsp read prop
# ----------------------------------------------------------------------

## @ingroup Input_Output-OpenVSP
def read_vsp_propeller(prop_id, units_type='SI',write_airfoil_file=True):#, number_of_radial_stations=20):
    """This reads an OpenVSP propeller geometry and writes it into a SUAVE propeller format.

    Assumptions:
    1. Written for OpenVSP 3.24.0

    Source:
    N/A

    Inputs:
    1. VSP 10-digit geom ID for prop.
    2. units_type set to 'SI' (default) or 'Imperial'.
    3. write_airfoil_file set to True (default) or False
    4. number_of_radial_stations is the radial discretization used to extract the propeller design from OpenVSP

    Outputs:
    Writes SUAVE propeller/rotor object, with these geometries, from VSP:
    	prop.
    		origin                                  [m] in all three dimensions
    		orientation				[deg] in all three dimensions
    		number_of_blades			[-]
    		tip_radius				[m]
    		hub_radius				[m]
    		twist_distribution			[deg]
    		chord_distribution			[m]
    		radius_distribution			[m]
    		sweep_distribution			[deg]
    		mid_chord_alignment			[m]
    		max_thickness_distribution		[m]
    		thickness_to_chord			[-]
    		blade_solidity				[-]
    		rotation			        [-]
    		VTOL_flag			        [-]


    		thickness_to_chord                      [-]
    		dihedral                                [radians]
    		symmetric                               <boolean>
    		tag                                     <string>
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


    # Check if this is a propeller or a lift rotor
    # Check if the thrust angle	is > 70 deg in pitch
    if vsp.GetParmVal( prop_id,'Y_Rotation','XForm') >= 70:
	# Assume lift rotor
	prop 	= SUAVE.Components.Energy.Converters.Lift_Rotor()
    else:
	# Instantiate a propeller
	prop 	= SUAVE.Components.Energy.Converters.Propeller()

    # Set the units
    if units_type == 'SI':
	units_factor = Units.meter * 1.
    elif units_type == 'imperial':
	units_factor = Units.foot * 1.
    elif units_type == 'inches':
	units_factor = Units.inch * 1.		

    # Apply a tag to the prop
    if vsp.GetGeomName(prop_id):
	tag = vsp.GetGeomName(prop_id)
	tag = tag.translate(t_table)
	prop.tag = tag
    else: 
	prop.tag = 'propgeom'

    # Propeller location (absolute)
    prop.origin 	= [[0.0,0.0,0.0]]
    prop.origin[0][0] 	= vsp.GetParmVal(prop_id, 'X_Location', 'XForm') * units_factor
    prop.origin[0][1] 	= vsp.GetParmVal(prop_id, 'Y_Location', 'XForm') * units_factor
    prop.origin[0][2] 	= vsp.GetParmVal(prop_id, 'Z_Location', 'XForm') * units_factor

    # Propeller orientation
    prop.orientation_euler_angles 	= [0.0,0.0,0.0]
    prop.orientation_euler_angles[0] 	= vsp.GetParmVal(prop_id, 'X_Rotation', 'XForm') * Units.degrees
    prop.orientation_euler_angles[1] 	= vsp.GetParmVal(prop_id, 'Y_Rotation', 'XForm') * Units.degrees
    prop.orientation_euler_angles[2] 	= vsp.GetParmVal(prop_id, 'Z_Rotation', 'XForm') * Units.degrees

    # Get the propeller parameter IDs
    parm_id    = vsp.GetGeomParmIDs(prop_id)
    parm_names = []
    for i in range(len(parm_id)):
	parm_name = vsp.GetParmName(parm_id[i])
	parm_names.append(parm_name)

    # Run the vsp Blade Element analysis
    vsp.SetStringAnalysisInput( "BladeElement" , "PropID" , (prop_id,) )
    rid = vsp.ExecAnalysis( "BladeElement" )
    Nc  = len(vsp.GetDoubleResults(rid,"YSection_000"))

    prop.number_points_around_airfoil = 2*Nc
    prop.CLi			      = vsp.GetParmVal(parm_id[parm_names.index('CLi')])
    prop.blade_solidity 	      = vsp.GetParmVal(parm_id[parm_names.index('Solidity')])
    prop.number_of_blades             = int(vsp.GetParmVal(parm_id[parm_names.index('NumBlade')]))

    prop.tip_radius                   = vsp.GetDoubleResults(rid, "Diameter" )[0] / 2 * units_factor
    prop.radius_distribution          = np.array(vsp.GetDoubleResults(rid, "Radius" )) * prop.tip_radius
    prop.radius_distribution[-1]      = 0.99 * prop.tip_radius # BEMT requires max nondimensional radius to be less than 1.0
    prop.hub_radius 		      = prop.radius_distribution[0]

    prop.chord_distribution           = np.array(vsp.GetDoubleResults(rid, "Chord" ))  * prop.tip_radius # vsp gives c/R
    prop.twist_distribution           = np.array(vsp.GetDoubleResults(rid, "Twist" ))  * Units.degrees
    prop.sweep_distribution 	      = np.array(vsp.GetDoubleResults(rid, "Sweep" ))
    prop.mid_chord_alignment          = np.tan(prop.sweep_distribution*Units.degrees)  * prop.radius_distribution
    prop.thickness_to_chord           = np.array(vsp.GetDoubleResults(rid, "Thick" ))
    prop.max_thickness_distribution   = prop.thickness_to_chord*prop.chord_distribution * units_factor
    prop.Cl_distribution              = np.array(vsp.GetDoubleResults(rid, "CLi" ))

    number_of_radial_stations        = len(prop.chord_distribution)

    # Extra data from VSP BEM for future use in BEMT
    prop.beta34 		= vsp.GetDoubleResults(rid, "Beta34" )[0]  # pitch at 3/4 radius
    prop.pre_cone 		= vsp.GetDoubleResults(rid, "Pre_Cone")[0]
    prop.rake 			= np.array(vsp.GetDoubleResults(rid, "Rake"))
    prop.skew 			= np.array(vsp.GetDoubleResults(rid, "Skew"))
    prop.axial 			= np.array(vsp.GetDoubleResults(rid, "Axial"))
    prop.tangential 		= np.array(vsp.GetDoubleResults(rid, "Tangential"))

    # Set prop rotation
    prop.rotation = 1

    # ---------------------------------------------
    # Rotor Airfoil
    # ---------------------------------------------
    if write_airfoil_file:
	print("Airfoil write not yet implemented. Defaulting to NACA 4412 airfoil for propeller cross section.") 

    return prop

## @ingroup Input_Output-OpenVSP
def write_vsp_propeller(prop, OML_set_ind):
    """This converts nacelles into OpenVSP format.
     
    N/A
    """     
    # unpack 
    prop_tag        = prop.tag 
    prop_x          = prop.origin[0][0]
    prop_y          = prop.origin[0][1]
    prop_z          = prop.origin[0][2]
    prop_x_rotation = prop.orientation_euler_angles[0]/Units.degrees    
    prop_y_rotation = prop.orientation_euler_angles[1]/Units.degrees    
    prop_z_rotation = prop.orientation_euler_angles[2]/Units.degrees   
    
 
    prop_id = vsp.AddGeom( "STACK")
    vsp.SetGeomName(prop_id,prop_tag)  

    # set nacelle relative location and rotation
    vsp.SetParmVal( prop_id,'Abs_Or_Relitive_flag','XForm',vsp.ABS)
    vsp.SetParmVal( prop_id,'X_Rotation','XForm',prop_x_rotation)
    vsp.SetParmVal( prop_id,'Y_Rotation','XForm',prop_y_rotation)
    vsp.SetParmVal( prop_id,'Z_Rotation','XForm',prop_z_rotation) 
    vsp.SetParmVal( prop_id,'X_Location','XForm',prop_x)
    vsp.SetParmVal( prop_id,'Y_Location','XForm',prop_y)
    vsp.SetParmVal( prop_id,'Z_Location','XForm',prop_z)     
    vsp.SetParmVal( prop_id,'Tess_U','Shape',radial_tesselation)
    vsp.SetParmVal( prop_id,'Tess_W','Shape',axial_tesselation)

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

    vsp.CutXSec(prop_id,4) # remove point section at end  
    vsp.CutXSec(prop_id,0) # remove point section at beginning 
    vsp.CutXSec(prop_id,1) # remove point section at beginning 
    for _ in range(num_segs-2): # add back the required number of sections
	vsp.InsertXSec(prop_id, 1, vsp.XS_ELLIPSE)          
	vsp.Update() 
    xsec_surf = vsp.GetXSecSurf(prop_id, 0 )  
    for i3 in reversed(range(num_segs)): 
	xsec = vsp.GetXSec( xsec_surf, i3 ) 
	if i3 == 0:
	    pass
	else:
	    vsp.SetParmVal(prop_id, "XDelta", "XSec_"+str(i3),x_delta[i3])
	    vsp.SetParmVal(prop_id, "ZDelta", "XSec_"+str(i3),z_delta[i3])  
	vsp.SetXSecWidthHeight( xsec, widths[i3], heights[i3])
	vsp.SetXSecTanAngles(xsec,vsp.XSEC_BOTH_SIDES,0,0,0,0)
	vsp.SetXSecTanSlews(xsec,vsp.XSEC_BOTH_SIDES,0,0,0,0)
	vsp.SetXSecTanStrengths( xsec, vsp.XSEC_BOTH_SIDES,0,0,0,0)     
	vsp.Update()          

    if ft_flag: 
	pass
    else:   
	# append front point  
	xsecsurf = vsp.GetXSecSurf(prop_id,0)
	vsp.ChangeXSecShape(xsecsurf,0,vsp.XS_POINT)
	vsp.Update()          
	xsecsurf = vsp.GetXSecSurf(prop_id,0)
	vsp.ChangeXSecShape(xsecsurf,num_segs-1,vsp.XS_POINT)
	vsp.Update()      

else: 
    prop_id = vsp.AddGeom( "BODYOFREVOLUTION")  
    vsp.SetGeomName(prop_id, prop_tag)

    # Origin 
    vsp.SetParmVal( prop_id,'Abs_Or_Relitive_flag','XForm',vsp.ABS)
    vsp.SetParmVal( prop_id,'X_Rotation','XForm',prop_x_rotation)
    vsp.SetParmVal( prop_id,'Y_Rotation','XForm',prop_y_rotation)
    vsp.SetParmVal( prop_id,'Z_Rotation','XForm',prop_z_rotation) 
    vsp.SetParmVal( prop_id,'X_Location','XForm',prop_x)
    vsp.SetParmVal( prop_id,'Y_Location','XForm',prop_y)
    vsp.SetParmVal( prop_id,'Z_Location','XForm',prop_z)  
    vsp.SetParmVal( prop_id,'Tess_U','Shape',radial_tesselation)
    vsp.SetParmVal( prop_id,'Tess_W','Shape',axial_tesselation)      

    # Length and overall diameter
    vsp.SetParmVal(prop_id,"Diameter","Design",diamater)
    if ft_flag:
	vsp.SetParmVal(prop_id,"Mode","Design",0.0)
    else:
	vsp.SetParmVal(prop_id,"Mode","Design",1.0) 

    if nacelle.naca_4_series_airfoil != None:
	if isinstance(nacelle.naca_4_series_airfoil, str) and len(nacelle.naca_4_series_airfoil) != 4:
	    raise AssertionError('Nacelle cowling airfoil must be of type < string > and length < 4 >')
	else: 
	    angle        = nacelle.cowling_airfoil_angle/Units.degrees 
	    camber       = float(nacelle.naca_4_series_airfoil[0])/100
	    camber_loc   = float(nacelle.naca_4_series_airfoil[1])/10
	    thickness    = float(nacelle.naca_4_series_airfoil[2:])/100

	    vsp.ChangeBORXSecShape(prop_id ,vsp.XS_FOUR_SERIES)
	    vsp.Update()
	    vsp.SetParmVal(prop_id,"Diameter","Design",diamater)
	    vsp.SetParmVal(prop_id,"Angle","Design",angle)
	    vsp.SetParmVal(prop_id, "Chord", "XSecCurve", length)
	    vsp.SetParmVal(prop_id, "ThickChord", "XSecCurve", thickness)
	    vsp.SetParmVal(prop_id, "Camber", "XSecCurve", camber )
	    vsp.SetParmVal(prop_id, "CamberLoc", "XSecCurve",camber_loc)  
	    vsp.Update()
    else:
	vsp.ChangeBORXSecShape(prop_id ,vsp.XS_SUPER_ELLIPSE)
	vsp.Update()
	if ft_flag:
	    vsp.SetParmVal(prop_id, "Super_Height", "XSecCurve", height) 
	    vsp.SetParmVal(prop_id,"Diameter","Design",diamater)
	else:
	    vsp.SetParmVal(prop_id, "Super_Height", "XSecCurve", diamater) 
	vsp.SetParmVal(prop_id, "Super_Width", "XSecCurve", length)
	vsp.SetParmVal(prop_id, "Super_MaxWidthLoc", "XSecCurve", 0.)
	vsp.SetParmVal(prop_id, "Super_M", "XSecCurve", 2.)
	vsp.SetParmVal(prop_id, "Super_N", "XSecCurve", 1.)  
     
    
	vsp.Update()      
    return 

## @ingroup Input_Output-OpenVSP
def vsp_read_propeller_bem(filename):
    """   This functions reads a .bem file from OpenVSP and saves it in the SUAVE propeller format
    Assumptions:
        None

    Source:
        None
    Inputs:
        OpenVSP .bem filename
    Outputs:
        SUAVE Propeller Data Structure     
    Properties Used:
        N/A
    """  
    # open newly written result files and read in aerodynamic properties 
    with open(filename,'r') as vsp_prop_file:  
	vsp_bem_lines   = vsp_prop_file.readlines()

	tag                  = vsp_bem_lines[0][0:20].strip('.')
	n_stations           = int(vsp_bem_lines[1][13:16].strip())  
	num_blades           = int(vsp_bem_lines[2][10:14].strip())  
	diameter             = float(vsp_bem_lines[3][9:21].strip()) 
	three_quarter_twist  = float(vsp_bem_lines[4][15:28].strip())   
	feather              = float(vsp_bem_lines[5][14:28].strip()) 
	precone              = float(vsp_bem_lines[6][15:28].strip()) 
	center               = list(vsp_bem_lines[7][7:44].strip().split(',')) 
	normal               = list(vsp_bem_lines[7][7:44].strip().split(','))  

	header     = 11 
	Radius_R   = np.zeros(n_stations)
	Chord_R    = np.zeros(n_stations)
	Twist_deg  = np.zeros(n_stations)
	Rake_R     = np.zeros(n_stations)
	Skew_R     = np.zeros(n_stations)
	Sweep      = np.zeros(n_stations)
	t_c        = np.zeros(n_stations)
	CLi        = np.zeros(n_stations)
	Axial      = np.zeros(n_stations)
	Tangential = np.zeros(n_stations) 

	for i in range(n_stations):
	    station       = list(vsp_bem_lines[header + i][0:120].strip().split(','))   
	    Radius_R[i]   = float(station[0])
	    Chord_R[i]    = float(station[1])
	    Twist_deg[i]  = float(station[2])
	    Rake_R[i]     = float(station[3])
	    Skew_R[i]     = float(station[4])
	    Sweep[i]      = float(station[5])
	    t_c[i]        = float(station[6])
	    CLi[i]        = float(station[7])
	    Axial[i]      = float(station[8])
	    Tangential[i] = float(station[9]) 

    # non dimensional radius cannot be 1.0 for bemt
    Radius_R[-1] = 0.99

    # unpack 
    prop = SUAVE.Components.Energy.Converters.Propeller()  
    prop.inputs                     = Data()
    prop.number_of_blades           = num_blades 
    prop.tag                        = tag 
    prop.tip_radius                 = diameter/2 
    prop.hub_radius                 = prop.tip_radius*Radius_R[0]  
    prop.design_Cl                  = np.mean(CLi)      
    prop.radius_distribution        = Radius_R*prop.tip_radius
    prop.chord_distribution         = Chord_R*prop.tip_radius
    prop.twist_distribution         = Twist_deg*Units.degrees 
    prop.mid_chord_alignment        = np.tan(Sweep*Units.degrees)*prop.radius_distribution    
    prop.thickness_to_chord         = t_c 
    prop.max_thickness_distribution = t_c*prop.chord_distribution
    prop.origin                     = [[float(center[0]) ,float(center[1]),float(center[2]) ]]
    prop.thrust_angle               = np.tan(float(normal[2])/-float(normal[0])) 
    prop.Cl_distribution            = CLi  

    return prop

## @ingroup Input_Output-OpenVSP
def write_vsp_propeller_bem(vsp_bem_filename,propeller):
    """   This functions writes a .bem file for OpenVSP
    Assumptions:
        None

    Source:
        None
    Inputs:
        OpenVSP .bem filename
        SUAVE Propeller Data Structure 
    Outputs:
        OpenVSP .bem file 
    Properties Used:
        N/A
    """    

    # unpack inputs 
    # Open the vsp_bem file after purging if it already exists
    purge_files([vsp_bem_filename]) 
    vsp_bem = open(vsp_bem_filename,'w')

    with open(vsp_bem_filename,'w') as vsp_bem:
	make_header_text(vsp_bem, propeller)

	make_section_text(vsp_bem,propeller)

	make_airfoil_text(vsp_bem,propeller)  

    return



## @ingroup Input_Output-OpenVSP
def make_header_text(vsp_bem,prop):  
    """This function writes the header of the OpenVSP .bem file
    Assumptions:
        None

    Source:
        None
    Inputs:
        vsp_bem - OpenVSP .bem file 
        prop    - SUAVE propeller data structure 

    Outputs:
        NA                
    Properties Used:
        N/A
    """      
    header_base = \
'''...{0}... 
Num_Sections: {1}
Num_Blade: {2}
Diameter: {3}
Beta 3/4 (deg): {4}
Feather (deg): 0.00000000
Pre_Cone (deg): 0.00000000
Center: {5}, {6}, {7}
Normal: {8}, {9}, {10}
''' 
    # Unpack inputs 
    name     = prop.tag
    N        = len(prop.radius_distribution)
    B        = prop.number_of_blades
    D        = prop.tip_radius*2
    beta     = np.round(prop.twist_distribution/Units.degrees,5)
    X        = prop.origin[0][0]
    Y        = prop.origin[0][1]    
    Z        = prop.origin[0][2]    
    Xn       = np.round(np.cos(np.pi- prop.thrust_angle ),5)
    Yn       = 0.0000
    Zn       = np.round(np.sin(np.pi- prop.thrust_angle ),5)

    beta_3_4  = np.interp(prop.tip_radius*0.75,prop.radius_distribution,beta)

    # Insert inputs into the template
    header_text = header_base.format(name,N,B,D,beta_3_4,X,Y,Z,Xn,Yn,Zn) 
    vsp_bem.write(header_text)    

    return   

## @ingroup Input_Output-OpenVSP
def make_section_text(vsp_bem,prop):
    """This function writes the sectional information of the propeller 
    Assumptions:
        None

    Source:
        None
    Inputs:
        vsp_bem - OpenVSP .bem file 
        prop    - SUAVE propeller data structure 

    Outputs:
        NA                                                
    Properties Used:
        N/A
    """  
    header = \
        '''Radius/R, Chord/R, Twist (deg), Rake/R, Skew/R, Sweep, t/c, CLi, Axial, Tangential\n''' 

    N          = len(prop.radius_distribution)
    r_R        = np.zeros(N)
    c_R        = np.zeros(N) 
    r_R        = prop.radius_distribution/prop.tip_radius
    c_R        = prop.chord_distribution/prop.tip_radius
    beta_deg   = prop.twist_distribution/Units.degrees 
    Rake_R     = np.zeros(N)
    Skew_R     = np.zeros(N)
    Sweep      = np.arctan(prop.mid_chord_alignment/prop.radius_distribution)
    t_c        = prop.thickness_to_chord
    CLi        = np.ones(N)*prop.design_Cl
    Axial      = np.zeros(N)
    Tangential = np.zeros(N)

    # Write propeller station imformation
    vsp_bem.write(header)       
    for i in range(N):
	section_text = format(r_R[i], '.7f')+ ", " + format(c_R[i], '.7f')+ ", " + format(beta_deg[i], '.7f')+ ", " +\
            format( Rake_R[i], '.7f')+ ", " + format(Skew_R[i], '.7f')+ ", " + format(Sweep[i], '.7f')+ ", " +\
            format(t_c[i], '.7f')+ ", " + format(CLi[i], '.7f') + ", "+ format(Axial[i], '.7f') + ", " +\
            format(Tangential[i], '.7f') + "\n"  
	vsp_bem.write(section_text)      

    return   

## @ingroup Input_Output-OpenVSP
def make_airfoil_text(vsp_bem,prop):   
    """This function writes the airfoil geometry into the vsp file
    Assumptions:
        None

    Source:
        None
    Inputs:
        vsp_bem - OpenVSP .bem file 
        prop    - SUAVE propeller data structure 

    Outputs:
        NA                
    Properties Used:
        N/A
    """ 

    N             = len(prop.radius_distribution)  
    airfoil_data  = import_airfoil_geometry(prop.airfoil_geometry)
    a_sec         = prop.airfoil_polar_stations
    for i in range(N):
	airfoil_station_header = '\nSection ' + str(i) + ' X, Y\n'  
	vsp_bem.write(airfoil_station_header)   

	airfoil_x     = airfoil_data.x_coordinates[int(a_sec[i])] 
	airfoil_y     = airfoil_data.y_coordinates[int(a_sec[i])] 

	for j in range(len(airfoil_x)): 
	    section_text = format(airfoil_x[j], '.7f')+ ", " + format(airfoil_y[j], '.7f') + "\n"  
	    vsp_bem.write(section_text)      
    return 
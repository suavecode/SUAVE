## @ingroup Input_Output-OpenVSP
# vsp_rotor.py

# Created:  Sep 2021, R. Erhard
# Modified:

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import MARC
from MARC.Core import Units , Data
import numpy as np
import scipy as sp
import string
from MARC.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.import_airfoil_geometry\
     import import_airfoil_geometry
try:
    import vsp as vsp
except ImportError:
    try:
        import openvsp as vsp
    except ImportError:
        # This allows MARC to build without OpenVSP
        pass

# This enforces lowercase names
chars = string.punctuation + string.whitespace
t_table = str.maketrans( chars          + string.ascii_uppercase ,
                         '_'*len(chars) + string.ascii_lowercase )

# ----------------------------------------------------------------------
#  vsp read prop
# ----------------------------------------------------------------------

## @ingroup Input_Output-OpenVSP
def read_vsp_rotor(prop_id, units_type='SI',write_airfoil_file=True):
    """This reads an OpenVSP rotor geometry and writes it into a MARC rotor format.

    Assumptions:
    1. Written for OpenVSP 3.24.0

    Source:
    N/A

    Inputs:
    1. VSP 10-digit geom ID for prop.
    2. units_type set to 'SI' (default) or 'Imperial'.
    3. write_airfoil_file set to True (default) or False
    4. number_of_radial_stations is the radial discretization used to extract the rotor design from OpenVSP

    Outputs:
    Writes MARC propeller/rotor object, with these geometries, from VSP:
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
    		thickness_to_chord                      [-]
                beta34                                  [radians]
                pre_cone                                [radians]
                rake                                    [radians]
                skew                                    [radians]
                axial                                   [radians]
                tangential                              [radians]
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


    # Check if this is a rotor or a lift rotor
    # Check if the thrust angle	is > 70 deg in pitch
    if vsp.GetParmVal( prop_id,'Y_Rotation','XForm') >= 70:
        # Assume lift rotor
        prop  	 = MARC.Components.Energy.Converters.Lift_Rotor() 
    else:
        # Instantiate a propeller
        prop 	= MARC.Components.Energy.Converters.Propeller() 

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


    scaling           = vsp.GetParmVal(prop_id, 'Scale', 'XForm')
    units_factor      = units_factor*scaling

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

    # Get the rotor parameter IDs
    parm_id    = vsp.GetGeomParmIDs(prop_id)
    parm_names = []
    for i in range(len(parm_id)):
        parm_name = vsp.GetParmName(parm_id[i])
        parm_names.append(parm_name)

    # Run the vsp Blade Element analysis
    vsp.SetStringAnalysisInput( "BladeElement" , "PropID" , (prop_id,) )
    rid = vsp.ExecAnalysis( "BladeElement" )
    Nc  = len(vsp.GetDoubleResults(rid,"YSection_000"))

    prop.vtk_airfoil_points           = 2*Nc
    prop.CLi                          = vsp.GetParmVal(parm_id[parm_names.index('CLi')])
    prop.blade_solidity               = vsp.GetParmVal(parm_id[parm_names.index('Solidity')])
    prop.number_of_blades             = int(vsp.GetParmVal(parm_id[parm_names.index('NumBlade')]))

    prop.tip_radius                   = vsp.GetDoubleResults(rid, "Diameter" )[0] / 2 * units_factor
    prop.radius_distribution          = np.array(vsp.GetDoubleResults(rid, "Radius" )) * prop.tip_radius

    prop.radius_distribution[-1]      = 0.99 * prop.tip_radius # BEMT requires max nondimensional radius to be less than 1.0
    if prop.radius_distribution[0] == 0.:
        start = 1
        prop.radius_distribution = prop.radius_distribution[start:]
    else:
        start = 0

    prop.hub_radius                   = prop.radius_distribution[0]

    prop.chord_distribution           = np.array(vsp.GetDoubleResults(rid, "Chord" ))[start:]  * prop.tip_radius # vsp gives c/R
    prop.twist_distribution           = np.array(vsp.GetDoubleResults(rid, "Twist" ))[start:]  * Units.degrees
    prop.sweep_distribution           = np.array(vsp.GetDoubleResults(rid, "Sweep" ))[start:]
    prop.mid_chord_alignment          = np.tan(prop.sweep_distribution*Units.degrees)  * prop.radius_distribution
    prop.thickness_to_chord           = np.array(vsp.GetDoubleResults(rid, "Thick" ))[start:]
    prop.max_thickness_distribution   = prop.thickness_to_chord*prop.chord_distribution * units_factor
    prop.Cl_distribution              = np.array(vsp.GetDoubleResults(rid, "CLi" ))[start:]

    # Extra data from VSP BEM for future use in BEVW
    prop.beta34                       = vsp.GetDoubleResults(rid, "Beta34" )[0]  # pitch at 3/4 radius
    prop.pre_cone                     = vsp.GetDoubleResults(rid, "Pre_Cone")[0]
    prop.rake                         = np.array(vsp.GetDoubleResults(rid, "Rake"))[start:]
    prop.skew                         = np.array(vsp.GetDoubleResults(rid, "Skew"))[start:]
    prop.axial                        = np.array(vsp.GetDoubleResults(rid, "Axial"))[start:]
    prop.tangential                   = np.array(vsp.GetDoubleResults(rid, "Tangential"))[start:]

    # Set prop rotation
    prop.rotation = 1

    # ---------------------------------------------
    # Rotor Airfoil
    # ---------------------------------------------
    if write_airfoil_file:
        print("Airfoil write not yet implemented. Defaulting to NACA 4412 airfoil for rotor cross section.")

    return prop

## @ingroup Input_Output-OpenVSP
def write_vsp_rotor_bem(vsp_bem_filename,rotor):
    """   This functions writes a .bem file for OpenVSP
    Assumptions:
        None

    Source:
        None
    Inputs:
        OpenVSP .bem filename
        MARC Propeller Data Structure
    Outputs:
        OpenVSP .bem file
    Properties Used:
        N/A
    """
    vsp_bem = open(vsp_bem_filename,'w')
    with open(vsp_bem_filename,'w') as vsp_bem:
        make_header_text(vsp_bem, rotor)

        make_section_text(vsp_bem,rotor)

        make_airfoil_text(vsp_bem,rotor)

    # Now import this prop
    vsp.ImportFile(vsp_bem_filename,vsp.IMPORT_BEM,'')

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
        prop    - MARC rotor data structure

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
    name      = prop.tag
    N         = len(prop.radius_distribution)
    B         = int(prop.number_of_blades)
    D         = np.round(prop.tip_radius*2,5)
    beta      = np.round(prop.twist_distribution/Units.degrees,5)
    X         = np.round(prop.origin[0][0],5)
    Y         = np.round(prop.origin[0][1],5)
    Z         = np.round(prop.origin[0][2],5)
    rotations = np.dot(prop.body_to_prop_vel(),np.array([-1,0,0])) # The sign is because props point opposite flow
    Xn        = np.round(rotations[0][0],5)
    Yn        = np.round(rotations[0][1],5)  
    Zn        = np.round(rotations[0][2],5)

    beta_3_4  = np.interp(prop.tip_radius*0.75,prop.radius_distribution,beta)

    # Insert inputs into the template
    header_text = header_base.format(name,N,B,D,beta_3_4,X,Y,Z,Xn,Yn,Zn)
    vsp_bem.write(header_text)

    return

## @ingroup Input_Output-OpenVSP
def make_section_text(vsp_bem,prop):
    """This function writes the sectional information of the rotor
    Assumptions:
        None

    Source:
        None
    Inputs:
        vsp_bem - OpenVSP .bem file
        prop    - MARC rotor data structure

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

    # Write rotor station imformation
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
        prop    - MARC rotor data structure

    Outputs:
        NA
    Properties Used:
        N/A
    """

    N             = len(prop.radius_distribution)
    airfoils      = prop.Airfoils
    a_loc         = prop.airfoil_polar_stations

    if len(airfoils.keys())>0:
        for i in range(N):
            airfoil_station_header = '\nSection ' + str(i) + ' X, Y\n'
            vsp_bem.write(airfoil_station_header)
            for j,airfoil in enumerate(airfoils): 
                if a_loc[i] == j: 
                    a_geo         = airfoil.geometry 
                    airfoil_x     = a_geo.x_coordinates 
                    airfoil_y     = a_geo.y_coordinates    
                    for k in range(len(airfoil_x)):
                        section_text = format(airfoil_x[k], '.7f')+ ", " + format(airfoil_y[k], '.7f') + "\n"
                        vsp_bem.write(section_text)                
                 
    return

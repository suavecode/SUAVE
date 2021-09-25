## @ingroup Input_Output-OpenVSP
# vsp_read_prop.py

# Created:  Jul 2021, M. Cunningham
# Modified: 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Units
from SUAVE.Components.Wings.Airfoils.Airfoil import Airfoil 
from SUAVE.Methods.Geometry.Two_Dimensional.Planform import wing_planform, wing_segmented_planform
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.compute_airfoil_polars \
     import compute_airfoil_polars

import vsp
import numpy as np
import string

# This enforces lowercase names
chars = string.punctuation + string.whitespace
t_table = str.maketrans( chars          + string.ascii_uppercase , 
                         '_'*len(chars) + string.ascii_lowercase )

# ----------------------------------------------------------------------
#  vsp read prop
# ----------------------------------------------------------------------

## @ingroup Input_Output-OpenVSP
def vsp_read_propeller(prop_id, units_type='SI',write_airfoil_file=True, number_of_radial_stations=20):
    """This reads an OpenVSP prop geometry and writes it into a SUAVE prop format.

    Assumptions:
    1. Written for OpenVSP 3.24.0

    Source:
    N/A

    Inputs:
    0. Pre-loaded VSP vehicle in memory, via vsp_read.
    1. VSP 10-digit geom ID for prop.
    2. units_type set to 'SI' (default) or 'Imperial'.
    3. write_airfoil_file set to True (default) or False
    4. number_of_radial_stations is the radial discretization used to extract the propeller design from OpenVSP
    5. number_of_chordwise_stations is the chordwise discretization (depending on design in OpenVSP, actual output may
       have more stations)
    Outputs:
    Writes SUAVE propeller/rotor object, with these geometries, from VSP:
    	prop.
    		origin                                  [m] in all three dimensions
    		orientation								[deg] in all three dimensions
    		number_of_blades						[-]
    		tip_radius								[m]
    		hub_radius								[m]
    		twist_distribution						[deg]
    		chord_distribution						[m]
    		radius_distribution						[m]
    		sweep_distribution						[deg]
    		mid_chord_alignment						[m]
    		max_thickness_distribution				[m]
    		thickness_to_chord						[-]
    		blade_solidity							[-]
    		rotation								[-]
    		VTOL_flag								[-]


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


    # Check if this is a propeller or a rotor
    # Check if the thrust angle	is > 70 deg in pitch
    if vsp.GetParmVal( prop_id,'Y_Rotation','XForm') >= 70:# Assume it is a lift rotor
        prop 				= SUAVE.Components.Energy.Converters.Lift_Rotor()
    else:
        # Instantiate a propeller
        prop 				= SUAVE.Components.Energy.Converters.Propeller()

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
    prop.origin 		= [[0.0,0.0,0.0]]
    prop.origin[0][0] 	= vsp.GetParmVal(prop_id, 'X_Location', 'XForm') * units_factor
    prop.origin[0][1] 	= vsp.GetParmVal(prop_id, 'Y_Location', 'XForm') * units_factor
    prop.origin[0][2] 	= vsp.GetParmVal(prop_id, 'Z_Location', 'XForm') * units_factor

    # Propeller orientation
    prop.orientation_euler_angles 		= [0.0,0.0,0.0]
    prop.orientation_euler_angles[0] 	= vsp.GetParmVal(prop_id, 'X_Rotation', 'XForm') * Units.degrees
    prop.orientation_euler_angles[1] 	= vsp.GetParmVal(prop_id, 'Y_Rotation', 'XForm') * Units.degrees
    prop.orientation_euler_angles[2] 	= vsp.GetParmVal(prop_id, 'Z_Rotation', 'XForm') * Units.degrees

    # Get the propeller parameter IDs
    parm_id = vsp.GetGeomParmIDs(prop_id)

    # Set number of radial and chordwise stations for running the OpenVSP BEM analysis
    number_of_chordwise_stations = 17 # chordwise discretization
    vsp.SetParmVal(parm_id[72], number_of_radial_stations)    # might be better to find the parameter by   
    vsp.SetParmVal(parm_id[73], number_of_chordwise_stations) # name than by hardcoding a specific number

    # Run the vsp Blade Element analysis
    vsp.SetStringAnalysisInput( "BladeElement" , "PropID" , (prop_id,) )
    rid = vsp.ExecAnalysis( "BladeElement" )
    len(vsp.GetDoubleResults(rid,"YSection_000")) # this does not always equal the specified number_of_chordwise_stations

    prop.CLi						= vsp.GetParmVal(parm_id[7])
    prop.blade_solidity 			= vsp.GetParmVal(parm_id[8])
    prop.number_of_blades           = int(vsp.GetParmVal(parm_id[36]))
    prop.tip_radius                 = vsp.GetDoubleResults(rid, "Diameter" )[0] / 2 * units_factor
    prop.radius_distribution        = np.array(vsp.GetDoubleResults(rid, "Radius" )) * prop.tip_radius
    prop.hub_radius 				= prop.radius_distribution[0]
    prop.radius_distribution[-1]	= 0.99 * prop.tip_radius # BEMT requires max nondimensional radius to be less than 1.0
    prop.chord_distribution         = np.array(vsp.GetDoubleResults(rid, "Chord" ))  * prop.tip_radius # vsp gives c/R
    prop.twist_distribution         = np.array(vsp.GetDoubleResults(rid, "Twist" ))  * Units.degrees
    prop.sweep_distribution 		= np.array(vsp.GetDoubleResults(rid, "Sweep" ))
    prop.mid_chord_alignment        = np.tan(prop.sweep_distribution*Units.degrees)  * prop.radius_distribution
    prop.thickness_to_chord         = np.array(vsp.GetDoubleResults(rid, "Thick" ))
    prop.max_thickness_distribution = prop.thickness_to_chord*prop.chord_distribution * units_factor

    prop.Cl_distribution            = np.array(vsp.GetDoubleResults(rid, "CLi" )) # What if this is just zeros?

    # Extra data from VSP BEM that SUAVE might not use...
    prop.beta34 					= vsp.GetDoubleResults(rid, "Beta34" )[0]  # pitch at 3/4 radius
    prop.pre_cone 					= vsp.GetDoubleResults(rid, "Pre_Cone")[0]
    prop.rake 						= np.array(vsp.GetDoubleResults(rid, "Rake"))
    prop.skew 						= np.array(vsp.GetDoubleResults(rid, "Skew"))
    prop.axial 						= np.array(vsp.GetDoubleResults(rid, "Axial"))
    prop.tangential 				= np.array(vsp.GetDoubleResults(rid, "Tangential"))

    # Prop rotation direction (need to check again)
    # 	- Propeller
    #		+1 for clockwise rotation when looking from behind
    #		-1 for counter-clockwise rotation when looking from behind
    # 	- Lift Rotor, thrust angle = 90 deg (same as propeller but rotated)
    #		+1 for right hand advancing blade
    #		-1 for left hand advancing blade
    if vsp.GetParmVal(parm_id[40]) == 1:
        prop.rotation = -1
    else:
        prop.rotation = 1


    # ---------------------------------------------
    # Rotor Airfoil
    # ---------------------------------------------
    # Use available geometry until airfoil import process (and polar data calculation) is determined
    prop.airfoil_geometry       = ['NACA_4412.txt']
    prop.airfoil_polars         = [['NACA_4412_polar_Re_50000.txt', 'NACA_4412_polar_Re_100000.txt', 
                                        'NACA_4412_polar_Re_200000.txt','NACA_4412_polar_Re_500000.txt', 
                                    'NACA_4412_polar_Re_1000000.txt']]
    prop.airfoil_polar_stations = [0] * number_of_radial_stations

    # compute airfoil polars for airfoils
    airfoil_polars              = compute_airfoil_polars(prop.airfoil_geometry, prop.airfoil_polars)
    prop.airfoil_cl_surrogates  = airfoil_polars.lift_coefficient_surrogates
    prop.airfoil_cd_surrogates  = airfoil_polars.drag_coefficient_surrogates
    prop.airfoil_flag           = True

    # Initial code (from prop_read_wing) to save the BEM airfoil files (needs work)
    # if write_airfoil_file==True:
    # 	vsp.WriteSeligAirfoil(str(prop.tag) + '_airfoil_XSec_' + str(jj) +'.dat', prop_id, float(jj/segment_num))
    # 	airfoil.coordinate_file    = str(prop.tag) + '_airfoil_XSec_' + str(jj) +'.dat'
    # 	airfoil.tag                = 'airfoil'
    #
    # Then run XFOIL for each airfoil to get the polars

    return prop



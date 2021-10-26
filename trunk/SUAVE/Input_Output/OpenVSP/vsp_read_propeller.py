## @ingroup Input_Output-OpenVSP
# vsp_read_propeller.py

# Created:  Sep 2021, R. Erhard
# Modified: 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Units
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.compute_airfoil_polars \
     import compute_airfoil_polars
import vsp
import os
import numpy as np
import string


from SUAVE.Components.Wings.Airfoils.Airfoil import Airfoil 
from SUAVE.Methods.Geometry.Two_Dimensional.Planform import wing_planform, wing_segmented_planform

# This enforces lowercase names
chars = string.punctuation + string.whitespace
t_table = str.maketrans( chars          + string.ascii_uppercase , 
                         '_'*len(chars) + string.ascii_lowercase )

# ----------------------------------------------------------------------
#  vsp read prop
# ----------------------------------------------------------------------

## @ingroup Input_Output-OpenVSP
def vsp_read_propeller(prop_id, units_type='SI',write_airfoil_file=True):#, number_of_radial_stations=20):
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



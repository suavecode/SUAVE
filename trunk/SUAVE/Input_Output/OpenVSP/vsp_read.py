## @ingroup Input_Output-OpenVSP
# vsp_read.py

# Created:  Jun 2018, T. St Francis
# Modified: Aug 2018, T. St Francis
#           Jan 2020, T. MacDonald
#           Jul 2020, E. Botero
#           Sep 2021, R. Erhard

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Input_Output.OpenVSP.vsp_propeller import read_vsp_propeller
from SUAVE.Input_Output.OpenVSP.vsp_fuselage  import read_vsp_fuselage
from SUAVE.Input_Output.OpenVSP.vsp_wing      import read_vsp_wing
from SUAVE.Input_Output.OpenVSP.vsp_nacelle   import read_vsp_nacelle

from SUAVE.Components.Energy.Networks.Lift_Cruise              import Lift_Cruise
from SUAVE.Components.Energy.Networks.Battery_Propeller        import Battery_Propeller

from SUAVE.Core import Units, Data
try:
    import vsp as vsp
except ImportError:
    # This allows SUAVE to build without OpenVSP
    pass


# ----------------------------------------------------------------------
#  vsp read
# ----------------------------------------------------------------------


## @ingroup Input_Output-OpenVSP
def vsp_read(tag, units_type='SI',specified_network=None): 
    """This reads an OpenVSP vehicle geometry and writes it into a SUAVE vehicle format.
    Includes wings, fuselages, and propellers.

    Assumptions:
    1. OpenVSP vehicle is composed of conventionally shaped fuselages, wings, and propellers. 
    1a. OpenVSP fuselage: generally narrow at nose and tail, wider in center). 
    1b. Fuselage is designed in VSP as it appears in real life. That is, the VSP model does not rely on
       superficial elements such as canopies, stacks, or additional fuselages to cover up internal lofting oddities.
    1c. This program will NOT account for multiple geometries comprising the fuselage. For example: a wingbox mounted beneath
       is a separate geometry and will NOT be processed.
    2. Fuselage origin is located at nose. VSP file origin can be located anywhere, preferably at the forward tip
       of the vehicle or in front (to make all X-coordinates of vehicle positive).
    3. Written for OpenVSP 3.21.1

    Source:
    N/A

    Inputs:
    1. A tag for an XML file in format .vsp3.
    2. Units_type set to 'SI' (default) or 'Imperial'
    3. User-specified network

    Outputs:
    Writes SUAVE vehicle with these geometries from VSP:    (All values default to SI. Any other 2nd argument outputs Imperial.)
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

    	Propellers.Propeller.
    		location[X,Y,Z]                            [radians]
    		rotation[X,Y,Z]                            [radians]
    		tip_radius                                 [m]
    	        hub_radius                                 [m]
    		thrust_angle                               [radians]

    Properties Used:
    N/A
    """  	

    vsp.ClearVSPModel() 
    vsp.ReadVSPFile(tag)	

    vsp_fuselages     = []
    vsp_wings         = []	
    vsp_props         = [] 
    vsp_nacelles      = [] 
    vsp_nacelle_type  = []
    
    vsp_geoms         = vsp.FindGeoms()
    geom_names        = []

    vehicle           = SUAVE.Vehicle()
    vehicle.tag       = tag 

    if units_type == 'SI':
        units_type = 'SI' 
    elif units_type == 'inches':
        units_type = 'inches'	
    else:
        units_type = 'imperial'	

    # The two for-loops below are in anticipation of an OpenVSP API update with a call for GETGEOMTYPE.
    # This print function allows user to enter VSP GeomID manually as first argument in vsp_read functions.

    print("VSP geometry IDs: ")	

    # Label each geom type by storing its VSP geom ID. 

    for geom in vsp_geoms: 
        geom_name = vsp.GetGeomName(geom)
        geom_names.append(geom_name)
        print(str(geom_name) + ': ' + geom)

    # --------------------------------
    # AUTOMATIC VSP ENTRY & PROCESSING
    # --------------------------------		

    for geom in vsp_geoms:
        geom_name = vsp.GetGeomName(geom)
        geom_type = vsp.GetGeomTypeName(str(geom))

        if geom_type == 'Fuselage':
            vsp_fuselages.append(geom)
        if geom_type == 'Wing':
            vsp_wings.append(geom)
        if geom_type == 'Propeller':
            vsp_props.append(geom) 
        if (geom_type == 'Stack') or (geom_type == 'BodyOfRevolution'):
            vsp_nacelle_type.append(geom_type)
            vsp_nacelles.append(geom) 
        
    # --------------------------------------------------			
    # Read Fuselages 
    # --------------------------------------------------			    
    for fuselage_id in vsp_fuselages:
        sym_planar = vsp.GetParmVal(fuselage_id, 'Sym_Planar_Flag', 'Sym') # Check for symmetry
        sym_origin = vsp.GetParmVal(fuselage_id, 'Sym_Ancestor_Origin_Flag', 'Sym') 
        if sym_planar == 2. and sym_origin == 1.:  
            num_fus  = 2 
            sym_flag = [1,-1]
        else: 
            num_fus  = 1 
            sym_flag = [1] 
        for fux_idx in range(num_fus):	# loop through fuselages on aircraft 
            fuselage = read_vsp_fuselage(fuselage_id,fux_idx,sym_flag[fux_idx],units_type)
            vehicle.append_component(fuselage)
        
    # --------------------------------------------------			    
    # Read Wings 
    # --------------------------------------------------			
    for wing_id in vsp_wings:
        wing = read_vsp_wing(wing_id, units_type)
        vehicle.append_component(wing)		 
        
    # --------------------------------------------------			    
    # Read Nacelles 
    # --------------------------------------------------			
    for nac_id, nacelle_id in enumerate(vsp_nacelles):
        nacelle = read_vsp_nacelle(nacelle_id,vsp_nacelle_type[nac_id], units_type)
        vehicle.append_component(nacelle)	  
    
    # --------------------------------------------------			    
    # Read Propellers/Rotors and assign to a network
    # --------------------------------------------------			
    # Initialize rotor network elements
    number_of_lift_rotor_engines = 0
    number_of_propeller_engines  = 0
    lift_rotors = Data()
    propellers  = Data() 
    for prop_id in vsp_props:
        prop = read_vsp_propeller(prop_id,units_type)
        prop.tag = vsp.GetGeomName(prop_id)
        if prop.orientation_euler_angles[1] >= 70 * Units.degrees:
            lift_rotors.append(prop)
            number_of_lift_rotor_engines += 1 
        else:
            propellers.append(prop)
            number_of_propeller_engines += 1  

    if specified_network == None:
        # If no network specified, assign a network
        if number_of_lift_rotor_engines>0 and number_of_propeller_engines>0:
            net = Lift_Cruise()
        else:
            net = Battery_Propeller() 
    else:
        net = specified_network

    # Create the rotor network
    if net.tag == "Lift_Cruise":
        # Lift + Cruise network
        for i in range(number_of_lift_rotor_engines):
            net.lift_rotors.append(lift_rotors[list(lift_rotors.keys())[i]])
        net.number_of_lift_rotor_engines = number_of_lift_rotor_engines	

        for i in range(number_of_propeller_engines):
            net.propellers.append(propellers[list(propellers.keys())[i]])
        net.number_of_propeller_engines = number_of_propeller_engines		

    elif net.tag == "Battery_Propeller":
        # Append all rotors as propellers for the battery propeller network
        for i in range(number_of_lift_rotor_engines):
            # Accounts for multicopter configurations
            net.propellers.append(lift_rotors[list(lift_rotors.keys())[i]])

        for i in range(number_of_propeller_engines):
            net.propellers.append(propellers[list(propellers.keys())[i]])

        net.number_of_propeller_engines = number_of_lift_rotor_engines + number_of_propeller_engines	

    vehicle.networks.append(net)

    return vehicle
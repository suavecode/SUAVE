## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
# generate_vortex_distribution.py
# 
# Created:  May 2018, M. Clarke
# Modified: Apr 2020, M. Clarke
#           Jun 2021, A. Blaufox

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# package imports 
import jax.numpy as jnp
import numpy as np
from jax import jit, lax
from jax.lax import dynamic_update_slice as DUS

from SUAVE.Core import  Data
from SUAVE.Components.Wings import All_Moving_Surface
from SUAVE.Components.Fuselages import Fuselage
from SUAVE.Components.Nacelles  import Nacelle
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.make_VLM_wings import make_VLM_wings
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.import_airfoil_geometry\
     import import_airfoil_geometry

# ----------------------------------------------------------------------
#  Generate Vortex Distribution
# ----------------------------------------------------------------------
## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
@jit
def generate_vortex_distribution(geometry,settings):
    ''' Compute the coordinates of panels, vortices , control points
    and geometry used to build the influence coefficient matrix. A 
    different discretization (n_sw and n_cw) may be defined for each type
    of major section (wings and fuselages). 
    
    Control surfaces are modelled as wings, but adapt their panel density 
    to that of the area in which they reside on their own wing.   

    Assumptions: 
    Below is a schematic of the coordinates of an arbitrary panel  
    
    XA1 ____________________________ XB1    
       |                            |
       |        bound vortex        |
    XAH|  ________________________  |XBH
       | |           XCH          | |
       | |                        | |
       | |                        | |     
       | |                        | |
       | |                        | |
       | |           0 <--control | |       
       | |          XC     point  | |  
       | |                        | |
   XA2 |_|________________________|_|XB2
         |                        |     
         |       trailing         |  
         |   <--  vortex   -->    |  
         |         legs           | 
             
    
    In addition, all control surfaces should be appended directly
       to the wing, not the wing segments    
    
    For control surfaces, "positve" deflection corresponds to the RH rule where the axis of rotation is the OUTBOARD-pointing hinge vector
    symmetry: the LH rule is applied to the reflected surface for non-ailerons. Ailerons follow a RH rule for both sides
    
    Source:  
    None

    Inputs:
    geometry.wings                                [Unitless]  
    settings.floating_point_precision             [np.dtype]
    
    Of the following settings, the user should define either the number_ atrributes or the wing_ and fuse_ attributes.
    settings.number_spanwise_vortices             - a base number of vortices to be applied to both wings and fuselages
    settings.number_chordwise_vortices            - a base number of vortices to be applied to both wings and fuselages
    settings.wing_spanwise_vortices               - the number of vortices to be applied to only the wings
    settings.wing_chordwise_vortices              - the number of vortices to be applied to only the wings
    settings.fuselage_spanwise_vortices           - the number of vortices to be applied to only the fuslages
    settings.fuselage_chordwise_vortices          - the number of vortices to be applied to only the fuselages 
       
    Outputs:                                   
    VD - vehicle vortex distribution              [Unitless] 

    Properties Used:
    N/A 
         
    '''
    # ---------------------------------------------------------------------------------------
    # STEP 0: Unpack settings
    # ---------------------------------------------------------------------------------------        
    #unpack other settings----------------------------------------------------
    spc            = settings.spanwise_cosine_spacing
    model_fuselage = settings.model_fuselage
    model_nacelle  = settings.model_nacelle
    precision      = settings.floating_point_precision
    
    show_prints    = settings.verbose if ('verbose' in settings.keys()) else False
    
    # unpack discretization settings------------------------------------------
    n_sw_global    = settings.number_spanwise_vortices  
    n_cw_global    = settings.number_chordwise_vortices
    n_sw_wing      = settings.wing_spanwise_vortices  
    n_cw_wing      = settings.wing_chordwise_vortices
    n_sw_fuse      = settings.fuselage_spanwise_vortices  
    n_cw_fuse      = settings.fuselage_chordwise_vortices
    
    #make sure n_cw and n_sw are both defined or not defined
    invalid_global_n   = bool(n_sw_global) != bool(n_cw_global) 
    invalid_wing_n     = bool(n_sw_wing)   != bool(n_cw_wing)
    invalid_fuse_n     = bool(n_sw_fuse)   != bool(n_cw_fuse)
    invalid_separate_n = bool(n_sw_wing)   != bool(n_sw_fuse)
    if invalid_global_n:
        raise AssertionError('If using global surface discretization, both n_sw and n_cw must be defined')
    elif invalid_wing_n or invalid_fuse_n:
        raise AssertionError('If using separate surface discretization, all n_sw and n_cw values must be defined')
    elif invalid_separate_n:
        raise AssertionError('If using separate surface discretization, both wing and fuselage discretization must be defined')
    
    #make sure that global and separate settings aren't both defined
    global_n_defined   = bool(n_sw_global) 
    separate_n_defined = bool(n_sw_wing)
    if global_n_defined == separate_n_defined:
        raise AssertionError('Specify either global or separate discretization')
    elif global_n_defined:
        n_sw_wing = n_sw_global
        n_cw_wing = n_cw_global
        n_sw_fuse = n_sw_global
        n_cw_fuse = n_cw_global
    else: #separate_n_defined
        #everything is already set up to use separate discretization
        pass
    
    # ---------------------------------------------------------------------------------------
    # STEP 0.5: Setup High level parameters
    # ---------------------------------------------------------------------------------------    
    VD = Data()
    
    
    # unpack geometry----------------------------------------------------------------
    # define point about which moment coefficient is computed
    if 'main_wing' in geometry.wings:
        c_bar      = geometry.wings['main_wing'].chords.mean_aerodynamic
        x_mac      = geometry.wings['main_wing'].aerodynamic_center[0] + geometry.wings['main_wing'].origin[0][0]
        z_mac      = geometry.wings['main_wing'].aerodynamic_center[2] + geometry.wings['main_wing'].origin[0][2]
        w_span     = geometry.wings['main_wing'].spans.projected
    else:
        c_bar  = 0.
        x_mac  = 0.
        w_span = 0.
        for wing in geometry.wings:
            if wing.vertical == False:
                if c_bar <= wing.chords.mean_aerodynamic:
                    c_bar  = wing.chords.mean_aerodynamic
                    x_mac  = wing.aerodynamic_center[0] + wing.origin[0][0]
                    z_mac  = wing.aerodynamic_center[2] + wing.origin[0][2]
                    w_span = wing.spans.projected

    x_cg       = geometry.mass_properties.center_of_gravity[0][0]
    z_cg       = geometry.mass_properties.center_of_gravity[0][2]
    
    # Do boolean math instead of an if statement for moment locations
    bool_cg = x_cg == 0.0
    
    x_m = bool_cg*x_mac + (1-bool_cg)*x_cg
    z_m = bool_cg*z_mac + (1-bool_cg)*z_cg

    VD.x_m    = x_m
    VD.z_m    = z_m
    VD.w_span = w_span
    VD.c_bar  = c_bar
    
    # ---------------------------------------------------------------------------------------
    # STEP 1: Define empty vectors for coordinates of panes, control points and bound vortices
    # ---------------------------------------------------------------------------------------

    VD.XAH    = jnp.empty(shape=[0,1], dtype=precision)
    VD.YAH    = jnp.empty(shape=[0,1], dtype=precision)
    VD.ZAH    = jnp.empty(shape=[0,1], dtype=precision)
    VD.XBH    = jnp.empty(shape=[0,1], dtype=precision)
    VD.YBH    = jnp.empty(shape=[0,1], dtype=precision)
    VD.ZBH    = jnp.empty(shape=[0,1], dtype=precision)
    VD.XCH    = jnp.empty(shape=[0,1], dtype=precision)
    VD.YCH    = jnp.empty(shape=[0,1], dtype=precision)
    VD.ZCH    = jnp.empty(shape=[0,1], dtype=precision)     
    VD.XA1    = jnp.empty(shape=[0,1], dtype=precision)
    VD.YA1    = jnp.empty(shape=[0,1], dtype=precision)  
    VD.ZA1    = jnp.empty(shape=[0,1], dtype=precision)
    VD.XA2    = jnp.empty(shape=[0,1], dtype=precision)
    VD.YA2    = jnp.empty(shape=[0,1], dtype=precision)    
    VD.ZA2    = jnp.empty(shape=[0,1], dtype=precision)    
    VD.XB1    = jnp.empty(shape=[0,1], dtype=precision)
    VD.YB1    = jnp.empty(shape=[0,1], dtype=precision)  
    VD.ZB1    = jnp.empty(shape=[0,1], dtype=precision)
    VD.XB2    = jnp.empty(shape=[0,1], dtype=precision)
    VD.YB2    = jnp.empty(shape=[0,1], dtype=precision)    
    VD.ZB2    = jnp.empty(shape=[0,1], dtype=precision)     
    VD.XAC    = jnp.empty(shape=[0,1], dtype=precision)
    VD.YAC    = jnp.empty(shape=[0,1], dtype=precision)
    VD.ZAC    = jnp.empty(shape=[0,1], dtype=precision) 
    VD.XBC    = jnp.empty(shape=[0,1], dtype=precision)
    VD.YBC    = jnp.empty(shape=[0,1], dtype=precision)
    VD.ZBC    = jnp.empty(shape=[0,1], dtype=precision) 
    VD.XC_TE  = jnp.empty(shape=[0,1], dtype=precision)
    VD.YC_TE  = jnp.empty(shape=[0,1], dtype=precision)
    VD.ZC_TE  = jnp.empty(shape=[0,1], dtype=precision)     
    VD.XA_TE  = jnp.empty(shape=[0,1], dtype=precision)
    VD.YA_TE  = jnp.empty(shape=[0,1], dtype=precision)
    VD.ZA_TE  = jnp.empty(shape=[0,1], dtype=precision) 
    VD.XB_TE  = jnp.empty(shape=[0,1], dtype=precision)
    VD.YB_TE  = jnp.empty(shape=[0,1], dtype=precision)
    VD.ZB_TE  = jnp.empty(shape=[0,1], dtype=precision)  
    VD.XC     = jnp.empty(shape=[0,1], dtype=precision)
    VD.YC     = jnp.empty(shape=[0,1], dtype=precision)
    VD.ZC     = jnp.empty(shape=[0,1], dtype=precision)    
    VD.FUS_XC = jnp.empty(shape=[0,1], dtype=precision)
    VD.FUS_YC = jnp.empty(shape=[0,1], dtype=precision)
    VD.FUS_ZC = jnp.empty(shape=[0,1], dtype=precision)      
    VD.CS     = jnp.empty(shape=[0,1], dtype=precision) 
    VD.X      = jnp.empty(shape=[0,1], dtype=precision)
    VD.Y      = jnp.empty(shape=[0,1], dtype=precision)
    VD.Z      = jnp.empty(shape=[0,1], dtype=precision)
    VD.Y_SW   = jnp.empty(shape=[0,1], dtype=precision)
    VD.DY     = jnp.empty(shape=[0,1], dtype=precision) 

    # empty vectors necessary for arbitrary discretization dimensions
    VD.n_w              = 0                            # number of wings counter (refers to wings, fuselages or other structures)  
    VD.n_cp             = 0                            # number of bound vortices (panels) counter 
    VD.n_sw             =  np.array([], dtype=jnp.int16) # array of the number of spanwise  strips in each wing
    VD.n_cw             = jnp.array([], dtype=jnp.int16) # array of the number of chordwise panels per strip in each wing
    VD.chordwise_breaks = jnp.array([], dtype=jnp.int32) # indices of the first panel in every strip      (given a list of all panels)
    VD.spanwise_breaks  = jnp.array([], dtype=jnp.int32) # indices of the first strip of panels in a wing (given chordwise_breaks)    
    VD.symmetric_wings  = jnp.array([], dtype=jnp.int32)
    
    VD.leading_edge_indices      = jnp.array([], dtype=bool)      # bool array of leading  edge indices (all false except for panels at leading  edge)
    VD.trailing_edge_indices     = jnp.array([], dtype=bool)      # bool array of trailing edge indices (all false except for panels at trailing edge)    
    VD.panels_per_strip          = jnp.array([], dtype=jnp.int16) # array of the number of panels per strip (RNMAX); this is assigned for all panels  
    VD.chordwise_panel_number    = jnp.array([], dtype=jnp.int16) # array of panels' numbers in their strips.     
    VD.chord_lengths             = jnp.array([], dtype=precision) # Chord length, this is assigned for all panels.
    VD.tangent_incidence_angle   = jnp.array([], dtype=precision) # Tangent Incidence Angles of the chordwise strip. LE to TE, ZETA
    VD.exposed_leading_edge_flag = jnp.array([], dtype=jnp.int16) # 0 or 1 per strip. 0 turns off leading edge suction for non-slat control surfaces
    
    # ---------------------------------------------------------------------------------------
    # STEP 2: Unpack aircraft wing geometry 
    # ---------------------------------------------------------------------------------------    
    VD.wing_areas  = [] # instantiate wing areas
    VD.vortex_lift = []
    
    #reformat/preprocess wings and control surfaces for VLM panelization
    VLM_wings = make_VLM_wings(geometry, settings)
    VD.VLM_wings = VLM_wings
    
    #generate panelization for each wing. Wings first, then control surface wings
    for wing in VD.VLM_wings:
        if not wing.is_a_control_surface:
            if show_prints: print('discretizing ' + wing.tag) 
            VD = generate_wing_vortex_distribution(VD,wing,n_cw_wing,n_sw_wing,spc,precision)            
                    
    for wing in VD.VLM_wings:
        if wing.is_a_control_surface:
            if show_prints:print('discretizing ' + wing.tag)
            VD = generate_wing_vortex_distribution(VD,wing,n_cw_wing,n_sw_wing,spc,precision)     
                    
    # ---------------------------------------------------------------------------------------
    # STEP 8: Unpack aircraft nacelle geometry
    # ---------------------------------------------------------------------------------------      
    VD.wing_areas = jnp.array(VD.wing_areas, dtype=precision)
    VD.n_fus      = 0
    for nac in geometry.nacelles:
        if show_prints: print('discretizing ' + nac.tag)
        VD = generate_fuselage_and_nacelle_vortex_distribution(VD,nac,n_cw_fuse,n_sw_fuse,precision,model_nacelle).s

    # ---------------------------------------------------------------------------------------
    # STEP 9: Unpack aircraft fuselage geometry
    # ---------------------------------------------------------------------------------------
    VD.wing_areas = jnp.array(VD.wing_areas, dtype=precision)
    for fus in geometry.fuselages:
        if show_prints: print('discretizing ' + fus.tag)
        VD = generate_fuselage_and_nacelle_vortex_distribution(VD,fus,n_cw_fuse,n_sw_fuse,precision,model_fuselage)

    # ---------------------------------------------------------------------------------------
    # STEP 10: Postprocess VD information
    # ---------------------------------------------------------------------------------------   
    
    total_sw = np.sum(VD.n_sw)

    VD['leading_edge_indices'] = LE_ind = jnp.where(VD['leading_edge_indices'],size=total_sw)

    # Compute Panel Areas and Normals
    VD.panel_areas = jnp.array(compute_panel_area(VD) , dtype=precision)
    VD.normals     = jnp.array(compute_unit_normal(VD), dtype=precision)  
    
    # Reshape chord_lengths
    VD.chord_lengths = jnp.atleast_2d(VD.chord_lengths) #need to be 2D for later calculations
    
    # Compute variables used in VORLAX
    X1c   = (VD.XA1+VD.XB1)/2
    X2c   = (VD.XA2+VD.XB2)/2
    Z1c   = (VD.ZA1+VD.ZB1)/2
    Z2c   = (VD.ZA2+VD.ZB2)/2
    SLOPE = (Z2c - Z1c)/(X2c - X1c)
    SLE   = SLOPE[LE_ind]   
    D     = jnp.sqrt((VD.YAH-VD.YBH)**2+(VD.ZAH-VD.ZBH)**2)[LE_ind]
    
    # Pack VORLAX variables
    VD.SLOPE = SLOPE
    VD.SLE   = SLE
    VD.D     = D
    
    # Do some final calculations for segmented breaks
    chord_arange   = jnp.arange(0,len(VD.chordwise_breaks))
    chord_breaksp1 = jnp.hstack((VD.chordwise_breaks,VD.n_cp))
    chord_repeats  = jnp.diff(chord_breaksp1)
    chord_segs     = jnp.repeat(chord_arange,chord_repeats,total_repeat_length=VD.n_cp)    
    
    span_arange   = jnp.arange(0,len(VD.spanwise_breaks))
    span_breaksp1 = jnp.hstack((VD.spanwise_breaks,sum(VD.n_sw)))
    span_repeats  = jnp.diff(span_breaksp1)
    span_segs     = jnp.repeat(span_arange,span_repeats,total_repeat_length=total_sw)    
        
    VD.chord_segs = chord_segs
    VD.span_segs  = span_segs
    VD.total_sw   = total_sw
    
    # Compute X and Z BAR ouside of generate_vortex_distribution to avoid requiring x_m and z_m as inputs
    VD.XBAR  = jnp.ones(total_sw) * x_m
    VD.ZBAR  = jnp.ones(total_sw) * z_m     
    VD.stripwise_panels_per_strip = VD.panels_per_strip[VD.leading_edge_indices]
    
    # For JAX some things have to be fixed
    VD['n_sw'] = tuple(VD['n_sw'])
    VD.static_keys  = ['n_sw','total_sw','n_w']
    
    if show_prints: print('finish discretization')            
    
    return VD 

# ----------------------------------------------------------------------
#  Discretize Wings
# ----------------------------------------------------------------------
## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
def generate_wing_vortex_distribution(VD,wing,n_cw,n_sw,spc,precision):
    """ This generates vortex distribution points for the given wing 

    Assumptions: 
    The wing is segmented and was made or modified by make_VLM_wings()
    
    For control surfaces, "positve" deflection corresponds to the RH rule where the axis of rotation is the OUTBOARD-pointing hinge vector
    symmetry: the LH rule is applied to the reflected surface for non-ailerons. Ailerons follow a RH rule for both sides
    
    The hinge_vector will only ever be calculated on the first strip of any control/all-moving surface. It is assumed that all control
    surfaces are trapezoids, thus needing only one hinge, and that all all-moving surfaces have exactly one point of rotation.

    Source:   
    None
    
    Inputs:   
    VD                   - vortex distribution
    wing                 - a Data object made or modified by make_VLM_wings() to mimick a Wing object
    
    Properties Used:
    N/A
    """ 
    wings = VD.VLM_wings
    
    # get geometry of wing  
    span          = wing.spans.projected
    sym_para      = wing.symmetric 
    vertical_wing = wing.vertical
    wing_origin   = wing.origin[0]
    VD.vortex_lift.append(wing.vortex_lift)

    # determine if vehicle has symmetry 
    if sym_para is True :
        span = span/2
        VD.vortex_lift.append(wing.vortex_lift)

    # ---------------------------------------------------------------------------------------
    # STEP 3: Get discretization control variables  
    # ---------------------------------------------------------------------------------------
    # get number of spanwise and chordwise panels for this wing
    n_sw = n_sw if (not wing.is_a_control_surface) else max(len(wing.y_coords_required)-1,1)
    n_cw = n_cw if (not wing.is_a_control_surface) else max(int(jnp.ceil(wing.chord_fraction*n_cw)),2)  
    
    # get y_coordinates (y-locations of the edges of each strip in wing-local coords)
    if spc == True: # discretize wing using cosine spacing     
        n               = jnp.linspace(n_sw+1,0,n_sw+1)         # vectorize
        thetan          = n*(jnp.pi/2)/(n_sw+1)                 # angular stations
        y_coordinates   = span*jnp.cos(thetan)                  # y locations based on the angular spacing
    else:           # discretize wing using linear spacing 
        y_coordinates   = jnp.linspace(0,span,n_sw+1) 

    # get span_breaks object
    span_breaks   = wing.span_breaks
    n_breaks      = len(span_breaks)

    # ---------------------------------------------------------------------------------------
    # STEP 4: Setup span_break and section arrays. A 'section' is the trapezoid between two
    #         span_breaks. A 'span_break' is described in the file make_VLM_wings.py
    # ---------------------------------------------------------------------------------------
    break_chord       = jnp.zeros(n_breaks)
    break_twist       = jnp.zeros(n_breaks)
    break_sweep       = jnp.zeros(n_breaks)
    break_dihedral    = jnp.zeros(n_breaks)
    break_camber_xs   = [] 
    break_camber_zs   = []
    break_x_offset    = jnp.zeros(n_breaks)
    break_z_offset    = jnp.zeros(n_breaks)
    break_spans       = jnp.zeros(n_breaks) 
    section_span      = jnp.zeros(n_breaks)
    section_area      = jnp.zeros(n_breaks)
    section_LE_cut    = jnp.zeros(n_breaks)
    section_TE_cut    = jnp.ones(n_breaks)
    span_breaks_cs_ID = []

    # ---------------------------------------------------------------------------------------
    # STEP 5:  Obtain sweep, chord, dihedral and twist at the beginning/end of each break.
    #          If applicable, append airfoil section VD and flap/aileron deflection angles.
    # --------------------------------------------------------------------------------------- 
    for i_break in range(n_breaks):   
        break_spans    = break_spans.at[i_break]   .set(span_breaks[i_break].span_fraction*span)
        break_chord    = break_chord.at[i_break]   .set(span_breaks[i_break].local_chord)
        break_twist    = break_twist.at[i_break]   .set(span_breaks[i_break].twist)
        break_dihedral = break_dihedral.at[i_break].set(span_breaks[i_break].dihedral_outboard)                    

        # get leading edge sweep. make_VLM wings should have precomputed this for all span_breaks
        is_not_last_break    = (i_break != n_breaks-1)
        break_sweep = break_sweep.at[i_break].set(span_breaks[i_break].sweep_outboard_LE if is_not_last_break else 0)

        # find span and area. All span_break offsets should be calculated in make_VLM_wings
        if i_break == 0:
            section_span   = section_span.at[i_break]  .set(0.0)
            break_x_offset = break_x_offset.at[i_break].set(0.0) 
            break_z_offset = break_z_offset.at[i_break].set(0.0)       
        else:
            section_span   = section_span.at[i_break]  .set(break_spans[i_break] - break_spans[i_break-1])
            section_area   = section_area.at[i_break]  .set(0.5*(break_chord[i_break-1] + break_chord[i_break])*section_span[i_break])
            break_x_offset = break_x_offset.at[i_break].set(span_breaks[i_break].x_offset)
            break_z_offset = break_z_offset.at[i_break].set(span_breaks[i_break].dih_offset)

        # Get airfoil section VD  
        if span_breaks[i_break].Airfoil: 
            airfoil_data = import_airfoil_geometry([span_breaks[i_break].Airfoil.airfoil.coordinate_file])    
            break_camber_zs.append(airfoil_data.camber_coordinates[0])
            break_camber_xs.append(airfoil_data.x_lower_surface[0]) 
        else:
            break_camber_zs.append(jnp.zeros(30))              
            break_camber_xs.append(jnp.linspace(0,1,30)) 
            
        span_breaks_cs_ID.append(span_breaks[i_break].cs_IDs[0,1])

        # Get control surface leading and trailing edge cute cuts: section__cuts[-1] should never be used in the following code
        section_LE_cut = section_LE_cut.at[i_break].set(span_breaks[i_break].cuts[0,1])
        section_TE_cut = section_TE_cut.at[i_break].set(span_breaks[i_break].cuts[1,1])
        
    # Make break_cambers jnp arrays
    break_camber_xs   = jnp.array(break_camber_xs)
    break_camber_zs   = jnp.array(break_camber_zs)
    span_breaks_cs_ID = jnp.array(span_breaks_cs_ID)

    VD.wing_areas.append(jnp.sum(section_area[:], dtype=precision))
    if sym_para is True :
        VD.wing_areas.append(jnp.sum(section_area[:], dtype=precision))            

    #Shift spanwise vortices onto section breaks  
    if len(y_coordinates) < n_breaks:
        raise ValueError('Not enough spanwise VLM stations for segment breaks')

    y_coords_required = break_spans if (not wing.is_a_control_surface) else jnp.array(sorted(wing.y_coords_required))  #control surfaces have additional required y_coords  
    shifted_idxs = jnp.zeros(len(y_coordinates))
    for y_req in y_coords_required:
        idx = (jnp.abs(y_coordinates - y_req) + shifted_idxs).argmin() #index of y-coord nearest to the span break
        shifted_idxs  = shifted_idxs.at[idx].set(jnp.inf)
        y_coordinates = y_coordinates.at[idx].set(y_req)

    y_coordinates = y_coordinates.sort()
    
    # ---------------------------------------------------------------------------------------
    # STEP 6: Define coordinates of panels horseshoe vortices and control points 
    # --------------------------------------------------------------------------------------- 
    y_a   = y_coordinates[:-1] 
    y_b   = y_coordinates[1:]             
    del_y = y_coordinates[1:] - y_coordinates[:-1] 
    
    
    # Let relevant control surfaces know which y-coords they are required to have----------------------------------
    if not wing.is_a_control_surface:
        for span_break in span_breaks:
            cs_IDs     = span_break.cs_IDs[:,1] #only the outboard control surfaces
    
            for cs_ID in cs_IDs[cs_IDs >= 0]:
                cs_tag     = wing.tag + '__cs_id_{}'.format(cs_ID)
                cs_wing    = wings[cs_tag]
                rel_offset = cs_wing.origin[0,1] - wing.origin[0][1] if not vertical_wing else cs_wing.origin[0,2] - wing.origin[0][2]
    
                rel_wing_ys = y_coordinates - rel_offset
    
                halfspan_size        = cs_wing.spans.projected * 0.5 if wing.symmetric else cs_wing.spans.projected
                directional_halfspan = jnp.sign(jnp.cos(cs_wing.dihedral))*halfspan_size #account for dihedral in quadrants II and III
    
                l_bound = jnp.minimum(0, directional_halfspan)
                r_bound = jnp.maximum(0, directional_halfspan)
    
                cs_wing.y_coords_required = rel_wing_ys[(l_bound<=rel_wing_ys) & (rel_wing_ys<=r_bound)]        
    
    
    # -------------------------------------------------------------------------------------------------------------
    # Run the strip contruction loop again if wing is symmetric. 
    # Reflection plane = x-y plane for vertical wings. Otherwise, reflection plane = x-z plane
    signs         = jnp.array([1, -1]) # acts as a multiplier for symmetry. -1 is only ever used for symmetric wings
    symmetry_mask = np.array([True,sym_para])
    for sym_sign_ind, sym_sign in enumerate(signs[symmetry_mask]):
        # create empty vectors for coordinates 
        xah   = jnp.zeros(n_cw*n_sw)
        yah   = jnp.zeros(n_cw*n_sw)
        zah   = jnp.zeros(n_cw*n_sw)
        xbh   = jnp.zeros(n_cw*n_sw)
        ybh   = jnp.zeros(n_cw*n_sw)
        zbh   = jnp.zeros(n_cw*n_sw)    
        xch   = jnp.zeros(n_cw*n_sw)
        ych   = jnp.zeros(n_cw*n_sw)
        zch   = jnp.zeros(n_cw*n_sw)    
        xa1   = jnp.zeros(n_cw*n_sw)
        ya1   = jnp.zeros(n_cw*n_sw)
        za1   = jnp.zeros(n_cw*n_sw)
        xa2   = jnp.zeros(n_cw*n_sw)
        ya2   = jnp.zeros(n_cw*n_sw)
        za2   = jnp.zeros(n_cw*n_sw)    
        xb1   = jnp.zeros(n_cw*n_sw)
        yb1   = jnp.zeros(n_cw*n_sw)
        zb1   = jnp.zeros(n_cw*n_sw)
        xb2   = jnp.zeros(n_cw*n_sw) 
        yb2   = jnp.zeros(n_cw*n_sw) 
        zb2   = jnp.zeros(n_cw*n_sw)    
        xac   = jnp.zeros(n_cw*n_sw)
        yac   = jnp.zeros(n_cw*n_sw)
        zac   = jnp.zeros(n_cw*n_sw)    
        xbc   = jnp.zeros(n_cw*n_sw)
        ybc   = jnp.zeros(n_cw*n_sw)
        zbc   = jnp.zeros(n_cw*n_sw)     
        xa_te = jnp.zeros(n_cw*n_sw)
        ya_te = jnp.zeros(n_cw*n_sw)
        za_te = jnp.zeros(n_cw*n_sw)    
        xb_te = jnp.zeros(n_cw*n_sw)
        yb_te = jnp.zeros(n_cw*n_sw)
        zb_te = jnp.zeros(n_cw*n_sw)  
        xc    = jnp.zeros(n_cw*n_sw) 
        yc    = jnp.zeros(n_cw*n_sw) 
        zc    = jnp.zeros(n_cw*n_sw) 
        x     = jnp.zeros((n_cw+1)*(n_sw+1)) # may have to change to make space for split if control surfaces are allowed to have more than two Segments
        y     = jnp.zeros((n_cw+1)*(n_sw+1)) 
        z     = jnp.zeros((n_cw+1)*(n_sw+1))         
        cs_w  = jnp.zeros(n_sw)
        

        # adjust origin for symmetry with special case for vertical symmetry
        if vertical_wing:
            wing_origin_x = wing_origin[0]
            wing_origin_y = wing_origin[1]
            wing_origin_z = wing_origin[2] * sym_sign
        else:
            wing_origin_x = wing_origin[0]
            wing_origin_y = wing_origin[1] * sym_sign
            wing_origin_z = wing_origin[2]       
            
        # ---------------------------------------------------------------------------------------------------------
        # Loop over each strip of panels in the wing
        
        
        # Setup items that will iterate in loop
        LE_inds        = jnp.zeros(n_cw*n_sw,dtype=bool)
        TE_inds        = jnp.zeros(n_cw*n_sw,dtype=bool)
        RNMAX          = jnp.zeros(n_cw*n_sw,dtype=jnp.int16)
        panel_numbers  = jnp.zeros(n_cw*n_sw,dtype=jnp.int16)
        chord_adjusted = jnp.zeros(n_cw*n_sw)
        tan_incidence  = jnp.zeros(n_cw*n_sw)
        exposed_leading_edge_flag = jnp.zeros(n_sw,dtype=jnp.int16)

        indices = [LE_inds,TE_inds,RNMAX,panel_numbers,chord_adjusted,tan_incidence,exposed_leading_edge_flag]
        
        coords = [xah,yah,zah,xbh,ybh,zbh,xch,ych,zch,xa1,ya1,za1,xa2,ya2,za2,xb1,yb1,zb1,xb2,yb2,zb2,xac,yac,zac,xbc,\
                  ybc,zbc,xa_te,ya_te,za_te,xb_te,yb_te,zb_te,xc,yc,zc,x,y,z,cs_w]

        
        def wing_s(idx_y,val):
            """ This is an inline wrapper
        
            Assumptions: 
            The same as generate_wing_vortex_distribution
        
            Source:   
            None
            
            Inputs:   
            MANY
            
            Properties Used:
            N/A
            """                 
            inds, i_break, cords = val
            inds, i_break, cords = wing_strip(i_break,wing,inds,cords,n_sw,y_a,y_b,idx_y,del_y,n_cw,break_spans,break_chord,break_twist,break_sweep,
               break_x_offset,break_z_offset,break_camber_xs,break_camber_zs,break_dihedral,section_span,
               section_LE_cut,section_TE_cut,sym_sign,sym_sign_ind,vertical_wing,span_breaks_cs_ID,precision)    
            
            return [inds, i_break, cords]
            
            
        
        i_break = 0           
        
        indices, i_break, coords = lax.fori_loop(0, n_sw, wing_s, [indices, i_break, coords])
        
        LE_inds,TE_inds,RNMAX,panel_numbers,chord_adjusted,tan_incidence,exposed_leading_edge_flag = indices
        
        VD.leading_edge_indices      = jnp.append(VD.leading_edge_indices     , LE_inds                  ) 
        VD.trailing_edge_indices     = jnp.append(VD.trailing_edge_indices    , TE_inds                  )            
        VD.panels_per_strip          = jnp.append(VD.panels_per_strip         , RNMAX                    )
        VD.chordwise_panel_number    = jnp.append(VD.chordwise_panel_number   , panel_numbers            )  
        VD.chord_lengths             = jnp.append(VD.chord_lengths            , chord_adjusted           )
        VD.tangent_incidence_angle   = jnp.append(VD.tangent_incidence_angle  , tan_incidence            )
        VD.exposed_leading_edge_flag = jnp.append(VD.exposed_leading_edge_flag, exposed_leading_edge_flag)
        

        xah,yah,zah,xbh,ybh,zbh,xch,ych,zch,xa1,ya1,za1,xa2,ya2,za2,xb1,yb1,zb1,xb2,yb2,zb2,xac,yac,zac,xbc,ybc,zbc,\
            xa_te,ya_te,za_te,xb_te,yb_te,zb_te,xc,yc,zc,x,y,z,cs_w = coords
    

    
        # adjusting coordinate axis so reference point is at the nose of the aircraft------------------------------
        xah = xah + wing_origin_x # x coordinate of left corner of bound vortex 
        yah = yah + wing_origin_y # y coordinate of left corner of bound vortex 
        zah = zah + wing_origin_z # z coordinate of left corner of bound vortex 
        xbh = xbh + wing_origin_x # x coordinate of right corner of bound vortex 
        ybh = ybh + wing_origin_y # y coordinate of right corner of bound vortex 
        zbh = zbh + wing_origin_z # z coordinate of right corner of bound vortex 
        xch = xch + wing_origin_x # x coordinate of center of bound vortex on panel
        ych = ych + wing_origin_y # y coordinate of center of bound vortex on panel
        zch = zch + wing_origin_z # z coordinate of center of bound vortex on panel  
    
        xa1 = xa1 + wing_origin_x # x coordinate of top left corner of panel
        ya1 = ya1 + wing_origin_y # y coordinate of bottom left corner of panel
        za1 = za1 + wing_origin_z # z coordinate of top left corner of panel
        xa2 = xa2 + wing_origin_x # x coordinate of bottom left corner of panel
        ya2 = ya2 + wing_origin_y # x coordinate of bottom left corner of panel
        za2 = za2 + wing_origin_z # z coordinate of bottom left corner of panel  
    
        xb1 = xb1 + wing_origin_x # x coordinate of top right corner of panel  
        yb1 = yb1 + wing_origin_y # y coordinate of top right corner of panel 
        zb1 = zb1 + wing_origin_z # z coordinate of top right corner of panel 
        xb2 = xb2 + wing_origin_x # x coordinate of bottom rightcorner of panel 
        yb2 = yb2 + wing_origin_y # y coordinate of bottom rightcorner of panel 
        zb2 = zb2 + wing_origin_z # z coordinate of bottom right corner of panel                   
    
        xac = xac + wing_origin_x  # x coordinate of control points on panel
        yac = yac + wing_origin_y  # y coordinate of control points on panel
        zac = zac + wing_origin_z  # z coordinate of control points on panel
        xbc = xbc + wing_origin_x  # x coordinate of control points on panel
        ybc = ybc + wing_origin_y  # y coordinate of control points on panel
        zbc = zbc + wing_origin_z  # z coordinate of control points on panel
    
        xc  = xc  + wing_origin_x  # x coordinate of control points on panel
        yc  = yc  + wing_origin_y  # y coordinate of control points on panel
        zc  = zc  + wing_origin_z  # y coordinate of control points on panel
        x   = x   + wing_origin_x  # x coordinate of control points on panel
        y   = y   + wing_origin_y  # y coordinate of control points on panel
        z   = z   + wing_origin_z  # y coordinate of control points on panel
    
        # find the location of the trailing edge panels of each wing-----------------------------------------------
        locations = ((jnp.linspace(1,n_sw,n_sw, endpoint = True) * n_cw) - 1).astype(int)
        xc_te1 = jnp.repeat(jnp.atleast_2d(xc [locations]), n_cw , axis = 0)
        yc_te1 = jnp.repeat(jnp.atleast_2d(yc [locations]), n_cw , axis = 0)
        zc_te1 = jnp.repeat(jnp.atleast_2d(zc [locations]), n_cw , axis = 0)        
        xa_te1 = jnp.repeat(jnp.atleast_2d(xa2[locations]), n_cw , axis = 0)
        ya_te1 = jnp.repeat(jnp.atleast_2d(ya2[locations]), n_cw , axis = 0)
        za_te1 = jnp.repeat(jnp.atleast_2d(za2[locations]), n_cw , axis = 0)
        xb_te1 = jnp.repeat(jnp.atleast_2d(xb2[locations]), n_cw , axis = 0)
        yb_te1 = jnp.repeat(jnp.atleast_2d(yb2[locations]), n_cw , axis = 0)
        zb_te1 = jnp.repeat(jnp.atleast_2d(zb2[locations]), n_cw , axis = 0)     
    
        xc_te = jnp.hstack(xc_te1.T)
        yc_te = jnp.hstack(yc_te1.T)
        zc_te = jnp.hstack(zc_te1.T)        
        xa_te = jnp.hstack(xa_te1.T)
        ya_te = jnp.hstack(ya_te1.T)
        za_te = jnp.hstack(za_te1.T)
        xb_te = jnp.hstack(xb_te1.T)
        yb_te = jnp.hstack(yb_te1.T)
        zb_te = jnp.hstack(zb_te1.T) 
    
        # find spanwise locations 
        y_sw = yc[locations]        
    
        # increment number of wings and panels
        n_panels = len(xch)
        VD.n_w  += 1             
        VD.n_cp += n_panels 
        
        # store this wing's discretization information  
        first_panel_ind  = VD.XAH.size
        first_strip_ind  = VD.chordwise_breaks.size
        chordwise_breaks = first_panel_ind + jnp.arange(n_panels)[0::n_cw]
        
        VD.chordwise_breaks = jnp.append(VD.chordwise_breaks, jnp.int32(chordwise_breaks))
        VD.spanwise_breaks  = jnp.append(VD.spanwise_breaks , jnp.int32(first_strip_ind ))            
        VD.n_sw             =  np.append(VD.n_sw            ,  np.int16(n_sw)            )
        VD.n_cw             = jnp.append(VD.n_cw            , jnp.int16(n_cw)            )
    
        # ---------------------------------------------------------------------------------------
        # STEP 7: Store wing in vehicle vector
        # --------------------------------------------------------------------------------------- 
        VD.XAH    = jnp.append(VD.XAH  , jnp.array(xah  , dtype=precision))
        VD.YAH    = jnp.append(VD.YAH  , jnp.array(yah  , dtype=precision))
        VD.ZAH    = jnp.append(VD.ZAH  , jnp.array(zah  , dtype=precision))
        VD.XBH    = jnp.append(VD.XBH  , jnp.array(xbh  , dtype=precision))
        VD.YBH    = jnp.append(VD.YBH  , jnp.array(ybh  , dtype=precision))
        VD.ZBH    = jnp.append(VD.ZBH  , jnp.array(zbh  , dtype=precision))
        VD.XCH    = jnp.append(VD.XCH  , jnp.array(xch  , dtype=precision))
        VD.YCH    = jnp.append(VD.YCH  , jnp.array(ych  , dtype=precision))
        VD.ZCH    = jnp.append(VD.ZCH  , jnp.array(zch  , dtype=precision))            
        VD.XA1    = jnp.append(VD.XA1  , jnp.array(xa1  , dtype=precision))
        VD.YA1    = jnp.append(VD.YA1  , jnp.array(ya1  , dtype=precision))
        VD.ZA1    = jnp.append(VD.ZA1  , jnp.array(za1  , dtype=precision))
        VD.XA2    = jnp.append(VD.XA2  , jnp.array(xa2  , dtype=precision))
        VD.YA2    = jnp.append(VD.YA2  , jnp.array(ya2  , dtype=precision))
        VD.ZA2    = jnp.append(VD.ZA2  , jnp.array(za2  , dtype=precision))        
        VD.XB1    = jnp.append(VD.XB1  , jnp.array(xb1  , dtype=precision))
        VD.YB1    = jnp.append(VD.YB1  , jnp.array(yb1  , dtype=precision))
        VD.ZB1    = jnp.append(VD.ZB1  , jnp.array(zb1  , dtype=precision))
        VD.XB2    = jnp.append(VD.XB2  , jnp.array(xb2  , dtype=precision))                
        VD.YB2    = jnp.append(VD.YB2  , jnp.array(yb2  , dtype=precision))        
        VD.ZB2    = jnp.append(VD.ZB2  , jnp.array(zb2  , dtype=precision))    
        VD.XC_TE  = jnp.append(VD.XC_TE, jnp.array(xc_te, dtype=precision))
        VD.YC_TE  = jnp.append(VD.YC_TE, jnp.array(yc_te, dtype=precision)) 
        VD.ZC_TE  = jnp.append(VD.ZC_TE, jnp.array(zc_te, dtype=precision))          
        VD.XA_TE  = jnp.append(VD.XA_TE, jnp.array(xa_te, dtype=precision))
        VD.YA_TE  = jnp.append(VD.YA_TE, jnp.array(ya_te, dtype=precision)) 
        VD.ZA_TE  = jnp.append(VD.ZA_TE, jnp.array(za_te, dtype=precision)) 
        VD.XB_TE  = jnp.append(VD.XB_TE, jnp.array(xb_te, dtype=precision))
        VD.YB_TE  = jnp.append(VD.YB_TE, jnp.array(yb_te, dtype=precision)) 
        VD.ZB_TE  = jnp.append(VD.ZB_TE, jnp.array(zb_te, dtype=precision))  
        VD.XAC    = jnp.append(VD.XAC  , jnp.array(xac  , dtype=precision))
        VD.YAC    = jnp.append(VD.YAC  , jnp.array(yac  , dtype=precision)) 
        VD.ZAC    = jnp.append(VD.ZAC  , jnp.array(zac  , dtype=precision)) 
        VD.XBC    = jnp.append(VD.XBC  , jnp.array(xbc  , dtype=precision))
        VD.YBC    = jnp.append(VD.YBC  , jnp.array(ybc  , dtype=precision)) 
        VD.ZBC    = jnp.append(VD.ZBC  , jnp.array(zbc  , dtype=precision))  
        VD.XC     = jnp.append(VD.XC   , jnp.array(xc   , dtype=precision))
        VD.YC     = jnp.append(VD.YC   , jnp.array(yc   , dtype=precision))
        VD.ZC     = jnp.append(VD.ZC   , jnp.array(zc   , dtype=precision))  
        VD.X      = jnp.append(VD.X    , jnp.array(x    , dtype=precision))
        VD.Y_SW   = jnp.append(VD.Y_SW , jnp.array(y_sw , dtype=precision))
        VD.Y      = jnp.append(VD.Y    , jnp.array(y    , dtype=precision))
        VD.Z      = jnp.append(VD.Z    , jnp.array(z    , dtype=precision))         
        VD.CS     = jnp.append(VD.CS   , jnp.array(cs_w , dtype=precision)) 
        VD.DY     = jnp.append(VD.DY   , jnp.array(del_y, dtype=precision))      
    #End symmetry loop
    VD.symmetric_wings = jnp.append(VD.symmetric_wings, int(sym_para))
    return VD

def wing_strip(i_break,wing,indices,coords,n_sw,y_a,y_b,idx_y,del_y,n_cw,break_spans,break_chord,break_twist,break_sweep,
               break_x_offset,break_z_offset,break_camber_xs,break_camber_zs,break_dihedral,section_span,
               section_LE_cut,section_TE_cut,sym_sign,sym_sign_ind,vertical_wing,span_breaks_cs_ID,precision):
    """ This generates vortex distribution points for the given strip of wing 

    Assumptions: 
    The same as generate_wing_vortex_distribution

    Source:   
    None
    
    Inputs:   
    MANY
    
    Properties Used:
    N/A
    """     
    
    
    xah,yah,zah,xbh,ybh,zbh,xch,ych,zch,xa1,ya1,za1,xa2,ya2,za2,xb1,yb1,zb1,xb2,yb2,zb2,xac,yac,zac,xbc,ybc,zbc,xa_te,\
        ya_te,za_te,xb_te,yb_te,zb_te,xc,yc,zc,x,y,z,cs_w = coords
    
    # define basic geometric values------------------------------------------------------------------------
    # inboard, outboard, and central panel values
    eta_a = (y_a[idx_y] - break_spans[i_break])  
    eta_b = (y_b[idx_y] - break_spans[i_break]) 
    eta   = (y_b[idx_y] - del_y[idx_y]/2 - break_spans[i_break]) 

    segment_chord_ratio = (break_chord[i_break+1] - break_chord[i_break])/section_span[i_break+1]
    segment_twist_ratio = (break_twist[i_break+1] - break_twist[i_break])/section_span[i_break+1]

    wing_chord_section_a  = break_chord[i_break] + (eta_a*segment_chord_ratio) 
    wing_chord_section_b  = break_chord[i_break] + (eta_b*segment_chord_ratio)
    wing_chord_section    = break_chord[i_break] + (eta*segment_chord_ratio)

    # x-positions based on whether the wing needs 'cuts' for its control sufaces
    nondim_x_stations = jnp.interp(jnp.linspace(0.,1.,num=n_cw+1), jnp.array([0.,1.]), jnp.array([section_LE_cut[i_break], section_TE_cut[i_break]]))
    x_stations_a      = nondim_x_stations * wing_chord_section_a  #x positions accounting for control surface cuts, relative to leading
    x_stations_b      = nondim_x_stations * wing_chord_section_b
    x_stations        = nondim_x_stations * wing_chord_section
    
    delta_x_a = (x_stations_a[-1] - x_stations_a[0])/n_cw  
    delta_x_b = (x_stations_b[-1] - x_stations_b[0])/n_cw      
    delta_x   = (x_stations[-1]   - x_stations[0]  )/n_cw             

    # define coordinates of horseshoe vortices and control points------------------------------------------
    xi_a1 = break_x_offset[i_break] + eta_a*jnp.tan(break_sweep[i_break]) + x_stations_a[:-1]                  # x coordinate of top left corner of panel
    xi_ah = break_x_offset[i_break] + eta_a*jnp.tan(break_sweep[i_break]) + x_stations_a[:-1] + delta_x_a*0.25 # x coordinate of left corner of panel
    xi_ac = break_x_offset[i_break] + eta_a*jnp.tan(break_sweep[i_break]) + x_stations_a[:-1] + delta_x_a*0.75 # x coordinate of bottom left corner of control point vortex  
    xi_a2 = break_x_offset[i_break] + eta_a*jnp.tan(break_sweep[i_break]) + x_stations_a[1:]                   # x coordinate of bottom left corner of bound vortex 
    xi_b1 = break_x_offset[i_break] + eta_b*jnp.tan(break_sweep[i_break]) + x_stations_b[:-1]                  # x coordinate of top right corner of panel      
    xi_bh = break_x_offset[i_break] + eta_b*jnp.tan(break_sweep[i_break]) + x_stations_b[:-1] + delta_x_b*0.25 # x coordinate of right corner of bound vortex         
    xi_bc = break_x_offset[i_break] + eta_b*jnp.tan(break_sweep[i_break]) + x_stations_b[:-1] + delta_x_b*0.75 # x coordinate of bottom right corner of control point vortex         
    xi_b2 = break_x_offset[i_break] + eta_b*jnp.tan(break_sweep[i_break]) + x_stations_b[1:]                   # x coordinate of bottom right corner of panel
    xi_ch = break_x_offset[i_break] + eta  *jnp.tan(break_sweep[i_break]) + x_stations[:-1]   + delta_x  *0.25 # x coordinate center of bound vortex of each panel 
    xi_c  = break_x_offset[i_break] + eta  *jnp.tan(break_sweep[i_break]) + x_stations[:-1]   + delta_x  *0.75 # x coordinate three-quarter chord control point for each panel

    #adjust for camber-------------------------------------------------------------------------------------    
    #format camber vars for wings vs control surface wings
    nondim_camber_x_coords = break_camber_xs[i_break] *1
    nondim_camber          = break_camber_zs[i_break] *1
    if wing.is_a_control_surface: #rescale so that airfoils get cut properly
        if not wing.is_slat:
            nondim_camber_x_coords -= 1 - wing.chord_fraction
        nondim_camber_x_coords /= wing.chord_fraction
        nondim_camber          /= wing.chord_fraction

    # adjustment of coordinates for camber
    section_camber_a  = nondim_camber*wing_chord_section_a  
    section_camber_b  = nondim_camber*wing_chord_section_b  
    section_camber_c  = nondim_camber*wing_chord_section             
    
    section_x_coord_a = nondim_camber_x_coords*wing_chord_section_a
    section_x_coord_b = nondim_camber_x_coords*wing_chord_section_b
    section_x_coord   = nondim_camber_x_coords*wing_chord_section

    z_c_a1 = jnp.interp((x_stations_a[:-1]                 ) ,section_x_coord_a, section_camber_a) 
    z_c_ah = jnp.interp((x_stations_a[:-1] + delta_x_a*0.25) ,section_x_coord_a, section_camber_a)
    z_c_ac = jnp.interp((x_stations_a[:-1] + delta_x_a*0.75) ,section_x_coord_a, section_camber_a) 
    z_c_a2 = jnp.interp((x_stations_a[1:]                  ) ,section_x_coord_a, section_camber_a) 
    z_c_b1 = jnp.interp((x_stations_b[:-1]                 ) ,section_x_coord_b, section_camber_b)   
    z_c_bh = jnp.interp((x_stations_b[:-1] + delta_x_b*0.25) ,section_x_coord_b, section_camber_b) 
    z_c_bc = jnp.interp((x_stations_b[:-1] + delta_x_b*0.75) ,section_x_coord_b, section_camber_b) 
    z_c_b2 = jnp.interp((x_stations_b[1:]                  ) ,section_x_coord_b, section_camber_b) 
    z_c_ch = jnp.interp((x_stations[:-1]   + delta_x  *0.25) ,section_x_coord  , section_camber_c) 
    z_c    = jnp.interp((x_stations[:-1]   + delta_x  *0.75) ,section_x_coord  , section_camber_c) 

    # adjust for dihedral and add to camber----------------------------------------------------------------    
    zeta_a1 = break_z_offset[i_break] + eta_a*jnp.tan(break_dihedral[i_break])  + z_c_a1  # z coordinate of top left corner of panel
    zeta_ah = break_z_offset[i_break] + eta_a*jnp.tan(break_dihedral[i_break])  + z_c_ah  # z coordinate of left corner of bound vortex  
    zeta_a2 = break_z_offset[i_break] + eta_a*jnp.tan(break_dihedral[i_break])  + z_c_a2  # z coordinate of bottom left corner of panel
    zeta_ac = break_z_offset[i_break] + eta_a*jnp.tan(break_dihedral[i_break])  + z_c_ac  # z coordinate of bottom left corner of panel of control point
    zeta_bc = break_z_offset[i_break] + eta_b*jnp.tan(break_dihedral[i_break])  + z_c_bc  # z coordinate of top right corner of panel of control point                          
    zeta_b1 = break_z_offset[i_break] + eta_b*jnp.tan(break_dihedral[i_break])  + z_c_b1  # z coordinate of top right corner of panel  
    zeta_bh = break_z_offset[i_break] + eta_b*jnp.tan(break_dihedral[i_break])  + z_c_bh  # z coordinate of right corner of bound vortex        
    zeta_b2 = break_z_offset[i_break] + eta_b*jnp.tan(break_dihedral[i_break])  + z_c_b2  # z coordinate of bottom right corner of panel                 
    zeta_ch = break_z_offset[i_break] + eta  *jnp.tan(break_dihedral[i_break])  + z_c_ch  # z coordinate center of bound vortex on each panel
    zeta    = break_z_offset[i_break] + eta  *jnp.tan(break_dihedral[i_break])  + z_c     # z coordinate three-quarter chord control point for each panel

    # adjust for twist-------------------------------------------------------------------------------------
    # pivot point is the leading edge before camber  
    pivot_x_a = break_x_offset[i_break] + eta_a*jnp.tan(break_sweep[i_break])             # x location of leading edge left corner of wing
    pivot_x_b = break_x_offset[i_break] + eta_b*jnp.tan(break_sweep[i_break])             # x location of leading edge right of wing
    pivot_x   = break_x_offset[i_break] + eta  *jnp.tan(break_sweep[i_break])             # x location of leading edge center of wing
    
    pivot_z_a = break_z_offset[i_break] + eta_a*jnp.tan(break_dihedral[i_break])          # z location of leading edge left corner of wing
    pivot_z_b = break_z_offset[i_break] + eta_b*jnp.tan(break_dihedral[i_break])          # z location of leading edge right of wing
    pivot_z   = break_z_offset[i_break] + eta  *jnp.tan(break_dihedral[i_break])          # z location of leading edge center of wing

    # adjust twist pivot line for control surface wings: offset leading edge to match that of the owning wing            
    if wing.is_a_control_surface and not wing.is_slat: #correction only leading for non-leading edge control surfaces since the LE is the pivot by default
        nondim_cs_LE = (1 - wing.chord_fraction)
        pivot_x_a   -= nondim_cs_LE *(wing_chord_section_a /wing.chord_fraction) 
        pivot_x_b   -= nondim_cs_LE *(wing_chord_section_b /wing.chord_fraction) 
        pivot_x     -= nondim_cs_LE *(wing_chord_section   /wing.chord_fraction) 

    # adjust coordinates for twist
    section_twist_a = break_twist[i_break] + (eta_a * segment_twist_ratio)               # twist at left side of panel
    section_twist_b = break_twist[i_break] + (eta_b * segment_twist_ratio)               # twist at right side of panel
    section_twist   = break_twist[i_break] + (eta   * segment_twist_ratio)               # twist at center local chord 

    xi_prime_a1    = pivot_x_a + jnp.cos(section_twist_a)*(xi_a1-pivot_x_a) + jnp.sin(section_twist_a)*(zeta_a1-pivot_z_a) # x coordinate transformation of top left corner
    xi_prime_ah    = pivot_x_a + jnp.cos(section_twist_a)*(xi_ah-pivot_x_a) + jnp.sin(section_twist_a)*(zeta_ah-pivot_z_a) # x coordinate transformation of bottom left corner
    xi_prime_ac    = pivot_x_a + jnp.cos(section_twist_a)*(xi_ac-pivot_x_a) + jnp.sin(section_twist_a)*(zeta_a2-pivot_z_a) # x coordinate transformation of bottom left corner of control point
    xi_prime_a2    = pivot_x_a + jnp.cos(section_twist_a)*(xi_a2-pivot_x_a) + jnp.sin(section_twist_a)*(zeta_a2-pivot_z_a) # x coordinate transformation of bottom left corner
    xi_prime_b1    = pivot_x_b + jnp.cos(section_twist_b)*(xi_b1-pivot_x_b) + jnp.sin(section_twist_b)*(zeta_b1-pivot_z_b) # x coordinate transformation of top right corner 
    xi_prime_bh    = pivot_x_b + jnp.cos(section_twist_b)*(xi_bh-pivot_x_b) + jnp.sin(section_twist_b)*(zeta_bh-pivot_z_b) # x coordinate transformation of top right corner 
    xi_prime_bc    = pivot_x_b + jnp.cos(section_twist_b)*(xi_bc-pivot_x_b) + jnp.sin(section_twist_b)*(zeta_b1-pivot_z_b) # x coordinate transformation of top right corner of control point                         
    xi_prime_b2    = pivot_x_b + jnp.cos(section_twist_b)*(xi_b2-pivot_x_b) + jnp.sin(section_twist_b)*(zeta_b2-pivot_z_b) # x coordinate transformation of botton right corner 
    xi_prime_ch    = pivot_x   + jnp.cos(section_twist)  *(xi_ch-pivot_x)   + jnp.sin(section_twist)  *(zeta_ch-pivot_z)   # x coordinate transformation of center of horeshoe vortex 
    xi_prime       = pivot_x   + jnp.cos(section_twist)  *(xi_c -pivot_x)   + jnp.sin(section_twist)  *(zeta   -pivot_z)   # x coordinate transformation of control point

    zeta_prime_a1  = pivot_z_a - jnp.sin(section_twist_a)*(xi_a1-pivot_x_a) + jnp.cos(section_twist_a)*(zeta_a1-pivot_z_a) # z coordinate transformation of top left corner
    zeta_prime_ah  = pivot_z_a - jnp.sin(section_twist_a)*(xi_ah-pivot_x_a) + jnp.cos(section_twist_a)*(zeta_ah-pivot_z_a) # z coordinate transformation of bottom left corner
    zeta_prime_ac  = pivot_z_a - jnp.sin(section_twist_a)*(xi_ac-pivot_x_a) + jnp.cos(section_twist_a)*(zeta_ac-pivot_z_a) # z coordinate transformation of bottom left corner
    zeta_prime_a2  = pivot_z_a - jnp.sin(section_twist_a)*(xi_a2-pivot_x_a) + jnp.cos(section_twist_a)*(zeta_a2-pivot_z_a) # z coordinate transformation of bottom left corner
    zeta_prime_b1  = pivot_z_b - jnp.sin(section_twist_b)*(xi_b1-pivot_x_b) + jnp.cos(section_twist_b)*(zeta_b1-pivot_z_b) # z coordinate transformation of top right corner 
    zeta_prime_bh  = pivot_z_b - jnp.sin(section_twist_b)*(xi_bh-pivot_x_b) + jnp.cos(section_twist_b)*(zeta_bh-pivot_z_b) # z coordinate transformation of top right corner 
    zeta_prime_bc  = pivot_z_b - jnp.sin(section_twist_b)*(xi_bc-pivot_x_b) + jnp.cos(section_twist_b)*(zeta_bc-pivot_z_b) # z coordinate transformation of top right corner                         
    zeta_prime_b2  = pivot_z_b - jnp.sin(section_twist_b)*(xi_b2-pivot_x_b) + jnp.cos(section_twist_b)*(zeta_b2-pivot_z_b) # z coordinate transformation of botton right corner 
    zeta_prime_ch  = pivot_z   - jnp.sin(section_twist)  *(xi_ch-pivot_x)   + jnp.cos(-section_twist) *(zeta_ch-pivot_z)   # z coordinate transformation of center of horseshoe
    zeta_prime     = pivot_z   - jnp.sin(section_twist)  *(xi_c -pivot_x)   + jnp.cos(-section_twist) *(zeta   -pivot_z)   # z coordinate transformation of control point
    
    # Define y-coordinate and other arrays-----------------------------------------------------------------
    # take normal value for first wing, then reflect over xz plane for a symmetric wing
    y_prime_as = (jnp.ones(n_cw+1)*y_a[idx_y]                ) *sym_sign          
    y_prime_a1 = (y_prime_as[:-1]                            ) *1       
    y_prime_ah = (y_prime_as[:-1]                            ) *1       
    y_prime_ac = (y_prime_as[:-1]                            ) *1          
    y_prime_a2 = (y_prime_as[:-1]                            ) *1        
    y_prime_bs = (jnp.ones(n_cw+1)*y_b[idx_y]                ) *sym_sign            
    y_prime_b1 = (y_prime_bs[:-1]                            ) *1         
    y_prime_bh = (y_prime_bs[:-1]                            ) *1         
    y_prime_bc = (y_prime_bs[:-1]                            ) *1         
    y_prime_b2 = (y_prime_bs[:-1]                            ) *1   
    y_prime_ch = (jnp.ones(n_cw)*(y_b[idx_y] - del_y[idx_y]/2)) *sym_sign
    y_prime    = (y_prime_ch                                 ) *1    
    
    # populate all corners of all panels. Right side only populated for last strip wing the wing
    xi_prime_as   = jnp.concatenate([xi_prime_a1,  jnp.array([xi_prime_a2  [-1]])])*1
    xi_prime_bs   = jnp.concatenate([xi_prime_b1,  jnp.array([xi_prime_b2  [-1]])])*1 
    zeta_prime_as = jnp.concatenate([zeta_prime_a1,jnp.array([zeta_prime_a2[-1]])])*1            
    zeta_prime_bs = jnp.concatenate([zeta_prime_b1,jnp.array([zeta_prime_b2[-1]])])*1   
    
    # Deflect control surfaces-----------------------------------------------------------------------------
    # note:    "positve" deflection corresponds to the RH rule where the axis of rotation is the OUTBOARD-pointing hinge vector
    # symmetry: the LH rule is applied to the reflected surface for non-ailerons. Ailerons follow a RH rule for both sides
    wing_is_all_moving = (not wing.is_a_control_surface) and issubclass(wing.wing_type, All_Moving_Surface)
    if wing.is_a_control_surface or wing_is_all_moving:
        
        #For the first strip of the wing, always need to find the hinge root point. The hinge root point and direction vector 
        #found here will not change for the rest of this control surface/all-moving surface. See docstring for reasoning.
        is_first_strip = (idx_y == 0)
        if is_first_strip:
            # get rotation points by iterpolating between strip corners --> le/te, ib/ob = leading/trailing edge, in/outboard
            ib_le_strip_corner = jnp.array([xi_prime_a1[0 ], y_prime_a1[0 ], zeta_prime_a1[0 ]])
            ib_te_strip_corner = jnp.array([xi_prime_a2[-1], y_prime_a2[-1], zeta_prime_a2[-1]])                    
            
            interp_fractions   = jnp.array([0.,    2.,    4.   ]) + wing.hinge_fraction
            interp_domains     = jnp.array([0.,1., 2.,3., 4.,5.])
            interp_ranges_ib   = jnp.array([ib_le_strip_corner, ib_te_strip_corner]).T.flatten()
            ib_hinge_point     = jnp.interp(interp_fractions, interp_domains, interp_ranges_ib)
            
            
            #Find the hinge_vector if this is a control surface or the user has not already defined and chosen to use a specific one                    
            if wing.is_a_control_surface:
                need_to_compute_hinge_vector = True
            else: #wing is an all-moving surface
                hinge_vector                 = jnp.array(wing.hinge_vector)
                hinge_vector_is_pre_defined  = (not wing.use_constant_hinge_fraction) and \
                                                not (hinge_vector==jnp.array([0.,0.,0.])).all()
                need_to_compute_hinge_vector = not hinge_vector_is_pre_defined  
                
            if need_to_compute_hinge_vector:
                ob_le_strip_corner = jnp.array([xi_prime_b1[0 ], y_prime_b1[0 ], zeta_prime_b1[0 ]])                
                ob_te_strip_corner = jnp.array([xi_prime_b2[-1], y_prime_b2[-1], zeta_prime_b2[-1]])                         
                interp_ranges_ob   = jnp.array([ob_le_strip_corner, ob_te_strip_corner]).T.flatten()
                ob_hinge_point     = jnp.interp(interp_fractions, interp_domains, interp_ranges_ob)
            
                use_root_chord_in_plane_normal = wing_is_all_moving and not wing.use_constant_hinge_fraction
                if use_root_chord_in_plane_normal: ob_hinge_point = ob_hinge_point.at[0].set(ib_hinge_point[0])
            
                hinge_vector       = ob_hinge_point - ib_hinge_point
                hinge_vector       = hinge_vector / jnp.linalg.norm(hinge_vector)   
            elif wing.vertical: #For a vertical all-moving surface, flip y and z of hinge vector before flipping again later
                HNG_1 = hinge_vector[1] * 1
                HNG_2 = hinge_vector[2] * 1
                
                hinge_vector =  hinge_vector.at[1].set(HNG_2)
                hinge_vector =  hinge_vector.at[2].set(HNG_1)
                
            #store hinge root point and direction vector
            wing.hinge_root_point = ib_hinge_point
            wing.hinge_vector     = hinge_vector
            #END first strip calculations
        
        # get deflection angle
        deflection_base_angle = wing.deflection      if (not wing.is_slat) else -wing.deflection
        symmetry_multiplier   = -wing.sign_duplicate if sym_sign_ind==1    else 1
        symmetry_multiplier  *= -1                   if vertical_wing      else 1
        deflection_angle      = deflection_base_angle * symmetry_multiplier
            
        # make quaternion rotation matrix
        quaternion   = make_hinge_quaternion(wing.hinge_root_point, wing.hinge_vector, deflection_angle)
        
        # rotate strips
        xi_prime_a1, y_prime_a1, zeta_prime_a1 = rotate_points_with_quaternion(quaternion, [xi_prime_a1,y_prime_a1,zeta_prime_a1])
        xi_prime_ah, y_prime_ah, zeta_prime_ah = rotate_points_with_quaternion(quaternion, [xi_prime_ah,y_prime_ah,zeta_prime_ah])
        xi_prime_ac, y_prime_ac, zeta_prime_ac = rotate_points_with_quaternion(quaternion, [xi_prime_ac,y_prime_ac,zeta_prime_ac])
        xi_prime_a2, y_prime_a2, zeta_prime_a2 = rotate_points_with_quaternion(quaternion, [xi_prime_a2,y_prime_a2,zeta_prime_a2])
                                                                                           
        xi_prime_b1, y_prime_b1, zeta_prime_b1 = rotate_points_with_quaternion(quaternion, [xi_prime_b1,y_prime_b1,zeta_prime_b1])
        xi_prime_bh, y_prime_bh, zeta_prime_bh = rotate_points_with_quaternion(quaternion, [xi_prime_bh,y_prime_bh,zeta_prime_bh])
        xi_prime_bc, y_prime_bc, zeta_prime_bc = rotate_points_with_quaternion(quaternion, [xi_prime_bc,y_prime_bc,zeta_prime_bc])
        xi_prime_b2, y_prime_b2, zeta_prime_b2 = rotate_points_with_quaternion(quaternion, [xi_prime_b2,y_prime_b2,zeta_prime_b2])
                                                                                           
        xi_prime_ch, y_prime_ch, zeta_prime_ch = rotate_points_with_quaternion(quaternion, [xi_prime_ch,y_prime_ch,zeta_prime_ch])
        xi_prime   , y_prime   , zeta_prime    = rotate_points_with_quaternion(quaternion, [xi_prime   ,y_prime   ,zeta_prime   ])
                                                                                           
        xi_prime_as, y_prime_as, zeta_prime_as = rotate_points_with_quaternion(quaternion, [xi_prime_as,y_prime_as,zeta_prime_as])
        xi_prime_bs, y_prime_bs, zeta_prime_bs = rotate_points_with_quaternion(quaternion, [xi_prime_bs,y_prime_bs,zeta_prime_bs])
    
    # reflect over the plane y = z for a vertical wing-----------------------------------------------------
    inverted_wing = -jnp.sign(break_dihedral[i_break] - jnp.pi/2)
    if vertical_wing:
        y_prime_a1, zeta_prime_a1 = zeta_prime_a1, inverted_wing*y_prime_a1
        y_prime_ah, zeta_prime_ah = zeta_prime_ah, inverted_wing*y_prime_ah
        y_prime_ac, zeta_prime_ac = zeta_prime_ac, inverted_wing*y_prime_ac
        y_prime_a2, zeta_prime_a2 = zeta_prime_a2, inverted_wing*y_prime_a2
                                                             
        y_prime_b1, zeta_prime_b1 = zeta_prime_b1, inverted_wing*y_prime_b1
        y_prime_bh, zeta_prime_bh = zeta_prime_bh, inverted_wing*y_prime_bh
        y_prime_bc, zeta_prime_bc = zeta_prime_bc, inverted_wing*y_prime_bc
        y_prime_b2, zeta_prime_b2 = zeta_prime_b2, inverted_wing*y_prime_b2
                                                             
        y_prime_ch, zeta_prime_ch = zeta_prime_ch, inverted_wing*y_prime_ch
        y_prime   , zeta_prime    = zeta_prime   , inverted_wing*y_prime
                                                             
        y_prime_as, zeta_prime_as = zeta_prime_as, inverted_wing*y_prime_as

        y_prime_bs = inverted_wing*y_prime_bs
        y_prime_bs, zeta_prime_bs = zeta_prime_bs, y_prime_bs
         
    # store coordinates of panels, horseshoeces vortices and control points relative to wing root----------
    xa1 = DUS(xa1, xi_prime_a1,(idx_y*n_cw,))    # top left corner of panel
    ya1 = DUS(ya1,y_prime_a1,(idx_y*n_cw,))
    za1 = DUS(za1,zeta_prime_a1,(idx_y*n_cw,))
    xah = DUS(xah,xi_prime_ah,(idx_y*n_cw,))     # left coord of horseshoe
    yah = DUS(yah,y_prime_ah,(idx_y*n_cw,))
    zah = DUS(zah,zeta_prime_ah,(idx_y*n_cw,))                    
    xac = DUS(xac,xi_prime_ac,(idx_y*n_cw,))     # left coord of control point
    yac = DUS(yac,y_prime_ac,(idx_y*n_cw,))
    zac = DUS(zac,zeta_prime_ac,(idx_y*n_cw,))
    xa2 = DUS(xa2,xi_prime_a2,(idx_y*n_cw,))     # bottom left corner of panel
    ya2 = DUS(ya2,y_prime_a2,(idx_y*n_cw,))
    za2 = DUS(za2,zeta_prime_a2,(idx_y*n_cw,))
                                     
    xb1 = DUS(xb1,xi_prime_b1,(idx_y*n_cw,))     # top right corner of panel
    yb1 = DUS(yb1,y_prime_b1,(idx_y*n_cw,))          
    zb1 = DUS(zb1,zeta_prime_b1,(idx_y*n_cw,))   
    xbh = DUS(xbh,xi_prime_bh,(idx_y*n_cw,))     # right coord of horseshoe
    ybh = DUS(ybh,y_prime_bh,(idx_y*n_cw,))          
    zbh = DUS(zbh,zeta_prime_bh,(idx_y*n_cw,))                    
    xbc = DUS(xbc,xi_prime_bc,(idx_y*n_cw,))     # right coord of control point
    ybc = DUS(ybc,y_prime_bc,(idx_y*n_cw,))                           
    zbc = DUS(zbc,zeta_prime_bc,(idx_y*n_cw,))   
    xb2 = DUS(xb2,xi_prime_b2,(idx_y*n_cw,))     # bottom right corner of panel
    yb2 = DUS(yb2,y_prime_b2,(idx_y*n_cw,))                        
    zb2 = DUS(zb2,zeta_prime_b2,(idx_y*n_cw,)) 
                                     
    xch = DUS(xch,xi_prime_ch,(idx_y*n_cw,))     # center coord of horseshoe
    ych = DUS(ych,y_prime_ch,(idx_y*n_cw,))                              
    zch = DUS(zch,zeta_prime_ch,(idx_y*n_cw,))
    xc  = DUS(xc ,xi_prime,(idx_y*n_cw,))        # center (true,(idx_y*n_cw,)) coord of control point
    yc  = DUS(yc ,y_prime,(idx_y*n_cw,))
    zc  = DUS(zc ,zeta_prime,(idx_y*n_cw,)) 
    x   = DUS(x,xi_prime_as,(idx_y*(n_cw+1),))     # x, y, z represent all all points of the corners of the panels, LE and TE inclusive
    y   = DUS(y,y_prime_as,(idx_y*(n_cw+1),))      # the final right corners get appended at last strip in wing, later
    z   = DUS(z,zeta_prime_as,(idx_y*(n_cw+1),))            

    cs_w = cs_w.at[idx_y].set(wing_chord_section)
           
    # store this strip's discretization information--------------------------------------------------------
    LE_inds        = jnp.full(n_cw, False)
    TE_inds        = jnp.full(n_cw, False)
    LE_inds        = LE_inds.at[0].set(True)
    TE_inds        = TE_inds.at[-1].set(True)
    
    RNMAX          = jnp.ones(n_cw, jnp.int16)*n_cw
    panel_numbers  = jnp.linspace(1,n_cw,n_cw, dtype=jnp.int16)
    
    LE_X           = (xi_prime_a1  [0 ] + xi_prime_b1  [0 ])/2
    LE_Z           = (zeta_prime_a1[0 ] + zeta_prime_b1[0 ])/2
    TE_X           = (xi_prime_a2  [-1] + xi_prime_b2  [-1])/2
    TE_Z           = (zeta_prime_a2[-1] + zeta_prime_b2[-1])/2           
    chord_adjusted = jnp.ones(n_cw) * jnp.sqrt((TE_X-LE_X)**2 + (TE_Z-LE_Z)**2) # CHORD in vorlax
    tan_incidence  = jnp.ones(n_cw) * (LE_Z-TE_Z)/(LE_X-TE_X)                  # ZETA  in vorlax
    chord_adjusted = jnp.array(chord_adjusted, dtype=precision)
    tan_incidence  = jnp.array(tan_incidence , dtype=precision)            
                
    is_a_slat         = wing.is_a_control_surface and wing.is_slat
    strip_has_no_slat = (not wing.is_a_control_surface) and (span_breaks_cs_ID[i_break] == -1) # wing's le, outboard control surface ID
    
    slat_cond                  = jnp.logical_or(is_a_slat,strip_has_no_slat)
    exposed_leading_edge_flag  = jnp.array([1],dtype=jnp.int16)*slat_cond 
    
    leading_edge_indices,trailing_edge_indices,panels_per_strip,chordwise_panel_number,chord_lengths, \
               tangent_incidence_angle,exposed_leading_edge_flags = indices
    
    leading_edge_indices       = DUS(leading_edge_indices,LE_inds,(idx_y*n_cw,))
    trailing_edge_indices      = DUS(trailing_edge_indices,TE_inds,(idx_y*n_cw,))      
    panels_per_strip           = DUS(panels_per_strip, RNMAX,(idx_y*n_cw,))
    chordwise_panel_number     = DUS(chordwise_panel_number,panel_numbers,(idx_y*n_cw,))  
    chord_lengths              = DUS(chord_lengths,chord_adjusted,(idx_y*n_cw,))
    tangent_incidence_angle    = DUS(tangent_incidence_angle,tan_incidence,(idx_y*n_cw,))
    exposed_leading_edge_flags = DUS(exposed_leading_edge_flags,exposed_leading_edge_flag,(idx_y,))
    
    indices = [leading_edge_indices,trailing_edge_indices,panels_per_strip,chordwise_panel_number,chord_lengths,
               tangent_incidence_angle,exposed_leading_edge_flags]
    
    


    #increment i_break if needed; check for end of wing----------------------------------------------------
    cond = y_b[idx_y] == break_spans[i_break+1]
    i_break += 1*cond
    
    # Functionally this doesn't do anything until the final break
    x = x.at[-(n_cw+1):].set(xi_prime_bs)
    y = y.at[-(n_cw+1):].set(y_prime_bs)
    z = z.at[-(n_cw+1):].set(zeta_prime_bs)    
    
    #End 'for each strip' loop     
    
    coords = [xah,yah,zah,xbh,ybh,zbh,xch,ych,zch,xa1,ya1,za1,xa2,ya2,za2,xb1,yb1,zb1,xb2,yb2,zb2,xac,yac,zac,xbc,ybc,
              zbc,xa_te,ya_te,za_te,xb_te,yb_te,zb_te,xc,yc,zc,x,y,z,cs_w]    
    
    return indices, i_break, coords

# ----------------------------------------------------------------------
#  Discretize Fuselage
# ----------------------------------------------------------------------
## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
def generate_fuselage_and_nacelle_vortex_distribution(VD,fus,n_cw,n_sw,precision,model_geometry=False):
    """ This generates the vortex distribution points on a fuselage or nacelle component
    Assumptions: 
    If nacelle has segments defined, the mean width and height of the nacelle is used
    Source:   
    None
    
    Inputs:   
    VD                   - vortex distribution
    
    Properties Used:
    N/A
    """    
    
    fhs_xa1 = jnp.zeros(n_cw*n_sw)
    fhs_ya1 = jnp.zeros(n_cw*n_sw)
    fhs_za1 = jnp.zeros(n_cw*n_sw)
    fhs_xa2 = jnp.zeros(n_cw*n_sw)
    fhs_ya2 = jnp.zeros(n_cw*n_sw)
    fhs_za2 = jnp.zeros(n_cw*n_sw)
    fhs_xb1 = jnp.zeros(n_cw*n_sw)
    fhs_yb1 = jnp.zeros(n_cw*n_sw)
    fhs_zb1 = jnp.zeros(n_cw*n_sw)
    fhs_yb2 = jnp.zeros(n_cw*n_sw)
    fhs_xb2 = jnp.zeros(n_cw*n_sw)
    fhs_zb2 = jnp.zeros(n_cw*n_sw)
    fhs_xah = jnp.zeros(n_cw*n_sw)
    fhs_yah = jnp.zeros(n_cw*n_sw)
    fhs_zah = jnp.zeros(n_cw*n_sw)
    fhs_xbh = jnp.zeros(n_cw*n_sw)
    fhs_ybh = jnp.zeros(n_cw*n_sw)
    fhs_zbh = jnp.zeros(n_cw*n_sw)
    fhs_xch = jnp.zeros(n_cw*n_sw)
    fhs_ych = jnp.zeros(n_cw*n_sw)
    fhs_zch = jnp.zeros(n_cw*n_sw)
    fhs_xc  = jnp.zeros(n_cw*n_sw)
    fhs_yc  = jnp.zeros(n_cw*n_sw)
    fhs_zc  = jnp.zeros(n_cw*n_sw)
    fhs_xac = jnp.zeros(n_cw*n_sw)
    fhs_yac = jnp.zeros(n_cw*n_sw)
    fhs_zac = jnp.zeros(n_cw*n_sw)
    fhs_xbc = jnp.zeros(n_cw*n_sw)
    fhs_ybc = jnp.zeros(n_cw*n_sw)
    fhs_zbc = jnp.zeros(n_cw*n_sw)
    fhs_x   = jnp.zeros((n_cw+1)*(n_sw+1))
    fhs_y   = jnp.zeros((n_cw+1)*(n_sw+1))
    fhs_z   = jnp.zeros((n_cw+1)*(n_sw+1))      

    fvs_xc    = jnp.zeros(n_cw*n_sw)
    fvs_zc    = jnp.zeros(n_cw*n_sw)
    fvs_yc    = jnp.zeros(n_cw*n_sw)   
    fvs_x     = jnp.zeros((n_cw+1)*(n_sw+1))
    fvs_y     = jnp.zeros((n_cw+1)*(n_sw+1))
    fvs_z     = jnp.zeros((n_cw+1)*(n_sw+1))   
    fus_v_cs  = jnp.zeros(n_sw)     
    
    # arrays to hold strip discretization values
    leading_edge_indices    = jnp.array([],dtype=bool)    
    trailing_edge_indices   = jnp.array([],dtype=bool)    
    panels_per_strip        = jnp.array([],dtype=jnp.int16)
    chordwise_panel_number  = jnp.array([],dtype=jnp.int16)
    chord_lengths           = jnp.array([],dtype=precision)               
    tangent_incidence_angle = jnp.array([],dtype=precision)               

    # geometry values
    origin     = fus.origin[0]

    # --TO DO-- model fuselage segments if defined, else use the following code
    
    # Horizontal Sections of fuselage
    fhs        = Data()        
    fhs.origin = jnp.zeros((n_sw+1,3))        
    fhs.chord  = jnp.zeros((n_sw+1))         
    fhs.sweep  = jnp.zeros((n_sw+1))     
                 
    fvs        = Data() 
    fvs.origin = jnp.zeros((n_sw+1,3))
    fvs.chord  = jnp.zeros((n_sw+1)) 
    fvs.sweep  = jnp.zeros((n_sw+1))

    if isinstance(fus, Fuselage):

        # Compute the curvature of the nose/tail given fineness ratio. Curvature is derived from general quadratic equation
        # This method relates the fineness ratio to the quadratic curve formula via a spline fit interpolation
        vec1               = jnp.array([2 , 1.5, 1.2 , 1])
        vec2               = jnp.array([1  ,1.57 , 3.2,  8])
        x                  = jnp.linspace(0,1,4)
        fus_nose_curvature = jnp.interp(jnp.interp(fus.fineness.nose,vec2,x), x , vec1)
        fus_tail_curvature = jnp.interp(jnp.interp(fus.fineness.tail,vec2,x), x , vec1)
        semispan_h = fus.width * 0.5
        semispan_v = fus.heights.maximum * 0.5
        si         = jnp.arange(1,((n_sw*2)+2))
        spacing    = jnp.cos((2*si - 1)/(2*len(si))*jnp.pi)
        h_array    = semispan_h*spacing[0:int((len(si)+1)/2)][::-1]
        v_array    = semispan_v*spacing[0:int((len(si)+1)/2)][::-1]

        for i in range(n_sw+1):
            fhs_cabin_length  = fus.lengths.total - (fus.lengths.nose + fus.lengths.tail)
            fhs.nose_length   = ((1 - ((abs(h_array[i]/semispan_h))**fus_nose_curvature ))**(1/fus_nose_curvature))*fus.lengths.nose
            fhs.tail_length   = ((1 - ((abs(h_array[i]/semispan_h))**fus_tail_curvature ))**(1/fus_tail_curvature))*fus.lengths.tail
            fhs.nose_origin   = fus.lengths.nose - fhs.nose_length
            fhs.origin        = fhs.origin.at[i,:].set(jnp.array([fhs.nose_origin , h_array[i], 0.])) # Local origin
            fhs.chord         = fhs.chord.at[i].set(fhs_cabin_length + fhs.nose_length + fhs.tail_length)

            fvs_cabin_length  = fus.lengths.total - (fus.lengths.nose + fus.lengths.tail)
            fvs.nose_length   = ((1 - ((abs(v_array[i]/semispan_v))**fus_nose_curvature ))**(1/fus_nose_curvature))*fus.lengths.nose
            fvs.tail_length   = ((1 - ((abs(v_array[i]/semispan_v))**fus_tail_curvature ))**(1/fus_tail_curvature))*fus.lengths.tail
            fvs.nose_origin   = fus.lengths.nose - fvs.nose_length
            fvs.origin        = fvs.origin.at[i,:].set(jnp.array([origin[0] + fvs.nose_origin , origin[1] , origin[2]+  v_array[i]]))
            fvs.chord         = fvs.chord.at[i].set(fvs_cabin_length + fvs.nose_length + fvs.tail_length)

        fhs.sweep = jnp.concatenate([jnp.arctan((fhs.origin[:,0][1:] - fhs.origin[:,0][:-1])/(fhs.origin[:,1][1:]  - fhs.origin[:,1][:-1])) ,jnp.zeros(1)])
        fvs.sweep = jnp.concatenate([jnp.arctan((fvs.origin[:,0][1:] - fvs.origin[:,0][:-1])/(fvs.origin[:,2][1:]  - fvs.origin[:,2][:-1])) ,jnp.zeros(1)])

    elif isinstance(fus, Nacelle):
        num_nac_segs = len(fus.Segments.keys())
        if num_nac_segs>1:
            widths  = jnp.zeros(num_nac_segs)
            heights = jnp.zeros(num_nac_segs)
            for i_seg in range(num_nac_segs):
                widths  = widths.at[i_seg].set(fus.Segments[i_seg].width)
                heights = heights.at[i_seg].set(fus.Segments[i_seg].height)
            mean_width   = jnp.mean(widths)
            mean_height  = jnp.mean(heights)
        else:
            mean_width   = fus.diameter
            mean_height  = fus.diameter
        length = fus.length

        # geometry values
        semispan_h = mean_width * 0.5
        semispan_v = mean_height * 0.5

        si         = jnp.arange(1,((n_sw*2)+2))
        spacing    = jnp.cos((2*si - 1)/(2*len(si))*jnp.pi)
        h_array    = semispan_h*spacing[0:int((len(si)+1)/2)][::-1]
        v_array    = semispan_v*spacing[0:int((len(si)+1)/2)][::-1]

        for i in range(n_sw+1):
            fhs.chord = fhs.chord.at[i].set(length)
            fvs.chord = fvs.chord.at[i].set(length)
            
    # ---------------------------------------------------------------------------------------
    # STEP 9: Define coordinates of panels horseshoe vortices and control points  
    # ---------------------------------------------------------------------------------------        
    fhs_eta_a = h_array[:-1] 
    fhs_eta_b = h_array[1:]            
    fhs_del_y = h_array[1:] - h_array[:-1]
    fhs_eta   = h_array[1:] - fhs_del_y/2

    fvs_eta_a = v_array[:-1] 
    fvs_eta_b = v_array[1:]                  
    fvs_del_y = v_array[1:] - v_array[:-1]
    fvs_eta   = v_array[1:] - fvs_del_y/2 

    fhs_cs = jnp.concatenate([fhs.chord,fhs.chord])
    fvs_cs = jnp.concatenate([fvs.chord,fvs.chord])
    
    fus_h_area = 0
    fus_v_area = 0    

    # define coordinates of horseshoe vortices and control points       
    for idx_y in range(n_sw):  
        idx_x = jnp.arange(n_cw)

        # fuselage horizontal section 
        delta_x_a = fhs.chord[idx_y]/n_cw      
        delta_x_b = fhs.chord[idx_y + 1]/n_cw    
        delta_x   = (fhs.chord[idx_y]+fhs.chord[idx_y + 1])/(2*n_cw)

        fhs_xi_a1 = fhs.origin[idx_y][0] + delta_x_a*idx_x                    # x coordinate of top left corner of panel
        fhs_xi_ah = fhs.origin[idx_y][0] + delta_x_a*idx_x + delta_x_a*0.25   # x coordinate of left corner of panel
        fhs_xi_a2 = fhs.origin[idx_y][0] + delta_x_a*idx_x + delta_x_a        # x coordinate of bottom left corner of bound vortex 
        fhs_xi_ac = fhs.origin[idx_y][0] + delta_x_a*idx_x + delta_x_a*0.75   # x coordinate of bottom left corner of control point vortex  
        fhs_xi_b1 = fhs.origin[idx_y+1][0] + delta_x_b*idx_x                  # x coordinate of top right corner of panel      
        fhs_xi_bh = fhs.origin[idx_y+1][0] + delta_x_b*idx_x + delta_x_b*0.25 # x coordinate of right corner of bound vortex         
        fhs_xi_b2 = fhs.origin[idx_y+1][0] + delta_x_b*idx_x + delta_x_b      # x coordinate of bottom right corner of panel
        fhs_xi_bc = fhs.origin[idx_y+1][0] + delta_x_b*idx_x + delta_x_b*0.75 # x coordinate of bottom right corner of control point vortex         
        fhs_xi_c  = (fhs.origin[idx_y][0] + fhs.origin[idx_y+1][0])/2  + delta_x*idx_x + delta_x*0.75   # x coordinate three-quarter chord control point for each panel
        fhs_xi_ch = (fhs.origin[idx_y][0] + fhs.origin[idx_y+1][0])/2  + delta_x*idx_x + delta_x*0.25   # x coordinate center of bound vortex of each panel 


        fhs_xa1 = fhs_xa1.at[idx_y*n_cw:(idx_y+1)*n_cw].set(fhs_xi_a1                       + fus.origin[0][0] )
        fhs_ya1 = fhs_ya1.at[idx_y*n_cw:(idx_y+1)*n_cw].set(jnp.ones(n_cw)*fhs_eta_a[idx_y]  + fus.origin[0][1])  
        fhs_za1 = fhs_za1.at[idx_y*n_cw:(idx_y+1)*n_cw].set(jnp.zeros(n_cw)                  + fus.origin[0][2])
        fhs_xa2 = fhs_xa2.at[idx_y*n_cw:(idx_y+1)*n_cw].set(fhs_xi_a2                       + fus.origin[0][0] )
        fhs_ya2 = fhs_ya2.at[idx_y*n_cw:(idx_y+1)*n_cw].set(jnp.ones(n_cw)*fhs_eta_a[idx_y]  + fus.origin[0][1]) 
        fhs_za2 = fhs_za2.at[idx_y*n_cw:(idx_y+1)*n_cw].set(jnp.zeros(n_cw)                  + fus.origin[0][2])     
        fhs_xb1 = fhs_xb1.at[idx_y*n_cw:(idx_y+1)*n_cw].set(fhs_xi_b1                       + fus.origin[0][0] ) 
        fhs_yb1 = fhs_yb1.at[idx_y*n_cw:(idx_y+1)*n_cw].set(jnp.ones(n_cw)*fhs_eta_b[idx_y]  + fus.origin[0][1]) 
        fhs_zb1 = fhs_zb1.at[idx_y*n_cw:(idx_y+1)*n_cw].set(jnp.zeros(n_cw)                  + fus.origin[0][2])
        fhs_xb2 = fhs_xb2.at[idx_y*n_cw:(idx_y+1)*n_cw].set(fhs_xi_b2                       + fus.origin[0][0] )
        fhs_yb2 = fhs_yb2.at[idx_y*n_cw:(idx_y+1)*n_cw].set(jnp.ones(n_cw)*fhs_eta_b[idx_y]  + fus.origin[0][1])
        fhs_zb2 = fhs_zb2.at[idx_y*n_cw:(idx_y+1)*n_cw].set(jnp.zeros(n_cw)                  + fus.origin[0][2])      
        fhs_xah = fhs_xah.at[idx_y*n_cw:(idx_y+1)*n_cw].set(fhs_xi_ah                       + fus.origin[0][0] ) 
        fhs_yah = fhs_yah.at[idx_y*n_cw:(idx_y+1)*n_cw].set(jnp.ones(n_cw)*fhs_eta_a[idx_y]  + fus.origin[0][1]) 
        fhs_zah = fhs_zah.at[idx_y*n_cw:(idx_y+1)*n_cw].set(jnp.zeros(n_cw)                  + fus.origin[0][2])            
        fhs_xbh = fhs_xbh.at[idx_y*n_cw:(idx_y+1)*n_cw].set(fhs_xi_bh                       + fus.origin[0][0] )
        fhs_ybh = fhs_ybh.at[idx_y*n_cw:(idx_y+1)*n_cw].set(jnp.ones(n_cw)*fhs_eta_b[idx_y]  + fus.origin[0][1]) 
        fhs_zbh = fhs_zbh.at[idx_y*n_cw:(idx_y+1)*n_cw].set(jnp.zeros(n_cw)                  + fus.origin[0][2])   
        fhs_xch = fhs_xch.at[idx_y*n_cw:(idx_y+1)*n_cw].set(fhs_xi_ch                       + fus.origin[0][0] )
        fhs_ych = fhs_ych.at[idx_y*n_cw:(idx_y+1)*n_cw].set(jnp.ones(n_cw)*fhs_eta[idx_y]    + fus.origin[0][1])               
        fhs_zch = fhs_zch.at[idx_y*n_cw:(idx_y+1)*n_cw].set(jnp.zeros(n_cw)                  + fus.origin[0][2])    
        fhs_xc  = fhs_xc .at[idx_y*n_cw:(idx_y+1)*n_cw].set(fhs_xi_c                        + fus.origin[0][0] )
        fhs_yc  = fhs_yc .at[idx_y*n_cw:(idx_y+1)*n_cw].set(jnp.ones(n_cw)*fhs_eta[idx_y]    + fus.origin[0][1]) 
        fhs_zc  = fhs_zc .at[idx_y*n_cw:(idx_y+1)*n_cw].set(jnp.zeros(n_cw)                  + fus.origin[0][2])      
        fhs_xac = fhs_xac.at[idx_y*n_cw:(idx_y+1)*n_cw].set(fhs_xi_ac                       + fus.origin[0][0] )
        fhs_yac = fhs_yac.at[idx_y*n_cw:(idx_y+1)*n_cw].set(jnp.ones(n_cw)*fhs_eta_a[idx_y]  + fus.origin[0][1])
        fhs_zac = fhs_zac.at[idx_y*n_cw:(idx_y+1)*n_cw].set(jnp.zeros(n_cw)                  + fus.origin[0][2])
        fhs_xbc = fhs_xbc.at[idx_y*n_cw:(idx_y+1)*n_cw].set(fhs_xi_bc                       + fus.origin[0][0] )
        fhs_ybc = fhs_ybc.at[idx_y*n_cw:(idx_y+1)*n_cw].set(jnp.ones(n_cw)*fhs_eta_b[idx_y]  + fus.origin[0][1])                            
        fhs_zbc = fhs_zbc.at[idx_y*n_cw:(idx_y+1)*n_cw].set(jnp.zeros(n_cw)                  + fus.origin[0][2])             
        fhs_x   = fhs_x.at[idx_y*(n_cw+1):(idx_y+1)*(n_cw+1)].set(jnp.concatenate([fhs_xi_a1,jnp.array([fhs_xi_a2[-1]])]) + fus.origin[0][0])
        fhs_y   = fhs_y.at[idx_y*(n_cw+1):(idx_y+1)*(n_cw+1)].set(jnp.ones(n_cw+1)*fhs_eta_a[idx_y]  + fus.origin[0][1])       
        fhs_z   = fhs_z.at[idx_y*(n_cw+1):(idx_y+1)*(n_cw+1)].set(jnp.zeros(n_cw+1)                  + fus.origin[0][2])

        # fuselage vertical section                      
        delta_x_a = fvs.chord[idx_y]/n_cw      
        delta_x_b = fvs.chord[idx_y + 1]/n_cw    
        delta_x   = (fvs.chord[idx_y]+fvs.chord[idx_y + 1])/(2*n_cw)   

        fvs_xi_a1 = fvs.origin[idx_y][0] + delta_x_a*idx_x                    # z coordinate of top left corner of panel
        fvs_xi_ah = fvs.origin[idx_y][0] + delta_x_a*idx_x + delta_x_a*0.25   # z coordinate of left corner of panel
        fvs_xi_a2 = fvs.origin[idx_y][0] + delta_x_a*idx_x + delta_x_a        # z coordinate of bottom left corner of bound vortex 
        fvs_xi_ac = fvs.origin[idx_y][0] + delta_x_a*idx_x + delta_x_a*0.75   # z coordinate of bottom left corner of control point vortex  
        fvs_xi_b1 = fvs.origin[idx_y+1][0] + delta_x_b*idx_x                    # z coordinate of top right corner of panel      
        fvs_xi_bh = fvs.origin[idx_y+1][0] + delta_x_b*idx_x + delta_x_b*0.25   # z coordinate of right corner of bound vortex         
        fvs_xi_b2 = fvs.origin[idx_y+1][0] + delta_x_b*idx_x + delta_x_b        # z coordinate of bottom right corner of panel
        fvs_xi_bc = fvs.origin[idx_y+1][0] + delta_x_b*idx_x + delta_x_b*0.75   # z coordinate of bottom right corner of control point vortex         
        fvs_xi_c  = (fvs.origin[idx_y][0] + fvs.origin[idx_y+1][0])/2 + delta_x *idx_x + delta_x*0.75     # z coordinate three-quarter chord control point for each panel
        fvs_xi_ch = (fvs.origin[idx_y][0] + fvs.origin[idx_y+1][0])/2 + delta_x *idx_x + delta_x*0.25     # z coordinate center of bound vortex of each panel 

        fvs_xc = fvs_xc.at[idx_y*n_cw:(idx_y+1)*n_cw].set(fvs_xi_c                       + fus.origin[0][0])
        fvs_zc = fvs_zc.at[idx_y*n_cw:(idx_y+1)*n_cw].set(jnp.ones(n_cw)*fvs_eta[idx_y]   + fus.origin[0][2])
        fvs_yc = fvs_yc.at[idx_y*n_cw:(idx_y+1)*n_cw].set(jnp.zeros(n_cw)                 + fus.origin[0][1]) 
        fvs_x  = fvs_x.at[idx_y*(n_cw+1):(idx_y+1)*(n_cw+1)].set(jnp.concatenate([fvs_xi_a1,jnp.array([fvs_xi_a2[-1]])]) + fus.origin[0][0])
        fvs_z  = fvs_z.at[idx_y*(n_cw+1):(idx_y+1)*(n_cw+1)].set(jnp.ones(n_cw+1)*fvs_eta_a[idx_y] + fus.origin[0][2])
        fvs_y  = fvs_y.at[idx_y*(n_cw+1):(idx_y+1)*(n_cw+1)].set(jnp.zeros(n_cw+1)                 + fus.origin[0][1])
        
        fus_h_area += ((fhs.chord[idx_y]+fhs.chord[idx_y + 1])/2)*(fhs_eta_b[idx_y] - fhs_eta_a[idx_y])
        fus_v_area += ((fvs.chord[idx_y]+fvs.chord[idx_y + 1])/2)*(fvs_eta_b[idx_y] - fvs_eta_a[idx_y])
        
        # store this strip's discretization information
        LE_inds        = jnp.full(n_cw, False)
        TE_inds        = jnp.full(n_cw, False)
        LE_inds        = LE_inds.at[0].set(True)
        TE_inds        = TE_inds.at[-1].set(True)
        
        RNMAX          = jnp.ones(n_cw, jnp.int16)*n_cw
        panel_numbers  = jnp.linspace(1,n_cw,n_cw, dtype=jnp.int16)
        
        i_LE, i_TE     = idx_y*n_cw, (idx_y+1)*n_cw-1
        LE_X           = (fhs_xa1[i_LE] + fhs_xb1[i_LE])/2
        LE_Z           = (fhs_za1[i_LE] + fhs_zb1[i_LE])/2
        TE_X           = (fhs_xa2[i_TE] + fhs_xb2[i_TE])/2
        TE_Z           = (fhs_za2[i_TE] + fhs_zb2[i_TE])/2           
        chord_adjusted = jnp.ones(n_cw) * jnp.sqrt((TE_X-LE_X)**2 + (TE_Z-LE_Z)**2) # CHORD in vorlax
        tan_incidence  = jnp.ones(n_cw) * (LE_Z-TE_Z)/(LE_X-TE_X)                  # ZETA  in vorlax      
        chord_adjusted = jnp.array(chord_adjusted, dtype=precision)
        tan_incidence  = jnp.array(tan_incidence , dtype=precision)
        
        leading_edge_indices    = jnp.append(leading_edge_indices   , LE_inds       ) 
        trailing_edge_indices   = jnp.append(trailing_edge_indices  , TE_inds       )            
        panels_per_strip        = jnp.append(panels_per_strip       , RNMAX         )
        chordwise_panel_number  = jnp.append(chordwise_panel_number , panel_numbers )  
        chord_lengths           = jnp.append(chord_lengths          , chord_adjusted)
        tangent_incidence_angle = jnp.append(tangent_incidence_angle, tan_incidence )        

    # xyz positions for the right side of this fuselage's outermost panels
    fhs_x = fhs_x.at[-(n_cw+1):].set(jnp.concatenate([fhs_xi_b1,jnp.array([fhs_xi_b2[-1]])])+ fus.origin[0][0])
    fhs_y = fhs_y.at[-(n_cw+1):].set(jnp.ones(n_cw+1)*fhs_eta_b[idx_y]  + fus.origin[0][1])        
    fhs_z = fhs_z.at[-(n_cw+1):].set(jnp.zeros(n_cw+1)                  + fus.origin[0][2]        )
    fvs_x = fvs_x.at[-(n_cw+1):].set(jnp.concatenate([fvs_xi_a1,jnp.array([fvs_xi_a2[-1]])]) + fus.origin[0][0])
    fvs_z = fvs_z.at[-(n_cw+1):].set(jnp.ones(n_cw+1)*fvs_eta_a[idx_y] + fus.origin[0][2]         )
    fvs_y = fvs_y.at[-(n_cw+1):].set(jnp.zeros(n_cw+1)                 + fus.origin[0][1]   )
    fhs_cs =  (fhs.chord[:-1]+fhs.chord[1:])/2
    fvs_cs =  (fvs.chord[:-1]+fvs.chord[1:])/2  
    
    # find the location of the trailing edge panels of each wing
    locations = ((jnp.linspace(1,n_sw,n_sw, endpoint = True) * n_cw) - 1).astype(int)
    fhs_xc_te1 = jnp.repeat(jnp.atleast_2d(fhs_xc[locations]), n_cw , axis = 0)
    fhs_yc_te1 = jnp.repeat(jnp.atleast_2d(fhs_yc[locations]), n_cw , axis = 0)
    fhs_zc_te1 = jnp.repeat(jnp.atleast_2d(fhs_zc[locations]), n_cw , axis = 0)        
    fhs_xa_te1 = jnp.repeat(jnp.atleast_2d(fhs_xa2[locations]), n_cw , axis = 0)
    fhs_ya_te1 = jnp.repeat(jnp.atleast_2d(fhs_ya2[locations]), n_cw , axis = 0)
    fhs_za_te1 = jnp.repeat(jnp.atleast_2d(fhs_za2[locations]), n_cw , axis = 0)
    fhs_xb_te1 = jnp.repeat(jnp.atleast_2d(fhs_xb2[locations]), n_cw , axis = 0)
    fhs_yb_te1 = jnp.repeat(jnp.atleast_2d(fhs_yb2[locations]), n_cw , axis = 0)
    fhs_zb_te1 = jnp.repeat(jnp.atleast_2d(fhs_zb2[locations]), n_cw , axis = 0)     
    
    fhs_xc_te = jnp.hstack(fhs_xc_te1.T)
    fhs_yc_te = jnp.hstack(fhs_yc_te1.T)
    fhs_zc_te = jnp.hstack(fhs_zc_te1.T)        
    fhs_xa_te = jnp.hstack(fhs_xa_te1.T)
    fhs_ya_te = jnp.hstack(fhs_ya_te1.T)
    fhs_za_te = jnp.hstack(fhs_za_te1.T)
    fhs_xb_te = jnp.hstack(fhs_xb_te1.T)
    fhs_yb_te = jnp.hstack(fhs_yb_te1.T)
    fhs_zb_te = jnp.hstack(fhs_zb_te1.T)     
    
    fhs_xc_te = jnp.concatenate([fhs_xc_te , fhs_xc_te ])
    fhs_yc_te = jnp.concatenate([fhs_yc_te , fhs_yc_te ])
    fhs_zc_te = jnp.concatenate([fhs_zc_te ,-fhs_zc_te ])                 
    fhs_xa_te = jnp.concatenate([fhs_xa_te , fhs_xa_te ])
    fhs_ya_te = jnp.concatenate([fhs_ya_te , fhs_ya_te ])
    fhs_za_te = jnp.concatenate([fhs_za_te ,-fhs_za_te ])            
    fhs_xb_te = jnp.concatenate([fhs_xb_te , fhs_xb_te ])
    fhs_yb_te = jnp.concatenate([fhs_yb_te , fhs_yb_te ])
    fhs_zb_te = jnp.concatenate([fhs_zb_te ,-fhs_zb_te ])    

    # Horizontal Fuselage Sections 
    wing_areas = []
    wing_areas.append(jnp.array(fus_h_area, dtype=precision))
    wing_areas.append(jnp.array(fus_h_area, dtype=precision))
    
    # store points of horizontal section of fuselage 
    fhs_cs  = jnp.concatenate([fhs_cs, fhs_cs])
    fhs_xah = jnp.concatenate([fhs_xah, fhs_xah])
    fhs_yah = jnp.concatenate([fhs_yah,-fhs_yah])
    fhs_zah = jnp.concatenate([fhs_zah, fhs_zah])
    fhs_xbh = jnp.concatenate([fhs_xbh, fhs_xbh])
    fhs_ybh = jnp.concatenate([fhs_ybh,-fhs_ybh])
    fhs_zbh = jnp.concatenate([fhs_zbh, fhs_zbh])
    fhs_xch = jnp.concatenate([fhs_xch, fhs_xch])
    fhs_ych = jnp.concatenate([fhs_ych,-fhs_ych])
    fhs_zch = jnp.concatenate([fhs_zch, fhs_zch])
    fhs_xa1 = jnp.concatenate([fhs_xa1, fhs_xa1])
    fhs_ya1 = jnp.concatenate([fhs_ya1,-fhs_ya1])
    fhs_za1 = jnp.concatenate([fhs_za1, fhs_za1])
    fhs_xa2 = jnp.concatenate([fhs_xa2, fhs_xa2])
    fhs_ya2 = jnp.concatenate([fhs_ya2,-fhs_ya2])
    fhs_za2 = jnp.concatenate([fhs_za2, fhs_za2])
    fhs_xb1 = jnp.concatenate([fhs_xb1, fhs_xb1])
    fhs_yb1 = jnp.concatenate([fhs_yb1,-fhs_yb1])    
    fhs_zb1 = jnp.concatenate([fhs_zb1, fhs_zb1])
    fhs_xb2 = jnp.concatenate([fhs_xb2, fhs_xb2])
    fhs_yb2 = jnp.concatenate([fhs_yb2,-fhs_yb2])            
    fhs_zb2 = jnp.concatenate([fhs_zb2, fhs_zb2])
    fhs_xac = jnp.concatenate([fhs_xac, fhs_xac])
    fhs_yac = jnp.concatenate([fhs_yac,-fhs_yac])
    fhs_zac = jnp.concatenate([fhs_zac, fhs_zac])            
    fhs_xbc = jnp.concatenate([fhs_xbc, fhs_xbc])
    fhs_ybc = jnp.concatenate([fhs_ybc,-fhs_ybc])
    fhs_zbc = jnp.concatenate([fhs_zbc, fhs_zbc])
    fhs_xc  = jnp.concatenate([fhs_xc , fhs_xc ])
    fhs_yc  = jnp.concatenate([fhs_yc ,-fhs_yc])
    fhs_zc  = jnp.concatenate([fhs_zc , fhs_zc ])     
    fhs_x   = jnp.concatenate([fhs_x  , fhs_x  ])
    fhs_y   = jnp.concatenate([fhs_y  ,-fhs_y ])
    fhs_z   = jnp.concatenate([fhs_z  , fhs_z  ])      
    
    if model_geometry == True:
        
        # increment fuslage lifting surface sections  
        VD.n_fus += 2    
        VD.n_cp  += len(fhs_xch)
        VD.n_w   += 2 
        
        # store this fuselage's discretization information 
        n_panels         = n_sw*n_cw
        first_panel_ind  = VD.XAH.size
        first_strip_ind  = [VD.chordwise_breaks.size, VD.chordwise_breaks.size+n_sw]
        chordwise_breaks =  first_panel_ind + jnp.arange(0,2*n_panels)[0::n_cw]        
        
        VD.chordwise_breaks = jnp.append(VD.chordwise_breaks, jnp.int32(chordwise_breaks))
        VD.spanwise_breaks  = jnp.append(VD.spanwise_breaks , jnp.int32(first_strip_ind ))            
        VD.n_sw             = jnp.append(VD.n_sw            , jnp.int16([n_sw, n_sw])    )
        VD.n_cw             = jnp.append(VD.n_cw            , jnp.int16([n_cw, n_cw])    )
        
        VD.leading_edge_indices      = jnp.append(VD.leading_edge_indices     , jnp.tile(leading_edge_indices        , 2) )
        VD.trailing_edge_indices     = jnp.append(VD.trailing_edge_indices    , jnp.tile(trailing_edge_indices       , 2) )           
        VD.panels_per_strip          = jnp.append(VD.panels_per_strip         , jnp.tile(panels_per_strip            , 2) )
        VD.chordwise_panel_number    = jnp.append(VD.chordwise_panel_number   , jnp.tile(chordwise_panel_number      , 2) ) 
        VD.chord_lengths             = jnp.append(VD.chord_lengths            , jnp.tile(chord_lengths               , 2) )
        VD.tangent_incidence_angle   = jnp.append(VD.tangent_incidence_angle  , jnp.tile(tangent_incidence_angle     , 2) ) 
        VD.exposed_leading_edge_flag = jnp.append(VD.exposed_leading_edge_flag, jnp.tile(jnp.ones(n_sw,dtype=jnp.int16), 2) )
    
        # Store fus in vehicle vector  
        VD.XAH    = jnp.append(VD.XAH  , jnp.array(fhs_xah  , dtype=precision))
        VD.YAH    = jnp.append(VD.YAH  , jnp.array(fhs_yah  , dtype=precision))
        VD.ZAH    = jnp.append(VD.ZAH  , jnp.array(fhs_zah  , dtype=precision))
        VD.XBH    = jnp.append(VD.XBH  , jnp.array(fhs_xbh  , dtype=precision))
        VD.YBH    = jnp.append(VD.YBH  , jnp.array(fhs_ybh  , dtype=precision))
        VD.ZBH    = jnp.append(VD.ZBH  , jnp.array(fhs_zbh  , dtype=precision))
        VD.XCH    = jnp.append(VD.XCH  , jnp.array(fhs_xch  , dtype=precision))
        VD.YCH    = jnp.append(VD.YCH  , jnp.array(fhs_ych  , dtype=precision))
        VD.ZCH    = jnp.append(VD.ZCH  , jnp.array(fhs_zch  , dtype=precision))     
        VD.XA1    = jnp.append(VD.XA1  , jnp.array(fhs_xa1  , dtype=precision))
        VD.YA1    = jnp.append(VD.YA1  , jnp.array(fhs_ya1  , dtype=precision))
        VD.ZA1    = jnp.append(VD.ZA1  , jnp.array(fhs_za1  , dtype=precision))
        VD.XA2    = jnp.append(VD.XA2  , jnp.array(fhs_xa2  , dtype=precision))
        VD.YA2    = jnp.append(VD.YA2  , jnp.array(fhs_ya2  , dtype=precision))
        VD.ZA2    = jnp.append(VD.ZA2  , jnp.array(fhs_za2  , dtype=precision))    
        VD.XB1    = jnp.append(VD.XB1  , jnp.array(fhs_xb1  , dtype=precision))
        VD.YB1    = jnp.append(VD.YB1  , jnp.array(fhs_yb1  , dtype=precision))
        VD.ZB1    = jnp.append(VD.ZB1  , jnp.array(fhs_zb1  , dtype=precision))
        VD.XB2    = jnp.append(VD.XB2  , jnp.array(fhs_xb2  , dtype=precision))                
        VD.YB2    = jnp.append(VD.YB2  , jnp.array(fhs_yb2  , dtype=precision))        
        VD.ZB2    = jnp.append(VD.ZB2  , jnp.array(fhs_zb2  , dtype=precision))  
        VD.XC_TE  = jnp.append(VD.XC_TE, jnp.array(fhs_xc_te, dtype=precision))
        VD.YC_TE  = jnp.append(VD.YC_TE, jnp.array(fhs_yc_te, dtype=precision)) 
        VD.ZC_TE  = jnp.append(VD.ZC_TE, jnp.array(fhs_zc_te, dtype=precision))          
        VD.XA_TE  = jnp.append(VD.XA_TE, jnp.array(fhs_xa_te, dtype=precision))
        VD.YA_TE  = jnp.append(VD.YA_TE, jnp.array(fhs_ya_te, dtype=precision)) 
        VD.ZA_TE  = jnp.append(VD.ZA_TE, jnp.array(fhs_za_te, dtype=precision)) 
        VD.XB_TE  = jnp.append(VD.XB_TE, jnp.array(fhs_xb_te, dtype=precision))
        VD.YB_TE  = jnp.append(VD.YB_TE, jnp.array(fhs_yb_te, dtype=precision)) 
        VD.ZB_TE  = jnp.append(VD.ZB_TE, jnp.array(fhs_zb_te, dtype=precision))      
        VD.XAC    = jnp.append(VD.XAC  , jnp.array(fhs_xac  , dtype=precision))
        VD.YAC    = jnp.append(VD.YAC  , jnp.array(fhs_yac  , dtype=precision)) 
        VD.ZAC    = jnp.append(VD.ZAC  , jnp.array(fhs_zac  , dtype=precision)) 
        VD.XBC    = jnp.append(VD.XBC  , jnp.array(fhs_xbc  , dtype=precision))
        VD.YBC    = jnp.append(VD.YBC  , jnp.array(fhs_ybc  , dtype=precision)) 
        VD.ZBC    = jnp.append(VD.ZBC  , jnp.array(fhs_zbc  , dtype=precision))  
        VD.XC     = jnp.append(VD.XC   , jnp.array(fhs_xc   , dtype=precision))
        VD.YC     = jnp.append(VD.YC   , jnp.array(fhs_yc   , dtype=precision))
        VD.ZC     = jnp.append(VD.ZC   , jnp.array(fhs_zc   , dtype=precision))  
        VD.CS     = jnp.append(VD.CS   , jnp.array(fhs_cs   , dtype=precision)) 
        VD.X      = jnp.append(VD.X    , jnp.array(fhs_x    , dtype=precision))  
        VD.Y      = jnp.append(VD.Y    , jnp.array(fhs_y    , dtype=precision))  
        VD.Z      = jnp.append(VD.Z    , jnp.array(fhs_z    , dtype=precision))
        
        VD.wing_areas = jnp.append(VD.wing_areas, jnp.array(wing_areas))
        
        VL = VD.vortex_lift
        VL.append(False)
        VL.append(False)
    
    
    return VD

# ----------------------------------------------------------------------
#  Panel Computations
# ----------------------------------------------------------------------
## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
def compute_panel_area(VD):
    """ This computes the area of the panels on the lifting surface of the vehicle 

    Assumptions: 
    None

    Source:   
    None
    
    Inputs:   
    VD                   - vortex distribution    
    
    Properties Used:
    N/A
    """     
    
    # create vectors for panel corders
    P1P2 = jnp.array([VD.XB1 - VD.XA1,VD.YB1 - VD.YA1,VD.ZB1 - VD.ZA1]).T
    P1P3 = jnp.array([VD.XA2 - VD.XA1,VD.YA2 - VD.YA1,VD.ZA2 - VD.ZA1]).T
    P2P3 = jnp.array([VD.XA2 - VD.XB1,VD.YA2 - VD.YB1,VD.ZA2 - VD.ZB1]).T
    P2P4 = jnp.array([VD.XB2 - VD.XB1,VD.YB2 - VD.YB1,VD.ZB2 - VD.ZB1]).T   
    
    # compute area of quadrilateral panel
    A_panel = 0.5*(jnp.linalg.norm(jnp.cross(P1P2,P1P3),axis=1) + jnp.linalg.norm(jnp.cross(P2P3, P2P4),axis=1))
    
    return A_panel


## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
def compute_unit_normal(VD):
    """ This computes the unit normal vector of each panel


    Assumptions: 
    None

    Source:
    None
    
    Inputs:   
    VD                   - vortex distribution    
    
    Properties Used:
    N/A
    """     

     # create vectors for panel
    P1P2 = jnp.array([VD.XB1 - VD.XA1,VD.YB1 - VD.YA1,VD.ZB1 - VD.ZA1]).T
    P1P3 = jnp.array([VD.XA2 - VD.XA1,VD.YA2 - VD.YA1,VD.ZA2 - VD.ZA1]).T

    cross = jnp.cross(P1P2,P1P3) 

    unit_normal = (cross.T / jnp.linalg.norm(cross,axis=1)).T

     # adjust Z values, no values should point down, flip vectors if so
    #condition = jnp.where(unit_normal[:,2]<0)
    cond = jnp.tile((unit_normal[:,2]<0)[:,jnp.newaxis],3)
    unit_normal = jnp.where(cond,-unit_normal,unit_normal)
    #unit_normal = unit_normal.at[condition,:].set(-unit_normal[condition,:])
    

    return unit_normal

# ----------------------------------------------------------------------
#  Rotation functions
# ----------------------------------------------------------------------
def rotate_points_about_line(point_on_line, direction_unit_vector, rotation_angle, points):
    """ This computes the location of given points after rotating about an arbitrary 
    line that passes through a given point. An important thing to note is that this
    function does not modify the original points. It instead makes copies of the points
    to rotate, rotates the copies, the outputs the copies as np.arrays.

    Assumptions: 
    None

    Source:   
    https://sites.google.com/site/glennmurray/Home/rotation-matrices-and-formulas/rotation-about-an-arbitrary-axis-in-3-dimensions
    
    Inputs:   
    point_on_line         - a list or array of size 3 corresponding to point coords (a,b,c)
    direction_unit_vector - a list or array of size 3 corresponding to unit vector  <u,v,w>
    rotation_angle        - angle of rotation in radians
    points                - a list or array of size 3 corresponding to the lists (xs, ys, zs)
                            where xs, ys, and zs are the (x,y,z) coords of the points 
                            that will be rotated
    
    Properties Used:
    N/A
    """       
    a,  b,  c  = point_on_line
    u,  v,  w  = direction_unit_vector
    xs, ys, zs = jnp.array(points[0]), jnp.array(points[1]), jnp.array(points[2])
    
    cos         = jnp.cos(rotation_angle)
    sin         = jnp.sin(rotation_angle)
    uvw_dot_xyz = u*xs + v*ys + w*zs
    
    xs_prime = (a*(v**2 + w**2) - u*(b*v + c*w - uvw_dot_xyz))*(1-cos)  +  xs*cos  +  (-c*v + b*w - w*ys + v*zs)*sin
    ys_prime = (b*(u**2 + w**2) - v*(a*u + c*w - uvw_dot_xyz))*(1-cos)  +  ys*cos  +  ( c*u - a*w + w*xs - u*zs)*sin
    zs_prime = (c*(u**2 + v**2) - w*(a*u + b*v - uvw_dot_xyz))*(1-cos)  +  zs*cos  +  (-b*u + a*v - v*xs + u*ys)*sin
    
    return xs_prime, ys_prime, zs_prime
    
def make_hinge_quaternion(point_on_line, direction_unit_vector, rotation_angle):
    """ This make a quaternion that will rotate a vector about a the line that 
    passes through the point 'point_on_line' and has direction 'direction_unit_vector'.
    The quat rotates 'rotation_angle' radians. The quat is meant to be multiplied by
    the vector [x  y  z  1]

    Assumptions: 
    None

    Source:   
    https://sites.google.com/site/glennmurray/Home/rotation-matrices-and-formulas/rotation-about-an-arbitrary-axis-in-3-dimensions
    
    Inputs:   
    point_on_line         - a list or array of size 3 corresponding to point coords (a,b,c)
    direction_unit_vector - a list or array of size 3 corresponding to unit vector  <u,v,w>
    rotation_angle        - angle of rotation in radians
    n_points              - number of points that will be rotated
    
    Properties Used:
    N/A
    """       
    a,  b,  c  = point_on_line
    u,  v,  w  = direction_unit_vector
    
    cos         = jnp.cos(rotation_angle)
    sin         = jnp.sin(rotation_angle)
    
    q11 = u**2 + (v**2 + w**2)*cos
    q12 = u*v*(1-cos) - w*sin
    q13 = u*w*(1-cos) + v*sin
    q14 = (a*(v**2 + w**2) - u*(b*v + c*w))*(1-cos)  +  (b*w - c*v)*sin
    
    q21 = u*v*(1-cos) + w*sin
    q22 = v**2 + (u**2 + w**2)*cos
    q23 = v*w*(1-cos) - u*sin
    q24 = (b*(u**2 + w**2) - v*(a*u + c*w))*(1-cos)  +  (c*u - a*w)*sin
    
    q31 = u*w*(1-cos) - v*sin
    q32 = v*w*(1-cos) + u*sin
    q33 = w**2 + (u**2 + v**2)*cos
    q34 = (c*(u**2 + v**2) - w*(a*u + b*v))*(1-cos)  +  (a*v - b*u)*sin    
    
    quat = jnp.array([[q11, q12, q13, q14],
                      [q21, q22, q23, q24],
                      [q31, q32, q33, q34],
                      [0. , 0. , 0. , 1. ]])
    
    return quat

def rotate_points_with_quaternion(quat, points):
    """ This rotates the points by a quaternion

    Assumptions: 
    None

    Source:   
    https://sites.google.com/site/glennmurray/Home/rotation-matrices-and-formulas/rotation-about-an-arbitrary-axis-in-3-dimensions
    
    Inputs:   
    quat     - a quaternion that will rotate the given points about a line which 
               is not necessarily at the origin  
    points   - a list or array of size 3 corresponding to the lists (xs, ys, zs)
               where xs, ys, and zs are the (x,y,z) coords of the points 
               that will be rotated
    
    Outputs:
    xs, ys, zs - np arrays of the rotated points' xyz coordinates
    
    Properties Used:
    N/A
    """     
    vectors = jnp.array([points[0],points[1],points[2],jnp.ones(len(points[0]))]).T
    x_primes, y_primes, z_primes = jnp.sum(quat[0]*vectors, axis=1), jnp.sum(quat[1]*vectors, axis=1), jnp.sum(quat[2]*vectors, axis=1)
    return x_primes, y_primes, z_primes
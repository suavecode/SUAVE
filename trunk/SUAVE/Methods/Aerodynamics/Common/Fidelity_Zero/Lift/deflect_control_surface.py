## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
# deflect_control_surface.py
# 
# Created:  Jul 2022, A. Blaufox & E. Botero
# Modified:
#           

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
from SUAVE.Components.Wings import All_Moving_Surface
from SUAVE.Core import Data
from .generate_VD_helpers import postprocess_VD

# ----------------------------------------------------------------------
#  Deflect Control Surface
# ----------------------------------------------------------------------

## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
def deflect_control_surfaces(VD,geometry,settings):
    """ 
    Goes through a vehicle and updates the control surface deflections in the VD. Crucially this rebuilds the VD as a
    postprocess step
    
    Assumptions: 

    Source:  


    Inputs: 
    VD - vehicle vortex distribution              [Unitless] 
    geometry.wings                                [Unitless]  
    settings.floating_point_precision             [np.dtype]

    Outputs:      
    VD - vehicle vortex distribution              [Unitless] 

    Properties Used:
    N/A
    """     
    
    # Loop over t ewings
    for wing in VD.VLM_wings:
        wing_is_all_moving = (not wing.is_a_control_surface) and issubclass(wing.wing_type, All_Moving_Surface)        
        if wing.is_a_control_surface or wing_is_all_moving:
            # Deflect the control surface
            VD, wing = deflect_control_surface(VD, wing)    
        
    VD = postprocess_VD(VD, settings)
    
    
    return VD


## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
def deflect_control_surface(VD,wing):
    """ 
    Deflects the panels of a vortex distribution that correspond to the given VLM_wing. 
    
    Assumptions: 
    If the user calls this function outside of generate_vortex_distribution,
    SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.postprocess_VD MUST be called
    right after

    Source:  


    Inputs: 
    VD                   - vehicle vortex distribution                    [Unitless] 
    wing                 - a VLM_wing object that was generated in the    [Unitless] 
                           original generate_vortex_distribution call. 
    wing.deflection_last - last deflection applied to this wing           [radians] 
    wing.deflection      - deflection to set this wing to.                [radians] 
    
    Outputs:      
    VD       - vehicle vortex distribution                    [Unitless] 
    wing     - VLM_wing object                                [Unitless] 


    Properties Used:
    N/A
    """     
    
    # Unpack number of strips for this wing
    n_sw     = wing.n_sw
    n_cw     = wing.n_cw
    sym_para = wing.symmetric    


    # Symmetry loop
    signs         = np.array([1, -1], dtype=int) # acts as a multiplier for symmetry. -1 is only ever used for symmetric wings
    symmetry_mask = [True,sym_para]
    for sym_sign in signs[symmetry_mask]:    
        
        # Pull out initial VD data points of surface
        condition      = VD.surface_ID      == wing.surface_ID*sym_sign
        condition_full = VD.surface_ID_full == wing.surface_ID*sym_sign
        xi_prime_a1    = VD.XA1[condition]
        xi_prime_ac    = VD.XAC[condition]
        xi_prime_ah    = VD.XAH[condition]
        xi_prime_a2    = VD.XA2[condition]
        y_prime_a1     = VD.YA1[condition]
        y_prime_ah     = VD.YAH[condition]
        y_prime_ac     = VD.YAC[condition]
        y_prime_a2     = VD.YA2[condition]
        zeta_prime_a1  = VD.ZA1[condition]
        zeta_prime_ah  = VD.ZAH[condition]
        zeta_prime_ac  = VD.ZAC[condition]
        zeta_prime_a2  = VD.ZA2[condition]
        xi_prime_b1    = VD.XB1[condition]
        xi_prime_bh    = VD.XBH[condition]
        xi_prime_bc    = VD.XBC[condition]
        xi_prime_b2    = VD.XB2[condition]
        y_prime_b1     = VD.YB1[condition]
        y_prime_bh     = VD.YBH[condition]
        y_prime_bc     = VD.YBC[condition]
        y_prime_b2     = VD.YB2[condition]
        zeta_prime_b1  = VD.ZB1[condition]
        zeta_prime_bh  = VD.ZBH[condition]
        zeta_prime_bc  = VD.ZBC[condition]
        zeta_prime_b2  = VD.ZB2[condition]
        xi_prime_ch    = VD.XCH[condition]
        xi_prime       = VD.XC [condition]
        y_prime_ch     = VD.YCH[condition]
        y_prime        = VD.YC [condition]
        zeta_prime_ch  = VD.ZCH[condition]
        zeta_prime     = VD.ZC [condition]
        
        X_as = np.zeros_like(VD.X[condition_full][:-(n_cw+1)])
        Y_as = np.zeros_like(VD.Y[condition_full][:-(n_cw+1)])
        Z_as = np.zeros_like(VD.Z[condition_full][:-(n_cw+1)])
        
        
        for idx_y in range(n_sw):
            start     , stop      = idx_y*n_cw    , (idx_y+1)*n_cw
            start_full, stop_full = idx_y*(n_cw+1), (idx_y+1)*(n_cw+1)
            
            # pack strip values
            raw_VD = Data()
            raw_VD.xi_prime_a1   = xi_prime_a1  [start:stop]
            raw_VD.xi_prime_ac   = xi_prime_ac  [start:stop]
            raw_VD.xi_prime_ah   = xi_prime_ah  [start:stop]
            raw_VD.xi_prime_a2   = xi_prime_a2  [start:stop]
            raw_VD.y_prime_a1    = y_prime_a1   [start:stop]
            raw_VD.y_prime_ah    = y_prime_ah   [start:stop]
            raw_VD.y_prime_ac    = y_prime_ac   [start:stop]
            raw_VD.y_prime_a2    = y_prime_a2   [start:stop]
            raw_VD.zeta_prime_a1 = zeta_prime_a1[start:stop]
            raw_VD.zeta_prime_ah = zeta_prime_ah[start:stop]
            raw_VD.zeta_prime_ac = zeta_prime_ac[start:stop]
            raw_VD.zeta_prime_a2 = zeta_prime_a2[start:stop]
            raw_VD.xi_prime_b1   = xi_prime_b1  [start:stop]
            raw_VD.xi_prime_bh   = xi_prime_bh  [start:stop]
            raw_VD.xi_prime_bc   = xi_prime_bc  [start:stop]
            raw_VD.xi_prime_b2   = xi_prime_b2  [start:stop]
            raw_VD.y_prime_b1    = y_prime_b1   [start:stop]
            raw_VD.y_prime_bh    = y_prime_bh   [start:stop]
            raw_VD.y_prime_bc    = y_prime_bc   [start:stop]
            raw_VD.y_prime_b2    = y_prime_b2   [start:stop]
            raw_VD.zeta_prime_b1 = zeta_prime_b1[start:stop]
            raw_VD.zeta_prime_bh = zeta_prime_bh[start:stop]
            raw_VD.zeta_prime_bc = zeta_prime_bc[start:stop]
            raw_VD.zeta_prime_b2 = zeta_prime_b2[start:stop]
            raw_VD.xi_prime_ch   = xi_prime_ch  [start:stop]
            raw_VD.xi_prime      = xi_prime     [start:stop]
            raw_VD.y_prime_ch    = y_prime_ch   [start:stop]
            raw_VD.y_prime       = y_prime      [start:stop]
            raw_VD.zeta_prime_ch = zeta_prime_ch[start:stop]
            raw_VD.zeta_prime    = zeta_prime   [start:stop]
            
            # deflect the surface
            raw_VD = deflect_control_surface_strip(wing, raw_VD, idx_y==0, sym_sign)
            
            # unpack strip values into surface values
            xi_prime_a1  [start:stop]  = raw_VD.xi_prime_a1  
            xi_prime_ac  [start:stop]  = raw_VD.xi_prime_ac  
            xi_prime_ah  [start:stop]  = raw_VD.xi_prime_ah  
            xi_prime_a2  [start:stop]  = raw_VD.xi_prime_a2  
            y_prime_a1   [start:stop]  = raw_VD.y_prime_a1   
            y_prime_ah   [start:stop]  = raw_VD.y_prime_ah   
            y_prime_ac   [start:stop]  = raw_VD.y_prime_ac   
            y_prime_a2   [start:stop]  = raw_VD.y_prime_a2   
            zeta_prime_a1[start:stop]  = raw_VD.zeta_prime_a1
            zeta_prime_ah[start:stop]  = raw_VD.zeta_prime_ah
            zeta_prime_ac[start:stop]  = raw_VD.zeta_prime_ac
            zeta_prime_a2[start:stop]  = raw_VD.zeta_prime_a2
        
            xi_prime_b1  [start:stop]  = raw_VD.xi_prime_b1  
            xi_prime_bh  [start:stop]  = raw_VD.xi_prime_bh  
            xi_prime_bc  [start:stop]  = raw_VD.xi_prime_bc  
            xi_prime_b2  [start:stop]  = raw_VD.xi_prime_b2  
            y_prime_b1   [start:stop]  = raw_VD.y_prime_b1   
            y_prime_bh   [start:stop]  = raw_VD.y_prime_bh   
            y_prime_bc   [start:stop]  = raw_VD.y_prime_bc   
            y_prime_b2   [start:stop]  = raw_VD.y_prime_b2   
            zeta_prime_b1[start:stop]  = raw_VD.zeta_prime_b1
            zeta_prime_bh[start:stop]  = raw_VD.zeta_prime_bh
            zeta_prime_bc[start:stop]  = raw_VD.zeta_prime_bc
            zeta_prime_b2[start:stop]  = raw_VD.zeta_prime_b2
               
            xi_prime_ch  [start:stop]  = raw_VD.xi_prime_ch  
            xi_prime     [start:stop]  = raw_VD.xi_prime     
            y_prime_ch   [start:stop]  = raw_VD.y_prime_ch   
            y_prime      [start:stop]  = raw_VD.y_prime      
            zeta_prime_ch[start:stop]  = raw_VD.zeta_prime_ch
            zeta_prime   [start:stop]  = raw_VD.zeta_prime   
            
            X_as[start_full:stop_full] = np.append(raw_VD.xi_prime_a1  , raw_VD.xi_prime_a2  [-1])
            Y_as[start_full:stop_full] = np.append(raw_VD.y_prime_a1   , raw_VD.y_prime_a2   [-1])
            Z_as[start_full:stop_full] = np.append(raw_VD.zeta_prime_a1, raw_VD.zeta_prime_a2[-1])
        
        # pack surface VD values into vehicle VD    
        VD.XA1[condition]    = xi_prime_a1    
        VD.XAC[condition]    = xi_prime_ac    
        VD.XAH[condition]    = xi_prime_ah    
        VD.XA2[condition]    = xi_prime_a2    
        VD.YA1[condition]    = y_prime_a1     
        VD.YAH[condition]    = y_prime_ah     
        VD.YAC[condition]    = y_prime_ac     
        VD.YA2[condition]    = y_prime_a2     
        VD.ZA1[condition]    = zeta_prime_a1  
        VD.ZAH[condition]    = zeta_prime_ah  
        VD.ZAC[condition]    = zeta_prime_ac  
        VD.ZA2[condition]    = zeta_prime_a2  
        VD.XB1[condition]    = xi_prime_b1    
        VD.XBH[condition]    = xi_prime_bh    
        VD.XBC[condition]    = xi_prime_bc    
        VD.XB2[condition]    = xi_prime_b2    
        VD.YB1[condition]    = y_prime_b1     
        VD.YBH[condition]    = y_prime_bh     
        VD.YBC[condition]    = y_prime_bc     
        VD.YB2[condition]    = y_prime_b2     
        VD.ZB1[condition]    = zeta_prime_b1  
        VD.ZBH[condition]    = zeta_prime_bh  
        VD.ZBC[condition]    = zeta_prime_bc  
        VD.ZB2[condition]    = zeta_prime_b2  
        VD.XCH[condition]    = xi_prime_ch    
        VD.XC [condition]    = xi_prime       
        VD.YCH[condition]    = y_prime_ch     
        VD.YC [condition]    = y_prime        
        VD.ZCH[condition]    = zeta_prime_ch  
        VD.ZC [condition]    = zeta_prime    
        
        X_last_bs = np.append(raw_VD.xi_prime_b1  , raw_VD.xi_prime_b2  [-1])
        Y_last_bs = np.append(raw_VD.y_prime_b1   , raw_VD.y_prime_b2   [-1])
        Z_last_bs = np.append(raw_VD.zeta_prime_b1, raw_VD.zeta_prime_b2[-1])
        
        VD.X[condition_full] = np.append(X_as, X_last_bs)
        VD.Y[condition_full] = np.append(Y_as, Y_last_bs)
        VD.Z[condition_full] = np.append(Z_as, Z_last_bs)
        
        
    wing.deflection_last = wing.deflection*1.
    
    VD.is_postprocessed = False
    
    return VD, wing



# ----------------------------------------------------------------------
#  Deflect Control Surface Strip
# ----------------------------------------------------------------------

## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
def deflect_control_surface_strip(wing, raw_VD, is_first_strip, sym_sign):
    """ Rotates existing points in the VD with respect to current values of a delta deflection

    Assumptions: 

    Source:  


    Inputs: 
    wing                 - a VLM_wing object that was generated in the    [Unitless] 
                           original generate_vortex_distribution call. 
    wing.deflection_last - last deflection applied to this wing           [radians]
    wing.deflection      - deflection to set this wing to.                [radians]
    
    raw_VD               - undeflected VD pertaining a strip of wing      [Unitless] 
    is_first_strip       - whether this is the first strip of wing        [Unitless]
    sym_sign             - 1 for original side, -1 for symmetric side     [Unitless]
    
    Outputs:      
    raw_VD - deflected VD values pertaining to wing         [Unitless] 


    Properties Used:
    N/A
    """    
    
    # Unpack
    vertical_wing = wing.vertical
    inverted_wing = wing.inverted_wing
    
    x_origin, y_origin, z_origin = wing.origin[0]
    
    y_origin *= sym_sign
    
    xi_prime_a1    = raw_VD.xi_prime_a1   
    xi_prime_ac    = raw_VD.xi_prime_ac   
    xi_prime_ah    = raw_VD.xi_prime_ah   
    xi_prime_a2    = raw_VD.xi_prime_a2   
    y_prime_a1     = raw_VD.y_prime_a1    
    y_prime_ah     = raw_VD.y_prime_ah    
    y_prime_ac     = raw_VD.y_prime_ac    
    y_prime_a2     = raw_VD.y_prime_a2    
    zeta_prime_a1  = raw_VD.zeta_prime_a1 
    zeta_prime_ah  = raw_VD.zeta_prime_ah 
    zeta_prime_ac  = raw_VD.zeta_prime_ac 
    zeta_prime_a2  = raw_VD.zeta_prime_a2 
                                          
    xi_prime_b1    = raw_VD.xi_prime_b1   
    xi_prime_bh    = raw_VD.xi_prime_bh   
    xi_prime_bc    = raw_VD.xi_prime_bc   
    xi_prime_b2    = raw_VD.xi_prime_b2   
    y_prime_b1     = raw_VD.y_prime_b1    
    y_prime_bh     = raw_VD.y_prime_bh    
    y_prime_bc     = raw_VD.y_prime_bc    
    y_prime_b2     = raw_VD.y_prime_b2    
    zeta_prime_b1  = raw_VD.zeta_prime_b1 
    zeta_prime_bh  = raw_VD.zeta_prime_bh 
    zeta_prime_bc  = raw_VD.zeta_prime_bc 
    zeta_prime_b2  = raw_VD.zeta_prime_b2 
                                          
    xi_prime_ch    = raw_VD.xi_prime_ch   
    xi_prime       = raw_VD.xi_prime      
    y_prime_ch     = raw_VD.y_prime_ch    
    y_prime        = raw_VD.y_prime       
    zeta_prime_ch  = raw_VD.zeta_prime_ch 
    zeta_prime     = raw_VD.zeta_prime    

    # flip over y = z for a vertical wing since deflection math assumes horizontal wing--------------------
    y_prime_a1, zeta_prime_a1 = flip_1(y_prime_a1, zeta_prime_a1, vertical_wing, inverted_wing)
    y_prime_ah, zeta_prime_ah = flip_1(y_prime_ah, zeta_prime_ah, vertical_wing, inverted_wing)
    y_prime_ac, zeta_prime_ac = flip_1(y_prime_ac, zeta_prime_ac, vertical_wing, inverted_wing)
    y_prime_a2, zeta_prime_a2 = flip_1(y_prime_a2, zeta_prime_a2, vertical_wing, inverted_wing)
    y_prime_b1, zeta_prime_b1 = flip_1(y_prime_b1, zeta_prime_b1, vertical_wing, inverted_wing)
    y_prime_bh, zeta_prime_bh = flip_1(y_prime_bh, zeta_prime_bh, vertical_wing, inverted_wing)
    y_prime_bc, zeta_prime_bc = flip_1(y_prime_bc, zeta_prime_bc, vertical_wing, inverted_wing)
    y_prime_b2, zeta_prime_b2 = flip_1(y_prime_b2, zeta_prime_b2, vertical_wing, inverted_wing)
    y_prime_ch, zeta_prime_ch = flip_1(y_prime_ch, zeta_prime_ch, vertical_wing, inverted_wing)
    y_prime   , zeta_prime    = flip_1(y_prime   , zeta_prime   , vertical_wing, inverted_wing)

    # Deflect control surfaces-----------------------------------------------------------------------------
    # note:    "positve" deflection corresponds to the RH rule where the axis of rotation is the OUTBOARD-pointing hinge vector
    # symmetry: the LH rule is applied to the reflected surface for non-ailerons. Ailerons follow a RH rule for both sides
    wing_is_all_moving = (not wing.is_a_control_surface) and issubclass(wing.wing_type, All_Moving_Surface)
        
    #For the first strip of the wing, always need to find the hinge root point. The hinge root point and direction vector 
    #found here will not change for the rest of this control surface/all-moving surface. See docstring for reasoning.
    if is_first_strip:
        # get rotation points by iterpolating between strip corners --> le/te, ib/ob = leading/trailing edge, in/outboard
        ib_le_strip_corner = np.array([xi_prime_a1[0 ], y_prime_a1[0 ], zeta_prime_a1[0 ]]) 
        ib_te_strip_corner = np.array([xi_prime_a2[-1], y_prime_a2[-1], zeta_prime_a2[-1]])                    
        
        interp_fractions   = np.array([0.,    2.,    4.   ]) + wing.hinge_fraction
        interp_domains     = np.array([0.,1., 2.,3., 4.,5.])
        interp_ranges_ib   = np.array([ib_le_strip_corner, ib_te_strip_corner]).T.flatten()
        ib_hinge_point     = np.interp(interp_fractions, interp_domains, interp_ranges_ib)
        
        #Find the hinge_vector if this is a control surface or the user has not already defined and chosen to use a specific one                    
        if wing.is_a_control_surface:
            need_to_compute_hinge_vector = True
        else: #wing is an all-moving surface
            hinge_vector                 = wing.hinge_vector
            hinge_vector_is_pre_defined  = (not wing.use_constant_hinge_fraction) and \
                                            not (hinge_vector==np.array([0.,0.,0.])).all()
            need_to_compute_hinge_vector = not hinge_vector_is_pre_defined  
            
        if need_to_compute_hinge_vector:
            ob_le_strip_corner = np.array([xi_prime_b1[0 ], y_prime_b1[0 ], zeta_prime_b1[0 ]])                
            ob_te_strip_corner = np.array([xi_prime_b2[-1], y_prime_b2[-1], zeta_prime_b2[-1]])                         
            interp_ranges_ob   = np.array([ob_le_strip_corner, ob_te_strip_corner]).T.flatten()
            ob_hinge_point     = np.interp(interp_fractions, interp_domains, interp_ranges_ob)
        
            use_root_chord_in_plane_normal = wing_is_all_moving and not wing.use_constant_hinge_fraction
            if use_root_chord_in_plane_normal: ob_hinge_point[0] = ib_hinge_point[0]
        
            hinge_vector       = ob_hinge_point - ib_hinge_point
            hinge_vector       = hinge_vector / np.linalg.norm(hinge_vector)   
        elif wing.vertical: #For a vertical all-moving surface, flip y and z of hinge vector before flipping again later
            hinge_vector[1], hinge_vector[2] = hinge_vector[2], hinge_vector[1] 
            
        #store hinge root point and direction vector
        wing.hinge_root_point = ib_hinge_point
        wing.hinge_vector     = hinge_vector 
        #END first strip calculations
    
    # get deflection angle
    ddeflection      = wing.deflection      - wing.deflection_last               # This is a delta deflection
    slat_multiplier  = (1 - wing.is_slat)   - wing.is_slat                       # Flip signs if it's a slat
    sym_multiplier   = (1 - (sym_sign==-1)) - wing.sign_duplicate*(sym_sign==-1) # If it's the symmetric side
    ver_multiplier   = (1 - vertical_wing)-1*vertical_wing                       # Vertical multiplier

    delta_deflection = slat_multiplier*sym_multiplier*ver_multiplier*ddeflection
        
    # make quaternion rotation matrix
    quaternion   = make_hinge_quaternion(wing.hinge_root_point, wing.hinge_vector, delta_deflection)
    
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
                                                                                                                             
    # flip over y = z again after deflecting---------------------------------------------------------------
    
    y_prime_a1, zeta_prime_a1 = flip_2(y_prime_a1, zeta_prime_a1, vertical_wing, inverted_wing)
    y_prime_ah, zeta_prime_ah = flip_2(y_prime_ah, zeta_prime_ah, vertical_wing, inverted_wing)
    y_prime_ac, zeta_prime_ac = flip_2(y_prime_ac, zeta_prime_ac, vertical_wing, inverted_wing)
    y_prime_a2, zeta_prime_a2 = flip_2(y_prime_a2, zeta_prime_a2, vertical_wing, inverted_wing)
    y_prime_b1, zeta_prime_b1 = flip_2(y_prime_b1, zeta_prime_b1, vertical_wing, inverted_wing)
    y_prime_bh, zeta_prime_bh = flip_2(y_prime_bh, zeta_prime_bh, vertical_wing, inverted_wing)
    y_prime_bc, zeta_prime_bc = flip_2(y_prime_bc, zeta_prime_bc, vertical_wing, inverted_wing)
    y_prime_b2, zeta_prime_b2 = flip_2(y_prime_b2, zeta_prime_b2, vertical_wing, inverted_wing)
    y_prime_ch, zeta_prime_ch = flip_2(y_prime_ch, zeta_prime_ch, vertical_wing, inverted_wing)
    y_prime   , zeta_prime    = flip_2(y_prime,    zeta_prime   , vertical_wing, inverted_wing)

    # Pack the VD
    raw_VD.xi_prime_a1   = xi_prime_a1      
    raw_VD.xi_prime_ac   = xi_prime_ac      
    raw_VD.xi_prime_ah   = xi_prime_ah      
    raw_VD.xi_prime_a2   = xi_prime_a2      
    raw_VD.y_prime_a1    = y_prime_a1       
    raw_VD.y_prime_ah    = y_prime_ah       
    raw_VD.y_prime_ac    = y_prime_ac       
    raw_VD.y_prime_a2    = y_prime_a2       
    raw_VD.zeta_prime_a1 = zeta_prime_a1    
    raw_VD.zeta_prime_ah = zeta_prime_ah    
    raw_VD.zeta_prime_ac = zeta_prime_ac    
    raw_VD.zeta_prime_a2 = zeta_prime_a2    
                                            
    raw_VD.xi_prime_b1   = xi_prime_b1      
    raw_VD.xi_prime_bh   = xi_prime_bh      
    raw_VD.xi_prime_bc   = xi_prime_bc      
    raw_VD.xi_prime_b2   = xi_prime_b2      
    raw_VD.y_prime_b1    = y_prime_b1       
    raw_VD.y_prime_bh    = y_prime_bh       
    raw_VD.y_prime_bc    = y_prime_bc       
    raw_VD.y_prime_b2    = y_prime_b2       
    raw_VD.zeta_prime_b1 = zeta_prime_b1    
    raw_VD.zeta_prime_bh = zeta_prime_bh    
    raw_VD.zeta_prime_bc = zeta_prime_bc    
    raw_VD.zeta_prime_b2 = zeta_prime_b2    
                                            
    raw_VD.xi_prime_ch   = xi_prime_ch      
    raw_VD.xi_prime      = xi_prime         
    raw_VD.y_prime_ch    = y_prime_ch       
    raw_VD.y_prime       = y_prime          
    raw_VD.zeta_prime_ch = zeta_prime_ch    
    raw_VD.zeta_prime    = zeta_prime       

    
    return raw_VD    

# ----------------------------------------------------------------------
#  Make Hinge Quaternion
# ----------------------------------------------------------------------

## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
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
    
    cos         = np.cos(rotation_angle)
    sin         = np.sin(rotation_angle)
    
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
    
    quat = np.array([[q11, q12, q13, q14],
                     [q21, q22, q23, q24],
                     [q31, q32, q33, q34],
                     [0. , 0. , 0. , 1. ]])
    
    return quat

# ----------------------------------------------------------------------
#  Rotate Points with Quaternion
# ----------------------------------------------------------------------

## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
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
    vectors = np.array([points[0],points[1],points[2],np.ones(len(points[0]))]).T
    x_primes, y_primes, z_primes = np.sum(quat[0]*vectors, axis=1), np.sum(quat[1]*vectors, axis=1), np.sum(quat[2]*vectors, axis=1)
    return x_primes, y_primes, z_primes
    
    
    
def flip_1(A,B,T1,T2):
    """ This swaps values based on a double boolean
    
    If T1 is false A and B are kept as is
    
    If T1 is true A and B are swapped
    If T2 is true 

    Assumptions: 
    None

    Source:   
    N/A
    
    
    Inputs:   
    input_A
    input_B
    bool1

    Outputs:
    swapped or not values
    
    Properties Used:
    N/A
    """
    
    NT1 = 1-T1
    
    return B*T1*T2+A*NT1, A*T1+B*NT1
    
def flip_2(A,B,T1,T2):
    """ This swaps values based on a double boolean
    
    If T1 is false A and B are kept as is
    
    If T1 is true A and B are swapped
    If T2 is true 

    Assumptions: 
    None

    Source:   
    N/A
    
    
    Inputs:   
    input_A
    input_B
    bool1

    Outputs:
    swapped or not values
    
    Properties Used:
    N/A
    """
    NT1 = 1-T1
    
    return B*T1+A*NT1, A*T1*T2+B*NT1
    

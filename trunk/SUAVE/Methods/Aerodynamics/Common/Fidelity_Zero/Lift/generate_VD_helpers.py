## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
# generate_VD_helpers.py
# 
# Created:  Jun 2021, A. Blaufox
# Modified: 
#           

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# package imports 
import numpy as np

# ----------------------------------------------------------------------
#  postprocess_VD
# ----------------------------------------------------------------------
## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift    
def postprocess_VD(VD, settings):
    """ 
    Recomputes data about the VD. Should be called any time VD panel 
    values (e.g. VD.XA1, VD.XCH, etc) are changed.

    Assumptions: 

    Source:  


    Inputs: 
    VD       - vehicle vortex distribution                    [Unitless] 
    settings - Vortex_Lattice.settings                        [Unitless] 
    
    Outputs:      
    VD       - vehicle vortex distribution                    [Unitless] 


    Properties Used:
    N/A
    """     
    #unpack
    precision  = settings.floating_point_precision   
    LE_ind     = VD.leading_edge_indices
    TE_ind     = VD.trailing_edge_indices
    strip_n_cw = VD.panels_per_strip[LE_ind]
    
    # @Emilio: assumes last VLM_wing is the last to get discretized & thus has highest surface_ID. 
    #          This is correct for now, but do we want to continue to assume this, or should this be 
    #           rewritten to allow any order of wings
    last_wing_ID = list(VD.VLM_wings.values())[-1].surface_ID
    is_VLM_wing  = np.abs(VD.surface_ID) <= last_wing_ID

    # Compute Panel Areas and Normals
    VD.panel_areas = np.array(compute_panel_area(VD) , dtype=precision)
    VD.normals     = np.array(compute_unit_normal(VD), dtype=precision)  
    
    # Reshape chord_lengths
    VD.chord_lengths = np.atleast_2d(VD.chord_lengths) #need to be 2D for later calculations
    
    # Compute panel-wise variables used in VORLAX
    X1c   = (VD.XA1+VD.XB1)/2
    X2c   = (VD.XA2+VD.XB2)/2
    Z1c   = (VD.ZA1+VD.ZB1)/2
    Z2c   = (VD.ZA2+VD.ZB2)/2
    SLOPE = (Z2c - Z1c)/(X2c - X1c)
    SLE   = SLOPE[LE_ind]    
    D    = np.sqrt((VD.YAH-VD.YBH)**2+(VD.ZAH-VD.ZBH)**2)[LE_ind]
    
    # Compute strip-wise values
    LE_X           = X1c[LE_ind]
    LE_Z           = Z1c[LE_ind]
    TE_X           = X2c[TE_ind]
    TE_Z           = Z2c[TE_ind]
    tan_incidence  = np.repeat((LE_Z-TE_Z)/(LE_X-TE_X), strip_n_cw) # ZETA  in vorlax
    chord_adjusted = np.repeat(np.sqrt((TE_X-LE_X)**2 + (TE_Z-LE_Z)**2), strip_n_cw) # CHORD in vorlax
    
    # Compute wing-only values
    wing_strip_n_cw  = VD.panels_per_strip[is_VLM_wing*LE_ind]
    
    XC_TE_wings  = np.repeat(VD.XC [is_VLM_wing*TE_ind], wing_strip_n_cw)
    YC_TE_wings  = np.repeat(VD.YC [is_VLM_wing*TE_ind], wing_strip_n_cw)
    ZC_TE_wings  = np.repeat(VD.ZC [is_VLM_wing*TE_ind], wing_strip_n_cw)
    XA_TE_wings  = np.repeat(VD.XA2[is_VLM_wing*TE_ind], wing_strip_n_cw)
    YA_TE_wings  = np.repeat(VD.YA2[is_VLM_wing*TE_ind], wing_strip_n_cw)
    ZA_TE_wings  = np.repeat(VD.ZA2[is_VLM_wing*TE_ind], wing_strip_n_cw)
    XB_TE_wings  = np.repeat(VD.XB2[is_VLM_wing*TE_ind], wing_strip_n_cw)
    YB_TE_wings  = np.repeat(VD.YB2[is_VLM_wing*TE_ind], wing_strip_n_cw)
    ZB_TE_wings  = np.repeat(VD.ZB2[is_VLM_wing*TE_ind], wing_strip_n_cw)
    
    Y_SW        = VD.YC[is_VLM_wing*TE_ind]
    
    # Pack VORLAX variables
    VD.SLOPE                   = SLOPE
    VD.SLE                     = SLE
    VD.D                       = D         
    VD.tangent_incidence_angle = tan_incidence
    VD.chord_lengths           = np.atleast_2d(chord_adjusted)
    VD.Y_SW                    = Y_SW
    
    VD.XC_TE[is_VLM_wing]  = XC_TE_wings
    VD.YC_TE[is_VLM_wing]  = YC_TE_wings
    VD.ZC_TE[is_VLM_wing]  = ZC_TE_wings
    VD.XA_TE[is_VLM_wing]  = XA_TE_wings
    VD.YA_TE[is_VLM_wing]  = YA_TE_wings
    VD.ZA_TE[is_VLM_wing]  = ZA_TE_wings
    VD.XB_TE[is_VLM_wing]  = XB_TE_wings
    VD.YB_TE[is_VLM_wing]  = YB_TE_wings
    VD.ZB_TE[is_VLM_wing]  = ZB_TE_wings
    
    VD.is_postprocessed = True
    
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
    P1P2 = np.array([VD.XB1 - VD.XA1,VD.YB1 - VD.YA1,VD.ZB1 - VD.ZA1]).T
    P1P3 = np.array([VD.XA2 - VD.XA1,VD.YA2 - VD.YA1,VD.ZA2 - VD.ZA1]).T
    P2P3 = np.array([VD.XA2 - VD.XB1,VD.YA2 - VD.YB1,VD.ZA2 - VD.ZB1]).T
    P2P4 = np.array([VD.XB2 - VD.XB1,VD.YB2 - VD.YB1,VD.ZB2 - VD.ZB1]).T   
    
    # compute area of quadrilateral panel
    A_panel = 0.5*(np.linalg.norm(np.cross(P1P2,P1P3),axis=1) + np.linalg.norm(np.cross(P2P3, P2P4),axis=1))
    
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
    P1P2 = np.array([VD.XB1 - VD.XA1,VD.YB1 - VD.YA1,VD.ZB1 - VD.ZA1]).T
    P1P3 = np.array([VD.XA2 - VD.XA1,VD.YA2 - VD.YA1,VD.ZA2 - VD.ZA1]).T

    cross = np.cross(P1P2,P1P3) 

    unit_normal = (cross.T / np.linalg.norm(cross,axis=1)).T

     # adjust Z values, no values should point down, flip vectors if so
    unit_normal[unit_normal[:,2]<0,:] = -unit_normal[unit_normal[:,2]<0,:]

    return unit_normal
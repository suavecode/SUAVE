## @ingroup Methods-Geometry-Two_Dimensional-Cross_Section-Airfoil
# compute_airfoil_properties.py
# 
# Created:  Mar 2019, M. Clarke
# Modified: Mar 2020, M. Clarke
#           Jan 2021, E. Botero
#           Jan 2021, R. Erhard
#           Nov 2021, R. Erhard

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core               import Data , Units, to_jnumpy
from SUAVE.Methods.Aerodynamics.AERODAS.pre_stall_coefficients import pre_stall_coefficients
from SUAVE.Methods.Aerodynamics.AERODAS.post_stall_coefficients import post_stall_coefficients 
from SUAVE.Methods.Aerodynamics.Airfoil_Panel_Method.airfoil_analysis                    import airfoil_analysis
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.import_airfoil_polars  import import_airfoil_polars 
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.compute_naca_4series   import compute_naca_4series  
from .import_airfoil_polars   import import_airfoil_polars 
import numpy as np

#from jax.scipy.interpolate import RegularGridInterpolator
import jax.numpy as jnp

## @ingroup Methods-Geometry-Two_Dimensional-Cross_Section-Airfoil
def compute_airfoil_properties(airfoil_geometry, airfoil_polar_files = None,use_pre_stall_data=True,linear_lift=False):
    """This computes the lift and drag coefficients of an airfoil in stall regimes using pre-stall
    characterstics and AERODAS formation for post stall characteristics. This is useful for 
    obtaining a more accurate prediction of wing and blade loading. Pre stall characteristics 
    are obtained in the form of a text file of airfoil polar data obtained from airfoiltools.com
    
    Assumptions:
    Uses AERODAS formulation for post stall characteristics 

    Source:
    Models of Lift and Drag Coefficients of Stalled and Unstalled Airfoils in Wind Turbines and Wind Tunnels
    by D Spera, 2008

    Inputs:
    a_geo                  <string>
    a_polar                <string>
    use_pre_stall_data     [Boolean]
           

    Outputs:
    airfoil_data.
        cl_polars          [unitless]
        cd_polars          [unitless]      
        aoa_sweep          [unitless]
    
    Properties Used:
    N/A
    """   
    
    Airfoil_Data   = Data()  
   
    # ----------------------------------------------------------------------------------------
    # Compute airfoil boundary layers properties 
    # ----------------------------------------------------------------------------------------   
    Airfoil_Data   = compute_boundary_layer_properties(airfoil_geometry,Airfoil_Data)
    num_polars     = len(Airfoil_Data.boundary_layer_reynolds_numbers)
     
    # ----------------------------------------------------------------------------------------
    # Compute extended cl and cd polars 
    # ----------------------------------------------------------------------------------------    
    if airfoil_polar_files != None : 
        # check number of polars per airfoil in batch
        num_polars   = 0 
        n_p = len(airfoil_polar_files)
        if n_p < 3:
            raise AttributeError('Provide three or more airfoil polars to compute surrogate') 
        num_polars = max(num_polars, n_p)         
         
        # read in polars from files overwrite panel code  
        airfoil_file_data                          = import_airfoil_polars(airfoil_polar_files)  
        Airfoil_Data.aoa_from_polar                = airfoil_file_data.angle_of_attacks
        Airfoil_Data.re_from_polar                 = airfoil_file_data.reynolds_number
        Airfoil_Data.lift_coefficients             = airfoil_file_data.lift_coefficients
        Airfoil_Data.drag_coefficients             = airfoil_file_data.drag_coefficients 
        
    # Get all of the coefficients for AERODAS wings
    AoA_sweep_deg         = np.linspace(-90,90,181)
    AoA_sweep_rad         = AoA_sweep_deg*Units.degrees    
    
    # Create an infinite aspect ratio wing
    geometry              = SUAVE.Components.Wings.Wing()
    geometry.aspect_ratio = np.inf
    geometry.section      = Data()  
    
    # Modify the "wing" slightly:
    geometry.thickness_to_chord = airfoil_geometry.thickness_to_chord  
    
    # AERODAS   
    CL = np.zeros((num_polars,len(AoA_sweep_deg)))
    CD = np.zeros((num_polars,len(AoA_sweep_deg)))  
    for j in range(num_polars):    
        airfoil_cl  = Airfoil_Data.lift_coefficients[j,:]
        airfoil_cd  = Airfoil_Data.drag_coefficients[j,:]   
        airfoil_aoa = Airfoil_Data.aoa_from_polar[j,:]/Units.degrees 
    
        # compute airfoil cl and cd for extended AoA range 
        CL[j,:],CD[j,:] = compute_extended_polars(airfoil_cl,airfoil_cd,airfoil_aoa,AoA_sweep_deg,geometry,use_pre_stall_data,linear_lift)  
         
    # ----------------------------------------------------------------------------------------
    # Store data 
    # ----------------------------------------------------------------------------------------    
    Airfoil_Data.reynolds_numbers    = to_jnumpy(Airfoil_Data.re_from_polar)
    Airfoil_Data.angle_of_attacks    = to_jnumpy(AoA_sweep_rad)
    Airfoil_Data.lift_coefficients   = CL 
    Airfoil_Data.drag_coefficients   = CD    
    
    return Airfoil_Data

def compute_extended_polars(airfoil_cl,airfoil_cd,airfoil_aoa,AoA_sweep_deg,geometry,use_pre_stall_data,linear_lift): 
    """ Computes the aerodynamic polars of an airfoil over an extended angle of attack range
    
    Assumptions:
    Uses AERODAS formulation for post stall characteristics 
    Source:
    Models of Lift and Drag Coefficients of Stalled and Unstalled Airfoils in Wind Turbines and Wind Tunnels
    by D Spera, 2008
        
    Inputs:
    airfoil_cl          [unitless]
    airfoil_cd          [unitless]
    airfoil_aoa         [degrees]
    AoA_sweep_deg       [unitless]
    geometry            [N/A]
    use_pre_stall_data  [boolean]
    Outputs: 
    CL                  [unitless]
    CD                  [unitless]       
    
    Properties Used:
    N/A
    """     
    
    # Create dummy settings and state
    settings                                              = Data()
    state                                                 = Data()
    state.conditions                                      = Data()
    state.conditions.aerodynamics                         = Data()
    state.conditions.aerodynamics.pre_stall_coefficients  = Data()
    state.conditions.aerodynamics.post_stall_coefficients = Data() 

    # computing approximate zero lift aoa
    AoA_sweep_radians    = AoA_sweep_deg*Units.degrees
    airfoil_cl_plus      = airfoil_cl[airfoil_cl>0]
    idx_zero_lift        = np.where(airfoil_cl == min(airfoil_cl_plus))[0][0]
    airfoil_cl_crossing  = airfoil_cl[idx_zero_lift-1:idx_zero_lift+1]
    airfoil_aoa_crossing = airfoil_aoa[idx_zero_lift-1:idx_zero_lift+1]
    try:
        A0  = np.interp(0,airfoil_cl_crossing, airfoil_aoa_crossing)* Units.deg 
    except:
        A0 = airfoil_aoa[idx_zero_lift] * Units.deg
    

    # max lift coefficent and associated aoa
    CL1max = np.max(airfoil_cl)
    idx_aoa_max_prestall_cl = np.where(airfoil_cl == CL1max)[0][0]
    ACL1 = airfoil_aoa[idx_aoa_max_prestall_cl] * Units.degrees

    # computing approximate lift curve slope
    linear_idxs = [int(np.where(airfoil_aoa==0)[0]),int(np.where(airfoil_aoa==4)[0])]
    cl_range    = airfoil_cl[linear_idxs]
    aoa_range   = airfoil_aoa[linear_idxs] * Units.degrees
    S1          = (cl_range[1]-cl_range[0])/(aoa_range[1]-aoa_range[0])

    # max drag coefficent and associated aoa
    CD1max  = np.max(airfoil_cd) 
    idx_aoa_max_prestall_cd = np.where(airfoil_cd == CD1max)[0][0]
    ACD1   = airfoil_aoa[idx_aoa_max_prestall_cd] * Units.degrees     
    
    # Find the point of lowest drag and the CD
    idx_CD_min = np.where(airfoil_cd==min(airfoil_cd))[0][0]
    ACDmin     = airfoil_aoa[idx_CD_min] * Units.degrees
    CDmin      = airfoil_cd[idx_CD_min]    
    
    # Setup data structures for this run
    ones = np.ones_like(AoA_sweep_radians)
    settings.section_zero_lift_angle_of_attack                = A0
    state.conditions.aerodynamics.angle_of_attack             = AoA_sweep_radians * ones 
    geometry.section.angle_attack_max_prestall_lift           = ACL1 * ones 
    geometry.pre_stall_maximum_drag_coefficient_angle         = ACD1 * ones 
    geometry.pre_stall_maximum_lift_coefficient               = CL1max * ones 
    geometry.pre_stall_maximum_lift_drag_coefficient          = CD1max * ones 
    geometry.section.minimum_drag_coefficient                 = CDmin * ones 
    geometry.section.minimum_drag_coefficient_angle_of_attack = ACDmin
    geometry.pre_stall_lift_curve_slope                       = S1
    
    # Get prestall coefficients
    CL1, CD1 = pre_stall_coefficients(state,settings,geometry)
    
    # Get poststall coefficents
    CL2, CD2 = post_stall_coefficients(state,settings,geometry)
    
    # Take the maxes
    CL_ij = np.fmax(CL1,CL2)
    CL_ij[AoA_sweep_radians<=A0] = np.fmin(CL1[AoA_sweep_radians<=A0],CL2[AoA_sweep_radians<=A0])
    
    CD_ij = np.fmax(CD1,CD2)
    
    # Pack this loop
    CL   = CL_ij
    CD   = CD_ij 
    
    if linear_lift:
        CL= S1*(AoA_sweep_radians-A0)
    
    if use_pre_stall_data == True:
        CL, CD = apply_pre_stall_data(AoA_sweep_deg, airfoil_aoa, airfoil_cl, airfoil_cd, CL, CD)
     
            
    return CL,CD
 
def apply_pre_stall_data(AoA_sweep_deg, airfoil_aoa, airfoil_cl, airfoil_cd, CL, CD):
    # Coefficients in pre-stall regime taken from experimental data:
    aoa_locs = (AoA_sweep_deg>=airfoil_aoa[0]) * (AoA_sweep_deg<=airfoil_aoa[-1])
    aoa_in_data = AoA_sweep_deg[aoa_locs]
    
    # if the data is within experimental use it, if not keep the surrogate values
    CL[aoa_locs] = airfoil_cl[abs(aoa_in_data[:,None] - airfoil_aoa[None,:]).argmin(axis=-1)]
    CD[aoa_locs] = airfoil_cd[abs(aoa_in_data[:,None] - airfoil_aoa[None,:]).argmin(axis=-1)]
    
    # remove kinks/overlap between pre- and post-stall                
    data_lb = np.where(CD == airfoil_cd[0])[0][0]
    data_ub = np.where(CD == airfoil_cd[-1])[0][-1]
    CD[0:data_lb] = np.maximum(CD[0:data_lb], CD[data_lb]*np.ones_like(CD[0:data_lb]))
    CD[data_ub:]  = np.maximum(CD[data_ub:],  CD[data_ub]*np.ones_like(CD[data_ub:])) 
    
    return CL, CD


## @ingroup Methods-Geometry-Two_Dimensional-Cross_Section-Airfoil
def compute_boundary_layer_properties(airfoil_geometry,Airfoil_Data): 
    '''Computes the boundary layer properties of an airfoil for a sweep of Reynolds numbers 
    and angle of attacks. 
    
    Source:
    None
    
    Assumptions:
    None 
    
    Inputs:
    airfoil_geometry   <data_structure>
    Airfoil_Data       <data_structure>
    
    Outputs:
    Airfoil_Data       <data_structure>
    
    Properties Used:
    N/A
    ''' 
    if airfoil_geometry == None:
        print('No airfoil defined, NACA 0012 surrogates will be used') 
        a_names                       = ['0012']                
        airfoil_geometry              = compute_naca_4series(a_names, npoints= 100)    
    
    AoA_sweep = np.array([-4,0,2,4,8,10,14])*Units.degrees 
    Re_sweep  = np.array([1,5,10,30,50,75,100,200])*1E4  
    AoA_vals  = np.tile(AoA_sweep[None,:],(len(Re_sweep) ,1))
    Re_vals   = np.tile(Re_sweep[:,None],(1, len(AoA_sweep)))     
    
    # run airfoil analysis  
    af_res  = airfoil_analysis(airfoil_geometry,AoA_vals,Re_vals) 
 
    Airfoil_Data.boundary_layer_lift_coefficients                   = af_res.cl
    Airfoil_Data.boundary_layer_drag_coefficients                   = af_res.cd
    Airfoil_Data.boundary_layer_moment_coefficients                 = af_res.cm 
    Airfoil_Data.boundary_layer_angle_of_attacks                    = to_jnumpy(AoA_sweep) 
    Airfoil_Data.boundary_layer_reynolds_numbers                    = to_jnumpy(Re_sweep)
    Airfoil_Data.boundary_layer_theta_lower_surface                 = af_res.theta 
    Airfoil_Data.boundary_layer_delta_lower_surface                 = af_res.delta  
    Airfoil_Data.boundary_layer_delta_star_lower_surface            = af_res.delta_star  
    Airfoil_Data.boundary_layer_cf_lower_surface                    = af_res.Ue_Vinf   
    Airfoil_Data.boundary_layer_Ue_Vinf_lower_surface               = af_res.cf   
    Airfoil_Data.boundary_layer_dcp_dx_lower_surface                = af_res.dcp_dx 
    Airfoil_Data.boundary_layer_theta_upper_surface                 = af_res.theta 
    Airfoil_Data.boundary_layer_delta_upper_surface                 = af_res.delta 
    Airfoil_Data.boundary_layer_delta_star_upper_surface            = af_res.delta_star   
    Airfoil_Data.boundary_layer_cf_upper_surface                    = af_res.Ue_Vinf    
    Airfoil_Data.boundary_layer_Ue_Vinf_upper_surface               = af_res.cf    
    Airfoil_Data.boundary_layer_dcp_dx_upper_surface                = af_res.dcp_dx   
    
    return Airfoil_Data
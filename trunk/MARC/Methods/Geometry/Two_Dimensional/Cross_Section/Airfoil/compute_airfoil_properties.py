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

import MARC
from MARC.Core                                                                          import Data , Units
from MARC.Methods.Aerodynamics.AERODAS.pre_stall_coefficients                           import pre_stall_coefficients
from MARC.Methods.Aerodynamics.AERODAS.post_stall_coefficients                          import post_stall_coefficients  
from MARC.Methods.Aerodynamics.Airfoil_Panel_Method.airfoil_analysis                    import airfoil_analysis
from MARC.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.import_airfoil_polars  import import_airfoil_polars 
from MARC.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.compute_naca_4series   import compute_naca_4series   
import numpy as np

## @ingroup Methods-Geometry-Two_Dimensional-Cross_Section-Airfoil
def compute_airfoil_properties(airfoil_geometry, airfoil_polar_files = None,use_pre_stall_data=True):
    """This computes the aerodynamic properties and coefficients of an airfoil in stall regimes using pre-stall
    characterstics and AERODAS formation for post stall characteristics. This is useful for 
    obtaining a more accurate prediction of wing and blade loading as well as aeroacoustics. Pre stall characteristics 
    are obtained in the form of a text file of airfoil polar data obtained from airfoiltools.com
    
    Assumptions:
        None 
        
    Source
        None
        
    Inputs:
    airfoil_geometry                        <data_structure>
    airfoil_polar_files                     <string>
    boundary_layer_files                    <string>
    use_pre_stall_data                      [Boolean]
    Outputs:
    airfoil_data.
        cl_polars                           [unitless]
        cd_polars                           [unitless]      
        aoa_sweep                           [unitless]
        
        # raw data                          [unitless]
        theta                 [unitless]
        delta                 [unitless]
        delta_star            [unitless] 
        sa_lower_surface                    [unitless]
        ue_lower_surface                    [unitless]
        cf                    [unitless]
        dcp_dx                [unitless] 
        Ret_lower_surface                   [unitless]
        H_lower_surface                     [unitless] 
    
    Properties Used:
    N/A
    """     
    Airfoil_Data   = Data()  
   
    # ----------------------------------------------------------------------------------------
    # Compute airfoil boundary layers properties 
    # ----------------------------------------------------------------------------------------   
    Airfoil_Data   = compute_boundary_layer_properties(airfoil_geometry,Airfoil_Data)
    num_polars     = len(Airfoil_Data.re_from_polar)
     
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
        Airfoil_Data.aoa_from_polar                = airfoil_file_data.aoa_from_polar
        Airfoil_Data.re_from_polar                 = airfoil_file_data.re_from_polar 
        Airfoil_Data.lift_coefficients             = airfoil_file_data.lift_coefficients
        Airfoil_Data.drag_coefficients             = airfoil_file_data.drag_coefficients 
        
    # Get all of the coefficients for AERODAS wings
    AoA_sweep_deg         = np.linspace(-14,90,105)
    AoA_sweep_rad         = AoA_sweep_deg*Units.degrees
    
    # AERODAS   
    CL = np.zeros((num_polars,len(AoA_sweep_deg)))
    CD = np.zeros((num_polars,len(AoA_sweep_deg)))  
    for j in range(num_polars):    
        airfoil_cl  = Airfoil_Data.lift_coefficients[j,:]
        airfoil_cd  = Airfoil_Data.drag_coefficients[j,:]   
        airfoil_aoa = Airfoil_Data.aoa_from_polar[j,:]/Units.degrees 
    
        # compute airfoil cl and cd for extended AoA range 
        CL[j,:],CD[j,:] = compute_extended_polars(airfoil_cl,airfoil_cd,airfoil_aoa,AoA_sweep_deg)  
         
    # ----------------------------------------------------------------------------------------
    # Store data 
    # ----------------------------------------------------------------------------------------    
    Airfoil_Data.reynolds_numbers    = Airfoil_Data.re_from_polar
    Airfoil_Data.angle_of_attacks    = AoA_sweep_rad 
    Airfoil_Data.lift_coefficients   = CL 
    Airfoil_Data.drag_coefficients   = CD    
        
    return Airfoil_Data
 
def compute_extended_polars(airfoil_cl,airfoil_cd,airfoil_aoa,AoA_sweep_deg): 
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
    
    Outputs: 
    CL                  [unitless]
    CD                  [unitless]       
    
    Properties Used:
    N/A
    """    
    
    # Create dummy settings and state
    start_loc = np.where(airfoil_aoa==-5)[0][0]
    end_loc   = np.where(airfoil_aoa== 8)[0][0]
    p1        = np.polyfit(airfoil_aoa[start_loc:end_loc], airfoil_cl[start_loc:end_loc], 1)
    CL        = p1[0]*AoA_sweep_deg + p1[1]  
    p2        = np.polyfit(airfoil_aoa[start_loc:end_loc], airfoil_cd[start_loc:end_loc], 4)
    CD        = p2[0]*(AoA_sweep_deg**4) + p2[1]*(AoA_sweep_deg**3) + p2[2]*(AoA_sweep_deg**2) + p2[3]*(AoA_sweep_deg**1) + p2[4] 
    
    return CL,CD

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
    af_res    = airfoil_analysis(airfoil_geometry,AoA_vals,Re_vals) 
    
    # store data
    Airfoil_Data.aoa_from_polar                       = np.tile(AoA_sweep[None,:],(len(Re_sweep),1)) 
    Airfoil_Data.re_from_polar                        = Re_sweep 
    Airfoil_Data.lift_coefficients                    = af_res.cl
    Airfoil_Data.drag_coefficients                    = af_res.cd
    Airfoil_Data.cm                                   = af_res.cm
    Airfoil_Data.boundary_layer                       = Data() 
    Airfoil_Data.boundary_layer.angle_of_attacks      = AoA_sweep 
    Airfoil_Data.boundary_layer.reynolds_numbers      = Re_sweep      
    Airfoil_Data.boundary_layer.theta                 = af_res.theta 
    Airfoil_Data.boundary_layer.delta                 = af_res.delta  
    Airfoil_Data.boundary_layer.delta_star            = af_res.delta_star  
    Airfoil_Data.boundary_layer.Ue_Vinf               = af_res.Ue_Vinf   
    Airfoil_Data.boundary_layer.cf                    = af_res.cf   
    Airfoil_Data.boundary_layer.dcp_dx                = af_res.dcp_dx 
    
    return Airfoil_Data
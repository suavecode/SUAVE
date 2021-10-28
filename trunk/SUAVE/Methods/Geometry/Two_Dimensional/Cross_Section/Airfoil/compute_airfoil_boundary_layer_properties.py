## @ingroup Methods-Geometry-Two_Dimensional-Cross_Section-Airfoil
# compute_airfoil_boundary_layer_properties.py
# 
# Created:  Oct 2021, M. Clarke 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core               import Data , Units 
from .import_airfoil_geometry import import_airfoil_geometry 
from .import_airfoil_polars   import import_airfoil_polars  
from SUAVE.Methods.Aerodynamics.Airfoil_Panel_Method.airfoil_analysis   import airfoil_analysis 
from scipy.interpolate        import RectBivariateSpline
import numpy as np

## @ingroup Methods-Geometry-Two_Dimensional-Cross_Section-Airfoil
def compute_airfoil_boundary_layer_properties(a_geo,npanel=100):
    """This computes the 
    Inputs:
    a_geo                  <string> 
           

    Outputs:
    airfoil_data.
        cl_polars          [unitless]
        cd_polars          [unitless]      
        aoa_sweep          [unitless]
    
    Properties Used:
    N/A
    """  
    
    num_airfoils     = len(a_geo)  
    airfoil_geometry = import_airfoil_geometry(a_geo, npoints = npanel+2) 
    
    airfoil_bl_surs  = Data()   
    lower_surface_theta_surs       = Data()
    lower_surface_delta_surs       = Data() 
    lower_surface_delta_star_surs  = Data()
    lower_surface_cf_surs          = Data() 
    lower_surface_Ue_surs          = Data()
    lower_surface_H_surs           = Data() 
    upper_surface_theta_surs       = Data()
    upper_surface_delta_surs       = Data() 
    upper_surface_delta_star_surs  = Data()
    upper_surface_cf_surs          = Data() 
    upper_surface_Ue_surs          = Data()
    upper_surface_H_surs           = Data() 
        
    Re         = np.linspace(1E1,5E6,4)
    AoA        = np.linspace(-5,15,5)*Units.degrees
    Re_batch   = np.atleast_2d(Re ).T
    AoA_batch  = np.atleast_2d(AoA).T    
    
    SMOOTHING = 0.1
    TE_idx = 2 #  Trailing Edge Index 
    
    for i in range(num_airfoils):  
        airfoil_geometry = import_airfoil_geometry([a_geo[i]], npoints = npanel+2) 
        AP  = airfoil_analysis(airfoil_geometry,AoA_batch,Re_batch, npanel, batch_analysis = True)  
        lower_surface_theta_surs[a_geo[i]]      = RectBivariateSpline(AoA,Re,AP.theta[TE_idx,:,:], s=SMOOTHING) 
        lower_surface_delta_surs[a_geo[i]]      = RectBivariateSpline(AoA,Re,AP.delta[TE_idx,:,:], s=SMOOTHING) 
        lower_surface_delta_star_surs[a_geo[i]] = RectBivariateSpline(AoA,Re,AP.delta_star[TE_idx,:,:], s=SMOOTHING) 
        lower_surface_cf_surs[a_geo[i]]         = RectBivariateSpline(AoA,Re,AP.Cf[TE_idx,:,:], s=SMOOTHING) 
        lower_surface_Ue_surs[a_geo[i]]         = RectBivariateSpline(AoA,Re,AP.Ue_Vinf[TE_idx,:,:], s=SMOOTHING) 
        lower_surface_H_surs[a_geo[i]]          = RectBivariateSpline(AoA,Re,AP.H[TE_idx,:,:], s=SMOOTHING)   
        upper_surface_theta_surs[a_geo[i]]      = RectBivariateSpline(AoA,Re,AP.theta[-TE_idx,:,:], s=SMOOTHING) 
        upper_surface_delta_surs[a_geo[i]]      = RectBivariateSpline(AoA,Re,AP.delta[-TE_idx,:,:], s=SMOOTHING) 
        upper_surface_delta_star_surs[a_geo[i]] = RectBivariateSpline(AoA,Re,AP.delta_star[-TE_idx,:,:], s=SMOOTHING) 
        upper_surface_cf_surs[a_geo[i]]         = RectBivariateSpline(AoA,Re,AP.Cf[-TE_idx,:,:], s=SMOOTHING) 
        upper_surface_Ue_surs[a_geo[i]]         = RectBivariateSpline(AoA,Re,AP.Ue_Vinf[-TE_idx,:,:], s=SMOOTHING) 
        upper_surface_H_surs[a_geo[i]]          = RectBivariateSpline(AoA,Re,AP.H[-TE_idx,:,:], s=SMOOTHING)           
        
    airfoil_bl_surs.lower_surface_theta_surrogates       = lower_surface_theta_surs
    airfoil_bl_surs.lower_surface_delta_surrogates       = lower_surface_delta_surs
    airfoil_bl_surs.lower_surface_delta_start_surrogates = lower_surface_delta_star_surs
    airfoil_bl_surs.lower_surface_cf_surrogates          = lower_surface_cf_surs
    airfoil_bl_surs.lower_surface_Ue_surrogates          = lower_surface_Ue_surs
    airfoil_bl_surs.lower_surface_H_surrogates           = lower_surface_H_surs 
    airfoil_bl_surs.upper_surface_theta_surrogates       = upper_surface_theta_surs
    airfoil_bl_surs.upper_surface_delta_surrogates       = upper_surface_delta_surs
    airfoil_bl_surs.upper_surface_delta_start_surrogates = upper_surface_delta_star_surs
    airfoil_bl_surs.upper_surface_cf_surrogates          = upper_surface_cf_surs
    airfoil_bl_surs.upper_surface_Ue_surrogates          = upper_surface_Ue_surs
    airfoil_bl_surs.upper_surface_H_surrogates           = upper_surface_H_surs 
    
    return airfoil_bl_surs
 
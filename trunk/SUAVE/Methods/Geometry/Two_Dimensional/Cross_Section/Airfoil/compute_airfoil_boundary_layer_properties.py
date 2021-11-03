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
from SUAVE.Methods.Utilities.Cubic_Spline_Blender import Cubic_Spline_Blender 
from SUAVE.Core import Data
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import RationalQuadratic, ConstantKernel, RBF, Matern
from sklearn import neighbors
from sklearn import svm, linear_model
import numpy as np

## @ingroup Methods-Geometry-Two_Dimensional-Cross_Section-Airfoil
def compute_airfoil_boundary_layer_properties(a_geo,npanel=100,surrogate_type = 'gaussian'):  
    """ Build a surrogate. Multiple options for models are available including:
        -Gaussian Processes
        -KNN
         
        
        Assumptions:
        None
        
        Source:
        N/A
        
        Inputs:
         
        
        Outputs:
         
        airfoil_bl_surs 
            lower_surface_theta_surrogates       [fun()]
            lower_surface_delta_surrogates       [fun()]
            lower_surface_delta_start_surrogates [fun()]
            lower_surface_cf_surrogates          [fun()]
            lower_surface_Ue_surrogates          [fun()]
            lower_surface_H_surrogates           [fun()]
            upper_surface_theta_surrogates       [fun()]
            upper_surface_delta_surrogates       [fun()]
            upper_surface_delta_start_surrogates [fun()]
            upper_surface_cf_surrogates          [fun()]
            upper_surface_Ue_surrogates          [fun()]
            upper_surface_H_surrogates           [fun()]
        
        Properties Used:
        Defaulted values
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
    
    xy         = np.zeros((len(AoA)*len(Re),2))  
    xy[:,0]    = np.repeat(Re,len(AoA), axis = 0)/1E5
    xy[:,1]    = np.tile(AoA,len(Re))
    
    TE_idx    = 2 #  Trailing Edge Index 
    
    for i in range(num_airfoils):  
        airfoil_geometry = import_airfoil_geometry([a_geo[i]], npoints = npanel+2) 
        AP  = airfoil_analysis(airfoil_geometry,AoA_batch,Re_batch, npanel, batch_analysis = True)   
        
        lower_surface_theta      = np.ravel(AP.theta[TE_idx,:,:]/np.mean(AP.theta[TE_idx,:,:]))
        lower_surface_delta      = np.ravel(AP.delta[TE_idx,:,:]/np.mean(AP.delta[TE_idx,:,:])) 
        lower_surface_delta_star = np.ravel(AP.delta_star[TE_idx,:,:]/np.mean(AP.delta_star[TE_idx,:,:]))
        lower_surface_cf         = np.ravel(AP.Cf[TE_idx,:,:]/np.mean(AP.Cf[TE_idx,:,:]))
        lower_surface_Ue         = np.ravel(AP.Ue_Vinf[TE_idx,:,:]/np.mean(AP.Ue_Vinf[TE_idx,:,:]))
        lower_surface_H          = np.ravel(AP.H[TE_idx,:,:]/np.mean(AP.H[TE_idx,:,:]))
        upper_surface_theta      = np.ravel(AP.theta[-TE_idx,:,:]/np.mean(AP.theta[-TE_idx,:,:]))
        upper_surface_delta      = np.ravel(AP.delta[-TE_idx,:,:]/np.mean(AP.delta[-TE_idx,:,:])) 
        upper_surface_delta_star = np.ravel(AP.delta_star[-TE_idx,:,:]/np.mean(AP.delta_star[-TE_idx,:,:]))  
        upper_surface_cf         = np.ravel(AP.Cf[-TE_idx,:,:]/np.mean(AP.Cf[-TE_idx,:,:])) 
        upper_surface_Ue         = np.ravel( AP.Ue_Vinf[-TE_idx,:,:]/np.mean( AP.Ue_Vinf[-TE_idx,:,:])) 
        upper_surface_H          = np.ravel(AP.H[-TE_idx,:,:]/np.mean(AP.H[-TE_idx,:,:]))       
    
    
        # Pick the type of process
        if surrogate_type  == 'gaussian':
            gp_kernel = Matern() 
            regr_ls_theta                = gaussian_process.GaussianProcessRegressor(kernel = gp_kernel)
            regr_ls_delta                = gaussian_process.GaussianProcessRegressor(kernel = gp_kernel)
            regr_ls_delta_star           = gaussian_process.GaussianProcessRegressor(kernel = gp_kernel)
            regr_ls_cf                   = gaussian_process.GaussianProcessRegressor(kernel = gp_kernel)
            regr_ls_Ue                   = gaussian_process.GaussianProcessRegressor(kernel = gp_kernel)
            regr_ls_H                    = gaussian_process.GaussianProcessRegressor(kernel = gp_kernel)
            regr_us_theta                = gaussian_process.GaussianProcessRegressor(kernel = gp_kernel)
            regr_us_delta                = gaussian_process.GaussianProcessRegressor(kernel = gp_kernel)
            regr_us_delta_star           = gaussian_process.GaussianProcessRegressor(kernel = gp_kernel)
            regr_us_cf                   = gaussian_process.GaussianProcessRegressor(kernel = gp_kernel)
            regr_us_Ue                   = gaussian_process.GaussianProcessRegressor(kernel = gp_kernel)
            regr_us_H                    = gaussian_process.GaussianProcessRegressor(kernel = gp_kernel)
            lower_surface_theta_sur      = regr_ls_theta.fit(xy, lower_surface_theta)
            lower_surface_delta_sur      = regr_ls_delta.fit(xy, lower_surface_delta) 
            lower_surface_delta_star_sur = regr_ls_delta_star.fit(xy, lower_surface_delta_star) 
            lower_surface_cf_sur         = regr_ls_cf.fit(xy, lower_surface_cf) 
            lower_surface_Ue_sur         = regr_ls_Ue.fit(xy, lower_surface_Ue) 
            lower_surface_H_sur          = regr_ls_H.fit(xy, lower_surface_H) 
            upper_surface_theta_sur      = regr_us_theta.fit(xy, upper_surface_theta)
            upper_surface_delta_sur      = regr_us_delta.fit(xy, upper_surface_delta) 
            upper_surface_delta_star_sur = regr_us_delta_star.fit(xy, upper_surface_delta_star) 
            upper_surface_cf_sur         = regr_us_cf.fit(xy, upper_surface_cf) 
            upper_surface_Ue_sur         = regr_us_Ue.fit(xy, upper_surface_Ue) 
            upper_surface_H_sur          = regr_us_H.fit(xy, upper_surface_H)   
    
        elif surrogate_type  == 'knn':
            regr_ls_theta                = neighbors.KNeighborsRegressor(n_neighbors=1,weights='distance')
            regr_ls_delta                = neighbors.KNeighborsRegressor(n_neighbors=1,weights='distance')
            regr_ls_delta_star           = neighbors.KNeighborsRegressor(n_neighbors=1,weights='distance')
            regr_ls_cf                   = neighbors.KNeighborsRegressor(n_neighbors=1,weights='distance')
            regr_ls_Ue                   = neighbors.KNeighborsRegressor(n_neighbors=1,weights='distance')
            regr_ls_H                    = neighbors.KNeighborsRegressor(n_neighbors=1,weights='distance')
            regr_us_theta                = neighbors.KNeighborsRegressor(n_neighbors=1,weights='distance')
            regr_us_delta                = neighbors.KNeighborsRegressor(n_neighbors=1,weights='distance')
            regr_us_delta_star           = neighbors.KNeighborsRegressor(n_neighbors=1,weights='distance')
            regr_us_cf                   = neighbors.KNeighborsRegressor(n_neighbors=1,weights='distance')
            regr_us_Ue                   = neighbors.KNeighborsRegressor(n_neighbors=1,weights='distance')
            regr_us_H                    = neighbors.KNeighborsRegressor(n_neighbors=1,weights='distance')
            lower_surface_theta_sur      = regr_ls_theta.fit(xy, lower_surface_theta)
            lower_surface_delta_sur      = regr_ls_delta.fit(xy, lower_surface_delta) 
            lower_surface_delta_star_sur = regr_ls_delta_star.fit(xy, lower_surface_delta_star) 
            lower_surface_cf_sur         = regr_ls_cf.fit(xy, lower_surface_cf) 
            lower_surface_Ue_sur         = regr_ls_Ue.fit(xy, lower_surface_Ue) 
            lower_surface_H_sur          = regr_ls_H.fit(xy, lower_surface_H) 
            upper_surface_theta_sur      = regr_us_theta.fit(xy, upper_surface_theta)
            upper_surface_delta_sur      = regr_us_delta.fit(xy, upper_surface_delta) 
            upper_surface_delta_star_sur = regr_us_delta_star.fit(xy, upper_surface_delta_star) 
            upper_surface_cf_sur         = regr_us_cf.fit(xy, upper_surface_cf) 
            upper_surface_Ue_sur         = regr_us_Ue.fit(xy, upper_surface_Ue) 
            upper_surface_H_sur          = regr_us_H.fit(xy, upper_surface_H)  

        elif surrogate_type  == 'svr':
            regr_ls_theta                = svm.SVR(C=500)
            regr_ls_delta                = svm.SVR(C=500)
            regr_ls_delta_star           = svm.SVR(C=500)
            regr_ls_cf                   = svm.SVR(C=500)
            regr_ls_Ue                   = svm.SVR(C=500)
            regr_ls_H                    = svm.SVR(C=500)
            regr_us_theta                = svm.SVR(C=500)
            regr_us_delta                = svm.SVR(C=500)
            regr_us_delta_star           = svm.SVR(C=500)
            regr_us_cf                   = svm.SVR(C=500)
            regr_us_Ue                   = svm.SVR(C=500)
            regr_us_H                    = svm.SVR(C=500) 
            lower_surface_theta_sur      = regr_ls_theta.fit(xy, lower_surface_theta)
            lower_surface_delta_sur      = regr_ls_delta.fit(xy, lower_surface_delta) 
            lower_surface_delta_star_sur = regr_ls_delta_star.fit(xy, lower_surface_delta_star) 
            lower_surface_cf_sur         = regr_ls_cf.fit(xy, lower_surface_cf) 
            lower_surface_Ue_sur         = regr_ls_Ue.fit(xy, lower_surface_Ue) 
            lower_surface_H_sur          = regr_ls_H.fit(xy, lower_surface_H) 
            upper_surface_theta_sur      = regr_us_theta.fit(xy, upper_surface_theta)
            upper_surface_delta_sur      = regr_us_delta.fit(xy, upper_surface_delta) 
            upper_surface_delta_star_sur = regr_us_delta_star.fit(xy, upper_surface_delta_star) 
            upper_surface_cf_sur         = regr_us_cf.fit(xy, upper_surface_cf) 
            upper_surface_Ue_sur         = regr_us_Ue.fit(xy, upper_surface_Ue) 
            upper_surface_H_sur          = regr_us_H.fit(xy, upper_surface_H)           
        
        lower_surface_theta_surs.append(lower_surface_theta_sur)
        lower_surface_delta_surs.append(lower_surface_delta_sur)
        lower_surface_delta_star_surs.append(lower_surface_delta_star_sur)
        lower_surface_cf_surs.append(lower_surface_cf_sur)
        lower_surface_Ue_surs.append(lower_surface_Ue_sur)
        lower_surface_H_surs.append(lower_surface_H_sur)
        upper_surface_theta_surs.append(upper_surface_theta_sur)
        upper_surface_delta_surs.append(upper_surface_delta_sur)
        upper_surface_delta_star_surs.append(upper_surface_delta_star_sur)
        upper_surface_cf_surs.append(upper_surface_cf_sur)
        upper_surface_Ue_surs.append(upper_surface_Ue_sur)
        upper_surface_H_surs.append(upper_surface_H_sur)     
        
        
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
 
 

# manage process with a driver function
def evaluate_thrust(self,state):
    """ Calculate thrust given the current state of the vehicle
    
        Assumptions:
        None
        
        Source:
        N/A
        
        Inputs:
        state [state()]
        
        Outputs:
        results.thrust_force_vector [newtons]
        results.vehicle_mass_rate   [kg/s]
        
        Properties Used:
        Defaulted values
    """            
    
    # Unpack the surrogate
    sfc_surrogate = self.sfc_surrogate
    thr_surrogate = self.thrust_surrogate
    
    # Unpack the conditions
    conditions = state.conditions
    # rescale altitude for proper surrogate performance
    altitude   = conditions.freestream.altitude/self.altitude_input_scale
    mach       = conditions.freestream.mach_number
    throttle   = conditions.propulsion.throttle
    
    cond = np.hstack([altitude,mach,throttle])
       
    if self.use_extended_surrogate:
        lo_blender = Cubic_Spline_Blender(0, .01)
        hi_blender = Cubic_Spline_Blender(0.99, 1)            
        sfc = self.extended_sfc_surrogate(sfc_surrogate, cond, lo_blender, hi_blender)
        thr = self.extended_thrust_surrogate(thr_surrogate, cond, lo_blender, hi_blender)
    else:
        sfc = sfc_surrogate.predict(cond)
        thr = thr_surrogate.predict(cond)

    sfc = sfc*self.sfc_input_scale*self.sfc_anchor_scale
    thr = thr*self.thrust_input_scale*self.thrust_anchor_scale
   
    F    = thr
    mdot = thr*sfc*self.number_of_engines
    
    if self.negative_throttle_values == False:
        F[throttle<=0.]    = 0.
        mdot[throttle<=0.] = 0.
       
    # Save the output
    results = Data()
    results.thrust_force_vector = self.number_of_engines * F * [np.cos(self.thrust_angle),0,-np.sin(self.thrust_angle)]
    results.vehicle_mass_rate   = mdot

    return results          

def build_surrogate(self):
    """ Build a surrogate. Multiple options for models are available including:
        -Gaussian Processes
        -KNN
        -SVR
        
        Assumptions:
        None
        
        Source:
        N/A
        
        Inputs:
        state [state()]
        
        Outputs:
        self.sfc_surrogate    [fun()]
        self.thrust_surrogate [fun()]
        
        Properties Used:
        Defaulted values
    """          
   

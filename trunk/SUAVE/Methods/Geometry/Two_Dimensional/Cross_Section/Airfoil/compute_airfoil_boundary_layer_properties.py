## @ingroup Methods-Geometry-Two_Dimensional-Cross_Section-Airfoil
# compute_airfoil_boundary_layer_properties.py
# 
# Created:  Oct 2021, M. Clarke 

# ----------------------------------------------------------------------
#  Imports
# ---------------------------------------------------------------------- 
import SUAVE
from SUAVE.Core               import Data , Units 
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.import_airfoil_geometry \
     import import_airfoil_geometry  
from SUAVE.Methods.Aerodynamics.Airfoil_Panel_Method.airfoil_analysis   import airfoil_analysis 
from scipy.interpolate        import RectBivariateSpline
from SUAVE.Methods.Utilities.Cubic_Spline_Blender import Cubic_Spline_Blender 
from SUAVE.Core import Data
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import RationalQuadratic, ConstantKernel, RBF, Matern
from sklearn import neighbors
from sklearn import svm 
import numpy as np

## @ingroup Methods-Geometry-Two_Dimensional-Cross_Section-Airfoil
def build_boundary_layer_surrogates(propeller,npanel=250,surrogate_type = 'gaussian'):  
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
     
    a_geo        = propeller.airfoil_geometry   
    a_loc        = propeller.airfoil_polar_stations  
    c            = propeller.chord_distribution        
    num_sec      = len(c)
    
    airfoil_bl_surs  = Data()   
    lower_surface_theta_surs       = []
    lower_surface_delta_surs       = []
    lower_surface_delta_star_surs  = []
    lower_surface_cf_surs          = []
    lower_surface_dcp_dx_surs      = []
    lower_surface_Ue_surs          = []
    lower_surface_H_surs           = []
    upper_surface_theta_surs       = []
    upper_surface_delta_surs       = []
    upper_surface_delta_star_surs  = []
    upper_surface_cf_surs          = []
    upper_surface_dcp_dx_surs      = []
    upper_surface_Ue_surs          = []
    upper_surface_H_surs           = []
        
    Re         = np.linspace(1E2,1E6,5)
    AoA        = np.linspace(-2,12,4)*Units.degrees
    Re_batch   = np.atleast_2d(Re ).T
    AoA_batch  = np.atleast_2d(AoA).T    
    
    xy         = np.zeros((len(AoA)*len(Re),2))  
    xy[:,0]    = np.repeat(Re,len(AoA), axis = 0)/1E5
    xy[:,1]    = np.tile(AoA,len(Re))
    
    TE_idx     = 4#  Trailing Edge Index 
    
    for i in range(num_sec):  
        airfoil_geometry = import_airfoil_geometry([a_geo[a_loc[i]]], npoints = npanel+2) 
        AP  = airfoil_analysis(airfoil_geometry,AoA_batch,Re_batch, npanel, batch_analysis = True)   
        
        # extract properties 
        lower_surface_theta      = AP.theta[TE_idx,:,:] 
        lower_surface_delta      = AP.delta[TE_idx,:,:] 
        lower_surface_delta_star = AP.delta_star[TE_idx,:,:] 
        lower_surface_cf         = AP.Cf[TE_idx,:,:] 
        lower_surface_Ue         = AP.Ue_Vinf[TE_idx,:,:] 
        lower_surface_H          = AP.H[TE_idx,:,:] 
        upper_surface_theta      = AP.theta[-TE_idx,:,:] 
        upper_surface_delta      = AP.delta[-TE_idx,:,:] 
        upper_surface_delta_star = AP.delta_star[-TE_idx,:,:] 
        upper_surface_cf         = AP.Cf[-TE_idx,:,:] 
        upper_surface_Ue         = AP.Ue_Vinf[-TE_idx,:,:] 
        upper_surface_H          = AP.H[-TE_idx,:,:]   
        x_surf                   = AP.x*c[i] 
        dp_dx_surf               = np.diff(AP.Cp)/np.diff(x_surf) 
        lower_surface_dcp_dx     = abs(dp_dx_surf[TE_idx,:,:])
        upper_surface_dcp_dx     = abs(dp_dx_surf[-TE_idx,:,:])
        
        # replace nans 0 with mean as a post post-processor  
        lower_surface_theta       = np.nan_to_num(lower_surface_theta)
        lower_surface_delta       = np.nan_to_num(lower_surface_delta)
        lower_surface_delta_star  = np.nan_to_num(lower_surface_delta_star)
        lower_surface_cf          = np.nan_to_num(lower_surface_cf)
        lower_surface_dcp_dx      = np.nan_to_num(lower_surface_dcp_dx)
        lower_surface_Ue          = np.nan_to_num(lower_surface_Ue)
        lower_surface_H           = np.nan_to_num(lower_surface_H)
        upper_surface_theta       = np.nan_to_num(upper_surface_theta)
        upper_surface_delta       = np.nan_to_num(upper_surface_delta)
        upper_surface_delta_star  = np.nan_to_num(upper_surface_delta_star)
        upper_surface_cf          = np.nan_to_num(upper_surface_cf)
        upper_surface_dcp_dx      = np.nan_to_num(upper_surface_dcp_dx)
        upper_surface_Ue          = np.nan_to_num(upper_surface_Ue)
        upper_surface_H           = np.nan_to_num(upper_surface_H)    

        lower_surface_theta[lower_surface_theta == 0]           = np.mean(lower_surface_theta)
        lower_surface_delta[lower_surface_delta == 0]           = np.mean(lower_surface_delta)
        lower_surface_delta_star[lower_surface_delta_star == 0] = np.mean(lower_surface_delta_star)
        lower_surface_cf[lower_surface_cf == 0]                 = np.mean(lower_surface_cf)
        lower_surface_dcp_dx[lower_surface_dcp_dx == 0]         = np.mean(lower_surface_dcp_dx)
        lower_surface_Ue[lower_surface_Ue == 0]                 = np.mean(lower_surface_Ue)
        lower_surface_H[lower_surface_H == 0]                   = np.mean(lower_surface_H)
        upper_surface_theta[upper_surface_theta == 0]           = np.mean(upper_surface_theta)
        upper_surface_delta[upper_surface_delta == 0]           = np.mean(upper_surface_delta)
        upper_surface_delta_star[upper_surface_delta_star== 0]  = np.mean(upper_surface_delta_star)
        upper_surface_cf[upper_surface_cf == 0]                 = np.mean(upper_surface_cf)
        upper_surface_dcp_dx[upper_surface_dcp_dx == 0]         = np.mean(upper_surface_dcp_dx)
        upper_surface_Ue[upper_surface_Ue == 0]                 = np.mean(upper_surface_Ue)
        upper_surface_H[upper_surface_H == 0]                   = np.mean(upper_surface_H) 
        
        
        lower_surface_theta      = np.atleast_2d(np.ravel(lower_surface_theta.T)).T
        lower_surface_delta      = np.atleast_2d(np.ravel(lower_surface_delta.T)).T
        lower_surface_delta_star = np.atleast_2d(np.ravel(lower_surface_delta_star.T)).T
        lower_surface_cf         = np.atleast_2d(np.ravel(lower_surface_cf.T)).T 
        lower_surface_dcp_dx     = np.atleast_2d(np.ravel(lower_surface_dcp_dx.T )).T
        lower_surface_Ue         = np.atleast_2d(np.ravel(lower_surface_Ue.T)).T 
        lower_surface_H          = np.atleast_2d(np.ravel(lower_surface_H.T )).T
        upper_surface_theta      = np.atleast_2d(np.ravel(upper_surface_theta.T )).T
        upper_surface_delta      = np.atleast_2d(np.ravel(upper_surface_delta.T )).T
        upper_surface_delta_star = np.atleast_2d(np.ravel(upper_surface_delta_star.T)).T  
        upper_surface_cf         = np.atleast_2d(np.ravel(upper_surface_cf.T )).T
        upper_surface_dcp_dx     = np.atleast_2d(np.ravel(upper_surface_dcp_dx.T )).T
        upper_surface_Ue         = np.atleast_2d(np.ravel(upper_surface_Ue.T)).T
        upper_surface_H          = np.atleast_2d(np.ravel(upper_surface_H.T)).T       
    
        # Pick the type of process
        if surrogate_type  == 'gaussian':
            gp_kernel = Matern() 
            regr_ls_theta                = gaussian_process.GaussianProcessRegressor(kernel = gp_kernel)
            regr_ls_delta                = gaussian_process.GaussianProcessRegressor(kernel = gp_kernel)
            regr_ls_delta_star           = gaussian_process.GaussianProcessRegressor(kernel = gp_kernel)
            regr_ls_cf                   = gaussian_process.GaussianProcessRegressor(kernel = gp_kernel)
            regr_ls_dcp_dx               = gaussian_process.GaussianProcessRegressor(kernel = gp_kernel)
            regr_ls_Ue                   = gaussian_process.GaussianProcessRegressor(kernel = gp_kernel)
            regr_ls_H                    = gaussian_process.GaussianProcessRegressor(kernel = gp_kernel)
            regr_us_theta                = gaussian_process.GaussianProcessRegressor(kernel = gp_kernel)
            regr_us_delta                = gaussian_process.GaussianProcessRegressor(kernel = gp_kernel)
            regr_us_delta_star           = gaussian_process.GaussianProcessRegressor(kernel = gp_kernel)
            regr_us_cf                   = gaussian_process.GaussianProcessRegressor(kernel = gp_kernel)
            regr_us_dcp_dx               = gaussian_process.GaussianProcessRegressor(kernel = gp_kernel)
            regr_us_Ue                   = gaussian_process.GaussianProcessRegressor(kernel = gp_kernel)
            regr_us_H                    = gaussian_process.GaussianProcessRegressor(kernel = gp_kernel)
            lower_surface_theta_sur      = regr_ls_theta.fit(xy, lower_surface_theta)
            lower_surface_delta_sur      = regr_ls_delta.fit(xy, lower_surface_delta) 
            lower_surface_delta_star_sur = regr_ls_delta_star.fit(xy, lower_surface_delta_star) 
            lower_surface_cf_sur         = regr_ls_cf.fit(xy, lower_surface_cf) 
            lower_surface_dcp_dx_sur     = regr_ls_dcp_dx.fit(xy, lower_surface_dcp_dx) 
            lower_surface_Ue_sur         = regr_ls_Ue.fit(xy, lower_surface_Ue) 
            lower_surface_H_sur          = regr_ls_H.fit(xy, lower_surface_H) 
            upper_surface_theta_sur      = regr_us_theta.fit(xy, upper_surface_theta)
            upper_surface_delta_sur      = regr_us_delta.fit(xy, upper_surface_delta) 
            upper_surface_delta_star_sur = regr_us_delta_star.fit(xy, upper_surface_delta_star) 
            upper_surface_cf_sur         = regr_us_cf.fit(xy, upper_surface_cf) 
            upper_surface_dcp_dx_sur     = regr_us_dcp_dx.fit(xy, upper_surface_dcp_dx) 
            upper_surface_Ue_sur         = regr_us_Ue.fit(xy, upper_surface_Ue) 
            upper_surface_H_sur          = regr_us_H.fit(xy, upper_surface_H)   
    
        elif surrogate_type  == 'knn':
            regr_ls_theta                = neighbors.KNeighborsRegressor(n_neighbors=1,weights='distance')
            regr_ls_delta                = neighbors.KNeighborsRegressor(n_neighbors=1,weights='distance')
            regr_ls_delta_star           = neighbors.KNeighborsRegressor(n_neighbors=1,weights='distance')
            regr_ls_cf                   = neighbors.KNeighborsRegressor(n_neighbors=1,weights='distance')
            regr_ls_dcp_dx               = neighbors.KNeighborsRegressor(n_neighbors=1,weights='distance')
            regr_ls_Ue                   = neighbors.KNeighborsRegressor(n_neighbors=1,weights='distance')
            regr_ls_H                    = neighbors.KNeighborsRegressor(n_neighbors=1,weights='distance')
            regr_us_theta                = neighbors.KNeighborsRegressor(n_neighbors=1,weights='distance')
            regr_us_delta                = neighbors.KNeighborsRegressor(n_neighbors=1,weights='distance')
            regr_us_delta_star           = neighbors.KNeighborsRegressor(n_neighbors=1,weights='distance')
            regr_us_cf                   = neighbors.KNeighborsRegressor(n_neighbors=1,weights='distance')
            regr_us_dcp_dx               = neighbors.KNeighborsRegressor(n_neighbors=1,weights='distance')
            regr_us_Ue                   = neighbors.KNeighborsRegressor(n_neighbors=1,weights='distance')
            regr_us_H                    = neighbors.KNeighborsRegressor(n_neighbors=1,weights='distance')
            lower_surface_theta_sur      = regr_ls_theta.fit(xy, lower_surface_theta)
            lower_surface_delta_sur      = regr_ls_delta.fit(xy, lower_surface_delta) 
            lower_surface_delta_star_sur = regr_ls_delta_star.fit(xy, lower_surface_delta_star) 
            lower_surface_cf_sur         = regr_ls_cf.fit(xy, lower_surface_cf) 
            lower_surface_dcp_dx_sur     = regr_ls_dcp_dx.fit(xy, lower_surface_dcp_dx) 
            lower_surface_Ue_sur         = regr_ls_Ue.fit(xy, lower_surface_Ue) 
            lower_surface_H_sur          = regr_ls_H.fit(xy, lower_surface_H) 
            upper_surface_theta_sur      = regr_us_theta.fit(xy, upper_surface_theta)
            upper_surface_delta_sur      = regr_us_delta.fit(xy, upper_surface_delta) 
            upper_surface_delta_star_sur = regr_us_delta_star.fit(xy, upper_surface_delta_star) 
            upper_surface_cf_sur         = regr_us_cf.fit(xy, upper_surface_cf) 
            upper_surface_dcp_dx_sur     = regr_us_dcp_dx.fit(xy, upper_surface_dcp_dx) 
            upper_surface_Ue_sur         = regr_us_Ue.fit(xy, upper_surface_Ue) 
            upper_surface_H_sur          = regr_us_H.fit(xy, upper_surface_H)  

        elif surrogate_type  == 'svr':
            regr_ls_theta                = svm.SVR(C=500)
            regr_ls_delta                = svm.SVR(C=500)
            regr_ls_delta_star           = svm.SVR(C=500)
            regr_ls_cf                   = svm.SVR(C=500)
            regr_ls_dcp_dx               = svm.SVR(C=500)
            regr_ls_Ue                   = svm.SVR(C=500)
            regr_ls_H                    = svm.SVR(C=500)
            regr_us_theta                = svm.SVR(C=500)
            regr_us_delta                = svm.SVR(C=500)
            regr_us_delta_star           = svm.SVR(C=500)
            regr_us_cf                   = svm.SVR(C=500)
            regr_us_dcp_dx               = svm.SVR(C=500)
            regr_us_Ue                   = svm.SVR(C=500)
            regr_us_H                    = svm.SVR(C=500) 
            lower_surface_theta_sur      = regr_ls_theta.fit(xy, lower_surface_theta)
            lower_surface_delta_sur      = regr_ls_delta.fit(xy, lower_surface_delta) 
            lower_surface_delta_star_sur = regr_ls_delta_star.fit(xy, lower_surface_delta_star) 
            lower_surface_cf_sur         = regr_ls_cf.fit(xy, lower_surface_cf) 
            lower_surface_dcp_dx_sur     = regr_ls_dcp_dx.fit(xy, lower_surface_dcp_dx) 
            lower_surface_Ue_sur         = regr_ls_Ue.fit(xy, lower_surface_Ue) 
            lower_surface_H_sur          = regr_ls_H.fit(xy, lower_surface_H) 
            upper_surface_theta_sur      = regr_us_theta.fit(xy, upper_surface_theta)
            upper_surface_delta_sur      = regr_us_delta.fit(xy, upper_surface_delta) 
            upper_surface_delta_star_sur = regr_us_delta_star.fit(xy, upper_surface_delta_star) 
            upper_surface_cf_sur         = regr_us_cf.fit(xy, upper_surface_cf) 
            upper_surface_dcp_dx_sur     = regr_us_dcp_dx.fit(xy, upper_surface_dcp_dx) 
            upper_surface_Ue_sur         = regr_us_Ue.fit(xy, upper_surface_Ue) 
            upper_surface_H_sur          = regr_us_H.fit(xy, upper_surface_H)           
        
        lower_surface_theta_surs.append(lower_surface_theta_sur)
        lower_surface_delta_surs.append(lower_surface_delta_sur)
        lower_surface_delta_star_surs.append(lower_surface_delta_star_sur)
        lower_surface_cf_surs.append(lower_surface_cf_sur)
        lower_surface_dcp_dx_surs.append(lower_surface_dcp_dx_sur)
        lower_surface_Ue_surs.append(lower_surface_Ue_sur)
        lower_surface_H_surs.append(lower_surface_H_sur)
        upper_surface_theta_surs.append(upper_surface_theta_sur)
        upper_surface_delta_surs.append(upper_surface_delta_sur)
        upper_surface_delta_star_surs.append(upper_surface_delta_star_sur)
        upper_surface_cf_surs.append(upper_surface_cf_sur)
        upper_surface_dcp_dx_surs.append(upper_surface_dcp_dx_sur)
        upper_surface_Ue_surs.append(upper_surface_Ue_sur)
        upper_surface_H_surs.append(upper_surface_H_sur)
        
    airfoil_bl_surs.lower_surface_theta_surrogates       = lower_surface_theta_surs
    airfoil_bl_surs.lower_surface_delta_surrogates       = lower_surface_delta_surs
    airfoil_bl_surs.lower_surface_delta_start_surrogates = lower_surface_delta_star_surs
    airfoil_bl_surs.lower_surface_cf_surrogates          = lower_surface_cf_surs
    airfoil_bl_surs.lower_surface_dcp_dx_surrogates      = lower_surface_dcp_dx_surs      
    airfoil_bl_surs.lower_surface_Ue_surrogates          = lower_surface_Ue_surs
    airfoil_bl_surs.lower_surface_H_surrogates           = lower_surface_H_surs 
    airfoil_bl_surs.upper_surface_theta_surrogates       = upper_surface_theta_surs
    airfoil_bl_surs.upper_surface_delta_surrogates       = upper_surface_delta_surs
    airfoil_bl_surs.upper_surface_delta_start_surrogates = upper_surface_delta_star_surs
    airfoil_bl_surs.upper_surface_dcp_dx_surrogates      = upper_surface_dcp_dx_surs      
    airfoil_bl_surs.upper_surface_cf_surrogates          = upper_surface_cf_surs
    airfoil_bl_surs.upper_surface_Ue_surrogates          = upper_surface_Ue_surs
    airfoil_bl_surs.upper_surface_H_surrogates           = upper_surface_H_surs  

    return airfoil_bl_surs 
 

# manage process with a driver function
def evaluate_boundary_layer_surrogates(propeller,AoA,Re):
    """  
        Defaulted values
    """            
    bl_surs = propeller.airfoil_bl_surrogates       
    a_loc   = propeller.airfoil_polar_stations  

    # Unpack the surrogate 
    ls_theta_surrogates       = bl_surs.lower_surface_theta_surrogates      
    ls_delta_surrogates       = bl_surs.lower_surface_delta_surrogates      
    ls_delta_star_surrogates  = bl_surs.lower_surface_delta_start_surrogates
    ls_cf_surrogates          = bl_surs.lower_surface_cf_surrogates  
    ls_dcp_dx_surrogates      = bl_surs.lower_surface_dcp_dx_surrogates           
    ls_Ue_surrogates          = bl_surs.lower_surface_Ue_surrogates         
    ls_H_surrogates           = bl_surs.lower_surface_H_surrogates          
    us_theta_surrogates       = bl_surs.upper_surface_theta_surrogates      
    us_delta_surrogates       = bl_surs.upper_surface_delta_surrogates      
    us_delta_star_surrogates  = bl_surs.upper_surface_delta_start_surrogates
    us_cf_surrogates          = bl_surs.upper_surface_cf_surrogates         
    us_dcp_dx_surrogates      = bl_surs.upper_surface_dcp_dx_surrogates      
    us_Ue_surrogates          = bl_surs.upper_surface_Ue_surrogates         
    us_H_surrogates           = bl_surs.upper_surface_H_surrogates          
     
    # rescale altitude for proper surrogate performance  
    num_cpts   = len(AoA[:,0,0])
    num_sec    = len(AoA[0,:,0])
    num_azi    = len(AoA[0,0,:])  
    
    ls_theta      = np.zeros((num_cpts,num_sec,num_azi))
    ls_delta      = np.zeros_like(ls_theta)
    ls_delta_star = np.zeros_like(ls_theta)
    ls_cf         = np.zeros_like(ls_theta)
    ls_dcp_dx     = np.zeros_like(ls_theta)
    ls_Ue         = np.zeros_like(ls_theta)
    ls_H          = np.zeros_like(ls_theta)
    us_theta      = np.zeros_like(ls_theta)
    us_delta      = np.zeros_like(ls_theta)
    us_delta_star = np.zeros_like(ls_theta)
    us_cf         = np.zeros_like(ls_theta)
    us_dcp_dx     = np.zeros_like(ls_theta)
    us_Ue         = np.zeros_like(ls_theta)
    us_H          = np.zeros_like(ls_theta) 
    
    for i in range(num_sec):
        Re_sec  = np.ravel(Re[:,i,:]/1E5)
        AoA_sec = np.ravel(AoA[:,i,:])
        cond = np.vstack([Re_sec,AoA_sec]).T  
        
        ls_theta[:,i,:]      = np.reshape(ls_theta_surrogates[a_loc[i]].predict(cond),(num_cpts,num_azi))
        ls_delta[:,i,:]      = np.reshape(ls_delta_surrogates[a_loc[i]].predict(cond) ,(num_cpts,num_azi))
        ls_delta_star[:,i,:] = np.reshape(ls_delta_star_surrogates[a_loc[i]].predict(cond),(num_cpts,num_azi))
        ls_cf[:,i,:]         = np.reshape(ls_cf_surrogates[a_loc[i]].predict(cond),(num_cpts,num_azi))
        ls_dcp_dx[:,i,:]     = np.reshape(ls_dcp_dx_surrogates[a_loc[i]].predict(cond),(num_cpts,num_azi))
        ls_Ue[:,i,:]         = np.reshape(ls_Ue_surrogates[a_loc[i]].predict(cond),(num_cpts,num_azi))
        ls_H[:,i,:]          = np.reshape(ls_H_surrogates[a_loc[i]].predict(cond),(num_cpts,num_azi))
        us_theta[:,i,:]      = np.reshape(us_theta_surrogates[a_loc[i]].predict(cond),(num_cpts,num_azi))
        us_delta[:,i,:]      = np.reshape(us_delta_surrogates[a_loc[i]].predict(cond),(num_cpts,num_azi))
        us_delta_star[:,i,:] = np.reshape(us_delta_star_surrogates[a_loc[i]].predict(cond),(num_cpts,num_azi))
        us_cf[:,i,:]         = np.reshape(us_cf_surrogates[a_loc[i]].predict(cond),(num_cpts,num_azi))
        us_dcp_dx[:,i,:]     = np.reshape(us_dcp_dx_surrogates[a_loc[i]].predict(cond),(num_cpts,num_azi))
        us_Ue[:,i,:]         = np.reshape(us_Ue_surrogates[a_loc[i]].predict(cond),(num_cpts,num_azi))
        us_H[:,i,:]          = np.reshape(us_H_surrogates[a_loc[i]].predict(cond),(num_cpts,num_azi))
    
    bl_results = Data()
    bl_results.ls_theta      = ls_theta           
    bl_results.ls_delta      = ls_delta      
    bl_results.ls_delta_star = ls_delta_star   
    bl_results.ls_cf         = ls_cf         
    bl_results.ls_dcp_dx     = ls_dcp_dx   
    bl_results.ls_Ue         = ls_Ue           
    bl_results.ls_H          = ls_H             
    bl_results.us_theta      = us_theta      
    bl_results.us_delta      = us_delta       
    bl_results.us_delta_star = us_delta_star 
    bl_results.us_cf         = us_cf 
    bl_results.us_dcp_dx     = us_dcp_dx 
    bl_results.us_Ue         = us_Ue         
    bl_results.us_H          = us_H          

    return bl_results        























   

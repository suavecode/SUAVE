## @ingroup Components-Energy-Networks
# Propulsor_Surrogate.py
#
# Created:  Mar 2017, E. Botero
# Modified: Jan 2020, T. MacDonald
#           May 2021, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# package imports
import numpy as np
from copy import deepcopy
from .Network import Network
from SUAVE.Methods.Utilities.Cubic_Spline_Blender import Cubic_Spline_Blender

from SUAVE.Core import Data
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import RationalQuadratic, ConstantKernel, RBF, Matern
from sklearn import neighbors
from sklearn import svm, linear_model

# ----------------------------------------------------------------------
#  Network
# ----------------------------------------------------------------------

## @ingroup Components-Energy-Networks
class Propulsor_Surrogate(Network):
    """ This is a way for you to load engine data from a source.
        A .csv file is read in, a surrogate made, that surrogate is used during the mission analysis.
        
        You need to use build surrogate first when setting up the vehicle to make this work.
        
        Assumptions:
        The input format for this should be Altitude, Mach, Throttle, Thrust, SFC
        
        Source:
        None
    """        
    def __defaults__(self):
        """ This sets the default values for the network to function
        
            Assumptions:
            None
            
            Source:
            N/A
            
            Inputs:
            None
            
            Outputs:
            None
            
            Properties Used:
            N/A
        """           
        self.engine_length            = None
        self.number_of_engines        = None
        self.tag                      = 'Engine_Deck_Surrogate'
        self.input_file               = None
        self.sfc_surrogate            = None
        self.thrust_surrogate         = None
        self.thrust_angle             = 0.0
        self.areas                    = Data()
        self.surrogate_type           = 'gaussian'
        self.altitude_input_scale     = 1.
        self.thrust_input_scale       = 1.
        self.sfc_anchor               = None
        self.sfc_anchor_scale         = 1.
        self.sfc_anchor_conditions    = np.array([[1.,1.,1.]])
        self.thrust_anchor            = None
        self.thrust_anchor_scale      = 1.
        self.thrust_anchor_conditions = np.array([[1.,1.,1.]])
        self.sfc_rubber_scale         = 1.
        self.use_extended_surrogate   = False
        self.sealevel_static_thrust   = 0.0
        self.negative_throttle_values = False
   
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
       
        # file name to look for
        file_name = self.input_file
       
        # Load the CSV file
        my_data = np.genfromtxt(file_name, delimiter=',')
       
        # Remove the header line
        my_data = np.delete(my_data,np.s_[0],axis=0)
       
        # Clean up to remove redundant lines
        b = np.ascontiguousarray(my_data).view(np.dtype((np.void, my_data.dtype.itemsize * my_data.shape[1])))
        _, idx = np.unique(b, return_index=True)
       
        my_data = my_data[idx]                
               
   
        xy  = my_data[:,:3] # Altitude, Mach, Throttle
        thr = np.transpose(np.atleast_2d(my_data[:,3])) # Thrust
        sfc = np.transpose(np.atleast_2d(my_data[:,4]))  # SFC
        
        self.altitude_input_scale = np.max(xy[:,0])
        self.thrust_input_scale   = np.max(thr)
        self.sfc_input_scale      = np.max(sfc)
        
        # normalize for better surrogate performance
        xy[:,0] /= self.altitude_input_scale
        thr     /= self.thrust_input_scale
        sfc     /= self.sfc_input_scale
       
       
        # Pick the type of process
        if self.surrogate_type  == 'gaussian':
            gp_kernel = Matern()
            regr_sfc = gaussian_process.GaussianProcessRegressor(kernel=gp_kernel)
            regr_thr = gaussian_process.GaussianProcessRegressor(kernel=gp_kernel)      
            thr_surrogate = regr_thr.fit(xy, thr)
            sfc_surrogate = regr_sfc.fit(xy, sfc)  
           
        elif self.surrogate_type  == 'knn':
            regr_sfc = neighbors.KNeighborsRegressor(n_neighbors=1,weights='distance')
            regr_thr = neighbors.KNeighborsRegressor(n_neighbors=1,weights='distance')
            sfc_surrogate = regr_sfc.fit(xy, sfc)
            thr_surrogate = regr_thr.fit(xy, thr)  
   
        elif self.surrogate_type  == 'svr':
            regr_thr = svm.SVR(C=500.)
            regr_sfc = svm.SVR(C=500.)
            sfc_surrogate  = regr_sfc.fit(xy, sfc)
            thr_surrogate  = regr_thr.fit(xy, thr)    
           
        elif self.surrogate_type == 'linear':
            regr_thr = linear_model.LinearRegression()
            regr_sfc = linear_model.LinearRegression()          
            sfc_surrogate  = regr_sfc.fit(xy, sfc)
            thr_surrogate  = regr_thr.fit(xy, thr)
            
        else:
            raise NotImplementedError('Selected surrogate method has not been implemented')
       
       
        if self.thrust_anchor is not None:
            cons = deepcopy(self.thrust_anchor_conditions)
            cons[0,0] /= self.altitude_input_scale
            base_thrust_at_anchor = thr_surrogate.predict(cons)
            self.thrust_anchor_scale = self.thrust_anchor/(base_thrust_at_anchor*self.thrust_input_scale)
            
        if self.sfc_anchor is not None:
            cons = deepcopy(self.sfc_anchor_conditions)
            cons[0,0] /= self.altitude_input_scale
            base_sfc_at_anchor = sfc_surrogate.predict(cons)
            self.sfc_anchor_scale = self.sfc_anchor/(base_sfc_at_anchor*self.sfc_input_scale)
       
        # Save the output
        self.sfc_surrogate    = sfc_surrogate
        self.thrust_surrogate = thr_surrogate   
        
    def extended_thrust_surrogate(self, thr_surrogate, cond, lo_blender, hi_blender):
        """ Fixes thrust values outside of the standard throttle range in order to provide
            reasonable values outside of the typical surrogate coverage area. 
            
            Assumptions:
            None
            
            Source:
            N/A
            
            Inputs:
            thr_surrogate     - Trained sklearn surrogate that outputs a scaled thrust value
            cond              - nx3 numpy array with input conditions for the surrogate
            lo_blender        - Cubic spline blending class that is used at the low throttle cutoff
            hi_blender        - Cubic spline blending class that is used at the high throttle cutoff
            
            Outputs:
            T                 [nondim]
            
            Properties Used:
            None
        """            
        # initialize
        cond_zero_eta      = deepcopy(cond)
        cond_one_eta       = deepcopy(cond)
        cond_zero_eta[:,2] = 0
        cond_one_eta[:,2]  = 1
        
        min_thrs = thr_surrogate.predict(cond_zero_eta)
        max_thrs = thr_surrogate.predict(cond_one_eta)
        dTdetas  = max_thrs - min_thrs
        
        etas          = cond[:,2]
        mask_low      = etas < 0
        mask_lo_blend = np.logical_and(etas >= 0, etas < 0.01)
        mask_mid      = np.logical_and(etas >= 0.01, etas < 0.99)
        mask_hi_blend = np.logical_and(etas >= 0.99, etas < 1)
        mask_high     = etas >= 1
        
        etas = np.atleast_2d(etas).T
        T = np.zeros_like(etas)
        
        # compute thrust
        T[mask_low] = min_thrs[mask_low] + etas[mask_low]*dTdetas[mask_low]
        
        if np.sum(mask_lo_blend) > 0:
            lo_weight = lo_blender.compute(etas[mask_lo_blend])
            T[mask_lo_blend] = (min_thrs[mask_lo_blend] + etas[mask_lo_blend]*dTdetas[mask_lo_blend])*lo_weight + \
                               thr_surrogate.predict(cond[mask_lo_blend])*(1-lo_weight)
        
        if np.sum(mask_mid) > 0:
            T[mask_mid] = thr_surrogate.predict(cond[mask_mid])
        
        if np.sum(mask_hi_blend) > 0:
            hi_weight = hi_blender.compute(etas[mask_hi_blend])
            T[mask_hi_blend] = thr_surrogate.predict(cond[mask_hi_blend])*hi_weight + \
                               (max_thrs[mask_hi_blend] + (etas[mask_hi_blend]-1)*dTdetas[mask_hi_blend])*(1-hi_weight)
        
        T[mask_high] = max_thrs[mask_high] + (etas[mask_high]-1)*dTdetas[mask_high]
        
        return T
    
    def extended_sfc_surrogate(self, sfc_surrogate, cond, lo_blender, hi_blender):
        """ Fixes sfc values outside of the standard throttle range in order to provide
            reasonable values outside of the typical surrogate coverage area. 
            
            Assumptions:
            None
            
            Source:
            N/A
            
            Inputs:
            sfc_surrogate     - Trained sklearn surrogate that outputs a scaled sfc value
            cond              - nx3 numpy array with input conditions for the surrogate
            lo_blender        - Cubic spline blending class that is used at the low throttle cutoff
            hi_blender        - Cubic spline blending class that is used at the high throttle cutoff
            
            Outputs:
            sfcs              [nondim]
            
            Properties Used:
            None
        """           
        # initialize
        cond_zero_eta      = deepcopy(cond)
        cond_one_eta       = deepcopy(cond)
        cond_zero_eta[:,2] = 0
        cond_one_eta[:,2]  = 1  
        
        etas          = cond[:,2]
        mask_low      = etas < 0
        mask_lo_blend = np.logical_and(etas >= 0, etas < 0.01)
        mask_mid      = np.logical_and(etas >= 0.01, etas < 0.99)
        mask_hi_blend = np.logical_and(etas >= 0.99, etas < 1)
        mask_high     = etas >= 1 
        
        etas = np.atleast_2d(etas).T
        sfcs = np.zeros_like(etas)
        
        # compute sfc
        if np.sum(mask_low) > 0:
            sfcs[mask_low] = sfc_surrogate.predict(cond_zero_eta[mask_low])
        
        if np.sum(mask_lo_blend) > 0:
            lo_weight = lo_blender.compute(etas[mask_lo_blend])
            sfcs[mask_lo_blend] = sfc_surrogate.predict(cond_zero_eta[mask_lo_blend])*lo_weight + \
                               sfc_surrogate.predict(cond[mask_lo_blend])*(1-lo_weight)
        
        if np.sum(mask_mid) > 0:
            sfcs[mask_mid] = sfc_surrogate.predict(cond[mask_mid])
        
        if np.sum(mask_hi_blend) > 0:
            hi_weight = hi_blender.compute(etas[mask_hi_blend])
            sfcs[mask_hi_blend] = sfc_surrogate.predict(cond[mask_hi_blend])*hi_weight + \
                               sfc_surrogate.predict(cond_one_eta[mask_hi_blend])*(1-hi_weight)
        
        if np.sum(mask_high) > 0:
            sfcs[mask_high] = sfc_surrogate.predict(cond_one_eta[mask_high])
            
        return sfcs   
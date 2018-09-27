## @ingroup Components-Energy-Networks
# Propulsor_Surrogate.py
#
# Created:  Mar 2017, E. Botero
# Modified:

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
import SUAVE

# package imports
import numpy as np
from SUAVE.Components.Propulsors.Propulsor import Propulsor

from SUAVE.Core import Data
import sklearn
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import RationalQuadratic 
from sklearn import neighbors
from sklearn import svm

# ----------------------------------------------------------------------
#  Network
# ----------------------------------------------------------------------

## @ingroup Components-Energy-Networks
class Propulsor_Surrogate(Propulsor):
    """ This is a way for you to load engine data from a source.
        A .csv file is read in, a surrogate made, that surrogate is used during the mission analysis.
        
        You need to use build surrogate first when setting up the vehicle to make this work.
    
        Assumptions:
        The input format for this should be Altitude, Mach, Throttle, Thrust, SFC
        
        Source:
        None
    """        
    def __defaults__(self): 
        """ This sets the default values for the network to function.
    
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
        self.nacelle_diameter  = None
        self.engine_length     = None
        self.number_of_engines = None
        self.tag               = 'Engine_Deck_Surrogate'
        self.input_file        = None
        self.sfc_surrogate     = None
        self.thrust_surrogate  = None
        self.thrust_angle      = 0.0
        self.areas             = Data()
        self.surrogate_type    = 'gaussian'
    
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
        altitude   = conditions.freestream.altitude
        mach       = conditions.freestream.mach_number
        throttle   = conditions.propulsion.throttle
        
        cond = np.hstack([altitude,mach,throttle])
        
        # Run the surrogate for a range of altitudes
        data_len = len(altitude)
        sfc = np.zeros([data_len,1])  
        thr = np.zeros([data_len,1]) 
        for ii,_ in enumerate(altitude):            
            sfc[ii] = sfc_surrogate.predict([np.array([altitude[ii][0],mach[ii][0],throttle[ii][0]])])    
            thr[ii] = thr_surrogate.predict([np.array([altitude[ii][0],mach[ii][0],throttle[ii][0]])])   #This is the fix when sklearn is update.

        
        F    = thr
        mdot = thr*sfc*self.number_of_engines
        
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
                
    
        xy  = my_data[1:,:3] # Altitude, Mach, Throttle
        thr = np.transpose(np.atleast_2d(my_data[1:,3])) # Thrust
        sfc = np.transpose(np.atleast_2d(my_data[1:,4]))  # SFC
        
        
        # Pick the type of process
        if self.surrogate_type  == 'gaussian': 
            gp_kernel_RQ = RationalQuadratic(length_scale=1.0, alpha=1.0)
            regr_sfc = gaussian_process.GaussianProcessRegressor(kernel=gp_kernel_RQ,normalize_y=True)
            regr_thr = gaussian_process.GaussianProcessRegressor(kernel=gp_kernel_RQ)      
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
        
        
        # Save the output
        self.sfc_surrogate    = sfc_surrogate
        self.thrust_surrogate = thr_surrogate     
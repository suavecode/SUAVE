# propulsor_surrogate.py
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
from sklearn import neighbors
from sklearn import svm

# ----------------------------------------------------------------------
#  Network
# ----------------------------------------------------------------------

# The input format for this should be Altitude, Mach, Throttle, Thrust, SFC

class Propulsor_Surrogate(Propulsor):
    def __defaults__(self): 
        self.nacelle_diameter  = None
        self.engine_length     = None
        self.number_of_engines = None
        self.tag               = 'network'
        self.input_file        = None
        self.sfc_surrogate     = None
        self.thrust_surrogate  = None
        self.areas             = Data()
    
    # manage process with a driver function
    def evaluate_thrust(self,state):
        
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
            sfc[ii] = sfc_surrogate.predict(np.array([altitude[ii][0],mach[ii][0],throttle[ii][0]]))    
            thr[ii] = thr_surrogate.predict(np.array([altitude[ii][0],mach[ii][0],throttle[ii][0]]))   
            
        ## Test plot, run a range of throttles
        #throttles = np.linspace(0.,1.)
        #alts  = 10000.*np.ones_like(throttles)
        #machs = 0.9*np.ones_like(throttles)
        
        #thrusts = thr_surrogate.predict(np.array([alts,machs,throttles]).T)   
        
        #import pylab as plt
        
        #plt.figure("throttles")
        #axes = plt.gca()  
        #axes.plot(throttles,thrusts)
        
        #plt.show()
        
        
        F    = thr
        mdot = thr*sfc*self.number_of_engines
        
        # Save the output
        results = Data()
        results.thrust_force_vector = self.number_of_engines * F * [1,0,0] 
        results.vehicle_mass_rate   = mdot
    
        return results          
    
    def build_surrogate(self):
        
        # file name to look for
        file_name = self.input_file
        
        # Load the CSV file
        my_data = np.genfromtxt(file_name, delimiter=',')
        
        # Clean up to remove redundant lines
        # Section 1
        #col1 = my_data[1:,:1]
        #col2 = my_data[1:,1:2]
        #col3 = my_data[1:,2:3]
        #for ii in xrange(0, len(col1)-1):
            #if (col1[ii] == col1[ii+1])&(col2[ii] == col2[ii+1])&(col3[ii] == col3[ii+1]):
                #np.delete(my_data,(ii), axis=0)

        b = np.ascontiguousarray(my_data).view(np.dtype((np.void, my_data.dtype.itemsize * my_data.shape[1])))
        _, idx = np.unique(b, return_index=True)
        
        my_data = my_data[idx]                
                
    
        xy  = my_data[1:,:3] # Altitude, Mach, Throttle
        thr = np.transpose(np.atleast_2d(my_data[1:,3])) # Thrust
        sfc = np.transpose(np.atleast_2d(my_data[1:,4]))  # SFC
        

        # Pick the type of process
        regr_sfc = gaussian_process.GaussianProcess(theta0=50.,thetaL=8.,thetaU=100.)
        regr_thr = gaussian_process.GaussianProcess(theta0=15.,thetaL=8.,thetaU=100.)                
        thr_surrogate = regr_thr.fit(xy, thr)
        sfc_surrogate = regr_sfc.fit(xy, sfc)          
        
        ## KNN
        #regr_sfc = neighbors.KNeighborsRegressor(n_neighbors=1,weights='distance')
        #regr_thr = neighbors.KNeighborsRegressor(n_neighbors=1,weights='distance')
        #sfc_surrogate = regr_sfc.fit(xy, sfc)
        #thr_surrogate = regr_thr.fit(xy, thr)  
    
        ## SVR
        #regr_thr = svm.SVR(C=500.)
        #regr_sfc = svm.SVR(C=500.)
        #sfc_surrogate  = regr_sfc.fit(xy, sfc)
        #thr_surrogate  = regr_thr.fit(xy, thr)           
        
        
        # Save the output
        self.sfc_surrogate    = sfc_surrogate
        self.thrust_surrogate = thr_surrogate     
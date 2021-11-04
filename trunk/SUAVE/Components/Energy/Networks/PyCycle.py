## @ingroup Components-Energy-Networks
# PyCycle.py
#
# Created:  Sep 2020, E. Botero
# Modified: 
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# package imports
import numpy as np
from copy import deepcopy

from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import Matern
from sklearn import neighbors
from sklearn import svm, linear_model

# SUAVE imports
from SUAVE.Core import Data, Units
from SUAVE.Components.Energy.Networks import Propulsor_Surrogate

# ----------------------------------------------------------------------
#  Network
# ----------------------------------------------------------------------

## @ingroup Components-Energy-Networks
class PyCycle(Propulsor_Surrogate):
    """ This is a way for you to run PyCycle, create a deck, and load the deck into a SUAVE surrogate
        
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
        self.tag                      = 'PyCycle_Engine'
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
        self.save_deck                = True
        self.evaluation_mach_alt      = [(0.8, 35000), (0.7, 35000),
                                         (0.7, 25000), (0.6, 25000),
                                         (0.6, 20000), (0.5, 20000), 
                                         (0.5, 10000), (0.4, 10000), (0.2, 10000),
                                         (0.2, 1000),  (0.4, 1000),  (0.6, 1000),
                                         (0.6, 0),     (0.4, 0),     (0.2, 0),     (0.001, 0)]
        self.evaluation_throttles     = np.array([1, 0.9, 0.8, .7])

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
        
        # unpack
        pycycle_problem = self.model
        
        
        pycycle_problem.set_solver_print(level=-1)
        pycycle_problem.set_solver_print(level=2, depth=0)        
        
        
        # Extract the data
        # Create lists that will turn into arrays
        Altitudes = []
        Machs     = []
        PCs       = []
        Thrust    = []
        TSFC      = []
        
        
        # if we added fc.dTS this would handle the deltaISA
        
        throttles = self.evaluation_throttles*1.

        for MN, alt in self.evaluation_mach_alt: 
    
            print('***'*10)
            print(f'* MN: {MN}, alt: {alt}')
            print('***'*10)
            pycycle_problem['OD_full_pwr.fc.MN'] = MN
            pycycle_problem['OD_full_pwr.fc.alt'] = alt
            pycycle_problem['OD_part_pwr.fc.MN'] = MN
            pycycle_problem['OD_part_pwr.fc.alt'] = alt
    
            for PC in throttles: 
                print(f'## PC = {PC}')
                pycycle_problem['OD_part_pwr.PC']  = PC
                pycycle_problem.run_model()
                #Save to our list for SUAVE
                Altitudes.append(alt)
                Machs.append(MN)
                PCs.append(PC)
                TSFC.append(pycycle_problem['OD_part_pwr.perf.TSFC'][0])
                Thrust.append(pycycle_problem['OD_part_pwr.perf.Fn'][0])

            throttles = np.flip(throttles)

        # Now setup into vectors
        Altitudes = np.atleast_2d(np.array(Altitudes)).T * Units.feet
        Mach      = np.atleast_2d(np.array(Machs)).T
        Throttle  = np.atleast_2d(np.array(PCs)).T
        thr       = np.atleast_2d(np.array(Thrust)).T * Units.lbf
        sfc       = np.atleast_2d(np.array(TSFC)).T   * Units['lbm/hr/lbf'] # lbm/hr/lbf converted to (kg/N/s)
        
        
        # Once we have the data the model must be deleted because pycycle models can't be deepcopied
        self.pop('model')
        
        # Concatenate all together and things will start to look like the propuslor surrogate soon
        my_data = np.concatenate([Altitudes,Mach,Throttle,thr,sfc],axis=1)
        
        if self.save_deck :
            # Write an engine deck
            np.savetxt("pyCycle_deck.csv", my_data, delimiter=",")
        
        print(my_data)
        
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
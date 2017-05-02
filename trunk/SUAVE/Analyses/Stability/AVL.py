# AVL.py
#
# Created:  Tim Momose, Dec 2014 
# Modified: Feb 2016, Andrew Wendorff


# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUAVE imports
import SUAVE
from SUAVE.Core import Units, Data
from SUAVE.Core import redirect

from SUAVE.Analyses.Mission.Segments.Conditions.Aerodynamics import Aerodynamics
from SUAVE.Analyses.Mission.Segments.Conditions.Conditions   import Conditions

from SUAVE.Methods.Aerodynamics.AVL.write_geometry   import write_geometry
from SUAVE.Methods.Aerodynamics.AVL.write_run_cases  import write_run_cases
from SUAVE.Methods.Aerodynamics.AVL.write_input_deck import write_input_deck
from SUAVE.Methods.Aerodynamics.AVL.run_analysis     import run_analysis
from SUAVE.Methods.Aerodynamics.AVL.translate_data   import translate_conditions_to_cases, translate_results_to_conditions
from SUAVE.Methods.Aerodynamics.AVL.purge_files      import purge_files
from SUAVE.Methods.Aerodynamics.AVL.Data.Results     import Results
from SUAVE.Methods.Aerodynamics.AVL.Data.Settings    import Settings
from SUAVE.Methods.Aerodynamics.AVL.Data.Cases       import Run_Case

# local imports
from Stability import Stability

# import SUAVE methods
from SUAVE.Methods.Flight_Dynamics.Static_Stability.Approximations.Tube_Wing.taw_cmalpha import taw_cmalpha
from SUAVE.Methods.Flight_Dynamics.Static_Stability.Approximations.Tube_Wing.taw_cnbeta import taw_cnbeta
from SUAVE.Methods.Flight_Dynamics.Static_Stability.Approximations.datcom import datcom
from SUAVE.Methods.Flight_Dynamics.Dynamic_Stability import Approximations as Approximations
from SUAVE.Methods.Flight_Dynamics.Dynamic_Stability import Full_Linearized_Equations as Full_Linearized_Equations
from SUAVE.Methods.Flight_Dynamics.Dynamic_Stability.Full_Linearized_Equations import Supporting_Functions as Supporting_Functions


# Package imports
import time
import pylab as plt
import os
import numpy as np
import sys
import sklearn
from sklearn import gaussian_process
from sklearn import neighbors
from sklearn import svm
from shutil import rmtree
from warnings import warn

## ----------------------------------------------------------------------
##  Class
## ----------------------------------------------------------------------

class AVL(Stability):
    """ SUAVE.Analyses.Aerodynamics.AVL
        aerodynamic model that performs a vortex lattice analysis using AVL
        (Athena Vortex Lattice, by Mark Drela of MIT).

        this class is callable, see self.__call__

    """

    def __defaults__(self):
        self.tag        = 'avl'
        self.keep_files = True

        self.settings = Settings()

        self.current_status = Data()
        self.current_status.batch_index = 0
        self.current_status.batch_file  = None
        self.current_status.deck_file   = None
        self.current_status.cases       = None
        
	self.settings.filenames.log_filename = sys.stdout
        self.settings.filenames.err_filename = sys.stderr
        
        # Conditions table, used for surrogate model training
        self.training = Data()        
        self.training.angle_of_attack  = np.array([-2.,3.,8.]) * Units.deg
        self.training.Mach             = np.array([0.3,0.7,0.85])
        self.training.lift_coefficient = None
        self.training.drag_coefficient = None
        self.training_file             = None
        
        # Surrogate model
        self.surrogates = Data()
        self.surrogates.moment_coefficient = None
	
	
	# Initialize quantities
    
	self.configuration = Data()
    
	self.geometry      = Data()
    
	self.stability_model = Data()
	self.stability_model.short_period = Data()
	self.stability_model.short_period.natural_frequency = 0.0
	self.stability_model.short_period.damping_ratio     = 0.0
	self.stability_model.phugoid = Data()
	self.stability_model.phugoid.damping_ratio     = 0.0
	self.stability_model.phugoid.natural_frequency = 0.0
	self.stability_model.roll_tau                  = 0.0
	self.stability_model.spiral_tau                = 0.0 
	self.stability_model.dutch_roll = Data()
	self.stability_model.dutch_roll.damping_ratio     = 0.0
	self.stability_model.dutch_roll.natural_frequency = 0.0
	

    def initialize(self):

        geometry = self.geometry
        self.tag      = 'avl_analysis_of_{}'.format(geometry.tag)
	configuration    = self.configuration
	stability_model  = self.stability_model
    
	configuration.mass_properties = geometry.mass_properties
    
	if geometry.has_key('fuel'): #fuel has been assigned(from weight statements)
	    configuration.fuel = geometry.fuel
	else: #assign as zero to allow things to run
	    fuel = SUAVE.Components.Physical_Component()
	    fuel.mass_properties.mass = 0.
	    configuration.fuel        = fuel	


        run_folder = self.settings.filenames.run_folder 
        
        # Sample training data
        self.sample_training()
    
        # Build surrogate
        self.build_surrogate()


        return

    def evaluate(self,conditions):

        # Unpack
        surrogates = self.surrogates  
	configuration   = self.configuration
        geometry        = self.geometry
	stability_model = self.stability_model	
	
	q             = conditions.freestream.dynamic_pressure
	Sref          = geometry.reference_area    
	velocity      = conditions.freestream.velocity
	density       = conditions.freestream.density
	Span          = geometry.wings['main_wing'].spans.projected
	mac           = geometry.wings['main_wing'].chords.mean_aerodynamic        
        mach          = conditions.freestream.mach_number
        AoA           = conditions.aerodynamics.angle_of_attack
	
	
        moment_model = surrogates.moment_coefficient
        

	configuration   = self.configuration
	stability_model = self.stability_model
	


	
	# set up data structures
	static_stability  = Data()
	dynamic_stability = Data()        
	
    
	#Run Analysis
	data_len = len(AoA)
	CM = np.zeros([data_len,1])
	for ii,_ in enumerate(AoA):
	    CM[ii] = moment_model.predict(np.array([AoA[ii][0],mach[ii][0]]))
	    static_stability.CM  = CM               
	
	
	print static_stability.CM
	if geometry.wings['vertical_stabilizer']:
	    static_stability.cn_beta[i]  = case_results.aerodynamics.cn_beta            
	    
    
	# Dynamic Stability
	if np.count_nonzero(configuration.mass_properties.moments_of_inertia.tensor) > 0:    
	    # Dynamic Stability Approximation Methods - valid for non-zero I tensor            
    
	    for i,_ in enumerate(aero):
		
		# Dynamic Stability
		dynamic_stability.cn_r[i]             = case_results.aerodynamics.Cn_r
		dynamic_stability.cl_p[i]             = case_results.aerodynamics.Cl_p
		dynamic_stability.cl_beta[i]          = case_results.aerodynamics.cl_beta
		dynamic_stability.cy_beta[i]          = 0
		dynamic_stability.cm_q[i]             = case_results.aerodynamics.Cm_q
		dynamic_stability.cm_alpha_dot[i]     = static_stability.cm_alpha[i]*(2*run_conditions.freestream.velocity/mac)
		dynamic_stability.cz_alpha[i]         = case_results.aerodynamics.cz_alpha
		
		dynamic_stability.cl_psi[i]           = aero.lift_coefficient[i]
		dynamic_stability.cL_u[i]             = 0
		dynamic_stability.cz_u[i]             = -2(aero.lift_coefficient[i] - velocity[i]*dynamic_stability.cL_u[i])  
		dynamic_stability.cz_alpha_dot[i]     = static_stability.cz_alpha[i]*(2*run_conditions.freestream.velocity/mac)
		dynamic_stability.cz_q[i]             = 2. * 1.1 * static_stability.cm_alpha[i]
		dynamic_stability.cx_u[i]             = -2. * aero.drag_coefficient[i]
		dynamic_stability.cx_alpha[i]         = aero.lift_coefficient[i] - conditions.lift_curve_slope[i]
    
		#stability_model.dutch_roll.damping_ratio[i]       = (1/(1 + (case_results.aerodynamics.dutch_roll_mode_1_imag /case_results.aerodynamics.dutch_roll_mode_1_real)**2))**0.5
		#stability_model.dutch_roll.natural_frequency[i]   =  - (case_results.aerodynamics.dutch_roll_mode_1_real/stability_model.dutch_roll.damping_ratio)
		#stability_model.dutch_roll.natural_frequency[i]   =  - (case_results.aerodynamics.short_period_mode_1_real/stability_model.short_period.damping_ratio)
		#stability_model.spiral_tau[i]                     =  1/case_results.aerodynamics.spiral_mode_real 
		#stability_model.roll_tau[i]                       =  1/case_results.aerodynamics.roll_mode_real
		#stability_model.short_period.damping_ratio[i]     =  (1/(1 + (case_results.aerodynamics.short_period_mode_1_imag /case_results.aerodynamics.short_period_mode_1_real)**2))**0.5
		#stability_model.short_period.natural_frequency[i] = - (case_results.aerodynamics.short_period_mode_1_real/stability_model.short_period.damping_ratio)
		#stability_model.phugoid.damping_ratio[i]          =  (1/(1 + (case_results.aerodynamics.phugoid_mode_mode_1_imag /case_results.aerodynamics.phugoid_mode_mode_1_real )**2))**0.5
		#stability_model.phugoid.natural_frequency[i]      = - ( case_results.aerodynamics.phugoid_mode_mode_1_real/stability_model.phugoid.damping_ratio)
	    
    
	# pack results
	results = Data()
	results.static  = static_stability
	results.dynamic = dynamic_stability
	
	return results, 0        
         
  
    def sample_training(self):
        
        # Unpack
        geometry = self.geometry
        training = self.training
        
        AoA  = training.angle_of_attack
        mach = training.Mach
        
        CL   = np.zeros([len(AoA)*len(mach),1])
        CD   = np.zeros([len(AoA)*len(mach),1])
        CM   = np.zeros([len(AoA)*len(mach),1])

        
        if self.training_file is None:
            # Calculate aerodynamics for table
            table_size = len(AoA)*len(mach)
            xy = np.zeros([table_size,2])
            count = 0
            time0 = time.time()
            
            for i,_ in enumerate(mach):
                for j,_ in enumerate(AoA):
                    xy[i*len(mach)+j,:] = np.array([AoA[j],mach[i]])
            for j,_ in enumerate(mach):
                # Set training conditions

                run_conditions = Aerodynamics()
                run_conditions.weights.total_mass           = geometry.mass_properties.max_takeoff
                run_conditions.freestream.density           = 1.225
                run_conditions.freestream.gravity           = 9.81          
                run_conditions.aerodynamics.angle_of_attack = AoA
                run_conditions.freestream.mach_number       = mach[j]
                run_conditions.freestream.velocity          = mach[j] * 340.29 #speed of sound
                
                #Run Analysis at AoA[i] and mach[j]
                results =  self.evaluate_conditions(run_conditions)
                
                # Obtain CD and CL # Store other variables here as well 
                CL[count*len(mach):(count+1)*len(mach),0] = results.aerodynamics.lift_coefficient[:,0]
                CD[count*len(mach):(count+1)*len(mach),0] = results.aerodynamics.drag_breakdown.induced.total[:,0]
                CM[count*len(mach):(count+1)*len(mach),0] = results.aerodynamics.pitch_moment_coefficient[:,0]
           
                
       
                count += 1
            
            time1 = time.time()
            
            print 'The total elapsed time to run AVL: '+ str(time1-time0) + '  Seconds'
        else:
            data_array = np.loadtxt(self.training_file)
            xy         = data_array[:,0:2]
            CL         = data_array[:,2:3]
            CD         = data_array[:,3:4]
            CM         = data_array[:,4:5]

        # Save the data
        np.savetxt(geometry.tag+'_data.txt',np.hstack([xy,CL,CD,CM]),fmt='%10.8f',header='AoA Mach CL CD')

        # Store training data
        training.coefficients = np.hstack([CL,CD,CM])
        training.grid_points  = xy
        

        return        

    def build_surrogate(self):

        # Unpack data
        training  = self.training
        AoA_data  = training.angle_of_attack
        mach_data = training.Mach
        CL_data   = training.coefficients[:,0]
        CD_data   = training.coefficients[:,1]
        CM_data   = training.coefficients[:,2]
        xy        = training.grid_points 
        
        # Gaussian Process New
        regr_cm = gaussian_process.GaussianProcess()
        regr_cl = gaussian_process.GaussianProcess()
        regr_cd = gaussian_process.GaussianProcess()
        cl_surrogate = regr_cl.fit(xy, CL_data)
        cd_surrogate = regr_cd.fit(xy, CD_data)
        cm_surrogate = regr_cm.fit(xy, CM_data) 
   

        self.surrogates.lift_coefficient = cl_surrogate
        self.surrogates.drag_coefficient = cd_surrogate
        self.surrogates.moment_coefficient = cm_surrogate
        
        # Standard subsonic test case
        AoA_points = np.linspace(-1.,7.,100)*Units.deg
        mach_points = np.linspace(.25,.9,100)      
        
        AoA_mesh,mach_mesh = np.meshgrid(AoA_points,mach_points)
        
        CL_sur = np.zeros(np.shape(AoA_mesh))
        CD_sur = np.zeros(np.shape(AoA_mesh))
        CM_sur = np.zeros(np.shape(AoA_mesh))
        
        
        for jj in range(len(AoA_points)):
            for ii in range(len(mach_points)):
                CL_sur[ii,jj] = cl_surrogate.predict(np.array([AoA_mesh[ii,jj],mach_mesh[ii,jj]]))
                CD_sur[ii,jj] = cd_surrogate.predict(np.array([AoA_mesh[ii,jj],mach_mesh[ii,jj]]))
                CM_sur[ii,jj] = cm_surrogate.predict(np.array([AoA_mesh[ii,jj],mach_mesh[ii,jj]]))
  
        return
        
              
# ----------------------------------------------------------------------
#  Helper Functions
# ----------------------------------------------------------------------
      
    def evaluate_conditions(self,run_conditions):
        """ process vehicle to setup geometry, condititon and configuration
    
            Inputs:
                run_conditions - DataDict() of aerodynamic conditions; until input
                method is finalized, will just assume mass_properties are always as 
                defined in self.features
    
            Outputs:
                results - a DataDict() of type 
                SUAVE.Analyses.Mission.Segments.Conditions.Aerodynamics(), augmented with
                case data on moment coefficients and control derivatives
    
            Assumptions:
    
        """
        
        # unpack
        run_folder      = os.path.abspath(self.settings.filenames.run_folder)
        output_template = self.settings.filenames.output_template
        batch_template  = self.settings.filenames.batch_template
        deck_template   = self.settings.filenames.deck_template
        stability_output_template = self.settings.filenames.stability_output_template
        
        # update current status
        self.current_status.batch_index += 1
        batch_index                      = self.current_status.batch_index
        self.current_status.batch_file   = batch_template.format(batch_index)
        self.current_status.deck_file    = deck_template.format(batch_index)
        
        # translate conditions
        cases                     = translate_conditions_to_cases(self,run_conditions)
        self.current_status.cases = cases        
        
        # case filenames
        for case in cases:
            case.result_filename = output_template.format(case.tag)
        
        case.eigen_result_filename = stability_output_template.format(batch_index)

            
    
        # write the input files
        with redirect.folder(run_folder,force=False):
            write_geometry(self)
            write_run_cases(self)
            write_input_deck(self)
    
            # RUN AVL!
            results_avl = run_analysis(self)
    
        # translate results
        results = translate_results_to_conditions(cases,results_avl)
    
        if not self.keep_files:
            rmtree( run_folder )
    
        return results

    
	

	
   

    
    


## @ingroup Analyses-Stability
# AVL.py
#
# Created: Apr 2017, M. Clarke 


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
from SUAVE.Methods.Aerodynamics.AVL.Data.Settings    import Settings
from SUAVE.Methods.Aerodynamics.AVL.Data.Cases       import Run_Case

# local imports 
from .Stability import Stability

# Package imports
import time
import pylab as plt
import os
import numpy as np
import sys
from scipy.interpolate import interp1d, interp2d, RectBivariateSpline
from shutil import rmtree

# ----------------------------------------------------------------------
#  Class
# ----------------------------------------------------------------------

## @ingroup Analyses-Stability
class AVL(Stability):
    """This builds a surrogate and computes moment using AVL.

    Assumptions:
    None

    Source:
    None
    """  

    def __defaults__(self):
        """This sets the default values and methods for the analysis.

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
        self.tag                                            = 'avl'
        self.keep_files                                     = True  
        
        self.current_status                                 = Data()
        self.current_status.batch_index                     = 0
        self.current_status.batch_file                      = None
        self.current_status.deck_file                       = None
        self.current_status.cases                           = None
        
        self.settings                                       = Settings()
        self.settings.filenames.log_filename                = sys.stdout
        self.settings.filenames.err_filename                = sys.stderr
        self.settings.spanwise_vortices                     = None
        self.settings.chordwise_vortices                    = None
            
        # Conditions table, used for surrogate model training
        self.training                                       = Data()        

        # Standard subsonic/transonic aircarft
        self.training.angle_of_attack                       = np.array([-2.,0., 2.,5., 7., 10.])*Units.degrees
        self.training.Mach                                  = np.array([0.05,0.15,0.25, 0.45,0.65,0.85])       
        self.training.moment_coefficient                    = None
        self.training.Cm_alpha_moment_coefficient           = None
        self.training.Cn_beta_moment_coefficient            = None
        self.training.neutral_point                         = None
        self.training_file                                  = None

        # Surrogate model
        self.surrogates                                     = Data()
        self.surrogates.moment_coefficient                  = None
        self.surrogates.Cm_alpha_moment_coefficient         = None
        self.surrogates.Cn_beta_moment_coefficient          = None      
        self.surrogates.neutral_point                       = None
        
        # Initialize quantities
        self.configuration                                  = Data()    
        self.geometry                                       = Data()
        
        self.stability_model                                = Data()
        self.stability_model.short_period                   = Data()
        self.stability_model.short_period.natural_frequency = 0.0
        self.stability_model.short_period.damping_ratio     = 0.0
        self.stability_model.phugoid                        = Data()
        self.stability_model.phugoid.damping_ratio          = 0.0
        self.stability_model.phugoid.natural_frequency      = 0.0
        self.stability_model.roll_tau                       = 0.0
        self.stability_model.spiral_tau                     = 0.0 
        self.stability_model.dutch_roll                     = Data()
        self.stability_model.dutch_roll.damping_ratio       = 0.0
        self.stability_model.dutch_roll.natural_frequency   = 0.0

        # Regression Status
        self.regression_flag                                = False

    def finalize(self):
        """Drives functions to get training samples and build a surrogate.

        Assumptions:
        None

        Source:
        N/A

        Inputs:
        None

        Outputs:
        self.tag = 'avl_analysis_of_{}'.format(geometry.tag)

        Properties Used:
        self.geometry.tag
        """          
        geometry                       = self.geometry
        self.tag                       = 'avl_analysis_of_{}'.format(geometry.tag)
        configuration                  = self.configuration
        stability_model                = self.stability_model
        configuration.mass_properties  = geometry.mass_properties
        if 'fuel' in geometry: #fuel has been assigned(from weight statements)
            configuration.fuel         = geometry.fuel
        else: #assign as zero to planes with no fuel such as UAVs
            fuel                       = SUAVE.Components.Physical_Component()
            fuel.mass_properties.mass  = 0.
            configuration.fuel         = fuel	

        # check if user specifies number of spanwise vortices
        if self.settings.spanwise_vortices == None:
            pass
        else:
            self.settings.discretization.defaults.wing.spanwise_vortices = self.settings.spanwise_vortices  
        
        # check if user specifies number of chordise vortices 
        if self.settings.chordwise_vortices == None:
            pass
        else:
            self.settings.discretization.defaults.wing.chordwise_vortices = self.settings.chordwise_vortices
                
        run_folder = self.settings.filenames.run_folder 
   
        # Sample training data
        self.sample_training()

        # Build surrogate
        self.build_surrogate()

        return

    def __call__(self,conditions):
        """Evaluates moment coefficient, stability deriviatives and neutral point using available surrogates.

        Assumptions:
        None

        Source:
        N/A

        Inputs:
        state.conditions.
          mach_number      [-]
          angle_of_attack  [radians]

        Outputs:
        results
            results.static_stability
            results.dynamic_stability
        

        Properties Used:
        self.surrogates.
           pitch_moment_coefficient [-] CM
           cm_alpha                 [-] Cm_alpha
           cn_beta                  [-] Cn_beta
           neutral_point            [-] NP

        """          
        
        # Unpack
        surrogates          = self.surrogates  
        geometry            = self.geometry
        stability_model     = self.stability_model
        configuration       = self.configuration
        
        q                   = conditions.freestream.dynamic_pressure
        Sref                = geometry.reference_area    
        velocity            = conditions.freestream.velocity
        density             = conditions.freestream.density
        Span                = geometry.wings['main_wing'].spans.projected
        mac                 = geometry.wings['main_wing'].chords.mean_aerodynamic        
        mach                = conditions.freestream.mach_number
        AoA                 = conditions.aerodynamics.angle_of_attack
        
        moment_model        = surrogates.moment_coefficient
        Cm_alpha_model      = surrogates.Cm_alpha_moment_coefficient
        Cn_beta_model       = surrogates.Cn_beta_moment_coefficient      
        neutral_point_model = surrogates.neutral_point
         
        # set up data structures
        static_stability    = Data()
        dynamic_stability   = Data()        

        #Run Analysis
        data_len            = len(AoA)
        CM                  = np.zeros([data_len,1])
        Cm_alpha            = np.zeros([data_len,1])
        Cn_beta             = np.zeros([data_len,1])
        NP                  = np.zeros([data_len,1]) 

        for ii,_ in enumerate(AoA):           
            CM[ii]          = moment_model(AoA[ii][0],mach[ii][0]) 
            Cm_alpha[ii]    = Cm_alpha_model(AoA[ii][0],mach[ii][0]) 
            Cn_beta[ii]     = Cn_beta_model(AoA[ii][0],mach[ii][0]) 
            NP[ii]          = neutral_point_model(AoA[ii][0],mach[ii][0]) 

        static_stability.CM       = CM
        static_stability.Cm_alpha = Cm_alpha 
        static_stability.Cn_beta  = Cn_beta   
        static_stability.NP       = NP  
       
        # pack results
        results         = Data()
        results.static  = static_stability
        results.dynamic = dynamic_stability
    
        return results   


    def sample_training(self):
        """Call methods to run AVL for sample point evaluation.

        Assumptions:
        None

        Source:
        N/A

        Inputs:
        see properties used

        Outputs:
        self.training.
          coefficients     [-] CM, Cm_alpha, Cn_beta
          neutral point    [-] NP
          grid_points      [radians,-] angles of attack and mach numbers 

        Properties Used:
        self.geometry.tag  <string>
        self.training.     
          angle_of_attack  [radians]
          Mach             [-]
        self.training_file (optional - file containing previous AVL data)
        """ 
        # Unpack
        geometry = self.geometry
        training = self.training
        
        AoA      = training.angle_of_attack
        mach     = training.Mach
        
        CM       = np.zeros((len(AoA),len(mach)))
        Cm_alpha = np.zeros_like(CM)  
        Cn_beta  = np.zeros_like(CM)  
        NP       = np.zeros_like(CM)  

        
        count      = 0 
        for i,_ in enumerate(mach):
            # Set training conditions
            run_conditions = Aerodynamics()
            run_conditions.weights.total_mass               = 0    # Currently set to zero. Used for dynamic analysis which is under development
            run_conditions.freestream.density               = 0    # Density not used in inviscid computation therefore set to zero. Used for dynamic analysis which is under development
            run_conditions.freestream.gravity               = 9.81          
            run_conditions.aerodynamics.angle_of_attack     = AoA
            run_conditions.freestream.mach_number           = mach[i]
             
            results =  self.evaluate_conditions(run_conditions)

            # Obtain CM Cm_alpha, Cn_beta and the Neutral Point # Store other variables here as well 
            CM [:,i]      = results.aerodynamics.pitch_moment_coefficient[:,0]
            Cm_alpha[:,i] = results.aerodynamics.cm_alpha[:,0]
            Cn_beta[:,i]  = results.aerodynamics.cn_beta[:,0]
            NP[:,i]       = results.aerodynamics.neutral_point[:,0]
  
        
        if self.training_file:
            data_array = np.loadtxt(self.training_file)
            CM_1D         = np.atleast_2d(data_array[:,0])
            Cm_alpha_1D   = np.atleast_2d(data_array[:,1])
            Cn_beta_1D    = np.atleast_2d(data_array[:,2])
            NP_1D         = np.atleast_2d(data_array[:,3])
            
            # convert from 1D to 2D
            CM       = np.reshape(CM_1D      , (len(AoA),-1))
            Cm_alpha = np.reshape(Cm_alpha_1D, (len(AoA),-1))
            Cn_beta  = np.reshape(Cn_beta_1D , (len(AoA),-1)) 
            NP       = np.reshape(NP_1D      , (len(AoA),-1))   
                                                              
        # Save the data for regression                           
        if self.save_regression_results: 
            # convert from 2D to 1D
            CM_1D       = CM.reshape([len(AoA)*len(mach),1]) 
            Cm_alpha_1D = Cm_alpha.reshape([len(AoA)*len(mach),1]) 
            Cn_beta_1D  = Cn_beta.reshape([len(AoA)*len(mach),1]) 
            NP_1D       = NP.reshape([len(AoA)*len(mach),1]) 
            np.savetxt(geometry.tag+'_stability_data.txt',np.hstack([CM_1D, Cm_alpha_1D,  Cn_beta_1D, NP_1D  ]),fmt='%10.8f',header='  CM       CM_alpha       CN_Beta      NP   ')
          
        # Save the data for regression 
        training_data = np.zeros((4,len(AoA),len(mach)))
        training_data[0,:,:] = CM      
        training_data[1,:,:] = Cm_alpha
        training_data[2,:,:] = Cn_beta 
        training_data[3,:,:] = NP
            
        # Store training data
        training.coefficients = training_data

        
        return        

    def build_surrogate(self):
        """Builds a surrogate based on sample evalations using a Guassian process.

        Assumptions:
        None

        Source:
        N/A

        Inputs:
        self.training.
          coefficients     [-] CM, Cm_alpha, Cn_beta 
          neutral point    [meters] NP
          grid_points      [radians,-] angles of attack and mach numbers 

        Outputs:
        self.surrogates.
          moment_coefficient             
          Cm_alpha_moment_coefficient    
          Cn_beta_moment_coefficient     
          neutral_point                     

        Properties Used:
        No others
        """  
        
        # Unpack data
        training  = self.training
        AoA_data  = training.angle_of_attack
        mach_data = training.Mach
        CM_data       = training.coefficients[0,:,:]
        Cm_alpha_data = training.coefficients[1,:,:]
        Cn_beta_data  = training.coefficients[2,:,:] 
        NP_data       = training.coefficients[3,:,:] 
        
        SMOOTHING = 0.1        
        self.surrogates.moment_coefficient          = RectBivariateSpline(AoA_data, mach_data, CM_data      , s=SMOOTHING) 
        self.surrogates.Cm_alpha_moment_coefficient = RectBivariateSpline(AoA_data, mach_data, Cm_alpha_data, s=SMOOTHING) 
        self.surrogates.Cn_beta_moment_coefficient  = RectBivariateSpline(AoA_data, mach_data, Cn_beta_data , s=SMOOTHING)  
        self.surrogates.neutral_point               = RectBivariateSpline(AoA_data, mach_data,  NP_data      , s=SMOOTHING) 
        
        return


# ----------------------------------------------------------------------
#  Helper Functions
# ----------------------------------------------------------------------

    def evaluate_conditions(self,run_conditions):
        """Process vehicle to setup geometry, condititon, and configuration.

        Assumptions:
        None

        Source:
        N/A

        Inputs:
        run_conditions <SUAVE data type> aerodynamic conditions; until input
                method is finalized, will assume mass_properties are always as 
                defined in self.features

        Outputs:
        results        <SUAVE data type>

        Properties Used:
        self.settings.filenames.
          run_folder
          output_template
          batch_template
          deck_template
        self.current_status.
          batch_index
          batch_file
          deck_file
          cases
        """  

        # unpack
        run_folder                       = os.path.abspath(self.settings.filenames.run_folder)
        output_template                  = self.settings.filenames.output_template
        batch_template                   = self.settings.filenames.batch_template
        deck_template                    = self.settings.filenames.deck_template
        
        # rename default avl aircraft tag
        self.settings.filenames.features = self.geometry._base.tag + '.avl'
        
        # update current status
        self.current_status.batch_index += 1
        batch_index                      = self.current_status.batch_index
        self.current_status.batch_file   = batch_template.format(batch_index)
        self.current_status.deck_file    = deck_template.format(batch_index)

        # control surfaces
        num_cs = 0       
        for wing in self.geometry.wings:
            for segment in wing.Segments:
                wing_segment =  wing.Segments[segment]
                section_cs = len(wing_segment.control_surfaces)
                if section_cs != 0:
                    cs_shift = True
                num_cs =  num_cs + section_cs

        # translate conditions
        cases                            = translate_conditions_to_cases(self,run_conditions)    
        for case in cases:
            cases[case].stability_and_control.number_control_surfaces = num_cs
        self.current_status.cases        = cases 

        # case filenames
        for case in cases:
            cases[case].result_filename  = output_template.format(case)

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











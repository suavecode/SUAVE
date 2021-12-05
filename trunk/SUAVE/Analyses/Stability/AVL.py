## @ingroup Analyses-Stability
# AVL.py
#
# Created:  Apr 2017, M. Clarke 
# Modified: Apr 2020, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUAVE imports
import SUAVE
from SUAVE.Core import Units, Data
from SUAVE.Core import redirect

from SUAVE.Analyses.Mission.Segments.Conditions.Aerodynamics import Aerodynamics
from SUAVE.Analyses.Mission.Segments.Conditions.Conditions   import Conditions

from SUAVE.Methods.Aerodynamics.AVL.write_geometry           import write_geometry
from SUAVE.Methods.Aerodynamics.AVL.write_mass_file          import write_mass_file
from SUAVE.Methods.Aerodynamics.AVL.write_run_cases          import write_run_cases
from SUAVE.Methods.Aerodynamics.AVL.write_input_deck         import write_input_deck
from SUAVE.Methods.Aerodynamics.AVL.run_analysis             import run_analysis
from SUAVE.Methods.Aerodynamics.AVL.translate_data           import translate_conditions_to_cases, translate_results_to_conditions
from SUAVE.Methods.Aerodynamics.AVL.purge_files              import purge_files
from SUAVE.Methods.Aerodynamics.AVL.Data.Settings            import Settings
from SUAVE.Methods.Aerodynamics.AVL.Data.Cases               import Run_Case
from SUAVE.Methods.Geometry.Two_Dimensional.Planform.populate_control_sections import populate_control_sections  
from SUAVE.Methods.Flight_Dynamics.Dynamic_Stability.compute_dynamic_flight_modes import  compute_dynamic_flight_modes
from SUAVE.Components.Wings.Control_Surfaces import Aileron , Elevator , Slat , Flap , Rudder 

# local imports 
from .Stability import Stability

# Package imports 
import os
import numpy as np
import sys  
from shutil import rmtree 
from scipy.interpolate import  RectBivariateSpline 

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
        self.tag                                    = 'avl' 
        
        self.current_status                         = Data()        
        self.current_status.batch_index             = 0
        self.current_status.batch_file              = None
        self.current_status.deck_file               = None
        self.current_status.cases                   = None      
        self.geometry                               = None   
                                                    
        self.settings                               = Settings()
        self.settings.filenames.log_filename        = sys.stdout
        self.settings.filenames.err_filename        = sys.stderr        
        self.settings.number_spanwise_vortices      = 20
        self.settings.number_chordwise_vortices     = 10
        self.settings.trim_aircraft                 = False 
        self.settings.print_output                  = False
                                                    
        # Regression Status      
        self.settings.keep_files                    = False
        self.settings.save_regression_results       = False          
        self.settings.regression_flag               = False 

        # Conditions table, used for surrogate model training
        self.training                               = Data()   
        
        # Standard subsonic/transonic aircarft
        self.training.angle_of_attack               = np.array([-2.,0., 2.,5., 7., 10.])*Units.degrees
        self.training.Mach                          = np.array([0.05,0.15,0.25, 0.45,0.65,0.85]) 
        self.settings.side_slip_angle               = 0.0
        self.settings.roll_rate_coefficient         = 0.0
        self.settings.pitch_rate_coefficient        = 0.0 
        self.settings.lift_coefficient              = None
        
        self.training.moment_coefficient            = None
        self.training.Cm_alpha_moment_coefficient   = None
        self.training.Cn_beta_moment_coefficient    = None
        self.training.neutral_point                 = None
        self.training_file                          = None
                                                    
        # Surrogate model
        self.surrogates                             = Data()
        self.surrogates.moment_coefficient          = None
        self.surrogates.Cm_alpha_moment_coefficient = None
        self.surrogates.Cn_beta_moment_coefficient  = None      
        self.surrogates.neutral_point               = None
    
        # Initialize quantities
        self.configuration                          = Data()    
        self.geometry                               = Data()
                                                    
    def finalize(self):
        """Drives functions to get training samples and build a surrogate.

        Assumptions:
        None

        Source:
        N/A

        Inputs:
        None

        Outputs:
        self.tag = 'avl_analysis_of_{}'.format( )

        Properties Used:
        self.geometry.tag
        """          
        geometry                                = self.geometry 
        self.tag                                = 'avl_analysis_of_{}'.format(geometry.tag) 
            
        # Sample training data
        self.sample_training()
        
        # Build surrogate
        self.build_surrogate()
    
        return

    def __call__(self,conditions):
        """Evaluates moment coefficient, stability and body axis deriviatives and neutral point using available surrogates.

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
            results.stability.static
            results.stability.dynamic
        

        Properties Used:
        self.surrogates.
           pitch_moment_coefficient [-] CM
           cm_alpha                 [-] Cm_alpha
           cn_beta                  [-] Cn_beta
           neutral_point            [-] NP

        """          
        
        # Unpack
        surrogates          = self.surrogates       
        Mach                = conditions.freestream.mach_number
        AoA                 = conditions.aerodynamics.angle_of_attack 
        moment_model        = surrogates.moment_coefficient
        Cm_alpha_model      = surrogates.Cm_alpha_moment_coefficient
        Cn_beta_model       = surrogates.Cn_beta_moment_coefficient      
        neutral_point_model = surrogates.neutral_point
        cg                  = self.geometry.mass_properties.center_of_gravity[0]
        MAC                 = self.geometry.wings.main_wing.chords.mean_aerodynamic
        
        # set up data structures
        static_stability    = Data()
        dynamic_stability   = Data()    

        #Run Analysis
        data_len            = len(AoA)
        CM                  = np.zeros([data_len,1])
        Cm_alpha            = np.zeros([data_len,1])
        Cn_beta             = np.zeros([data_len,1])
        NP                  = np.zeros([data_len,1]) 

        for i,_ in enumerate(AoA):           
            CM[i]       = moment_model(AoA[i][0],Mach[i][0])[0]  
            Cm_alpha[i] = Cm_alpha_model(AoA[i][0],Mach[i][0])[0]  
            Cn_beta[i]  = Cn_beta_model(AoA[i][0],Mach[i][0])[0]  
            NP[i]       = neutral_point_model(AoA[i][0],Mach[i][0])[0]    
                
        static_stability.CM            = CM
        static_stability.Cm_alpha      = Cm_alpha 
        static_stability.Cn_beta       = Cn_beta   
        static_stability.neutral_point = NP 
        static_stability.static_margin = (NP - cg)/MAC    
 
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
          grid_points      [radians,-] angles of attack and Mach numbers 

        Properties Used:
        self.geometry.tag  <string>
        self.training.     
          angle_of_attack  [radians]
          Mach             [-]
        self.training_file (optional - file containing previous AVL data)
        """ 
        # Unpack
        run_folder             = os.path.abspath(self.settings.filenames.run_folder)
        geometry               = self.geometry
        training               = self.training 
        trim_aircraft          = self.settings.trim_aircraft  
        AoA                    = training.angle_of_attack
        Mach                   = training.Mach
        side_slip_angle        = self.settings.side_slip_angle
        roll_rate_coefficient  = self.settings.roll_rate_coefficient
        pitch_rate_coefficient = self.settings.pitch_rate_coefficient
        lift_coefficient       = self.settings.lift_coefficient
        atmosphere             = SUAVE.Analyses.Atmospheric.US_Standard_1976()
        atmo_data              = atmosphere.compute_values(altitude = 0.0)         
                               
        CM                     = np.zeros((len(AoA),len(Mach)))
        Cm_alpha               = np.zeros_like(CM)
        Cn_beta                = np.zeros_like(CM)
        NP                     = np.zeros_like(CM)
       
        # remove old files in run directory  
        if os.path.exists('avl_files'):
            if not self.settings.regression_flag:
                rmtree(run_folder)
                
        for i,_ in enumerate(Mach):
            # Set training conditions
            run_conditions = Aerodynamics()
            run_conditions.freestream.density                  = atmo_data.density[0,0] 
            run_conditions.freestream.gravity                  = 9.81            
            run_conditions.freestream.speed_of_sound           = atmo_data.speed_of_sound[0,0]  
            run_conditions.freestream.velocity                 = Mach[i] * run_conditions.freestream.speed_of_sound
            run_conditions.freestream.mach_number              = Mach[i] 
            run_conditions.aerodynamics.side_slip_angle        = side_slip_angle
            run_conditions.aerodynamics.angle_of_attack        = AoA 
            run_conditions.aerodynamics.roll_rate_coefficient  = roll_rate_coefficient
            run_conditions.aerodynamics.lift_coefficient       = lift_coefficient
            run_conditions.aerodynamics.pitch_rate_coefficient = pitch_rate_coefficient
            
            #Run Analysis at AoA[i] and Mach[i]
            results =  self.evaluate_conditions(run_conditions, trim_aircraft)

            # Obtain CM Cm_alpha, Cn_beta and the Neutral Point 
            CM[:,i]       = results.aerodynamics.Cmtot[:,0]
            Cm_alpha[:,i] = results.stability.static.Cm_alpha[:,0]
            Cn_beta[:,i]  = results.stability.static.Cn_beta[:,0]
            NP[:,i]       = results.stability.static.neutral_point[:,0]
        
        if self.training_file:
            # load data 
            data_array   = np.loadtxt(self.training_file)  
            CM_1D        = np.atleast_2d(data_array[:,0]) 
            Cm_alpha_1D  = np.atleast_2d(data_array[:,1])            
            Cn_beta_1D   = np.atleast_2d(data_array[:,2])
            NP_1D        = np.atleast_2d(data_array[:,3])
            
            # convert from 1D to 2D
            CM        = np.reshape(CM_1D, (len(AoA),-1))
            Cm_alpha  = np.reshape(Cm_alpha_1D, (len(AoA),-1))
            Cn_beta   = np.reshape(Cn_beta_1D , (len(AoA),-1))
            NP        = np.reshape(NP_1D , (len(AoA),-1))
        
        # Save the data for regression 
        if self.settings.save_regression_results:
            # convert from 2D to 1D
            CM_1D       = CM.reshape([len(AoA)*len(Mach),1]) 
            Cm_alpha_1D = Cm_alpha.reshape([len(AoA)*len(Mach),1])  
            Cn_beta_1D  = Cn_beta.reshape([len(AoA)*len(Mach),1])         
            NP_1D       = Cn_beta.reshape([len(AoA)*len(Mach),1]) 
            np.savetxt(geometry.tag+'_stability_data.txt',np.hstack([CM_1D,Cm_alpha_1D, Cn_beta_1D,NP_1D ]),fmt='%10.8f',header='   CM       Cm_alpha       Cn_beta       NP ')
        
        # Store training data
        # Save the data for regression
        training_data = np.zeros((4,len(AoA),len(Mach)))
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
          grid_points      [radians,-] angles of attack and Mach numbers 

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
        training                                    = self.training
        AoA_data                                    = training.angle_of_attack
        mach_data                                   = training.Mach
        CM_data                                     = training.coefficients[0,:,:]
        Cm_alpha_data                               = training.coefficients[1,:,:]
        Cn_beta_data                                = training.coefficients[2,:,:]
        NP_data                                     = training.coefficients[3,:,:]
        
        self.surrogates.moment_coefficient          = RectBivariateSpline(AoA_data, mach_data, CM_data      ) 
        self.surrogates.Cm_alpha_moment_coefficient = RectBivariateSpline(AoA_data, mach_data, Cm_alpha_data) 
        self.surrogates.Cn_beta_moment_coefficient  = RectBivariateSpline(AoA_data, mach_data, Cn_beta_data ) 
        self.surrogates.neutral_point               = RectBivariateSpline(AoA_data, mach_data, NP_data      )  
                                                       
        return

    
# ----------------------------------------------------------------------
#  Helper Functions
# ----------------------------------------------------------------------
        
    def evaluate_conditions(self,run_conditions, trim_aircraft ):
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
        run_script_path                  = run_folder.rstrip('avl_files').rstrip('/')
        aero_results_template_1          = self.settings.filenames.aero_output_template_1       # 'stability_axis_derivatives_{}.dat' 
        aero_results_template_2          = self.settings.filenames.aero_output_template_2       # 'surface_forces_{}.dat'
        aero_results_template_3          = self.settings.filenames.aero_output_template_3       # 'strip_forces_{}.dat'   
        aero_results_template_4          = self.settings.filenames.aero_output_template_4       # 'body_axis_derivatives_{}.dat'     
        dynamic_results_template_1       = self.settings.filenames.dynamic_output_template_1    # 'eigen_mode_{}.dat'
        dynamic_results_template_2       = self.settings.filenames.dynamic_output_template_2    # 'system_matrix_{}.dat'
        batch_template                   = self.settings.filenames.batch_template
        deck_template                    = self.settings.filenames.deck_template 
        print_output                     = self.settings.print_output 
        
        # rename defaul avl aircraft tag
        self.tag                         = 'avl_analysis_of_{}'.format(self.geometry.tag) 
        self.settings.filenames.features = self.geometry._base.tag + '.avl'
        self.settings.filenames.mass_file= self.geometry._base.tag + '.mass'
        
        # update current status
        self.current_status.batch_index += 1
        batch_index                      = self.current_status.batch_index
        self.current_status.batch_file   = batch_template.format(batch_index)
        self.current_status.deck_file    = deck_template.format(batch_index)
               
        # control surfaces
        num_cs       = 0
        cs_names     = []
        cs_functions = []
        control_surfaces = False
        for wing in self.geometry.wings: # this parses through the wings to determine how many control surfaces does the vehicle have 
            if wing.control_surfaces:
                control_surfaces = True 
                wing = populate_control_sections(wing)     
                num_cs_on_wing = len(wing.control_surfaces)
                num_cs +=  num_cs_on_wing
                for ctrl_surf in wing.control_surfaces:
                    cs_names.append(ctrl_surf.tag)  
                    if (type(ctrl_surf) ==  Slat):
                        ctrl_surf_function  = 'slat'
                    elif (type(ctrl_surf) ==  Flap):
                        ctrl_surf_function  = 'flap' 
                    elif (type(ctrl_surf) ==  Aileron):
                        ctrl_surf_function  = 'aileron'                          
                    elif (type(ctrl_surf) ==  Elevator):
                        ctrl_surf_function  = 'elevator' 
                    elif (type(ctrl_surf) ==  Rudder):
                        ctrl_surf_function = 'rudder'                      
                    cs_functions.append(ctrl_surf_function)   
        
        # translate conditions
        cases                            = translate_conditions_to_cases(self, run_conditions)    
        for case in cases:
            case.stability_and_control.number_control_surfaces = num_cs
            case.stability_and_control.control_surface_names   = cs_names
        self.current_status.cases        = cases  
        
       # write casefile names using the templates defined in MACE/Analyses/AVL/AVL_Data_Classes/Settings.py 
        for case in cases:  
            case.aero_result_filename_1     = aero_results_template_1.format(case.tag)      # 'stability_axis_derivatives_{}.dat'  
            case.aero_result_filename_2     = aero_results_template_2.format(case.tag)      # 'surface_forces_{}.dat'
            case.aero_result_filename_3     = aero_results_template_3.format(case.tag)      # 'strip_forces_{}.dat'  
            case.aero_result_filename_4     = aero_results_template_4.format(case.tag)      # 'body_axis_derivatives_{}.dat'
            case.eigen_result_filename_1    = dynamic_results_template_1.format(case.tag)   # 'eigen_mode_{}.dat'
            case.eigen_result_filename_2    = dynamic_results_template_2.format(case.tag)   # 'system_matrix_{}.dat'
        
        # write the input files
        with redirect.folder(run_folder,force=False):
            write_geometry(self,run_script_path)
            write_mass_file(self,run_conditions)
            write_run_cases(self,trim_aircraft)
            write_input_deck(self, trim_aircraft,control_surfaces)

            # RUN AVL!
            results_avl = run_analysis(self,print_output)
    
        # translate results
        results = translate_results_to_conditions(cases,results_avl)
        
        # -----------------------------------------------------------------------------------------------------------------------                     
        # Dynamic Stability & System Matrix Computation
        # -----------------------------------------------------------------------------------------------------------------------      
        # Dynamic Stability
        if np.count_nonzero(self.geometry.mass_properties.moments_of_inertia.tensor) > 0:  
                results = compute_dynamic_flight_modes(results,self.geometry,run_conditions,cases)        
             
        if not self.settings.keep_files:
            rmtree( run_folder )           
 
        return results

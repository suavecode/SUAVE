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
import pylab as plt
import os
import numpy as np
import sys
import sklearn
from sklearn import gaussian_process
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
        self.tag                                    = 'avl'
        self.keep_files                             = False
        self.save_regression_results                = False   
        
        self.current_status                         = Data()        
        self.current_status.batch_index             = 0
        self.current_status.batch_file              = None
        self.current_status.deck_file               = None
        self.current_status.cases                   = None      
        self.geometry                               = None   
                                                    
        self.settings                               = Settings()
        self.settings.filenames.log_filename        = sys.stdout
        self.settings.filenames.err_filename        = sys.stderr        
        self.settings.spanwise_vortices             = 20
        self.settings.chordwise_vortices            = 10
        self.settings.trim_aircraft                 = False 
                                                    
        # Conditions table, used for surrogate model training
        self.training                               = Data()   
        
        # Standard subsonic/transonic aircarft
        self.training.angle_of_attack               = np.array([-2.,0., 2.,5., 7., 10.])*Units.degrees
        self.training.Mach                          = np.array([0.05,0.15,0.25, 0.45,0.65,0.85]) 
        
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
                                                    
        # Regression Status                         
        self.regression_flag                        = False 

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
        configuration.mass_properties  = geometry.mass_properties
        if 'fuel' in geometry: #fuel has been assigned(from weight statements)
            configuration.fuel         = geometry.fuel
        else: #assign as zero to planes with no fuel such as UAVs
            fuel                       = SUAVE.Components.Physical_Component()
            fuel.mass_properties.mass  = 0.
            configuration.fuel         = fuel	
            
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
        geometry            = self.geometry 
        
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
            CM[ii]          = moment_model.predict([np.array([AoA[ii][0],mach[ii][0]])])
            Cm_alpha[ii]    = Cm_alpha_model.predict([np.array([AoA[ii][0],mach[ii][0]])])
            Cn_beta[ii]     = Cn_beta_model.predict([np.array([AoA[ii][0],mach[ii][0]])])
            NP[ii]          = neutral_point_model.predict([np.array([AoA[ii][0],mach[ii][0]])])     
            
        static_stability.CM            = CM
        static_stability.Cm_alpha      = Cm_alpha 
        static_stability.Cn_beta       = Cn_beta   
        static_stability.neutral_point = NP 
 
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
        run_folder    = os.path.abspath(self.settings.filenames.run_folder)
        geometry      = self.geometry
        training      = self.training 
        trim_aircraft = self.settings.trim_aircraft            
        
        AoA           = training.angle_of_attack
        mach          = training.Mach
                      
        CM            = np.zeros([len(AoA)*len(mach),1])
        Cm_alpha      = np.zeros([len(AoA)*len(mach),1])
        Cn_beta       = np.zeros([len(AoA)*len(mach),1]) 
        NP            = np.zeros([len(AoA)*len(mach),1]) 

        # Calculate aerodynamics for table
        table_size    = len(AoA)*len(mach)
        xy            = np.zeros([table_size,2])
        count         = 0
        
        # remove old files in run directory  
        if os.path.exists('avl_files'):
            if not self.regression_flag:
                rmtree(run_folder)
        
        for i,_ in enumerate(mach):
            for j,_ in enumerate(AoA):
                xy[i*len(mach)+j,:] = np.array([AoA[j],mach[i]])
        for j,_ in enumerate(mach):
            # Set training conditions
            run_conditions = Aerodynamics()
            run_conditions.freestream.density           = 1.2
            run_conditions.freestream.gravity           = 9.81        
            run_conditions.aerodynamics.angle_of_attack = AoA 
            run_conditions.freestream.speed_of_sound    = 343.
            run_conditions.aerodynamics.side_slip_angle = 0
            run_conditions.freestream.velocity          = mach[j] * run_conditions.freestream.speed_of_sound
            run_conditions.freestream.mach_number       = mach[j] 
            
            #Run Analysis at AoA[i] and mach[j]
            results =  self.evaluate_conditions(run_conditions, trim_aircraft)

            # Obtain CM Cm_alpha, Cn_beta and the Neutral Point # Store other variables here as well 
            CM[count*len(mach):(count+1)*len(mach),0]       = results.aerodynamics.Cmtot[:,0]
            Cm_alpha[count*len(mach):(count+1)*len(mach),0] = results.stability.static.Cm_alpha[:,0]
            Cn_beta[count*len(mach):(count+1)*len(mach),0]  = results.stability.static.Cn_beta[:,0]
            NP[count*len(mach):(count+1)*len(mach),0]       = results.stability.static.neutral_point[:,0]

            count += 1
        
        if self.training_file:
            data_array = np.loadtxt(self.training_file)
            xy         = data_array[:,0:2]
            CM         = data_array[:,2:3]
            Cm_alpha   = data_array[:,3:4]
            Cn_beta    = data_array[:,4:5]
            NP         = data_array[:,5:6]
        
        # Save the data for regression 
        if self.save_regression_results:
            np.savetxt(geometry.tag+'_data_stability.txt',np.hstack([xy,CM,Cm_alpha, Cn_beta,NP ]),fmt='%10.8f',header='     AoA        Mach        CM       Cm_alpha       Cn_beta       NP ')
        
        # Store training data
        training.coefficients = np.hstack([CM,Cm_alpha,Cn_beta,NP])
        training.grid_points  = xy

        
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
          moment_coefficient             <Guassian process surrogate>
          Cm_alpha_moment_coefficient    <Guassian process surrogate>
          Cn_beta_moment_coefficient     <Guassian process surrogate>
          neutral_point                  <Guassian process surrogate>       

        Properties Used:
        No others
        """  
        # Unpack data
        training                                    = self.training
        AoA_data                                    = training.angle_of_attack
        mach_data                                   = training.Mach
        CM_data                                     = training.coefficients[:,0]
        Cm_alpha_data                               = training.coefficients[:,1]
        Cn_beta_data                                = training.coefficients[:,2]
        NP_data                                     = training.coefficients[:,3]	
        xy                                          = training.grid_points 

        # Gaussian Process New
        regr_cm                                     = gaussian_process.GaussianProcessRegressor()
        regr_cm_alpha                               = gaussian_process.GaussianProcessRegressor()
        regr_cn_beta                                = gaussian_process.GaussianProcessRegressor()
        regr_np                                     = gaussian_process.GaussianProcessRegressor()

        cm_surrogate                                = regr_cm.fit(xy, CM_data) 
        cm_alpha_surrogate                          = regr_cm_alpha.fit(xy, Cm_alpha_data) 
        cn_beta_surrogate                           = regr_cn_beta.fit(xy, Cn_beta_data)
        neutral_point_surrogate                     = regr_np.fit(xy, NP_data)

        self.surrogates.moment_coefficient          = cm_surrogate
        self.surrogates.Cm_alpha_moment_coefficient = cm_alpha_surrogate
        self.surrogates.Cn_beta_moment_coefficient  = cn_beta_surrogate   
        self.surrogates.neutral_point               = neutral_point_surrogate
        
        return

    
# ----------------------------------------------------------------------
#  Helper Functions
# ----------------------------------------------------------------------
        
    def evaluate_conditions(self,run_conditions, trim_aircraft  ):
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
        for wing in self.geometry.wings: # this parses through the wings to determine how many control surfaces does the vehicle have 
            if wing.control_surfaces:
                wing = populate_control_sections(wing)     
                num_cs_on_wing = len(wing.control_surfaces)
                num_cs +=  num_cs_on_wing
                for cs in wing.control_surfaces:
                    ctrl_surf = wing.control_surfaces[cs]     
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
            cases[case].stability_and_control.number_control_surfaces = num_cs
            cases[case].stability_and_control.control_surface_names   = cs_names
        self.current_status.cases        = cases  
        
       # write casefile names using the templates defined in MACE/Analyses/AVL/AVL_Data_Classes/Settings.py 
        for case in cases:  
            cases[case].aero_result_filename_1     = aero_results_template_1.format(case)        # 'stability_axis_derivatives_{}.dat'  
            cases[case].aero_result_filename_2     = aero_results_template_2.format(case)        # 'surface_forces_{}.dat'
            cases[case].aero_result_filename_3     = aero_results_template_3.format(case)        # 'strip_forces_{}.dat'  
            cases[case].aero_result_filename_4     = aero_results_template_4.format(case)        # 'body_axis_derivatives_{}.dat'
            cases[case].eigen_result_filename_1    = dynamic_results_template_1.format(case)     # 'eigen_mode_{}.dat'
            cases[case].eigen_result_filename_2    = dynamic_results_template_2.format(case)     # 'system_matrix_{}.dat'
        
        # write the input files
        with redirect.folder(run_folder,force=False):
            write_geometry(self,run_script_path)
            write_mass_file(self,run_conditions)
            write_run_cases(self,trim_aircraft)
            write_input_deck(self, trim_aircraft)

            # RUN AVL!
            results_avl = run_analysis(self)
    
        # translate results
        results = translate_results_to_conditions(cases,results_avl)
        
        # -----------------------------------------------------------------------------------------------------------------------                     
        # Dynamic Stability & System Matrix Computation
        # -----------------------------------------------------------------------------------------------------------------------      
        # Dynamic Stability
        if np.count_nonzero(self.geometry.mass_properties.moments_of_inertia.tensor) > 0:  
                results = compute_dynamic_flight_modes(results,self.geometry,run_conditions,cases)        
             
        if not self.keep_files:
            rmtree( run_folder )           
 
        return results

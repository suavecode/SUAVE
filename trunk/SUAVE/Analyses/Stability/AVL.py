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
from SUAVE.Methods.Aerodynamics.AVL.write_mass_file  import write_mass_file
from SUAVE.Methods.Aerodynamics.AVL.write_run_cases  import write_run_cases
from SUAVE.Methods.Aerodynamics.AVL.write_input_deck import write_input_deck
from SUAVE.Methods.Aerodynamics.AVL.run_analysis     import run_analysis
from SUAVE.Methods.Aerodynamics.AVL.translate_data   import translate_conditions_to_cases, translate_results_to_conditions
from SUAVE.Methods.Aerodynamics.AVL.purge_files      import purge_files
from SUAVE.Methods.Aerodynamics.AVL.Data.Settings    import Settings
from SUAVE.Methods.Aerodynamics.AVL.Data.Cases       import Run_Case
from SUAVE.Components.Wings.Control_Surface          import append_ctrl_surf_to_wing_segments 

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
from warnings import warn
from control.matlab import * # control toolbox needed in python. Run "pip (or pip3) install control"

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
        self.settings.Trim                          = False
        self.settings.Eigen_Modes                   = False
                                                    
        # Conditions table, used for surrogate model training
        self.training                               = Data()   
        
        # Standard subsonic/transolic aircarft
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
            
        static_stability.CM       = CM
        static_stability.Cm_alpha = Cm_alpha 
        static_stability.Cn_beta  = Cn_beta   
        static_stability.NP       = NP 
 
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
        geometry    = self.geometry
        training    = self.training 
        Trim        = self.settings.Trim
        Eigen_Modes = self.settings.Eigen_Modes              
        
        AoA         = training.angle_of_attack
        mach        = training.Mach
        
        CM          = np.zeros([len(AoA)*len(mach),1])
        Cm_alpha    = np.zeros([len(AoA)*len(mach),1])
        Cn_beta     = np.zeros([len(AoA)*len(mach),1]) 
        NP          = np.zeros([len(AoA)*len(mach),1]) 

        # Calculate aerodynamics for table
        table_size = len(AoA)*len(mach)
        xy         = np.zeros([table_size,2])
        count      = 0
        
        for i,_ in enumerate(mach):
            for j,_ in enumerate(AoA):
                xy[i*len(mach)+j,:] = np.array([AoA[j],mach[i]])
        for j,_ in enumerate(mach):
            # Set training conditions
            run_conditions = Aerodynamics()
            run_conditions.weights.total_mass           = geometry.mass_properties.mass
            run_conditions.freestream.density           = 1.2
            run_conditions.freestream.gravity           = 9.81        
            run_conditions.aerodynamics.angle_of_attack = AoA 
            run_conditions.aerodynamics.side_slip_angle = 0 
            run_conditions.freestream.mach_number       = mach[j]
            
            #Run Analysis at AoA[i] and mach[j]
            results =  self.evaluate_conditions(geometry,run_conditions, Trim, Eigen_Modes)

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
        # np.savetxt(geometry.tag+'_data_stability.txt',np.hstack([xy,CM,Cm_alpha, Cn_beta,NP ]),fmt='%10.8f',header='     AoA        Mach        CM       Cm_alpha       Cn_beta       NP ')
        
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
        
    def evaluate_conditions(self,geometry,run_conditions, Trim , Eigen_Modes):
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
        aero_results_template_1          = self.settings.filenames.aero_output_template_1       # 'stability_derivatives_{}.dat'
        aero_results_template_2          = self.settings.filenames.aero_output_template_2       # 'body_axis_derivatives_{}.dat'
        aero_results_template_3          = self.settings.filenames.aero_output_template_3       # 'total_forces_{}.dat'
        aero_results_template_4          = self.settings.filenames.aero_output_template_4       # 'surface_forces_{}.dat'
        aero_results_template_5          = self.settings.filenames.aero_output_template_5       # 'strip_forces_{}.dat'         
        aero_results_template_6          = self.settings.filenames.aero_output_template_6       # 'element_forces_{}.dat'
        aero_results_template_7          = self.settings.filenames.aero_output_template_7       # 'body_forces_{}.dat'
        aero_results_template_8          = self.settings.filenames.aero_output_template_8       # 'hinge_moments_{}.dat' 
        aero_results_template_9          = self.settings.filenames.aero_output_template_9       # 'strip_shear_moment_{}.dat'   
        dynamic_results_template_1       = self.settings.filenames.dynamic_output_template_1    # 'eigen_mode_{}.dat'
        dynamic_results_template_2       = self.settings.filenames.dynamic_output_template_2    # 'system_matrix_{}.dat'
        batch_template                   = self.settings.filenames.batch_template
        deck_template                    = self.settings.filenames.deck_template 
        
        # rename defaul avl aircraft tag
        self.tag                         = 'avl_analysis_of_{}'.format(geometry.tag) 
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
        LongMode_idx = []
        LatMode_idx  = []
        for wing in self.geometry.wings: # this parses through the wings to determine how many control surfaces does the vehicle have 
            if wing.control_surfaces:
                wing = append_ctrl_surf_to_wing_segments(wing)             
            for seg in wing.Segments:
                section_cs = len(wing.Segments[seg].control_surfaces)
                if section_cs != 0:
                    for cs_idx in wing.Segments[seg].control_surfaces:
                        cs_names.append(cs_idx.tag)  
                        cs_functions.append(cs_idx.function) 
                        if cs_idx.function == 'flap' or cs_idx.function == 'elevator' or cs_idx.function == 'slat': 
                            LongMode_idx.append(num_cs)
                        if cs_idx.function == 'aileron' or cs_idx.function == 'rudder': 
                            LatMode_idx.append(num_cs)   
                        num_cs  +=  1
        
        # translate conditions
        cases                            = translate_conditions_to_cases(self,geometry,run_conditions)    
        for case in cases:
            cases[case].stability_and_control.number_control_surfaces = num_cs
            cases[case].stability_and_control.control_surface_names   = cs_names
        self.current_status.cases        = cases  
        
       # write casefile names using the templates defined in MACE/Analyses/AVL/AVL_Data_Classes/Settings.py 
        for case in cases:  
            cases[case].aero_result_filename_1     = aero_results_template_1.format(case)        # 'stability_derivatives_{}.dat'
            cases[case].aero_result_filename_2     = aero_results_template_2.format(case)        # 'body_axis_derivatives_{}.dat'
            cases[case].aero_result_filename_3     = aero_results_template_3.format(case)        # 'total_forces_{}.dat'
            cases[case].aero_result_filename_4     = aero_results_template_4.format(case)        # 'surface_forces_{}.dat'
            cases[case].aero_result_filename_5     = aero_results_template_5.format(case)        # 'strip_forces_{}.dat'            
            cases[case].aero_result_filename_6     = aero_results_template_6.format(case)        # 'element_forces_{}.dat'
            cases[case].aero_result_filename_7     = aero_results_template_7.format(case)        # 'body_forces_{}.dat'
            cases[case].aero_result_filename_8     = aero_results_template_8.format(case)        # 'hinge_moments_{}.dat'
            cases[case].aero_result_filename_9     = aero_results_template_9.format(case)        # 'strip_shear_moment_{}.dat'     
            cases[case].eigen_result_filename_1    = dynamic_results_template_1.format(case)     # 'eigen_mode_{}.dat'
            cases[case].eigen_result_filename_2    = dynamic_results_template_2.format(case)     # 'system_matrix_{}.dat'

        # write the input files
        with redirect.folder(run_folder,force=False):
            write_geometry(self,run_script_path)
            write_mass_file(self,run_conditions)
            write_run_cases(self,Trim)
            write_input_deck(self, Trim, Eigen_Modes)

            # RUN AVL!
            results_avl = run_analysis(self,Eigen_Modes)
    
        # translate results
        results = translate_results_to_conditions(cases,results_avl,Eigen_Modes)
    
        if not self.keep_files:
            rmtree( run_folder )
            
        # -----------------------------------------------------------------------------------------------------------------------                     
        # Dynamic Stability & System Matrix Computation
        # -----------------------------------------------------------------------------------------------------------------------      
        if Eigen_Modes:
            # Unpack aircraft Properties 
            b_ref  = results.b_ref
            c_ref  = results.c_ref
            S_ref  = results.S_ref 
            Ixx    = self.geometry.mass_properties.moments_of_inertia[0][0]
            Iyy    = self.geometry.mass_properties.moments_of_inertia[1][1]
            Izz    = self.geometry.mass_properties.moments_of_inertia[2][2]     
            
            # unpack FLight Conditions  
            rho    = run_conditions.freestream.density
            u0     = run_conditions.freestream.velocity
            q0     = 0.5 * rho * u0**2
            # -----------------------------------------------------------------------------------------------------------------------                     
            # longitudinal Modes 
            # -----------------------------------------------------------------------------------------------------------------------  
            # Build longitudinal EOM A Matrix (stability axis)
            ALon = np.atleast_3d(results.stability.dynamic.A_matrix_LongModes) 
            BLon = np.atleast_3d(results.stability.dynamic.B_matrix_LongModes) 
            CLon = np.repeat(np.atleast_3d(np.identity(6)),len(results.aerodynamics.lift_coefficient),axis = 2)
            DLon = np.zeros_like(BLon)
                
            # Find phugoid
            phugoidFreqHz             = results.stability.dynamic.phugoid_mode_1_real/(2*np.pi)
            phugoidDamping            = np.cos(np.arctan(results.stability.dynamic.phugoid_mode_1_real/results.stability.dynamic.phugoid_mode_1_imag))
            phugoidTimeDoubleHalf     = np.log(2) / (abs(phugoidDamping)* 2 * np.pi *phugoidFreqHz)   
            
            # Find short period
            shortPeriodFreqHz         = np.sqrt(results.stability.dynamic.short_period_mode_1_real**2 +results.stability.dynamic.short_period_mode_1_imag**2)
            shortPeriodDamping        = np.cos(np.arctan(results.stability.dynamic.short_period_mode_1_real/results.stability.dynamic.short_period_mode_1_imag))
            shortPeriodTimeDoubleHalf = np.log(2) /(abs(shortPeriodDamping)* 2 * np.pi *shortPeriodFreqHz)  
            
            # Build longitudinal state space system
            lonSys = ss(ALon,BLon,CLon,DLon)
            
            # Build longitudinal EOM A Matrix (stability axis)
            ALat = np.atleast_3d(results.stability.dynamic.A_matrix_LatModes) 
            BLat = np.atleast_3d(results.stability.dynamic.B_matrix_LatModes) 
            CLat = np.repeat(np.atleast_3d(np.identity(6)),len(results.aerodynamics.lift_coefficient),axis = 2)
            DLat = np.zeros_like(BLon)
            
            # -----------------------------------------------------------------------------------------------------------------------                     
            # Lateral Modes 
            # -----------------------------------------------------------------------------------------------------------------------  
            # Find dutch roll  
            dutchRollFreqHz             = results.stability.dynamic.dutch_roll_mode_1_real/(2*np.pi)
            dutchRollDamping            = np.cos(np.arctan(results.stability.dynamic.dutch_roll_mode_1_real/results.stability.dynamic.dutch_roll_mode_1_imag))
            dutchRollTimeDoubleHalf     = np.log(2) / (abs(dutchRollDamping) * 2 * np.pi * dutchRollFreqHz) 
            
            # Find roll mode
            rollSubsistenceFreqHz       = results.stability.dynamic.roll_mode/(2*np.pi)
            rollSubsistenceDamping      = 1.
            rollSubsistenceTimeConstant = 1 /(2 * np.pi * rollSubsistenceFreqHz * rollSubsistenceDamping  )  
            
            # Find spiral mode      
            spiralFreqHz                = results.stability.dynamic.spiral_mode/(2*np.pi)
            spiralDamping               = 1.
            spiralTimeDoubleHalf        = np.log(2) / (abs(spiralDamping)*2 * np.pi *spiralFreqHz)
                
            # Build lateral state space system
            latSys = ss(ALat,BLat,CLat,DLat) 
            
            # Inertial coupling susceptibility
            # See Etkin & Reid pg. 118
            Mw     = 0.5 * rho * u0 * c_ref * S_ref * results.stability.static.cm_alpha;
            Nv     = 0.5 * rho * u0 * b_ref * S_ref * results.stability.static.cn_beta;      
            results.stability.dynamic.pMax = min(min(np.sqrt(-Mw * u0 /(Izz - Ixx))) , min(np.sqrt(Nv * u0/(Iyy-Ixx)))) # check with Monica (removed -ve sign on Nv)
            
            # -----------------------------------------------------------------------------------------------------------------------  
            # Store Results
            # -----------------------------------------------------------------------------------------------------------------------  
            results.stability.dynamic.lonSys                      = lonSys                                     
            results.stability.dynamic.phugoidFreqHz               = phugoidFreqHz
            results.stability.dynamic.phugoidDamp                 = phugoidDamping
            results.stability.dynamic.phugoidTimeDoubleHalf       = phugoidTimeDoubleHalf
            results.stability.dynamic.shortPeriodFreqHz           = shortPeriodFreqHz
            results.stability.dynamic.shortPeriodDamp             = shortPeriodDamping
            results.stability.dynamic.shortPeriodTimeDoubleHalf   = shortPeriodTimeDoubleHalf
             
            results.stability.dynamic.latSys                      = latSys
            results.stability.dynamic.dutchRollFreqHz             = dutchRollFreqHz
            results.stability.dynamic.dutchRollDamping            = dutchRollDamping 
            results.stability.dynamic.dutchRollTimeDoubleHalf     = dutchRollTimeDoubleHalf
            results.stability.dynamic.rollSubsistenceFreqHz       = rollSubsistenceFreqHz
            results.stability.dynamic.rollSubsistenceTimeConstant = rollSubsistenceTimeConstant
            results.stability.dynamic.rollSubsistenceDamning      = rollSubsistenceDamping
            results.stability.dynamic.spiralFreqHz                = spiralFreqHz
            results.stability.dynamic.spiralTimeDoubleHalf        = spiralTimeDoubleHalf
            results.stability.dynamic.spiralDamping               = spiralDamping 
    
        return results

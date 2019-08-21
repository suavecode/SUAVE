## @ingroup Analyses-Aerodynamics
# AVL_Inviscid.py
#
# Created:  Apr 2017, M. Clarke 
# Modified: Jan 2018, W. Maier
#           Oct 2018, M. Clarke
#           Aug 2019, M. Clarke
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

# Package imports
import pylab as plt
import os
import sklearn
from sklearn import gaussian_process
import numpy as np
import sys
from shutil import rmtree
from warnings import warn
from control.matlab import * # control toolbox needed in python. Run "pip (or pip3) install control"

# ----------------------------------------------------------------------
#  Class
# ----------------------------------------------------------------------
## @ingroup Analyses-Aerodynamics
class AVL_Inviscid(Aerodynamics):
    """This builds a surrogate and computes lift using AVL.

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
        self.tag                             = 'avl'
        self.keep_files                      = True
                
        self.current_status                  = Data()        
        self.current_status.batch_index      = 0
        self.current_status.batch_file       = None
        self.current_status.deck_file        = None
        self.current_status.cases            = None      
        self.geometry                        = None   
        
        self.settings                        = Settings()
        self.settings.filenames.log_filename = sys.stdout
        self.settings.filenames.err_filename = sys.stderr        
        self.settings.spanwise_vortices      = 20
        self.settings.chordwise_vortices     = 10
        self.settings.Trim                   = False
        self.settings.Eigen_Modes            = False
        
        # Conditions table, used for surrogate model training
        self.training                        = Data()   
        
        # Standard subsonic/transolic aircarft
        self.training.angle_of_attack        = np.array([-2.,0., 2.,5., 7., 10.])*Units.degrees
        self.training.Mach                   = np.array([0.05,0.15,0.25, 0.45,0.65,0.85]) 
        
        self.training.lift_coefficient       = None
        self.training.drag_coefficient       = None
        self.training.span_efficiency_factor = None
        self.training_file                   = None
        
        # Surrogate model
        self.surrogates                      = Data()
        
        # Regression Status
        self.regression_flag                 = False

    def initialize(self,spanwise_vortices,chordwise_vortices):
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
        geometry     = self.geometry
        self.tag     = 'avl_analysis_of_{}'.format(geometry.tag)
        run_folder   = self.settings.filenames.run_folder    
            
        # Sample training data
        self.sample_training()
    
        # Build surrogate
        self.build_surrogate()

        return

    def evaluate(self,state,settings,geometry):
        """Evaluates lift and drag using available surrogates.

        Assumptions:
        Returned drag values are currently not meaningful.

        Source:
        N/A

        Inputs:
        state.conditions.
          mach_number      [-]
          angle_of_attack  [radians]

        Outputs:
        inviscid_lift      [-] CL
        inviscid_drag      [-] CD
        span_efficiency    [-] e
        
        Properties Used:
        self.surrogates.
          lift_coefficient       [-] CL
          drag_coefficient       [-] CD
          span_efficiency_factor [-] e
          
        """  
        # Unpack
        surrogates    = self.surrogates        
        conditions    = state.conditions
        
        mach          = conditions.freestream.mach_number
        AoA           = conditions.aerodynamics.angle_of_attack
        lift_model    = surrogates.lift_coefficient
        drag_model    = surrogates.drag_coefficient
        e_model       = surrogates.span_efficiency_factor
        
        # Inviscid lift
        data_len        = len(AoA)
        inviscid_lift   = np.zeros([data_len,1])
        inviscid_drag   = np.zeros([data_len,1])    
        span_efficiency = np.zeros([data_len,1]) 
        
        for ii,_ in enumerate(AoA):
            inviscid_lift[ii]   = lift_model.predict([np.array([AoA[ii][0],mach[ii][0]])])  
            inviscid_drag[ii]   = drag_model.predict([np.array([AoA[ii][0],mach[ii][0]])])
            span_efficiency[ii] = e_model.predict([np.array([AoA[ii][0],mach[ii][0]])])
        
        # Store inviscid lift results     
        conditions.aerodynamics.lift_breakdown.inviscid_wings_lift       = Data()    
        conditions.aerodynamics.lift_breakdown.inviscid_wings_lift       = inviscid_lift
        state.conditions.aerodynamics.lift_coefficient                   = inviscid_lift
        state.conditions.aerodynamics.lift_breakdown.compressible_wings  = inviscid_lift
        
        # Store inviscid drag results  
        ar            = geometry.wings['main_wing'].aspect_ratio
        state.conditions.aerodynamics.inviscid_drag_coefficient          = inviscid_drag
        state.conditions.aerodynamics.drag_breakdown.induced = Data(
            total                  = inviscid_drag   ,
            span_efficiency_factor = span_efficiency ,
        )        
                
        return inviscid_lift
        

    def sample_training(self):
        """Call methods to run AVL for sample point evaluation.

        Assumptions:
        Returned drag values are not meaningful.

        Source:
        N/A

        Inputs:
        see properties used

        Outputs:
        self.training.
          coefficients     [-] CL and CD
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
                    
        CL          = np.zeros([len(AoA)*len(mach),1])
        CD          = np.zeros([len(AoA)*len(mach),1])
        e           = np.zeros([len(AoA)*len(mach),1])
        
        # Calculate aerodynamics for table
        table_size  = len(AoA)*len(mach)
        xy          = np.zeros([table_size,2])  
        count       = 0
        
        for i,_ in enumerate(mach):
            for j,_ in enumerate(AoA):
                xy[i*len(mach)+j,:] = np.array([AoA[j],mach[i]])
        for j,_ in enumerate(mach):
            # Set training conditions
            run_conditions = Aerodynamics()
            if geometry.mass_properties.mass == 0:
                run_conditions.weights.total_mass = geometry.mass_properties.max_takeoff
            elif geometry.mass_properties.max_takeoff == 0:
                run_conditions.weights.total_mass = geometry.mass_properties.mass
            else:
                raise AttributeError("Specify Vehicle Mass")
            run_conditions.freestream.density           = 1.2
            run_conditions.freestream.gravity           = 9.81        
            run_conditions.aerodynamics.angle_of_attack = AoA 
            run_conditions.aerodynamics.side_slip_angle = 0 
            run_conditions.freestream.mach_number       = mach[j]
            
            #Run Analysis at AoA[i] and mach[j]
            results =  self.evaluate_conditions(geometry,run_conditions, Trim, Eigen_Modes)
            
            # Obtain CD , CL and e  
            CL[count*len(mach):(count+1)*len(mach),0]   = results.aerodynamics.lift_coefficient[:,0]
            CD[count*len(mach):(count+1)*len(mach),0]   = results.aerodynamics.drag_breakdown.induced.total[:,0]      
            e[count*len(mach):(count+1)*len(mach),0]    = results.aerodynamics.drag_breakdown.induced.efficiency_factor[:,0]  
            
            count += 1
        
        if self.training_file:
            data_array = np.loadtxt(self.training_file)
            xy         = data_array[:,0:2]
            CL         = data_array[:,2:3]
            CD         = data_array[:,3:4]            
            e          = data_array[:,4:5]
            
        # Save the data for regression
        # np.savetxt(geometry.tag+'_data_aerodynamics.txt',np.hstack([xy,CL,CD,e]),fmt='%10.8f',header='   AoA      Mach     CL     CD    e ')
        
        # Store training data
        training.coefficients = np.hstack([CL,CD,e])
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
          coefficients             [-] CL and CD
          span efficiency factor   [-] e 
          grid_points              [radians,-] angles of attack and mach numbers 

        Outputs:
        self.surrogates.
          lift_coefficient       <Guassian process surrogate>
          drag_coefficient       <Guassian process surrogate>
          span_efficiency_factor <Guassian process surrogate>

        Properties Used:
        No others
        """   
        # Unpack data
        training                         = self.training
        CL_data                          = training.coefficients[:,0]
        CD_data                          = training.coefficients[:,1]
        e_data                           = training.coefficients[:,2]
        xy                               = training.grid_points 
        
        # Gaussian Process New
        regr_cl                          = gaussian_process.GaussianProcessRegressor()
        regr_cd                          = gaussian_process.GaussianProcessRegressor()
        regr_e                           = gaussian_process.GaussianProcessRegressor()
        
        cl_surrogate                     = regr_cl.fit(xy, CL_data)
        cd_surrogate                     = regr_cd.fit(xy, CD_data)
        e_surrogate                      = regr_e.fit(xy, e_data)
        
        self.surrogates.lift_coefficient = cl_surrogate
        self.surrogates.drag_coefficient = cd_surrogate
        self.surrogates.span_efficiency_factor = e_surrogate  
        
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
        
        # clear folder of old files 
        rmtree(run_folder)               
        
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
        EigenModes = False # Do not run eigen mode analysis for aero analysis
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
            
        return results

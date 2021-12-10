## @ingroup Analyses-Aerodynamics
# AVL_Inviscid.py
#
# Created:  Apr 2017, M. Clarke 
# Modified: Jan 2018, W. Maier
#           Oct 2018, M. Clarke
#           Aug 2019, M. Clarke
#           Apr 2020, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUAVE imports
import SUAVE
from SUAVE.Core import Units, Data
from SUAVE.Core import redirect

from SUAVE.Analyses.Mission.Segments.Conditions.Aerodynamics import Aerodynamics
from SUAVE.Analyses.Mission.Segments.Conditions.Conditions   import Conditions

from SUAVE.Methods.Aerodynamics.AVL.write_geometry            import write_geometry
from SUAVE.Methods.Aerodynamics.AVL.write_mass_file           import write_mass_file
from SUAVE.Methods.Aerodynamics.AVL.write_run_cases           import write_run_cases
from SUAVE.Methods.Aerodynamics.AVL.write_input_deck          import write_input_deck
from SUAVE.Methods.Aerodynamics.AVL.run_analysis              import run_analysis
from SUAVE.Methods.Aerodynamics.AVL.translate_data            import translate_conditions_to_cases, translate_results_to_conditions
from SUAVE.Methods.Aerodynamics.AVL.purge_files               import purge_files
from SUAVE.Methods.Aerodynamics.AVL.Data.Settings             import Settings
from SUAVE.Methods.Aerodynamics.AVL.Data.Cases                import Run_Case
from SUAVE.Methods.Geometry.Two_Dimensional.Planform.populate_control_sections   import populate_control_sections  
from SUAVE.Components.Wings.Control_Surfaces import Aileron , Elevator , Slat , Flap , Rudder 

# Package imports 
import os 
import numpy as np
import sys
from shutil import rmtree 
from scipy.interpolate import  RectBivariateSpline 

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
        self.tag                                = 'avl'    
                                                
        self.current_status                     = Data()        
        self.current_status.batch_index         = 0
        self.current_status.batch_file          = None
        self.current_status.deck_file           = None
        self.current_status.cases               = None      
        self.geometry                           = None   
                                                
        self.settings                           = Settings()
        self.settings.filenames.log_filename    = sys.stdout
        self.settings.filenames.err_filename    = sys.stderr        
        self.settings.number_spanwise_vortices  = 20
        self.settings.number_chordwise_vortices = 10
        self.settings.trim_aircraft             = False 
        self.settings.side_slip_angle           = 0.0
        self.settings.roll_rate_coefficient     = 0.0
        self.settings.pitch_rate_coefficient    = 0.0
        self.settings.lift_coefficient          = None
        self.settings.print_output              = False 
        
        # Regression Status
        self.settings.keep_files                = False
        self.settings.save_regression_results   = False          
        self.settings.regression_flag           = False 
        
        # Conditions table, used for surrogate model training
        self.training                           = Data()   
        
        # Standard subsonic/transolic aircarft
        self.training.angle_of_attack           = np.array([-2.,0., 2.,5., 7., 10.])*Units.degrees
        self.training.Mach                      = np.array([0.05,0.15,0.25, 0.45,0.65,0.85]) 
                                                
        self.training.lift_coefficient          = None
        self.training.drag_coefficient          = None
        self.training.span_efficiency_factor    = None
        self.training_file                      = None
        
        # Surrogate model
        self.surrogates                         = Data()

    def initialize(self,number_spanwise_vortices,number_chordwise_vortices,keep_files,save_regression_results,regression_flag,
                   print_output,trim_aircraft,side_slip_angle,roll_rate_coefficient,pitch_rate_coefficient,lift_coefficient):
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

        self.settings.keep_files                = keep_files
        self.settings.save_regression_results   = save_regression_results
        self.settings.regression_flag           = regression_flag       
        self.settings.trim_aircraft             = trim_aircraft   
        self.settings.print_output              = print_output   
        self.settings.number_spanwise_vortices  = number_spanwise_vortices  
        self.settings.number_chordwise_vortices = number_chordwise_vortices
        self.settings.side_slip_angle           = side_slip_angle 
        self.settings.roll_rate_coefficient     = roll_rate_coefficient 
        self.settings.pitch_rate_coefficient    = pitch_rate_coefficient
        self.settings.lift_coefficient          =  lift_coefficient
        
        self.tag     = 'avl_analysis_of_{}'.format(geometry.tag)  
        
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
        Mach          = conditions.freestream.mach_number
        AoA           = conditions.aerodynamics.angle_of_attack
        lift_model    = surrogates.lift_coefficient
        drag_model    = surrogates.drag_coefficient
        e_model       = surrogates.span_efficiency_factor
        
        # Inviscid lift
        data_len        = len(AoA)
        inviscid_lift   = np.zeros([data_len,1])
        inviscid_drag   = np.zeros([data_len,1])    
        span_efficiency = np.zeros([data_len,1]) 
        
        for i,_ in enumerate(AoA): 
            inviscid_lift[i]   = lift_model(AoA[i][0],Mach[i][0])[0] 
            inviscid_drag[i]   = drag_model(AoA[i][0],Mach[i][0])[0] 
            span_efficiency[i] = e_model(AoA[i][0],Mach[i][0])[0] 
        
        # Store inviscid lift results     
        conditions.aerodynamics.lift_breakdown.inviscid_wings_lift = Data()
        conditions.aerodynamics.lift_breakdown.compressible_wings  = Data()
        conditions.aerodynamics.lift_breakdown.inviscid_wings_lift = inviscid_lift
        conditions.aerodynamics.lift_coefficient                   = inviscid_lift
        
        Sref = geometry.reference_area
        for wing in geometry.wings.values():
            wing_area                                                            = wing.areas.reference
            conditions.aerodynamics.lift_breakdown.compressible_wings[wing.tag]  = inviscid_lift*(wing_area/Sref)

         
        # Store inviscid drag results   
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
        
        len_AoA = len(AoA)
        len_Mach = len(Mach)
        
        CL = np.zeros((len_AoA,len_Mach))
        CD = np.zeros_like(CL)  
        e  = np.zeros_like(CL)
        
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
            run_conditions.freestream.mach_number              = Mach[i]
            run_conditions.freestream.velocity                 = Mach[i] * run_conditions.freestream.speed_of_sound
            run_conditions.aerodynamics.side_slip_angle        = side_slip_angle
            run_conditions.aerodynamics.angle_of_attack        = AoA 
            run_conditions.aerodynamics.roll_rate_coefficient  = roll_rate_coefficient
            run_conditions.aerodynamics.lift_coefficient       = lift_coefficient
            run_conditions.aerodynamics.pitch_rate_coefficient = pitch_rate_coefficient
            
            
            #Run Analysis at AoA[i] and Mach[j]
            results =  self.evaluate_conditions(run_conditions, trim_aircraft)
            
            # Obtain CD , CL and e
            CL[:,i] = results.aerodynamics.lift_coefficient[:,0]
            CD[:,i] = results.aerodynamics.drag_breakdown.induced.total[:,0]      
            e [:,i] = results.aerodynamics.drag_breakdown.induced.efficiency_factor[:,0]  
        
        if self.training_file:
            # load data 
            data_array    = np.loadtxt(self.training_file)  
            CL_1D         = np.atleast_2d(data_array[:,0]) 
            CD_1D         = np.atleast_2d(data_array[:,1])            
            e_1D          = np.atleast_2d(data_array[:,2])
            
            # convert from 1D to 2D
            CL = np.reshape(CL_1D, (len_AoA,-1))
            CD = np.reshape(CD_1D, (len_AoA,-1))
            e  = np.reshape(e_1D , (len_AoA,-1))
        
        # Save the data for regression
        if self.settings.save_regression_results: 
            # convert from 2D to 1D
            CL_1D = CL.reshape([len_AoA*len_Mach,1]) 
            CD_1D = CD.reshape([len_AoA*len_Mach,1])  
            e_1D  = e.reshape([len_AoA*len_Mach,1]) 
            np.savetxt(geometry.tag+'_aero_data.txt',np.hstack([CL_1D,CD_1D,e_1D]),fmt='%10.8f',header='  CL      CD      e  ')
          
        # Save the data for regression
        training_data = np.zeros((3,len_AoA,len_Mach))
        training_data[0,:,:] = CL 
        training_data[1,:,:] = CD 
        training_data[2,:,:] = e  
            
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
          coefficients             [-] CL and CD
          span efficiency factor   [-] e 
          grid_points              [radians,-] angles of attack and Mach numbers 
        Outputs:
        self.surrogates.
          lift_coefficient        
          drag_coefficient        
          span_efficiency_factor  
        Properties Used:
        No others
        """    
        # Unpack data
        training  = self.training
        AoA_data  = training.angle_of_attack 
        mach_data = training.Mach 
        CL_data   = training.coefficients[0,:,:]
        CDi_data  = training.coefficients[1,:,:]
        e_data    = training.coefficients[2,:,:]  
       
        self.surrogates.lift_coefficient       = RectBivariateSpline(AoA_data, mach_data, CL_data )   
        self.surrogates.drag_coefficient       = RectBivariateSpline(AoA_data, mach_data, CDi_data)  
        self.surrogates.span_efficiency_factor = RectBivariateSpline(AoA_data, mach_data, e_data  )    
        
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
                for cs in wing.control_surfaces:
                    ctrl_surf = cs    
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
        cases                            = translate_conditions_to_cases(self,run_conditions)    
        for case in cases:
            case.stability_and_control.number_control_surfaces = num_cs
            case.stability_and_control.control_surface_names   = cs_names
        self.current_status.cases                              = cases  
        
       # write case filenames using the templates defined in SUAVE/Analyses/Aerodynamics/AVL/Data/Settings.py 
        for case in cases:  
            case.aero_result_filename_1     = aero_results_template_1.format(case.tag)        # 'stability_axis_derivatives_{}.dat'  
            case.aero_result_filename_2     = aero_results_template_2.format(case.tag)        # 'surface_forces_{}.dat'
            case.aero_result_filename_3     = aero_results_template_3.format(case.tag)        # 'strip_forces_{}.dat'          
            case.aero_result_filename_4     = aero_results_template_4.format(case.tag)        # 'body_axis_derivatives_{}.dat'            
            case.eigen_result_filename_1    = dynamic_results_template_1.format(case.tag)     # 'eigen_mode_{}.dat'
            case.eigen_result_filename_2    = dynamic_results_template_2.format(case.tag)     # 'system_matrix_{}.dat'
        
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
    
        if not self.settings.keep_files:
            rmtree( run_folder )
            
        return results

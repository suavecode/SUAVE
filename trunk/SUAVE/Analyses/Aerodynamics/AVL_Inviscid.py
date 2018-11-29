## @ingroup Analyses-Aerodynamics
# AVL_Inviscid.py
#
# Created:  Apr 2017, M. Clarke 
# Modified: Jan 2018, W. Maier

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

# Package imports
import time
import pylab as plt
import os
import sklearn
from sklearn import gaussian_process
import numpy as np
import sys
from shutil import rmtree
from warnings import warn

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
        
        self.settings                        = Settings()
        
        self.current_status                  = Data()
        
        self.current_status.batch_index      = 0
        self.current_status.batch_file       = None
        self.current_status.deck_file        = None
        self.current_status.cases            = None      
        self.geometry                        = None   
        
        self.settings.filenames.log_filename = sys.stdout
        self.settings.filenames.err_filename = sys.stderr
        
        # Default number of spanwise and chordwise votices
        self.settings.spanwise_vortices      = None
        self.settings.chordwise_vortices     = None
        
        # Conditions table, used for surrogate model training
        self.training                        = Data()   
        
        # Standard subsonic/transolic aircarft
        self.training.angle_of_attack        = np.array([-2.,0., 2.,5., 7., 10.])*Units.degrees
        self.training.Mach                   = np.array([0.05,0.15,0.25, 0.45,0.65,0.85]) 
        
        self.training.lift_coefficient       = None
        self.training.drag_coefficient       = None
        self.training_file                   = None
        
        # Surrogate model
        self.surrogates                      = Data()
        
        # Regression Status
        self.regression_flag                 = False

    def initialize(self):
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

        Properties Used:
        self.surrogates.
          lift_coefficient [-] CL
          drag_coefficient [-] CD
        """  
        # Unpack
        surrogates    = self.surrogates        
        conditions    = state.conditions
        
        mach          = conditions.freestream.mach_number
        AoA           = conditions.aerodynamics.angle_of_attack
        lift_model    = surrogates.lift_coefficient
        drag_model    = surrogates.drag_coefficient
        
        # Inviscid lift
        data_len      = len(AoA)
        inviscid_lift = np.zeros([data_len,1])
        for ii,_ in enumerate(AoA):
            inviscid_lift[ii] = lift_model.predict([np.array([AoA[ii][0],mach[ii][0]])]) #sklearn update fix
            
        conditions.aerodynamics.lift_breakdown.inviscid_wings_lift       = Data()    
        conditions.aerodynamics.lift_breakdown.inviscid_wings_lift       = inviscid_lift

        state.conditions.aerodynamics.lift_coefficient                   = inviscid_lift
        state.conditions.aerodynamics.lift_breakdown.compressible_wings  = inviscid_lift
        
        # Inviscid drag, zeros are a placeholder for possible future implementation
        inviscid_drag                                                    = np.zeros([data_len,1])        
        state.conditions.aerodynamics.inviscid_drag_coefficient          = inviscid_drag
        
        return inviscid_lift, inviscid_drag
        

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
        geometry = self.geometry
        training = self.training   
        
        AoA      = training.angle_of_attack
        mach     = training.Mach   
        
        CL       = np.zeros([len(AoA)*len(mach),1])
        CD       = np.zeros([len(AoA)*len(mach),1])

        # Calculate aerodynamics for table
        table_size = len(AoA)*len(mach)
        xy         = np.zeros([table_size,2])
        count      = 0
        time0      = time.time()
        
        for i,_ in enumerate(mach):
            for j,_ in enumerate(AoA):
                xy[i*len(mach)+j,:] = np.array([AoA[j],mach[i]])
        for j,_ in enumerate(mach):
            # Set training conditions
            run_conditions = Aerodynamics()
            run_conditions.weights.total_mass           = 0     # Currently set to zero. Used for dynamic analysis which is under development
            run_conditions.freestream.density           = 0     # Density not used in inviscid computation therefore set to zero. Used for dynamic analysis which is under development
            run_conditions.freestream.gravity           = 9.81        
            run_conditions.aerodynamics.angle_of_attack = AoA 
            run_conditions.freestream.mach_number       = mach[j]
            
            #Run Analysis at AoA[i] and mach[j]
            results =  self.evaluate_conditions(run_conditions)
            
            # Obtain CD and CL # Store other variables here as well 
            CL[count*len(mach):(count+1)*len(mach),0]   = results.aerodynamics.lift_coefficient[:,0]
            CD[count*len(mach):(count+1)*len(mach),0]   = results.aerodynamics.drag_breakdown.induced.total[:,0]      
       
            count += 1
        
        time1 = time.time()
        
        print('The total elapsed time to run AVL: '+ str(time1-time0) + '  Seconds')
        
        if self.training_file:
            data_array = np.loadtxt(self.training_file)
            xy         = data_array[:,0:2]
            CL         = data_array[:,2:3]
            CD         = data_array[:,3:4]
            
        # Save the data for regression
        #np.savetxt(geometry.tag+'_data_aerodynamics.txt',np.hstack([xy,CL,CD]),fmt='%10.8f',header='   AoA      Mach     CL     CD ')
        
        # Store training data
        training.coefficients = np.hstack([CL,CD])
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
          coefficients     [-] CL and CD
          grid_points      [radians,-] angles of attack and mach numbers 

        Outputs:
        self.surrogates.
          lift_coefficient <Guassian process surrogate>
          drag_coefficient <Guassian process surrogate>

        Properties Used:
        No others
        """   
        # Unpack data
        training                         = self.training
        AoA_data                         = training.angle_of_attack
        mach_data                        = training.Mach
        CL_data                          = training.coefficients[:,0]
        CD_data                          = training.coefficients[:,1]
        xy                               = training.grid_points 
        
        # Gaussian Process New
        regr_cl                          = gaussian_process.GaussianProcessRegressor()
        regr_cd                          = gaussian_process.GaussianProcessRegressor()
        cl_surrogate                     = regr_cl.fit(xy, CL_data)
        cd_surrogate                     = regr_cd.fit(xy, CD_data)
        self.surrogates.lift_coefficient = cl_surrogate
        self.surrogates.drag_coefficient = cd_surrogate  

    
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
        
        # check if user specifies number of spanwise vortices
        if self.settings.spanwise_vortices == None: 
            spanwise_elements  = self.settings.discretization.defaults.wing.spanwise_elements
        else:
            spanwise_elements  = self.settings.spanwise_vortices
        
        # check if user specifies number of chordise vortices 
        if self.settings.chordwise_vortices == None: 
            chordwise_elements  = self.settings.discretization.defaults.wing.chordwise_elements
        else:
            chordwise_elements  = self.settings.chordwise_vortices
        
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
            write_geometry(self,spanwise_elements,chordwise_elements)
            write_run_cases(self)
            write_input_deck(self)
    
            # RUN AVL!
            results_avl = run_analysis(self)
    
        # translate results
        results = translate_results_to_conditions(cases,results_avl)
    
        if not self.keep_files:
            rmtree( run_folder )
    
        return results

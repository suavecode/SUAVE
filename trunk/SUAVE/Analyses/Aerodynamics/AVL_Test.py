# AVL_Test_Script.py
#
# Created:  Tim Momose, Dec 2014 
# Modified: Feb 2016, Andrew Wendorff

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import os
import numpy as np
from shutil import rmtree
from warnings import warn

# SUAVE imports
from SUAVE.Core import Data
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

from Aerodynamics import Aerodynamics as Aero_Analysis


# ----------------------------------------------------------------------
#  Class
# ----------------------------------------------------------------------

class AVL_Test(Aero_Analysis):
    """ SUAVE.Analyses.Aerodynamics.AVL_Test
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
        
        self.features = None


    def finalize(self):

        geometry = self.geometry
        self.tag      = 'avl_analysis_of_{}'.format(geometry.tag)

        run_folder = self.settings.filenames.run_folder
        if os.path.exists(run_folder):
            if self.keep_files:
                warn('deleting old avl run files',Warning)
            rmtree(run_folder)
        os.mkdir(run_folder)

        return


    def evaluate(self,state,**args):
        
        # unpack
        conditions = state.conditions
        results = self.evaluate_conditions(conditions)
        
        # pack conditions
        state.conditions.aerodynamics.lift_coefficient         = results.conditions.aerodynamics.lift_coefficient
        state.conditions.aerodynamics.drag_coefficient         = results.conditions.aerodynamics.drag_coefficient
        state.conditions.aerodynamics.pitch_moment_coefficient = results.conditions.aerodynamics.pitch_moment_coefficient

        return results


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


    def __call__(self,*args,**kwarg):
        return self.evaluate(*args,**kwarg)
    
    
    initialize = finalize

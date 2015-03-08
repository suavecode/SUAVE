# AVL_Callable.py
#
# Created:  Tim Momose, Dec 2014


# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import os
import numpy as np

# SUAVE imports
from SUAVE.Core import Data
import SUAVE.Plugins.VyPy.tools.redirect as redirect

from SUAVE.Analyses.Missions.Segments.Conditions.Aerodynamics import Aerodynamics
from SUAVE.Analyses.Missions.Segments.Conditions.Conditions   import Conditions

from SUAVE.Methods.Aerodynamics.AVL.write_geometry   import write_geometry
from SUAVE.Methods.Aerodynamics.AVL.write_run_cases  import write_run_cases
from SUAVE.Methods.Aerodynamics.AVL.write_input_deck import write_input_deck
from SUAVE.Methods.Aerodynamics.AVL.run_analysis     import run_analysis
from SUAVE.Methods.Aerodynamics.AVL.translate_data   import translate_conditions_to_cases, translate_results_to_conditions
from SUAVE.Methods.Aerodynamics.AVL.purge_files      import purge_files
from SUAVE.Methods.Aerodynamics.AVL.Data.Results     import Results
from SUAVE.Methods.Aerodynamics.AVL.Data.Settings    import Settings
from SUAVE.Methods.Aerodynamics.AVL.Data.Cases       import Run_Case


# ----------------------------------------------------------------------
#  Class
# ----------------------------------------------------------------------

class AVL_Callable(Data):
    """ SUAVE.Methods.Aerodynamics.AVL.AVL_Callable
        aerodynamic model that performs a vortex lattice analysis using AVL
        (Athena Vortex Lattice, by Mark Drela of MIT).

        this class is callable, see self.__call__

    """

    def __defaults__(self):
        self.tag        = 'avl'
        self.keep_files = True

        self.settings = Settings()

        self.analysis_temps = Data()
        self.analysis_temps.current_batch_index = 0
        self.analysis_temps.current_batch_file  = None
        self.analysis_temps.current_cases       = None


    def initialize(self,vehicle):

        self.features = vehicle
        self.tag      = 'avl_analysis_of_{}'.format(vehicle.tag)
        self.settings.filenames.run_folder = \
            os.path.abspath(self.settings.filenames.run_folder)
        if not os.path.exists(self.settings.filenames.run_folder):
            os.mkdir(self.settings.filenames.run_folder)

        return


    def evaluate(self,run_conditions):
        """ process vehicle to setup geometry, condititon and configuration

            Inputs:
                run_conditions - DataDict() of aerodynamic conditions; until input
                method is finalized, will just assume mass_properties are always as 
                defined in self.features

            Outputs:
                results - a DataDict() of type 
                SUAVE.Analyses.Missions.Segments.Conditions.Aerodynamics(), augmented with
                case data on moment coefficients and control derivatives

            Assumptions:

        """
        #assert cases is not None and len(cases) , 'run_case container is empty or None'
        self.analysis_temps.current_batch_index  += 1
        self.analysis_temps.current_batch_file = self.settings.filenames.batch_template.format(self.analysis_temps.current_batch_index)
        cases = translate_conditions_to_cases(self,run_conditions)
        self.analysis_temps.current_cases = cases        

        for case in cases:
            case.result_filename = self.settings.filenames.output_template.format(case.tag)

        with redirect.folder(self.settings.filenames.run_folder,[],[],False):
            write_geometry(self)
            write_run_cases(self)
            write_input_deck(self)

            results_avl = run_analysis(self)

        results = translate_results_to_conditions(cases,results_avl)

        if not self.keep_files:
            from shutil import rmtree
            rmtree(os.path.abspath(self.settings.filenames.run_folder))

        return results

    def __call__(self,*args,**kwarg):
        return self.evaluate(*args,**kwarg)
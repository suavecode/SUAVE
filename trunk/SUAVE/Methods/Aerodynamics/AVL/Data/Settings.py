## @ingroup Methods-Aerodynamics-AVL-Data
#Settings.py
# 
# Created:  Dec 2014, T. Momose
# Modified: Jan 2016, E. Botero
#           Oct 2018, M. Clarke
#           Aug 2019, M. Clarke
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import SUAVE
from SUAVE.Core import Data
from .Cases     import Run_Case

# ------------------------------------------------------------
#   Configuration
# ------------------------------------------------------------

## @ingroup Methods-Aerodynamics-AVL-Data
class Settings(Data):
        """ A class that defines important settings that call the AVL executable in addition to the 
        format of the result, batch and geometry filenames
        
        Assumptions:
            None
    
        Source:
            None
    
        Inputs:
            None
    
        Outputs:
            None
    
        Properties Used:
            N/A
        """    

        def __defaults__(self):
                """ Defines naming convention for files created/used by AVL to compute analysus
        
                Assumptions:
                    None
        
                Source:
                    None
        
                Inputs:
                    None
        
                Outputs:
                    None
        
                Properties Used:
                    N/A
                """  
                self.run_cases                           = Run_Case.Container()
                self.filenames                           = Data()
                self.flow_symmetry                       = Data()
                self.discretization                      = Data()
                self.number_control_surfaces             = 0
                
                self.filenames.avl_bin_name              = 'avl' # to call avl from command line. If avl is not on the system path, include absolute path to the avl binary i.e. '/your/path/to/avl'
                self.filenames.run_folder                = 'avl_files'  
                self.filenames.features                  = 'aircraft.avl'
                self.filenames.mass_file                 = 'aircraft.mass'
                self.filenames.batch_template            = 'batch_{0:03d}.run'
                self.filenames.deck_template             = 'commands_{0:03d}.deck' 
                self.filenames.aero_output_template_1    = 'stability_derivatives_{}.dat'
                self.filenames.aero_output_template_2    = 'body_axis_derivatives_{}.dat'
                self.filenames.aero_output_template_3    = 'total_forces_{}.dat'
                self.filenames.aero_output_template_4    = 'surface_forces_{}.dat'
                self.filenames.aero_output_template_5    = 'strip_forces_{}.dat'                
                self.filenames.aero_output_template_6    = 'element_forces_{}.dat'
                self.filenames.aero_output_template_7    = 'body_forces_{}.dat'
                self.filenames.aero_output_template_8    = 'hinge_moments_{}.dat'
                self.filenames.aero_output_template_9    = 'strip_shear_moment_{}.dat'
                self.filenames.dynamic_output_template_1 = 'eigen_mode_{}.dat'
                self.filenames.dynamic_output_template_2 = 'system_matrix_{}.dat'
                self.filenames.case_template             = 'case_{0:03d}_{1:02d}'
                self.filenames.log_filename              = 'avl_log.dat'
                self.filenames.err_filename              = 'avl_err.dat'
                
                self.flow_symmetry.xz_plane              = 0	# Symmetry across the xz-plane, y=0
                self.flow_symmetry.xy_parallel           = 0    # Symmetry across the z=z_symmetry_plane plane
                self.flow_symmetry.z_symmetry_plane      = 0.0

                

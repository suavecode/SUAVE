## @ingroup Methods-Aerodynamics-AVL-Data
# Configuration.py
# 
# Created:  Oct 2014, T. Momose
# Modified: Jan 2016, E. Botero
#           Apr 2017, M. Clarke
#           Aug 2019, M. Clarke
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Data


# ------------------------------------------------------------
#   Configuration
# ------------------------------------------------------------

## @ingroup Methods-Aerodynamics-AVL-Data
class Configuration(Data):
	"""A data class defining the reference parameters of the aircraft geometry and 
	flight configuration 

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
		""" Defines the data structure and defaults for mass properties of the aircraft 
		in AVL

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
		self.tag = 'configuration'
		self.parasite_drag                = 0.0
		
		self.reference_values             = Data()     
		self.reference_values.sref        = 0.0        # [m]
		self.reference_values.bref        = 0.0        # [m]
		self.reference_values.cref        = 0.0        # [m]
		self.reference_values.cg_coords   = [0.,0.,0.] # [m]
		
		self.mass_properties              = Data()
		self.mass_properties.inertial     = Data()
		self.mass_properties.mass         = 0.0        # [kg]
		self.mass_properties.inertial.Ixx = 0.0        # [kg.m^2]   
		self.mass_properties.inertial.Iyy = 0.0        # [kg.m^2]   
		self.mass_properties.inertial.Izz = 0.0        # [kg.m^2]   
		self.mass_properties.inertial.Ixy = 0.0        # [kg.m^2]   
		self.mass_properties.inertial.Iyz = 0.0        # [kg.m^2]   
		self.mass_properties.inertial.Izx = 0.0        # [kg.m^2]
		
		self.symmetry_settings            = Data()
		self.symmetry_settings.Iysym      = 0	# Assumed y-symmetry of solution 
							# (1: symmetric, -1: antisymmetric, 0: no symmetry assumed)
		self.symmetry_settings.Izsym      = 0	# Assumed z-symmetry of solution
		self.symmetry_settings.Zsym       = 0.0	# z-coordinate of plane of z-symmetry
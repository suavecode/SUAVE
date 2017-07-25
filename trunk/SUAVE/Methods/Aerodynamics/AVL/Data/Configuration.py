# Configuration.py
# 
# Created:  Oct 2014, T. Momose
# Modified: Jan 2016, E. Botero
#           Apr 2017, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Data


# ------------------------------------------------------------
#   Configuration
# ------------------------------------------------------------

class Configuration(Data):
	
	def __defaults__(self):
		
		self.tag = 'configuration'
		self.parasite_drag                = 0.0
		
		self.reference_values             = Data()
		self.reference_values.sref        = 0.0
		self.reference_values.bref        = 0.0
		self.reference_values.cref        = 0.0
		self.reference_values.cg_coords   = [0.,0.,0.]
		
		self.mass_properties              = Data()
		self.mass_properties.inertial     = Data()
		self.mass_properties.mass         = 0.0
		self.mass_properties.inertial.Ixx = 0.0
		self.mass_properties.inertial.Iyy = 0.0
		self.mass_properties.inertial.Izz = 0.0
		self.mass_properties.inertial.Ixy = 0.0
		self.mass_properties.inertial.Iyz = 0.0
		self.mass_properties.inertial.Izx = 0.0
		
		self.symmetry_settings            = Data()
		self.symmetry_settings.Iysym      = 0	# Assumed y-symmetry of solution 
							# (1: symmetric, -1: antisymmetric, 0: no symmetry assumed)
		self.symmetry_settings.Izsym      = 0	# Assumed z-symmetry of solution
		self.symmetry_settings.Zsym       = 0.0	# z-coordinate of plane of z-symmetry

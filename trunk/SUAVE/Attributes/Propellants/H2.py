""" H2.py: Physical properties of H2 for propulsion use """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from Propellant import Propellant

# ----------------------------------------------------------------------
#  Class
# ----------------------------------------------------------------------

class H2(Propellant):

    """ Physical properties of H2 for propulsion use """

    def __defaults__(self):

        self.tag='H2'
        self.MolecularMass=2
        self.R=8314/2  #gas constant [J/kg-k]
        self.phase='liquid'  #either liquid or gas
        self.specific_energy=0. #specific energy in [J/kg]
        self.density=0.
        self.volume=0.
        
        if self.phase=='liquid':
            self.specific_energy=141.86E6 #[J/kg], using higher heating value
            self.density=70.99 #[kg/m^3]
            
        if self.phase=='gas':
            self.specific_energy=123E6# [J/kg]
            self.P=700E5   #tank pressure [Pa]
            
            self.T=293     #temperature [K]
            self.z=1.4699  #compressibility factor
            self.density=self.P/(self.R*self.T*self.z)
                
                
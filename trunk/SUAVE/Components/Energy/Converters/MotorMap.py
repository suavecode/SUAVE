#Created by T. MacDonald 4/1/14
#Last modified 6/23/14 
#AIAA 2014-0536
#Electric Propulsion Modeling for Conceptual Aircraft Design
#Robert A. McDonald

""" SUAVE.Attrubtes.Components.Energy.Conversion.MotorMap
"""

# ------------------------------------------------------------
#  Imports 
# ------------------------------------------------------------

from Converter import Converter

# ------------------------------------------------------------
#  Energy Conversion
# ------------------------------------------------------------

class GenConverter(Converter):
    """ SUAVE.Attributes.Components.Energy.Conversion.Component
    """
    def __defaults__(self):
        self.tag = 'General Electrical Converter'
	self.omegah = 1 # RPM at max efficiency
	self.Qh = 1 # Torque at max efficiency
	self.etah = 0.95 # Max efficiency
	self.k0 = .5 # Parasite loss ratio
	self.kQ = 2 # Ratio of max rated torque to Qh
	self.kP = 2 # Ratio of max rated power to power at max efficiency
	self.komega = 2 # Ratio of max rated RPM to omegah
	# Parameters for reasonable match to empricial results
	self.C0 = self.k0*self.omegah*self.Qh/6*(1-self.etah)/self.etah
	self.C1 = -3*self.C0/2/self.omegah + self.Qh*(1-self.etah)/4/self.etah
	self.C2 = self.C0/2/self.omegah**3 + self.Qh*(1-self.etah)/4/self.etah/self.omegah**2
	self.C3 = self.omegah*(1-self.etah)/2/self.Qh/self.etah	
	 	
    def eta(self,omega,Q):
	# Calculation of motor efficiency at given RPM and Torque
	PL = self.C0+self.C1*omega+self.C2*omega**3+self.C3*Q**2
	return omega*Q/(omega*Q+PL)

    def __call__(self,omegah,Qh,etah,k0,kQ,kP,komega):
	C0 = k0*omegah*Qh/6*(1-etah)/etah
	self.C0 = C0
	self.C1 = -3*C0/2/omegah + Qh*(1-etah)/4/etah
	self.C2 = C0/2/omegah**3 + Qh*(1-etah)/4/etah/omegah**2
	self.C3 = omegah*(1-etah)/2/Qh/etah
		
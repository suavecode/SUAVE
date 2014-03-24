""" Burner.py: Burner Propulsor Segment """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
from Segment import Segment
from SUAVE.Attributes.Propellants import Jet_A

# ----------------------------------------------------------------------
#  Class
# ----------------------------------------------------------------------

class Burner(Segment):   

    """ A Burner Segment of a Propulsor """

    def __defaults__(self):

        self.tag = 'Burner'
        self.propellant = Jet_A()
        self.oxidizer = 'Air'
        self.fuel_ox_ratio_max = 0.0
        self.Tt_limit = None                # K

    def __call__(self,thermo,power):

        """  Burner(): populates final p_t and T_t values based on initial p_t, T_t values and efficiencies
    
         Inputs:    self.pt[0] = stagnation pressure at burner inlet            (float)     (required)
                    self.Tt[0] = stagnation temperature at burner inlet         (float)     (required)
                    self.eta = burner efficiency                                (float)     (required) default = 1.0 (ideal)
                    self.pt_ratio = burner pt ratio (pt loss)                   (float)     (required) default = 1.0 (ideal)

         Outputs:   self.pt[1] = stagnation pressure at burner outlet           (float)     
                    self.Tt[1] = stagnation temperature at burner outlet        (float)                                                                                                  

        """

        # unpack
        i = self.i; f = self.f
        hf = self.propellant.specific_energy    # J/kg

        if self.active:

            # check limit data
            max_T = False; chemistry = True
            if self.Tt_limit is not None:
                if len(self.Tt_limit) == 1:
                    if self.Tt_limit > thermo.Tt[i]:
                        max_T = True; chemistry = False
                    else:
                        print "Error in Burner: maximum temperature must >= inlet temperature"
                        print "Ignoring burner temperature limit, setting fuel burn to zero"
                        self.active = False
                else:
                    print "Error in Burner: maximum temperature must be a single defined value"
                    print "Ignoring burner temperature limit, setting fuel burn to zero"
                    self.active = False

        if self.active:

            # burner max temperature limit
            if max_T:            
                thermo.ht[f] = self.Tt_limit*thermo.cp[f]
                self.fuel_ox_ratio_max = (thermo.ht[f] - thermo.ht[i])/(self.eta*hf - thermo.ht[i])
                
            # burner limited by chemistry
            elif chemistry:        
                self.fuel_ox_ratio_max = self.propellant.max_mass_fraction[self.oxidizer]    
                thermo.ht[f] = (self.fuel_ox_ratio_max*hf*self.eta + thermo.ht[i])/(1 + self.fuel_ox_ratio_max)

            # compute Tt and pt        
            thermo.Tt[f] = thermo.ht[f]/thermo.cp[f]
            thermo.pt[f] = thermo.pt[i]*self.pt_ratio
            power.fuel_ox_ratio_max += self.fuel_ox_ratio_max
        
        # burner inactive, add dummy segment
        else:
            thermo.ht[f] = thermo.ht[i]
            thermo.Tt[f] = thermo.Tt[i]
            thermo.pt[f] = thermo.pt[i]
        
        return
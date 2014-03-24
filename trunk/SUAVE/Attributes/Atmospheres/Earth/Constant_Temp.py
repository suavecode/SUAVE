""" Constant_Temp.py: Constant Temperature Atmosphere """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Attributes.Atmospheres import Atmosphere

# ----------------------------------------------------------------------
#  Earth at Constant Temperature
# ----------------------------------------------------------------------

class Constant_Temp(Atmosphere):
    """ SUAVE.Attributes.Atmospheres.Constant_Temp
        ISA Pressure variation with constant (or input) outside temperature
    """
    #def __setattr__(self,k,v):
        #allow = ['alt', 'dt', 'oat']
        #if k in allow:
            #self.__dict__[k] = v
            #self._calcAtmosphere()
        #else:
            ##protected attribute
            #print '%s is calculated, not input' % k
    ##end __setattr__
    
    #def _calcAirTemperature(self):
        #""" Temperature directly input
        #"""
        #pass
    
    #def _calcAirPressure(self):
        #""" Calculate pressure variation in standard atmosphere based on perfect gas law and
        #static atmosphere assumption
        #"""
        #isa = IntlStandardAtmosphere(self.alt)
        #self.__dict__['pressure'] = isa.pressure   
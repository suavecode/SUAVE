## @ingroup Attributes-Solids
# Copper.py
#
# Created: Feb 2020, K. Hamilton
# Modified: Jan 2022, S. Claridge

#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------

from .Solid import Solid
from SUAVE.Core import Units
from scipy import interpolate
from array import *
import numpy as np

#-------------------------------------------------------------------------------
# RRR=50 OFHC Copper Class
#-------------------------------------------------------------------------------

## @ingroup Attributes-Solid
class Copper(Solid):

    """ Physical Constants Specific to copper RRR=50 OFHC
    
    Assumptions:
    None
    
    Source:
    "PROPERTIES OF SELECTED MATERIALS AT CRYOGENIC TEMPERATURES" Peter E. Bradley and Ray Radebaugh
    "A copper resistance temperature scale" Dauphinee, TM and Preston-Thomas, H
    
    Inputs:
    N/A
    
    Outputs:
    N/A
    
    Properties Used:
    None
    """

    def __defaults__(self):
        """Sets material properties at instantiation.
        
        Assumptions:
        None

        Source:
        N/A

        Inputs:
        N/A

        Outputs:
        N/A

        Properties Used:
        None
        """

        self.density                    =     8960.0        # [kg/(m**3)]
        self.conductivity_electrical    = 58391886.09       # [mhos/m]
        self.conductivity_thermal       =      392.4        # [W/(m*K)]
        self.interpolate                = False

        # Lookup table arrays. Temperature in K, thermal conductivity in W/(m*K)
        temperatures   = np.array([4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 120.0, 140.0, 160.0, 180.0, 200.0, 220.0, 240.0, 260.0, 280.0, 300.0])
        conductivities = np.array([3.204, 4.668, 6.223, 7.781, 9.273, 10.64, 11.85, 12.87, 13.68, 14.44, 11.63, 8.636, 6.7, 5.611, 5.003, 4.651, 4.439, 4.218, 4.116, 4.06, 4.026, 4.001, 3.982, 3.965, 3.95, 3.936, 3.924])

        # Function that interpolates the lookup table data
        self.c_thermal = interpolate.interp1d(temperatures, conductivities, kind = 'cubic', fill_value='extrapolate')


        # Lookup table. Temperature in K, conductivity in mhos/m
        temperatures   = np.array([4.2, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 38.0, 40.0, 42.0, 44.0, 46.0, 48.0, 50.0, 52.0, 54.0, 56.0, 58.0, 60.0, 64.0, 68.0, 72.0, 76.0, 80.0, 85.0, 90.0, 95.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 190.0, 200.0, 210.0, 220.0, 230.0, 240.0, 250.0, 260.0, 270.0, 273.16, 280.0, 290.0, 300.0, 310.0, 320.0])
        conductivities = np.array([62706513.73, 59649351.1, 58823529.41, 57862233.7, 56756705.27, 55547010.82, 54265718.98, 52838319.53, 51265727.5, 49621943.91, 47879990.68, 46119256.86, 44295117.43, 42462678.59, 40594025.68, 38689780.55, 36839664.93, 34982975.23, 33178363.63, 29768952.17, 26747761.29, 23951445.97, 21412914.59, 19097216.66, 17144994.46, 15418051.98, 13912657.74, 12602171.91, 11436522.41, 10413826.85, 9526550.123, 8071135.431, 6915841.211, 6030094.818, 5295406.175, 4719663.549, 4129948.887, 3661837.678, 3283955.516, 2972855.668, 2494427.41, 2147259.712, 1884972.1, 1680269.452, 1516360.227, 1382622.667, 1270747.745, 1176213.988, 1095370.181, 1025134.636, 963654.7402, 909308.7026, 860944.9011, 817561.3102, 778513.0909, 742977.6209, 710652.6341, 701058.8235, 681102.5197, 653862.9927, 628729.7528, 605483.2867, 583918.8609])

        # Function that interpolates the lookup table data
        self.c_electrical = interpolate.interp1d(temperatures, conductivities, kind = 'cubic', fill_value='extrapolate')



    def thermal_conductivity(self, temperature):
                
        # Create output variable
        conductivity = self.c_thermal(temperature)

        return conductivity
        


    # lookup table and interpolator for estimating the electrical conductivity of copper at cryogenic temperatures.
    def electrical_conductivity(self, temperature):

        
        conductivity = self.c_electrical(temperature)
        
        return conductivity
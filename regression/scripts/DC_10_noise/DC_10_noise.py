# DC_10_noise.py

import SUAVE
import numpy as np
from SUAVE.Core import Units
from SUAVE.Methods.Noise import Correlations as Correlations
from SUAVE.Core import (
    Data, Container
)

def main():

    weight_landing    = 300000 * Units.lbs
    number_of_engines = 3.
    thrust_sea_level  = 40000 * Units.force_pounds
    thrust_landing    = 0.45 * thrust_sea_level
    
    noise = Correlations.shevell(weight_landing, number_of_engines, thrust_sea_level, thrust_landing)
    
    actual = Data()
    actual.takeoff   = 99.982372547196633
    actual.side_line = 97.482372547196633
    actual.landing   = 105.69577388532885
    
    # Shevell Data
    #actual.takeoff   = 99.7
    #actual.side_line = 97.2
    #actual.landing   = 105.2    

    error = Data()
    error.takeoff = (actual.takeoff - noise.takeoff)/actual.takeoff
    error.side_line = (actual.side_line - noise.side_line)/actual.side_line
    error.landing = (actual.landing - noise.landing)/actual.landing
    
    for k,v in list(error.items()):
        assert(np.abs(v)<1e-6)
        
    return

# ----------------------------------------------------------------------        
#   Call Main
# ----------------------------------------------------------------------    

if __name__ == '__main__':
    main()
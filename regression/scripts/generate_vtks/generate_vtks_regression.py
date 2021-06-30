# VTK_Test.py
# 
# Created:  Jun 2021, R. Erhard
# Modified: 

""" generates vtk files for X57 aircraft 
"""
# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Units

import numpy as np
import pylab as plt
import sys

from SUAVE.Plots.Mission_Plots import *  
from SUAVE.Plots.Geometry_Plots.plot_vehicle import plot_vehicle  
from SUAVE.Plots.Geometry_Plots.plot_vehicle_vlm_panelization  import plot_vehicle_vlm_panelization

sys.path.append('../Vehicles') 
from X57_Maxwell import vehicle_setup, configs_setup 


# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------
def main():
    # run test with helical fixed wake model
    x57 = vehicle_setup()
    generate_vtks(x57)
    
    return 


if __name__ == '__main__': 
    main()    
    plt.show()
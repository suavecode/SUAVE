# VTK_Test.py
# 
# Created:  Jun 2021, R. Erhard
# Modified: 

""" generates vtk files for X57 aircraft 
"""
# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import pylab as plt
import sys


from SUAVE.Time_Accurate.Simulations.save_vehicle_vtk import save_vehicle_vtk
sys.path.append('../Vehicles') 
from X57_Maxwell import vehicle_setup, configs_setup 


# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------
def main():
    # run test with helical fixed wake model
    x57 = vehicle_setup()
    save_vehicle_vtk(x57)
    
    return 


if __name__ == '__main__': 
    main()    
    plt.show()
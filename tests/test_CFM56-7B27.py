""" test_CFM56-7B27.py: size a CFM International 56-7B27 Engine """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import sys
sys.path.append('../trunk')
import SUAVE
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
#  Main
# ----------------------------------------------------------------------

def main():
    
    # initialize the engine
    CFM56_7B27 = SUAVE.Components.Propulsors.Turbojet()
    
    # segment total pressure properties 
    CFM56_7B27.Inlet.pt_ratio = 0.99
    CFM56_7B27.Compressors.LowPressure.pt_ratio = 2.0 ###
    CFM56_7B27.Compressors.HighPressure.pt_ratio = 2.0 ###
    CFM56_7B27.Burner.pt_ratio = 0.90
    CFM56_7B27.Turbines.HighPressure.pt_ratio = 0.5 ####
    CFM56_7B27.Nozzle.pt_ratio = 0.99

    # segment efficiencies
    CFM56_7B27.Compressors.LowPressure.eta = 0.90
    CFM56_7B27.Compressors.HighPressure.eta = 0.90
    CFM56_7B27.Burner.eta = 0.95
    CFM56_7B27.Turbines.HighPressure.eta = 0.95
    CFM56_7B27.Turbines.LowPressure.eta = 0.95

    # set design reference point: static sea level (default)
    atmosphere = SUAVE.Attributes.Atmospheres.Earth.US_Standard_1976()
    M_design = 0.0
    h_design = 0.0
    F_sls = 121e3       # N

    # initialize engine properties
    print CFM56_7B27.initialize(F_sls,atmosphere,h_design,M_design)

    # plot solution
    #title = "Thrust Angle History"
    #plt.figure(0)
    #for i in range(len(results.Segments)):
    #    plt.plot(results.Segments[i].t/60,np.degrees(results.Segments[i].gamma),'bo-')
    #plt.xlabel('Time (mins)'); plt.ylabel('Thrust Angle (deg)'); plt.title(title)
    #plt.grid(True)

    #title = "Throttle History"
    #plt.figure(1)
    #for i in range(len(results.Segments)):
    #    plt.plot(results.Segments[i].t/60,results.Segments[i].eta,'bo-')
    #plt.xlabel('Time (mins)'); plt.ylabel('Throttle'); plt.title(title)
    #plt.grid(True)

    #title = "Angle of Attack History"
    #plt.figure(2)
    #for i in range(len(results.Segments)):
    #    plt.plot(results.Segments[i].t/60,np.degrees(results.Segments[i].alpha),'bo-')
    #plt.xlabel('Time (mins)'); plt.ylabel('Angle of Attack (deg)'); plt.title(title)
    #plt.grid(True)

    #title = "Fuel Burn"
    #plt.figure(3)
    #for i in range(len(results.Segments)):
    #    plt.plot(results.Segments[i].t/60,flight.m0 - results.Segments[i].m,'bo-')
    #plt.xlabel('Time (mins)'); plt.ylabel('Fuel Burn (kg)'); plt.title(title)
    #plt.grid(True)

    #title = "Fuel Burn Rate"
    #plt.figure(4)
    #for i in range(len(results.Segments)):
    #    plt.plot(results.Segments[i].t/60,results.Segments[i].mdot,'bo-')
    #plt.xlabel('Time (mins)'); plt.ylabel('Fuel Burn Rate (kg/s)'); plt.title(title)
    #plt.grid(True)

    #plt.show()
    
    return

# call main
if __name__ == '__main__':
    main()


""" Results.py: Methods post-processing results """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

def compute_energies(results,summary=False):

    # evaluate each segment 
    for i in range(len(results.Segments)):

        segment = results.Segments[i]
        segment.P_fuel, segment.P_e = segment.config.Propulsors.power_flow(segment.eta,segment)

        # time integration operator
        I = segment.numerics.I*segment.dt

        # raw propellant energy consumed 
        segment.energy.propellant = np.dot(I,segment.P_fuel)[-1]

        # raw elecrical energy consumed  
        segment.energy.electric = np.dot(I,segment.P_e)[-1]

        # energy to gravity 
        segment.energy.gravity = np.dot(I,segment.m*segment.g*segment.vectors.V[:,2])[-1]   # J

        # energy to drag
        segment.energy.drag = np.dot(I,segment.D*segment.V)[-1]                             # J

        if summary:
            print " "
            print "####### Energy Summary: Segment " + str(i) + " #######"
            print " "
            print "Propellant energy used = " + str(segment.energy.propellant/1e6) + " MJ"
            print "Electrical energy used = " + str(segment.energy.electric/1e6) + " MJ"
            print "Energy lost to gravity = " + str(segment.energy.gravity/1e6) + " MJ"
            print "Energy lost to drag    = " + str(segment.energy.drag/1e6) + " MJ"
            print " "
            print "#########################################"
            print " "

    return        

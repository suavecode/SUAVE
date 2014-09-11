""" Results.py: Methods post-processing results """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
from SUAVE.Structure                    import Data, Data_Exception
# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

def compute_energies(results,summary=False):

    # evaluate each segment 
    for i in range(len(results.Segments)):

        segment = results.Segments[i]
        eta=segment.conditions.propulsion.throttle[:,0]
        state = Data()
        state.q  = segment.conditions.freestream.dynamic_pressure[:,0]
        state.g0 = segment.conditions.freestream.gravity[:,0]
        state.V  = segment.conditions.freestream.velocity[:,0]
        state.M  = segment.conditions.freestream.mach_number[:,0]
        state.T  = segment.conditions.freestream.temperature[:,0]
        state.p  = segment.conditions.freestream.pressure[:,0]
        
        
        segment.P_fuel, segment.P_e = segment.config.Propulsors.power_flow(eta,state)
        
        # time integration operator
        '''
        print segment.numerics
        I = segment.numerics.integrate_time

        # raw propellant energy consumed 
        segment.energy.propellant = np.dot(I,segment.P_fuel)[-1]

        # raw electrical energy consumed  
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
        '''
    return        

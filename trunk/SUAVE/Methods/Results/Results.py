""" Results.py: Methods post-processing results """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
import Units

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

def compute_efficiencies(results,summary=False):

    # evaluate each segment 
    for i in range(len(results.Segments)):

        segment = results.Segments[i]
        segment.P_fuel, segment.P_e = segment.config.Propulsors.power_flow(segment.eta,segment)

        # propulsive efficiency (mass flow)
        mask = (segment.P_fuel > 0.0)
        segment.efficiency.propellant = np.zeros_like(segment.F)
        segment.efficiency.propellant[mask] = segment.F[mask]*segment.V[mask]/segment.P_fuel[mask]

        # propulsive efficiency (electric) 
        mask = (segment.P_e > 0.0)
        segment.efficiency.electric = np.zeros_like(segment.F)
        segment.efficiency.electric[mask] = segment.F[mask]*segment.V[mask]/segment.P_e[mask]

    return

def compute_velocity_increments(results,summary=False):

    # evaluate each segment 
    for i in range(len(results.Segments)):

        segment = results.Segments[i]
        segment.P_fuel, segment.P_e = segment.config.Propulsors.power_flow(segment.eta,segment)

        # time integration operator
        I = segment.integrate

        # DV to gravity 
        segment.DV.gravity = np.dot(I,segment.g*segment.vectors.V[:,2])         # m/s

        # DV to drag
        segment.DV.drag = np.dot(I,segment.D*segment.V/segment.m)               # m/s

        # DV to drag
        FV = np.sum(segment.vectors.F*segment.vectors.V,axis=1)
        segment.DV.thrust = np.dot(I,FV/segment.m)                              # m/s

    return

def compute_alpha(results):

    for i in range(len(results.Segments)):
        results.Segments[i].alpha = results.Segments[i].gamma - results.Segments[i].psi

    return
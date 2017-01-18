# propeller_map.py
# 
# Created:  Jan 2016, E. Botero
# Modified:

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import SUAVE
import numpy as np
from SUAVE.Core import Units, Data

from sklearn import gaussian_process

# ----------------------------------------------------------------------
#  Propeller Map
# ----------------------------------------------------------------------

def propeller_map(prop, points):
    
    #Unpack    
    R          = prop.tip_radius  
    altitudes  = points.altitudes
    velocities = points.velocities
    omega      = points.omega
    
    # Diameter
    D = R*2 
    
    # Number of points
    N = len(altitudes)*len(velocities)*len(omega)
    
    # Make a vector of all of the conditions
    vels = []
    alts = []
    oms  = []
    J    = []
    for ii in xrange(0,len(altitudes)):
        for jj in xrange(0,len(velocities)):
            for kk in xrange(0,len(omega)):
                alts.append(altitudes[ii])
                vels.append(velocities[jj])
                oms.append(omega[kk])
                J.append(velocities[jj]/(omega[kk]*D)) # Advance Ratio
    
    
    Aerodynamic = SUAVE.Analyses.Mission.Segments.Aerodynamic()
    ones_row  = Aerodynamic.state.ones_row
    Aerodynamic.state.numerics.number_control_points = N
    Aerodynamic.state.conditions.propulsion.throttle = 1. * ones_row(1)
    Aerodynamic.process.initialize.expand_state(Aerodynamic,Aerodynamic.state)
    Aerodynamic.state.conditions.frames.inertial.velocity_vector[:,0] = vels
    Aerodynamic.state.conditions.freestream.altitude = alts
    
    Aerodynamic.analyses.atmosphere = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    Aerodynamic.process.iterate.conditions.atmosphere(Aerodynamic,Aerodynamic.state)
    Aerodynamic.process.iterate.conditions.orientations(Aerodynamic,Aerodynamic.state)
    
    prop.inputs.omega = np.array(oms).reshape([N,1])
            
    thrust, torque, power, Cp = prop.spin(Aerodynamic.state.conditions)
    
    cond = Aerodynamic.state.conditions
    eta = cond.propulsion.etap
    
    Cpflat = Cp.flatten()
    
    xyz = np.vstack([alts,J,Cpflat]).T

    gp = gaussian_process.GaussianProcess()
    
    surr = gp.fit(xyz,eta)
    
    return surr
    
    
    
if __name__ == '__main__': 

    from SUAVE.Methods.Propulsion import propeller_design
    
    # Design the Propeller
    prop                     = SUAVE.Components.Energy.Converters.Propeller()
    prop.number_blades       = 2.0 
    prop.freestream_velocity = 50.0
    prop.angular_velocity    = 2000.*(2.*np.pi/60.0)
    prop.tip_radius          = 1.5
    prop.hub_radius          = 0.05
    prop.design_Cl           = 0.7 
    prop.design_altitude     = 0.0 * Units.km
    prop.design_thrust       = 0.0
    prop.design_power        = 7000.
    prop                     = propeller_design(prop)  

    
    points = Data()
    points.altitudes  = np.array([0, 1000, 2000])
    points.velocities = np.array([1, 50.])
    points.omega      = np.array([2000. * Units.rpm])
    
    surrogate = propeller_map(prop, points)
    
    print surrogate.predict([0,0.1,.1])
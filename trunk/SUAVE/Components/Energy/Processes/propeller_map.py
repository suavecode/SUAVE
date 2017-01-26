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
from sklearn import neighbors
from sklearn import svm
import pylab as plt

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
    
    # Advance ratios
    n = np.array(omega)/(2*np.pi)
    Js = np.array([velocities])/(np.transpose([n])*D)
    
    Js, indices = np.unique(Js,return_index=True)
    vel = np.tile(velocities,len(omega))
    vel = vel[indices]
    
    # Number of points
    N = len(Js)*len(altitudes)
    
    # Make a vector of all of the conditions
    alts = []
    J    = []
    vels = []
    for ii in xrange(0,len(altitudes)):
        for jj in xrange(0,len(Js)):
            alts.append(altitudes[ii])
            vels.append(vel[jj])
            J.append(Js[jj])

    Aerodynamic = SUAVE.Analyses.Mission.Segments.Aerodynamic()
    ones_row  = Aerodynamic.state.ones_row
    Aerodynamic.state.numerics.number_control_points = N
    Aerodynamic.state.conditions.propulsion.throttle = 1. * ones_row(1)
    Aerodynamic.process.initialize.expand_state(Aerodynamic,Aerodynamic.state)
    Aerodynamic.state.conditions.frames.inertial.velocity_vector[:,0] = vels
    Aerodynamic.state.conditions.freestream.altitude[:,0] = np.array(alts)
    
    Aerodynamic.analyses.atmosphere = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    Aerodynamic.process.iterate.conditions.atmosphere(Aerodynamic,Aerodynamic.state)
    Aerodynamic.process.iterate.conditions.orientations(Aerodynamic,Aerodynamic.state)
    
    prop.inputs.omega = np.array([np.array(vels)/(np.array(J)*D)]).T * 2.*np.pi
            
    thrust, torque, power, Cp = prop.spin(Aerodynamic.state.conditions)
    
    cond = Aerodynamic.state.conditions
    eta = cond.propulsion.etap
    
    Cpflat = Cp.flatten()
    
    xyz = np.vstack([J,np.array(alts)]).T
    
    #gp_eta = gaussian_process.GaussianProcess()
    #gp_cp = gaussian_process.GaussianProcess()
    
    #gp_eta = svm.SVR(C=500.)
    #gp_cp  = svm.SVR(C=500.)
    
    gp_eta = neighbors.KNeighborsRegressor(n_neighbors=1,weights='distance')
    gp_cp  = neighbors.KNeighborsRegressor(n_neighbors=1,weights='distance')
    
    surr_eta = gp_eta.fit(xyz,eta)
    surr_cp  = gp_cp.fit(xyz,Cp)
    
    prop.surrogate.efficiency        = surr_eta
    prop.surrogate.power_coefficient = surr_cp
    
    # Test the surrogate
    eta_sur = np.zeros(np.shape(xyz))
    
    Jnew  = np.linspace(1.0, 3.5, 100)
    altnew = np.linspace(14000, 16000, 100)
    
    Jmesh, altmesh = np.meshgrid(Jnew,altnew)
    
    etas = np.zeros(np.shape(Jmesh))
    cps  = np.zeros(np.shape(Jmesh))
    
    for ii in xrange(len(Jnew)):
        for jj in xrange(len(altnew)):
            etas[ii,jj] = surr_eta.predict(np.array([Jmesh[ii,jj],altmesh[ii,jj]]))
            cps[ii,jj] = surr_cp.predict(np.array([Jmesh[ii,jj],altmesh[ii,jj]]))
            
    fig = plt.figure("Eta")
    plt_handle = plt.contourf(Jmesh,altmesh,etas,levels=None)
    cbar = plt.colorbar()
    plt.scatter(xyz[:,0],xyz[:,1])
    plt.xlabel('J')
    plt.xlabel('Altitude')
    
    #fig = plt.figure("Eta")
    #plt_handle = plt.contourf(Jmesh,altmesh,cps,levels=None)
    #cbar = plt.colorbar()
    #plt.scatter(xyz[:,0],xyz[:,1])
    #plt.xlabel('J')
    #plt.xlabel('Altitude')
    
    plt.show()
    
    return prop
    
    
    
#if __name__ == '__main__': 

    #from SUAVE.Methods.Propulsion import propeller_design
    
    ## Design the Propeller
    #prop                     = SUAVE.Components.Energy.Converters.Propeller()
    #prop.number_blades       = 2.0 
    #prop.freestream_velocity = 50.0
    #prop.angular_velocity    = 2000.*(2.*np.pi/60.0)
    #prop.tip_radius          = 1.5
    #prop.hub_radius          = 0.05
    #prop.design_Cl           = 0.7 
    #prop.design_altitude     = 0.0 * Units.km
    #prop.design_thrust       = 0.0
    #prop.design_power        = 7000.
    #prop                     = propeller_design(prop)  

    
    #points = Data()
    #points.altitudes  = np.array([0, 1000, 2000])
    #points.velocities = np.array([1, 50.])
    #points.omega      = np.array([2000. * Units.rpm])
    
    #surrogate = propeller_map(prop, points)
    
    #print surrogate.predict([0,0.1,.1])
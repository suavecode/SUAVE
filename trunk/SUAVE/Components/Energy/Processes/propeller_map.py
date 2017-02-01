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
    
    # Run the propeller
    thrust, torque, power, J, Cp, eta, alts  = run_propeller(prop, points)
    
    # Build a surrogate
    prop = build_surrogate(prop, J, Cp, eta, alts)
    
    # Test Surrogate
    test_surrogate(prop,thrust,torque,power,J,Cp,eta,alts)
    
    
    return prop

def run_propeller(prop,points):
    
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
    
    return thrust, torque, power, J, Cp, eta, alts 


def test_surrogate(prop,thrust, torque, power, J, Cp, eta, alts ):
    
    surr_eta = prop.surrogate.efficiency 
    surr_cp  = prop.surrogate.power_coefficient
    
    
    Jnew  = np.linspace(1.0, 2.4, 40)
    altnew = np.linspace(14000, 16000, 40)
    
    Jmesh, altmesh = np.meshgrid(Jnew,altnew)
    
    etas = np.zeros(np.shape(Jmesh))
    cps  = np.zeros(np.shape(Jmesh))
    
    eta_prop = np.zeros(np.shape(Jmesh))
    cp_prop  = np.zeros(np.shape(Jmesh))  
    
    vel = 35.
    
    points = Data()
    
    R = prop.tip_radius 
    D = R*2 
    
    
    for ii in xrange(len(Jnew)):
        for jj in xrange(len(altnew)):
            jjj  = Jmesh[ii,jj]
            altj = altmesh[ii,jj]
            
            n = vel/(jjj*D)
            omega = n*2*np.pi
            
            points.altitudes  = [altj]
            points.velocities = [vel] 
            points.omega      = [omega]
            
            etas[ii,jj] = surr_eta.predict(np.array([jjj,altj]))
            cps[ii,jj] = surr_cp.predict(np.array([jjj,altj]))
            
            thrust, torque, power, J, Cp, eta, alts  = run_propeller(prop, points)
            
            eta_prop[ii,jj] = eta
            cp_prop[ii,jj]  = Cp
            
    fig = plt.figure("Eta")
    plt_handle = plt.contourf(Jmesh,altmesh,cps,levels=None)
    cbar = plt.colorbar()
    #plt.scatter(xyz[:,0],xyz[:,1])
    plt.xlabel('J')
    plt.ylabel('Altitude')
    
    fig = plt.figure("Eta_Real")
    plt_handle = plt.contourf(Jmesh,altmesh,cp_prop,levels=None)
    cbar = plt.colorbar()
    #plt.scatter(xyz[:,0],xyz[:,1])
    plt.xlabel('J')
    plt.ylabel('Altitude')
    
    plt.show()    
    
    return 
    

def build_surrogate(prop, J, Cp, eta, alts):
    
    xyz = np.vstack([J,np.array(alts)]).T
    
    #surr_eta = gaussian_process.GaussianProcess()
    #surr_cp = gaussian_process.GaussianProcess()
    
    #surr_eta = svm.SVR(C=500.)
    #surr_cp  = svm.SVR(C=500.)
    
    surr_eta = neighbors.KNeighborsRegressor(n_neighbors=1,weights='distance')
    surr_cp  = neighbors.KNeighborsRegressor(n_neighbors=1,weights='distance')

    prop.surrogate.efficiency        = surr_eta.fit(xyz,eta)
    prop.surrogate.power_coefficient = surr_cp.fit(xyz,Cp)
  
    return prop

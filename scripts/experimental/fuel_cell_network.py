# fuel_cell_network.py
# 
# Created:  Tim MacDonald, Jul 2014
# Modified:  
# Adapted from solar_enery_network_5.py

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
import SUAVE

# package imports
import numpy as np
import scipy as sp
from scipy import optimize
from SUAVE.Attributes import Units
import copy

from SUAVE.Structure import (
Data, Container, Data_Exception, Data_Warning,
)

# ----------------------------------------------------------------------
#   Alternate Approach 
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
#  Energy Component Class
# ----------------------------------------------------------------------
from SUAVE.Components import Physical_Component

class Energy_Component(Physical_Component):
    def __defaults__(self):
        
        # function handles for input
        self.inputs  = Data()
        
        # function handles for output
        self.outputs = Data()
        
        return

class Fuel_Cell(Energy_Component):
    def __defaults__(self):
        self.efficiency = 0.8
        self.inputs.propellant = SUAVE.Attributes.Propellants.Jet_A1()
        self.max_mdot = 2.0
        
    def power(self,conditions):
        spec_energy = self.inputs.propellant.specific_energy
        eta = copy.copy(conditions.propulsion.throttle)
        if np.any(abs(eta)>100):
            print 'Warning: Fuel Cell throttle values outside +-100 have not been tested'
        self.outputs.power = spec_energy*eta*self.max_mdot*self.efficiency
        if abs(eta[-1]-1.0) > 0.01:
            a = 0
        if abs(self.outputs.power[-1]) < 5*10**7:
            a = 0                 
        self.outputs.mdot = eta*self.max_mdot

class Motor(Energy_Component):
    def __defaults__(self):
        self.efficiency = 0.95
        
    def power(self,conditions): # to be designed to take eta
        self.outputs.power = self.inputs.powerin*self.efficiency
        
class Propulsor(Energy_Component):
    def __defaults__(self):
        self.A0 = (.75)**2*np.pi
                 
        
    def evaluate(self,conditions):
        def fM(M):
            fm = ((gamma+1)/2)**((gamma+1)/2/(gamma-1))*(M/(1+(gamma-1)/2*M**2)**((gamma+1)/2/(gamma-1)))
            return fm
        
        def PtP(M):
            Pt = (1 + (gamma-1)/2*M**2)**(gamma/(gamma-1))
            return Pt
        
        def TtT(M):
            Tt = (1 + (gamma-1)/2*M**2)
            return Tt         
        M0 = conditions.freestream.mach_number
        a0 = conditions.freestream.speed_of_sound
        U0 = M0*a0
        P0 = conditions.freestream.pressure
        T0 = conditions.freestream.temperature
        rho0 = conditions.freestream.density
        mdot = U0*rho0*self.A0
        power = self.inputs.power
        # More code here
        gamma = 1.4
        R = 286.9
        Cp = 1005.0
        Pt0 = P0*PtP(M0)
        Tt0 = T0*TtT(M0)
        a0 = np.sqrt(gamma*R*T0)
        U0 = a0*M0
        mdot = gamma/((gamma+1)/2)**((gamma+1)/2/(gamma-1))*(Pt0*self.A0/np.sqrt(gamma*R*Tt0))*fM(M0)
        
        M2 = 0.2                                # --------- These values don't matter for a simple model
        A2 = self.A0*fM(M0)/fM(M2)
        Tt2 = Tt0
        Pt2 = Pt0
        
        P = power # this is power, not pressure
        p = P/mdot
        dTt = p/Cp
        neg_flag = np.zeros_like(dTt)
        neg_flag[dTt < 0] = 1
        dTt[dTt < 0.0] = -dTt[dTt < 0.0]
        Tt3 = Tt2 + dTt
        Pt3 = Pt2*(Tt3/Tt2)**(gamma/(gamma-1))
        M3 = 0.5                                # ---------
        A3 = A2*fM(M2)/fM(M3)*Pt2/Pt3*np.sqrt(Tt3/Tt2)
        
        Pe = P0
        Pte = Pt3
        Tte = Tt3
        
        #print Tt3/Tt2
    
        Me = np.sqrt(2.0*Pte*(Pte/Pe)**(-1/gamma)-2.0*Pe)/np.sqrt((gamma-1)*Pe)
        Ae = A3*fM(M3)/fM(Me)
        Te = Tte/TtT(Me)
        ae = np.sqrt(gamma*R*Te)
        Ue = Me*ae
        e = 2*U0/(U0+Ue)
        
        F = gamma*M0**2*(Me/M0*np.sqrt(Te/T0)-1)*P0*self.A0
        F[neg_flag == 1] = -F[neg_flag == 1]
        F = F[:,0]
        mdot = mdot[:,0]
        P = np.zeros_like(F)        
        F[np.isnan(F)] = conditions.propulsion.throttle*-1.0*power/2e2
        
        return F, mdot, P, e
              
    
        

# the network
class Network(Data):
    def __defaults__(self):
        self.propellant  = None
        self.fuel_cell   = None
        self.motor       = None
        self.propulsor   = None
        self.payload     = None
        self.nacelle_dia = 0.0
        self.tag         = 'Network'
    
    # manage process with a driver function
    def evaluate(self,eta,conditions):
    
        # unpack
        propellant  = self.propellant
        fuel_cell   = self.fuel_cell
        motor       = self.motor
        propulsor   = self.propulsor
        payload     = self.payload
        
        # step 1

        # step 2
        #if abs(fuel_cell.outputs.power[-1]) < 5*10**7:
            #a = 0        
        fuel_cell.power(conditions)
        if abs(fuel_cell.outputs.power[-1]) < 5*10**7:
            a = 0        
        fuel_cell.power_generated = copy.copy(fuel_cell.outputs.power)
            
        # link
        motor.inputs.powerin = fuel_cell.outputs.power
        mdot = fuel_cell.outputs.mdot
        # step 3
        motor.power(conditions)
        # link
        propulsor.inputs.power =  motor.outputs.power
        #print(motor.outputs.omega)
        # step 6
        F, mdotP, P, e = propulsor.evaluate(conditions)
        # Package for solver
        mdot = mdot[:,0]
        self.propulsive_efficiency = e
        
        return F, mdot, P
            
    __call__ = evaluate


if __name__ == '__main__': 
    
    import pylab as plt
    
    conditions = Data()
    conditions.freestream = Data()
    conditions.propulsion = Data()
    conditions.freestream.mach_number = np.array([[ 0.64      ],
       [ 0.64393343],
       [ 0.65556182],
       [ 0.67437694],
       [ 0.69955649],
       [ 0.73      ],
       [ 0.76437694],
       [ 0.80118488],
       [ 0.83881512],
       [ 0.87562306],
       [ 0.91      ],
       [ 0.94044351],
       [ 0.96562306],
       [ 0.98443818],
       [ 0.99606657],
       [ 1.        ]])
    conditions.freestream.speed_of_sound = np.array([[ 322.29006967],
       [ 322.06007003],
       [ 321.37918466],
       [ 320.27449942],
       [ 318.79030104],
       [ 316.98676835],
       [ 314.93809751],
       [ 312.73003297],
       [ 310.45678941],
       [ 308.21738035],
       [ 306.11141944],
       [ 304.23452929],
       [ 302.67357371],
       [ 301.50200932],
       [ 300.77571113],
       [ 300.529644  ]])
    conditions.freestream.pressure = np.array([[ 57221.83377333],
       [ 56794.0307044 ],
       [ 55544.48314305],
       [ 53570.02114594],
       [ 51017.22743163],
       [ 48063.59369909],
       [ 44896.84385616],
       [ 41696.15276154],
       [ 38618.03968527],
       [ 35788.28305363],
       [ 33299.76086698],
       [ 31215.04096889],
       [ 29571.99707767],
       [ 28390.67796417],
       [ 27679.94588536],
       [ 27442.82907028]])
    conditions.freestream.temperature = np.array([[ 258.4662925 ],
       [ 258.09751949],
       [ 257.00735603],
       [ 255.24355615],
       [ 252.88336642],
       [ 250.03012222],
       [ 246.80869967],
       [ 243.36002755],
       [ 239.83490438],
       [ 236.38739481],
       [ 233.16809633],
       [ 230.31757046],
       [ 227.9602233 ],
       [ 226.19889871],
       [ 225.11041527],
       [ 224.74223652]])
    conditions.freestream.density = np.array([[ 0.77125139],
       [ 0.76657908],
       [ 0.75289338],
       [ 0.73114772],
       [ 0.70280475],
       [ 0.66967179],
       [ 0.63371421],
       [ 0.59687699],
       [ 0.56093939],
       [ 0.52741764],
       [ 0.49751948],
       [ 0.4721445 ],
       [ 0.45191804],
       [ 0.43724352],
       [ 0.42835885],
       [ 0.42538511]])
    eta = np.array([[0.0]]*16)
    eta[:,0] = np.linspace(-100.0,105.0,16)
    conditions.propulsion.throttle = eta
    
    net = Network()
    
    net.propellant = SUAVE.Attributes.Propellants.Jet_A1()
    net.fuel_cell = Fuel_Cell()
    net.motor = Motor()
    net.propulsor = Propulsor()   
    F,mdot,P = net.evaluate(eta,conditions)
    #print F
    #print mdot
    #print P
    
    fig = plt.figure("Throttle and Fuel Burn")
    tot_energy = 0.0
    Thrust = F
        
    axes = plt.gca()  
    axes.plot( eta , Thrust , 'bo-' )
    axes.set_xlabel('Throttle')
    axes.set_ylabel('Thrust')
    axes.grid(True)
    plt.show()
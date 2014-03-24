""" ClimbDescentConstantMach.py: climb or descent segment at constant Mach """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
# from SUAVE.Plugins.ADiPy import *
from scipy import interpolate
from SUAVE.Attributes.Missions.Segments import Segment
from SUAVE.Methods.Utilities.Chebyshev import cosine_space, chebyshev_data
from SUAVE.Methods.Constraints import horizontal_force, vertical_force

# ----------------------------------------------------------------------
#  Class
# ----------------------------------------------------------------------

class Constant_Mach(Segment):

    """ Segment: constant Mach climb or descent """

    def __defaults__(self):
        self.tag = 'Segment: constant Mach climb or descent'

    def initialize(self):

        err = False; guess = []

        # check altitude
        try: 
            self.altitude
        except AttributeError:
            print "Error in cruise segment: no altitude defined."
            return True
        else:
            if np.shape(self.altitude):
                if len(np.shape(self.altitude)) == 2:
                    print "Error in climb / descent segment: altitude must be a vector of length two."
                    return True
                elif len(np.shape(self.altitude)) == 1:
                    if len(self.altitude) != 2:
                        print "Error in climb / descent segment: altitude must be a vector of length two."
                        return True
   
        # atmospheric conditions
        alt = cosine_space(self.options.N,self.altitude[0],self.altitude[1])        # km
        self.compute_atmosphere(alt)
                              
        # check Minf
        try: 
            self.Minf
        except AttributeError:
            have_M = False
            print "Error in climb / descent segment: Mach must be defined."
            return True
        else:
            have_M = True
            if not np.isscalar(self.Minf):
                print "Error in climb / descent segment: Mach must be a scalar."
                return True
        
        # check climb angle
        try:
            self.psi
        except AttributeError:
            have_angle = False
        else:
            have_angle = True
            if not np.isscalar(self.psi):    
                print "Error in climb / descent segment: climb angle must be a scalar."
                return True
            else:
                self.psi = np.radians(self.psi)*np.ones(self.options.N)

        # check climb / descent rate
        try: 
            self.rate
        except AttributeError:
            have_rate = False
        else:
            have_rate = True
            if not np.isscalar(self.rate):    
                print "Error in climb / descent segment: climb / descent rate must be a scalar."
                return True             

        # create "raw" Chebyshev data (0 ---> 1)  
        x, D, I = chebyshev_data(self.options.N,True)

        self.dz = (self.altitude[1] - self.altitude[0])*1e3                                  # m
        V2 = (self.a*self.Minf)**2                                                      # m^2/s^2

        # velocity components and final time
        if have_angle and have_rate:
            print "Error in climb / descent segment: Mach, climb / descent angle, and climb / descent rate are all defined. Please disambiguate."
            return True
        elif have_angle and not have_rate:
            Vx = np.sqrt(V2/(np.tan(self.psi)**2 + 1))                                # m/s
            Vz = np.sqrt(V2 - Vx**2)                                                    # m/s
            self.psi *= np.sign(self.dz)                                    # rad
            self.dt = np.dot(I*self.dz,1/Vz)[-1]                               # s 
        elif not have_angle and have_rate:
            Vz = self.rate*np.sign(self.dz)                                                  # m/s
            Vx = np.sqrt(V2 - Vz**2)                                                    # m/s
            self.psi = np.arctan(Vz/Vx)                                                      # rad
            self.dt = np.abs(self.dz)/self.rate                                              # s
        else:
            print "Error in climb / descent segment: insufficient input data defined."
            print "Any two of: velocity, climb / descent angle, and climb / descent rate must be defined."
            return True

        # allocate vectors
        self.initialize_vectors()

        # gravity
        self.compute_gravity(alt,self.planet)

        # freestream conditions
        self.vectors.V[:,0] = Vx                                              # m/s
        self.vectors.V[:,2] = Vz                                              # m/s
        self.compute_freestream()

        ## set up constraints 
        #dVdt = self.Minf*Vz*np.dot(D/self.dz,self.a)                               # m/s^2
        #self.RFx = HorizontalForce();    self.RFx.F_mg = dVdt*np.cos(self.psi)/self.g
        #self.RFz = VerticalForce();      self.RFz.F_mg = dVdt*np.sin(self.psi)/self.g

        #print Vx, Vz
        #print self.tf
        #print psi
        #print dVdt
        #print RFx.F_mg, RFz.F_mg
        #raw_input()

        if not err:

            # create guess
            gamma = np.ones(self.options.N)*self.psi
            eta = np.ones(self.options.N)*0.50
            self.guess = np.append(gamma,eta)

        return err

    def unpack(self,x):

        # unpack state data
        x_state = []; tf = []
        x_control = np.reshape(x,(self.options.N,2),order="F")
    
        return x_state, x_control, tf

    def dynamics(self,x_state,x_control,D,I):
        pass

    def constraints(self,x_state,x_control,D,I):

        # initialize needed arrays
        N = self.options.N
        if self.complex:
            self.vectors.D = np.zeros((N,3)) + 0j       # drag vector (N)
            self.vectors.L = np.zeros((N,3)) + 0j       # lift vector (N)
            self.vectors.u = np.zeros((N,3)) + 0j       # thrust unit tangent vector
            self.vectors.F = np.zeros((N,3)) + 0j       # thrust vector (N)
            self.vectors.Ftot = np.zeros((N,3)) + 0j    # total force vector (N)

        else:    
            self.vectors.D = np.zeros((N,3))            # drag vector (N)
            self.vectors.L = np.zeros((N,3))            # lift vector (N)
            self.vectors.u = np.zeros((N,3))            # thrust unit tangent vector
            self.vectors.F = np.zeros((N,3))            # thrust vector (N)
            self.vectors.Ftot = np.zeros((N,3))         # total force vector (N)

        # unpack control data
        gamma = x_control[:,0]; eta = x_control[:,1]

        # set up thrust vector
        self.vectors.u[:,0] = np.cos(gamma) 
        self.vectors.u[:,2] = np.sin(gamma)

        # aero data 
        self.compute_aero(gamma - self.psi)

        # compute aero vectors
        self.compute_aero_vectors_2D()

        # propulsion data & thrust vectos
        self.compute_propulsion(eta)
        self.compute_thrust_vectors_2D()

        # mass
        # self.compute_mass(self.m0,I)
        self.m = self.m0 - np.dot(self.numerics.I*self.dz, \
            self.mdot/self.vectors.V[:,2])
        # print self.m
        # raw_input()
        # total up forces
        self.compute_forces()

        # evaluate constraints 
        if self.complex:
            R = np.zeros_like(x_control) + 0j
        else:
            R = np.zeros_like(x_control)

        Dz = self.numerics.D/self.dz
        dVdt = self.Minf*self.vectors.V[:,2]*np.dot(Dz,self.a)            # m/s^2
        R[:,0] = horizontal_force(self,dVdt*np.cos(self.psi)/self.g) 
        R[:,1] = vertical_force(self,dVdt*np.sin(self.psi)/self.g)  

        # evaluate constraints 
        #if not self.complex:
        #    print gamma
        #    print eta
        #    print R
        #    print self.vectors.Ftot
        #    print self.vectors.L
        #    raw_input()
        
        return R

    def solution(self,x):
        
        # unpack vector
        x_state, x_control, dt = self.unpack(x)

        # unpack control data
        gamma = x_control[:,0]; eta = x_control[:,1]

        # operators
        self.t = self.t0 + self.numerics.t*self.dt
        self.differentiate = self.numerics.D/self.dt 
        self.integrate = self.numerics.I*self.dt

        # resample control data to Chebyshev time points
        Iz = self.numerics.I*self.dz
        t = np.dot(Iz,1/self.vectors.V[:,2])
        gamma_of_t = interpolate.InterpolatedUnivariateSpline(t,gamma)
        self.gamma = gamma_of_t(self.t)
        eta_of_t = interpolate.InterpolatedUnivariateSpline(t,eta)
        self.eta = eta_of_t(self.t)
        del Iz, t, gamma_of_t, eta_of_t

        # initialize arrays    
        self.vectors.D = np.zeros((self.options.N,3))            # drag vector (N)
        self.vectors.L = np.zeros((self.options.N,3))            # lift vector (N)
        self.vectors.u = np.zeros((self.options.N,3))            # thrust unit tangent vector
        self.vectors.F = np.zeros((self.options.N,3))            # thrust vector (N)
        self.vectors.Ftot = np.zeros((self.options.N,3))         # total force vector (N)

        # thrust vector
        self.vectors.u[:,0] = np.cos(gamma) 
        self.vectors.u[:,2] = np.sin(gamma)

        # aero data 
        self.compute_aero(self.gamma - self.psi)

        # compute aero vectors
        self.compute_aero_vectors_2D()

        # propulsion data & thrust vectos
        self.compute_propulsion(self.eta)
        self.compute_thrust_vectors_2D()

        # mass
        self.compute_mass(self.m0,self.integrate)

        # total up forces
        self.compute_forces()

        # position
        self.vectors.r[:,0] = np.dot(self.integrate,self.vectors.V[:,0])
        self.vectors.r[:,2] = np.dot(self.integrate,self.vectors.V[:,2])+self.altitude[0]*1e3

        # accelerations
        self.vectors.a[:,0] = np.dot(self.differentiate,self.vectors.V[:,0]) 
        self.vectors.a[:,2] = np.dot(self.differentiate,self.vectors.V[:,2])

        return 

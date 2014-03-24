""" ClimbDescentConstantSpeed.py: climb or descent segment at constant speed """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
from SUAVE.Attributes.Missions.Segments import Segment
from SUAVE.Methods.Utilities.Chebyshev import cosine_space
from SUAVE.Methods.Constraints import horizontal_force, vertical_force

# ----------------------------------------------------------------------
#  Class
# ----------------------------------------------------------------------

class Constant_Speed(Segment):

    """ Segment: constant speed climb or descent """

    def __defaults__(self):
        self.tag = 'Segment: constant speed climb or descent'

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
                              
        # check Vinf
        try: 
            self.Vinf
        except AttributeError:
            have_V = False
        else:
            have_V = True
            if not np.isscalar(self.Vinf):
                print "Error in climb / descent segment: velocity must be a scalar."
                return True
            else:
                self.Minf = self.Vinf/self.a

        # check climb angle
        try:
            self.psi
        except AttributeError:
            have_angle = False
        else:
            have_angle = True
            self.psi = np.radians(self.psi)
            if not np.isscalar(self.psi):    
                print "Error in climb / descent segment: climb angle must be a scalar."
                return True

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

        # input logic
        if have_V and have_angle and have_rate:
            raise RuntimeError , \
                """ Error in cruise segment: '%s'.
                    velocity, climb / descent angle, and climb / descent rate all defined. Please disambiguate.
                """ % self.tag
        elif have_V and have_angle and not have_rate:
            self.rate = self.Vinf*np.sin(np.radians(self.angle))
        elif have_V and not have_angle and have_rate:
            self.psi = np.arcsin(self.rate/self.Vinf)
        elif not have_V and have_angle and have_rate:
            self.Vinf = self.rate/np.sin(np.radians(self.angle))
        else:
            raise RuntimeError , \
                """ Error in climb / descent segment: '%s'
                    Insufficient input data defined.
                    Any two of: velocity, climb / descent angle, and climb / descent rate must be defined. 
                """ % self.tag

        # derived quantities 
        dz = (self.altitude[1] - self.altitude[0])*1e3                                  # m
        self.dt = np.abs(dz)/self.rate
        Vz = dz/self.dt
        Vx = self.Vinf*np.cos(self.psi)

        # allocate vectors
        self.initialize_vectors()

        # gravity
        self.compute_gravity(alt,self.planet)

        # freestream conditions
        self.vectors.V[:,0] = Vx                                              # m/s
        self.vectors.V[:,2] = Vz                                              # m/s
        self.compute_freestream()

        # lift direction
        self.compute_lift_direction_2D()

        if not err:

            # create guess
            gamma = np.ones(self.options.N)*self.psi*np.sign(dz)
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

        # unpack state data
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
        self.compute_mass(self.m0,I)

        # total up forces
        self.compute_forces()

        # evaluate constraints 
        if self.complex:
            R = np.zeros_like(x_control) + 0j
        else:
            R = np.zeros_like(x_control)

        R[:,0] = horizontal_force(self) 
        R[:,1] = vertical_force(self)  

        return R

    def solution(self,x):
        
        # unpack vector
        x_state, x_control, tf = self.unpack(x)

        # operators        
        self.t = self.t0 + self.numerics.t*self.dt
        self.differentiate = self.numerics.D/self.dt 
        self.integrate = self.numerics.I*self.dt

        # call dynamics function to fill state data
        self.constraints(x_state,x_control,self.differentiate,self.integrate)

        # state and orientation data
        self.vectors.r[:,0] = np.dot(self.integrate,self.vectors.V[:,0])
        self.vectors.r[:,2] = np.dot(self.integrate,self.vectors.V[:,2])+self.altitude[0]*1e3

        # flight angle
        self.psi = self.psi*np.ones_like(self.t)

        # accelerations
        self.vectors.a = np.zeros_like(self.vectors.u)

        # controls
        self.gamma = x_control[:,0]
        self.eta = x_control[:,1]

        return 

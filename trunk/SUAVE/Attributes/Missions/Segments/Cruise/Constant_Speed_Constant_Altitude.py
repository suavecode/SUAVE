""" CruiseConstantSpeedConstantAltitude.py: constant speed, constant altitude cruise """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
from SUAVE.Attributes.Missions.Segments import Segment
from SUAVE.Methods.Constraints import horizontal_force, vertical_force

# ----------------------------------------------------------------------
#  Class
# ----------------------------------------------------------------------

class Constant_Speed_Constant_Altitude(Segment):

    """ Segment: constant speed, constant altitude cruise """

    def __defaults__(self):
        self.tag = 'Segment: constant speed, constant altitude cruise'

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
                    print "Error in cruise segment: altitude must be a scalar."
                    return True
                elif len(np.shape(self.altitude)) == 1:
                    if len(self.altitude) > 1:
                        print "Error in cruise segment: altitude must be a scalar."
                        return True
                
        # check Vinf / Minf
        try: 
            self.Vinf
        except AttributeError:
            have_V = False
        else:
            have_V = True
            if np.shape(self.Vinf):
                if len(np.shape(self.Vinf)) == 2:
                    print "Error in cruise segment: velocity must be a scalar."
                    return True
                elif len(np.shape(self.Vinf)) == 1:
                    if len(self.Vinf) > 1:
                        print "Error in cruise segment: velocity must be a scalar."
                        return True

        try: 
            self.Minf
        except AttributeError:
            have_M = False
        else:
            have_M = True
            if np.shape(self.Minf):
                if len(np.shape(self.Minf)) == 2:
                    print "Error in cruise segment: velocity must be a scalar."
                    return True
                elif len(np.shape(self.Minf)) == 1:
                    if len(self.Minf) > 1:
                        print "Error in cruise segment: velocity must be a scalar."
                        return True

        # allocate vectors
        self.initialize_vectors()

        # gravity
        self.compute_gravity(self.altitude,self.planet)

        # freestream conditions
        self.compute_atmosphere(self.altitude)

        if have_V and have_M:
            print "Error in cruise segment: velocity and Mach both defined. Please disambiguate."
            return True
        elif have_V and not have_M:
            self.Minf = self.Vinf/self.a
        elif not have_V and have_M:
            self.Vinf = self.Minf*self.a
        elif not have_V and not have_M:
            print "Error in cruise segment: no velocity or Mach defined."
            return True

        # final time
        self.dt = (self.range*1e3)/self.Vinf      # s

        # freestream conditions
        self.vectors.V[:,0] = self.Vinf
        self.compute_freestream()

        # lift direction
        self.compute_lift_direction_2D()

        if not err:

            # create guess
            self.guess = np.append(np.zeros(self.options.N),np.ones(self.options.N)*0.50)

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

        # aero data (gamma = alpha in this case)
        self.compute_aero(gamma)

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
        self.vectors.r[:,2] = np.ones(self.options.N)*self.altitude*1e3

        # flight angle
        self.psi = np.zeros_like(self.t)

        # accelerations
        self.vectors.a = np.zeros_like(self.vectors.u)

        # controls
        self.gamma = x_control[:,0]
        self.eta = x_control[:,1]

        return 

class Constant_Mach_Constant_Altitude(Constant_Speed_Constant_Altitude):

    def __defaults__(self):
        self.tag = 'Segment: constant Mach, constant altitude cruise'
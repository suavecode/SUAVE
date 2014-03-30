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
        
        ## TODO: subclass this check out        
        
        # check number of inputs
        inputs = []
        for key in ['altitude','Vinf','Minf']:
            if self.has_key(key):
                inputs.append(key)
        if len(inputs) > 2: raise RuntimeError, 'too many inputs, must pick too: altitude, Vinf, Minf'
        if not 'altitude' in inputs: raise RuntimeError, 'did not define altitude'
        
        # allocate vectors
        self.initialize_vectors()
        
        # gravity
        self.compute_gravity(self.altitude,self.planet)

        # freestream atmoshperic conditions
        self.compute_atmosphere(self.altitude)
        
        # fill in undefinined freestream speed
        if 'Vinf' in inputs:
            self.Minf = self.Vinf/self.a
        elif 'Minf' in inputs:
            self.Vinf = self.Minf*self.a
        
        # final time
        self.dt = (self.range*1e3)/self.Vinf      # s

        # freestream conditions
        self.vectors.V[:,0] = self.Vinf
        self.compute_freestream()

        # lift direction
        self.compute_lift_direction_2D()

        # create guess - THIS DRIVES THE NUMBER OF UNKOWNS
        N = self.options.N
        self.guess = np.hstack([ np.zeros(N)     ,  # gamma - thrust angle
                                 np.ones(N)*0.50 ]) # eta   - throttle

        return 

    def unpack(self,x):
        """ 
        """
        
        # unpack state data
        x_state = []; tf = []
        x_control = np.reshape(x, (self.options.N,2), order="F")
    
        return x_state, x_control, tf

    def dynamics(self,x_state,x_control,D,I):
        pass

    def constraints(self,x_state,x_control,D,I):

        # unpack state data
        gamma = x_control[:,0]; 
        eta   = x_control[:,1]

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
        self.integrate     = self.numerics.I*self.dt

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
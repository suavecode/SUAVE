""" Glide.py: flight without thrust or control vector """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
from SUAVE.Analyses.Missions.Segments import Segment
from SUAVE.Methods.Flight_Dynamics import equations_of_motion

# ----------------------------------------------------------------------
#  Class
# ----------------------------------------------------------------------

class Glide(Segment):

    """ Segment: flight without thrust or control vector """

    def __defaults__(self):
        self.tag = 'Segment: flight without thrust or control vector'

    def initialize(self):

        err = False

        return err

    def unpack(self,x):

        # unpack state data
        tf = x[-1]; x = x[0:-1]
        x_state = np.reshape(x,(self.options.N-1,2*self.dofs),order="F")
        x_state = np.append([self.z0],x_state,axis=0)

        x_control = []
    
        return x_state, x_control, tf

    def dynamics(self,x_state,x_control,D,I):
        
        # unpack state data
        y = x_state[:,2]
    
        # set up thrust vector
        self.initialize_vectors(len(y))

        # gravity
        self.compute_gravity(x_state[:,2],self.planet)

        # freestream conditions
        self.compute_atmosphere(y)

        # freestream conditions
        self.compute_freestream(x_state[:,(1,3)])

        # aero data (no orientation)
        self.compute_aero(self.alpha)

        # compute aero vectors
        self.compute_aero_vectors_2D()

        # compute "propulsion" (free flight in this case)
        self.F = 0.0; self.mdot = 0.0

        # mass
        self.compute_mass(self.m0,I)

        # total up forces
        self.compute_forces()

        return equations_of_motion(self)

    def constraints(self,x_state,x_control,D,I):
        pass

    #def final_condition(self,x_state,x_control,D,I):
        #pass

    def solution(self,x):
        
        # unpack vector
        x_state, x_control, tf = self.unpack(x)

        # operators
        self.t = self.t0 + self.numerics.t*(tf - self.t0)
        D = self.numerics.D/tf 
        I = self.numerics.I*tf

        # call dynamics function to fill state data
        dzdt = self.dynamics(x_state,x_control,D,I)

        # state and orientation data
        self.vectors.r[:,0] = x_state[:,0]
        self.vectors.V[:,0] = x_state[:,1]
        self.vectors.r[:,2] = x_state[:,2]
        self.vectors.V[:,2] = x_state[:,3]
        self.vectors.u = self.vectors.v.copy()

        # accelerations
        self.vectors.a = np.zeros_like(self.vectors.Ftot)
        self.vectors.a[:,0] = dzdt[:,1]
        self.vectors.a[:,2] = dzdt[:,3]

        return 
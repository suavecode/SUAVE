""" Segment.py: Class for storing mission segment data """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
from SUAVE.Structure                                import Data, Data_Exception
from SUAVE.Structure                                import Container as ContainerBase
from SUAVE.Attributes.Planets                       import Planet
from SUAVE.Attributes.Atmospheres                   import Atmosphere

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

class Segment(Data):
    """ Top-level Mission Segment """
    def __defaults__(self):
        self.tag = 'Segment'
        self.dofs = 2
        self.planet = Planet()
        self.atmosphere = Atmosphere()
        self.config = []
        self.dynamics = []
        self.controls = []
        self.complex = False
        self.jacobian = "complex"

        # numerical options
        self.options = Data()
        self.options.tag = 'Options'
        self.options.tol_solution = 1e-8
        self.options.tol_BCs = 1e-8
        self.options.N = 16

        # numerical options
        self.numerics = Data()
        self.numerics.tag = 'Numerics'

        # vector quantities 
        self.vectors = Data()

        # energies 
        self.energy = Data()

        # efficiencies 
        self.efficiency = Data()

        # velocity increments
        self.DV = Data()

    def initialize_vectors(self):

        N = self.options.N

        # initialize arrays
        if self.complex:
            self.vectors.r = np.zeros((N,3)) + 0j       # position vector (m)
            self.vectors.V = np.zeros((N,3)) + 0j       # velocity vector (m/s)
            self.vectors.D = np.zeros((N,3)) + 0j       # drag vector (N)
            self.vectors.L = np.zeros((N,3)) + 0j       # lift vector (N)
            self.vectors.v = np.zeros((N,3)) + 0j       # velocity unit tangent vector
            self.vectors.l = np.zeros((N,3)) + 0j       # lift unit tangent vector
            self.vectors.u = np.zeros((N,3)) + 0j       # thrust unit tangent vector
            self.vectors.F = np.zeros((N,3)) + 0j       # thrust vector (N)
            self.vectors.Ftot = np.zeros((N,3)) + 0j    # total force vector (N)
            self.vectors.a = np.zeros((N,3)) + 0j       # acceleration (m/s^2)
            self.alpha = np.zeros(N) + 0j               # alpha (rad)

        else:    
            self.vectors.r = np.zeros((N,3))            # position vector (m)
            self.vectors.V = np.zeros((N,3))            # velocity vector (m/s)
            self.vectors.D = np.zeros((N,3))            # drag vector (N)
            self.vectors.L = np.zeros((N,3))            # lift vector (N)
            self.vectors.v = np.zeros((N,3))            # velocity unit tangent vector
            self.vectors.l = np.zeros((N,3))            # lift unit tangent vector
            self.vectors.u = np.zeros((N,3))            # thrust unit tangent vector
            self.vectors.F = np.zeros((N,3))            # thrust vector (N)
            self.vectors.Ftot = np.zeros((N,3))         # total force vector (N)
            self.vectors.a = np.zeros((N,3))            # acceleration (m/s^2)
            self.alpha = np.zeros(N)                    # alpha (rad)

        return

    def compute_atmosphere(self,altitude):

        # atmospheric properties
        #if not np.shape(altitude):
        #    altitude = np.array([altitude])
        #if len(altitude) == 1:
        #    p, T, rho, a, mew = self.atmosphere.compute_values(altitude)
        #    self.p = p*np.ones(self.options.N)
        #    self.T = T*np.ones(self.options.N)
        #    self.rho = rho*np.ones(self.options.N)
        #    self.a = a*np.ones(self.options.N)
        #    self.mew = mew*np.ones(self.options.N)
        #elif len(altitude) == self.options.N:
        self.p, self.T, self.rho, self.a, self.mew = \
            self.atmosphere.compute_values(altitude)
        #else: 
        #    print "error: altitude larray length does not match velocity"
        #    return
        return

    def assign_velocity(self,V):

        # figure out DOFs
        if not np.shape(V):
            V = np.array([V]); N = 1
        else:
            if len(np.shape(V)) == 2:
                N, dofs = np.shape(V)
            elif len(np.shape(V)) == 1:
                dofs = np.shape(V)[0]; N = 1
            else:
                print "Something wrong with V array"
                return []

        if N == 1:
            if len(V) == 1:
                self.vectors.V[:,0] = V         # Vx
            elif len(V) == 2:
                self.vectors.V[:,0] = V[0]      # Vx
                self.vectors.V[:,2] = V[1]      # Vz
            elif len(V) == 3:
                self.vectors.V[:,0] = V[0]      # Vx
                self.vectors.V[:,1] = V[1]      # Vy
                self.vectors.V[:,2] = V[2]      # Vz
        elif N == self.N:
            if dofs == 1:
                self.vectors.V[:,0] = V         # Vx
            elif dofs == 2:
                self.vectors.V[:,0] = V[:,0]    # Vx
                self.vectors.V[:,2] = V[:,1]    # Vz
            elif dofs == 3:
                self.vectors.V = V              # V
        else: 
            print "error: velocity vector size does not match"
            return

        return

    def compute_freestream(self):

        # velocity magnitude
        V2 = np.sum(self.vectors.V**2,axis=1)
        self.V = np.sqrt(V2)

        # velocity unit tangent vector
        self.vectors.v[:,0] = self.vectors.V[:,0]/self.V
        self.vectors.v[:,1] = self.vectors.V[:,1]/self.V
        self.vectors.v[:,2] = self.vectors.V[:,2]/self.V

        # dynamic pressure
        self.q = 0.5*self.rho*V2                                    # Pa

        # Mach
        if np.isscalar(self.a):
            if self.a == 0:
                self.M = np.inf
            else:
                self.M = self.V/self.a
        else:
            self.M = np.ones(len(self.a))*np.inf
            mask = self.a != 0
            self.M[mask] = self.V[mask]/self.a[mask]

        # Re
        if np.isscalar(self.mew):
            if self.mew == 0:
                self.Re = np.inf
            else:
                self.Re = self.rho*self.V/self.mew
        else:
            self.Re = np.ones(len(self.a))*np.inf
            mask = self.mew != 0
            self.Re[mask] = self.rho[mask]*self.V[mask]/self.mew[mask]  # per m

        return

    def compute_oreintation(self,u):

        # figure out DOFs
        if len(np.shape(u)) == 2:
            N, dofs = np.shape(u)
        elif len(np.shape(u)) == 1:
            dofs = np.shape(u)[0]; N = 1 
        else:
            print "Something wrong with V array"
            return []

        if N == 1:
            if len(u) == 1:
                self.vectors.u[:,0] = 1.0       # horizontal, input ignored
            elif len(u) == 2:
                self.vectors.u[:,0] = u[0]      # ux
                self.vectors.u[:,2] = u[1]      # uz
            elif len(u) == 3:
                self.vectors.u[:,0] = u[0]      # Vx
                self.vectors.u[:,1] = u[1]      # Vy
                self.vectors.u[:,2] = u[2]      # Vz
        elif N == self.N:
            if dofs == 1:
                self.vectors.u[:,0] = 1.0       # horizontal, input ignored
            elif dofs == 2:
                self.vectors.u[:,0] = u[:,0]    # ux
                self.vectors.u[:,2] = u[:,1]    # uz
            elif dofs == 3:
                self.vectors.u = u              # u
        else: 
            print "error: thrust vector size does not match"
            return

        # normalize u
        umag = np.sqrt(np.sum(self.vectors.u**2,axis=1))
        self.vectors.u[:,0] = self.vectors.u[:,0]/umag
        self.vectors.u[:,1] = self.vectors.u[:,1]/umag
        self.vectors.u[:,2] = self.vectors.u[:,2]/umag

        return
    
    def compute_aero(self,alpha):

        # get scalar aero properties
        self.CD, self.CL = self.config.Aerodynamics(alpha,self)     # nondimensional

        self.F_aero = self.q*self.config.Aerodynamics.S             # N
        self.D = self.CD*self.F_aero                                # N
        self.L = self.CL*self.F_aero                                # N

        return 

    def compute_alpha_2D(self):  # needs some work - MC

        # compute angle of attack and lift vectors   
        for i in range(self.N):

            # determine vehicle plane unit vector
            n_hat = np.cross(self.vectors.v[i,:],self.vectors.u[i,:])
            n_hat_norm = np.linalg.norm(n_hat)
            self.alpha[i] = 0.0

            if n_hat_norm > 0.0:

                # angle of attack
                self.alpha[i] = np.arccos(np.dot(self.vectors.v[i,:],self.vectors.u[i,:]))
                self.alpha[i] = self.alpha[i]*np.sign(n_hat[1])

        return

    def compute_lift_direction_2D(self):
        
        # compute lift unit vectors   
        self.vectors.l[:,0] = -self.vectors.v[:,2]
        self.vectors.l[:,2] = self.vectors.v[:,0]

        return

    def compute_aero_vectors_2D(self):

        # drag vector
        self.vectors.D[:,0] = -self.D*self.vectors.v[:,0]
        self.vectors.D[:,2] = -self.D*self.vectors.v[:,2]

        # lift unit vector
        self.vectors.l[:,0] = -self.vectors.v[:,2]
        self.vectors.l[:,2] = self.vectors.v[:,0]

        # lift vector
        self.vectors.L[:,0] = self.L*self.vectors.l[:,0]
        self.vectors.L[:,2] = self.L*self.vectors.l[:,2]

        return

    def compute_thrust_vectors_2D(self):

        # thurst vectors
        self.vectors.F[:,0] = self.F*self.vectors.u[:,0]
        self.vectors.F[:,2] = self.F*self.vectors.u[:,2]

        return

    def compute_aero_vectors_3D(self):

        # compute angle of attack and lift vectors   
        for i in range(self.N):

            # determine vehicle plane unit vector
            n_hat = np.cross(self.vectors.v[i,:],self.vectors.u[i,:])
            n_hat_norm = np.linalg.norm(n_hat)

            if n_hat_norm == 0.0:

                # u parallel to v, assume lift points vertically
                self.alpha[i] = 0.0; self.vectors.l[i,2] = 1.0

            else:

                # lift unit vector
                n_hat = n_hat/n_hat_norm
                self.vectors.l[i,:] = np.cross(n_hat,self.vectors.v[i,:])

                # angle of attack
                print n_hat
                self.alpha[i] = np.arccos(np.dot(self.vectors.v[i,:],self.vectors.u[i,:]))
                self.alpha[i] = self.alpha[i]*np.sign(np.dot(self.vectors.l[i,:],self.vectors.u[i,:]))

        # drag vector
        self.vectors.D[:,0] = -self.D*self.vectors.v[:,0]
        self.vectors.D[:,1] = -self.D*self.vectors.v[:,1]
        self.vectors.D[:,2] = -self.D*self.vectors.v[:,2]

        # lift vector
        self.vectors.L[:,0] = self.L*self.vectors.l[:,0]
        self.vectors.L[:,1] = self.L*self.vectors.l[:,1]
        self.vectors.L[:,2] = self.L*self.vectors.l[:,2]

        return

    def compute_propulsion(self,eta):

        N = len(eta)

        try: 
            self.config.Propulsors
        except AttributeError:
            self.F = np.zeros(N); self.mdot = np.zeros(N); self.P = np.zeros(N)        
        else:
            if not self.config.Propulsors:
                self.F = np.zeros(N); self.mdot = np.zeros(N); self.P = np.zeros(N);      
            else:
                self.F, self.mdot, self.P = self.config.Propulsors(eta,self)

        return 

    def compute_propulsion_vectors(self):

        # requires freestream and propulstion, returns thrust vectors
        self.vectors.F[:,0] = self.thrust*self.vectors.u[:,0]
        self.vectors.F[:,1] = self.thrust*self.vectors.u[:,1]
        self.vectors.F[:,2] = self.thrust*self.vectors.u[:,2]

        return
    
    def compute_gravity(self,altitude,planet):
 
        # gravity
        self.g = planet.sea_level_gravity        # m/s^2 (placeholder for better g models)
        self.g0 = planet.sea_level_gravity       # m/s^2
        
        return

    def compute_mass(self,m0,I):

        # treat constant mdot
        if not np.shape(self.mdot):
            self.mdot *= np.ones(self.N)
        elif len(self.mdot) == 1:
            self.mdot *= np.ones(self.N)

        self.m = m0 - np.dot(I,self.mdot)

        return self.m

    def compute_forces(self):

        self.vectors.Ftot = self.vectors.F + self.vectors.L + self.vectors.D

        return

# ----------------------------------------------------------------------
#  Handle Linking
# ----------------------------------------------------------------------

class Container(ContainerBase):
    pass

Segment.Container = Container

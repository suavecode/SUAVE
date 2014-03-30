
def compute_atmosphere(self,altitude):

    self.p, self.T, self.rho, self.a, self.mew = \
        self.atmosphere.compute_values(altitude)

    return

def compute_freestream(self):

    # velocity magnitude
    V2 = np.sum(self.vectors.V**2,axis=1)
    self.V = np.sqrt(V2)

    # velocity unit tangent vector
    self.vectors.v = self.vectors.V / self.V[:,None]

    # dynamic pressure
    self.q = 0.5*self.rho*V2 # Pa

    # Mach number
    self.M = np.ones(len(self.a))*np.inf
    mask = self.a > 0 # avoid non-phsyical values
    self.M[mask] = self.V[mask]/self.a[mask]

    # Reynolds number
    self.Re = np.ones(len(self.a))*np.inf
    mask = self.mew > 0 # avoid non-phsyical values
    self.Re[mask] = self.rho[mask]*self.V[mask]/self.mew[mask]  # per m

    return

def compute_aero(self,alpha):

    # CALL AERODYNAMICS MODEL
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
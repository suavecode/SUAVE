""" test_pseudospectral.py: test pseudospectral boundary value solution with test problem """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import SUAVE
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
#  Main
# ----------------------------------------------------------------------

def main():
   
    # create "vehicle"
    cannonball = SUAVE.Vehicle()
    cannonball.tag = "Projectile"
    cannonball.Mass_Props.mass = 1.0
    cannonball.S = 1.0

    # create "configuration"
    config = cannonball.new_configuration("only")
    config.Functions.Aero = aero

    # define problem to be solved
    projectile = SUAVE.Analyses.Mission.Segments.Glide()
    projectile.tag = "Projectile Motion with Drag"
    projectile.final_condition = landing
    projectile.atmosphere = SUAVE.Attributes.Atmospheres.Atmosphere()
    projectile.atmosphere.rho = 1.0
    projectile.planet.sea_level_gravity = 1.0
    projectile.alpha = 0.0
    projectile.m0 = cannonball.Mass_Props.mass
    projectile.config = config

    # initial conditions
    z0 = np.zeros(4)
    z0[0] = 0.0             # x(0)
    z0[1] = 1.0             # Vx(0)
    z0[2] = 0.0             # y(0)
    z0[3] = 1.0             # Vy(0)          
    
    # projectile.pack = pack
    projectile.z0 = z0
    projectile.t0 = 0.0             # initial time
    tf0 = 2.0                       # final time guess

    # define options (accept defaults)
    N = 32; projectile.options.N = N

    # create initial guess (no drag solution)
    g = 1.0; zguess = np.zeros((N,4))
    t = projectile.t0 + 0.5*(1 - np.cos(np.pi*np.arange(0,N)/(N-1)))*(tf0 - projectile.t0)
    zguess[:,0] = z0[1]*t                       # x
    zguess[:,1] = z0[1]*np.ones(N)              # Vx
    zguess[:,2] = z0[3]*t - 0.5*g*t**2          # y
    zguess[:,3] = z0[3]*np.ones(N) - g*t        # Vy
    projectile.guess = zguess[1:].flatten('F')
    projectile.guess = np.append(projectile.guess,tf0-projectile.t0)

    # solve problem
    SUAVE.Methods.Utilities.pseudospectral(projectile)
    
    # unpack results
    t = projectile.t
    x = projectile.vectors.r[:,0]
    y = projectile.vectors.r[:,2]
    Vx = projectile.vectors.V[:,0]
    Vy = projectile.vectors.V[:,2]

    # plot solution
    title = projectile.tag
    plt.subplot(211)
    plt.plot(t,x,t,Vx,t,y,t,Vy)
    plt.xlabel('t'); plt.ylabel('z'); plt.title(title)
    plt.legend(('x', 'dx/dt', 'y', 'dy/dt'),'lower left')
    plt.grid(True)       
    
    plt.subplot(212)
    plt.plot(x,y,'o-')
    plt.plot(zguess[:,0],zguess[:,2])
    plt.xlabel('x'); plt.ylabel('y'); plt.title(title)
    plt.legend(('with drag', 'no drag'),'upper left')
    plt.grid(True)
     
    plt.grid(True)
    plt.show()

    return

# constant CD aero model
def aero(self,alpha,segment): 
    
    CL = 0.0
    CD = 0.5

    return CD, CL

# final condition: hit the ground
def landing(self,x_state,x_control,D,I):
        
        y = x_state[:,2]

        return y[-1]      # y(tf) = 0

# call main
if __name__ == '__main__':
    main()

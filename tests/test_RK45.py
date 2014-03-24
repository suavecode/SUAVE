""" test_RK45.py: test RK45 integration with test problem """

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
   
    # initial conditions
    z0 = np.zeros(4)
    z0[0] = 0.0             # x(0)
    z0[1] = 1.0             # Vx(0)
    z0[2] = 0.0             # y(0)
    z0[3] = 1.0             # Vy(0)

    # minimum conditions
    zmin = np.zeros(4)
    zmin[0] = None          # x (no minimum constraint)
    zmin[1] = 0.0           # Vx > 0 
    zmin[2] = 0.0           # y > 0 (hit the groud)
    zmin[3] = None          # Vy (no minimum constraint)            

    # maximum conditions
    zmax = np.zeros(4)
    zmax[0] = None          # x (no maximum constraint)
    zmax[1] = None          # Vx (no maximum constraint) 
    zmax[2] = None          # y (no maximum constraint)
    zmax[3] = None          # Vy (no maximum constraint)     

    # define problem to be solved
    projectile = SUAVE.Methods.Utilities.Problem()
    projectile.tag = "Projectile Motion with Drag"
    projectile.f = dzdt
    projectile.z0 = z0
    projectile.zmin = zmin
    projectile.zmax = zmax
    projectile.h0 = 0.05
    projectile.t0 = 0.0             # initial time
    projectile.tf = 3.0             # maximum time
    projectile.config = None

    # define options (accept defaults)
    options = SUAVE.Methods.Utilities.Options()

    # solve problem
    solution = SUAVE.Methods.Utilities.runge_kutta_45(projectile,options)
    
    # unpack results
    t = solution.t
    z = solution.z

    print "Solution ended because: " + solution.exit.reason + \
        " for variable " + str(solution.exit.j)
    print "Final error on boundary = " + str(solution.exit.err)

    # plot solution
    title = projectile.tag
    plt.subplot(211)
    plt.plot(t,z[:,0],t,z[:,1],t,z[:,2],t,z[:,3])
    plt.xlabel('t'); plt.ylabel('z'); plt.title(title)
    plt.legend(('x', 'dx/dt', 'y', 'dy/dt'),'lower left')
    plt.grid(True)       
    
    plt.subplot(212)
    plt.plot(z[:,0],z[:,2],'o-')
    plt.xlabel('x'); plt.ylabel('y'); plt.title(title)
    plt.grid(True)
     
    plt.grid(True)
    plt.show()

    return

# test problem: projectile motion with drag
def dzdt(self,t,z):

    m = 1.0; g = 1.0; CD = 0.5

    # unpack state data
    x = z[0]; Vx = z[1]; y = z[2]; Vy = z[3]
    V2 = Vx**2 + Vy**2; V = np.sqrt(V2)

    # drag force
    D = CD*V2

    dzdt = np.zeros(len(z))
    dzdt[0] = Vx
    dzdt[1] = -(D*Vx/V)/m
    dzdt[2] = Vy
    dzdt[3] = -(D*Vy/V)/m - g

    return dzdt

# call main
if __name__ == '__main__':
    main()

#boeing_concept.py

# Created:  Feb 2021, E. Botero
# Modified: 


# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt  

from SUAVE.Core import Data, Units
from SUAVE.Input_Output.OpenVSP.vsp_read import vsp_read
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift import VLM as VLM

# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------

def main():

    # First import the geometry
    #vehicle = vsp_read('boeing_n_d_t.vsp3',units_type='inches')
    vehicle = vsp_read('boeing_B_only.vsp3',units_type='inches')
    
    #vehicle.fuselages.pop('fueslage')
    #vehicle.wings.pop('tail')
    #vehicle.fuselages.fueslage.origin[0][2] = .25
    
    vehicle.wings.gross_wing_b.vortex_lift = True
        
    #vehicle.reference_area = vehicle.wings.gross_wing_b__t___d_.areas.reference
    vehicle.reference_area = 2*158.13
    
    # Setup conditions
    conditions = setup_conditions()
    
    # Run
    results, CP = analyze(vehicle, conditions)
    
    # Plot the CP's
    plot_CP(CP,conditions,vehicle)

    
    print(results)
    
    
    return


# ----------------------------------------------------------------------
#  setup_conditions
# ----------------------------------------------------------------------


def setup_conditions():
        
    #aoas  = np.array([-2,0,2,4,6,-2,0,2,4,6,-2,0,2,4,6,-2,0,2,4,6,-2,0,2,4,6,-2,0,2,4,6]) * Units.degrees
    #machs = np.array([0.4,0.4,0.4,0.4,0.4,0.8,0.8,0.8,0.8,0.8,1.4,1.4,1.4,1.4,1.4,1.6,1.6,1.6,1.6,1.6,1.8,1.8,1.8,1.8,1.8,2,2,2,2,2])
    
    
    #aoas  = np.array([6.,2.,2.,6.]) * Units.degrees
    #machs = np.array([0.4,1.,2.0,2.0])    
    
    #aoas  = np.array([6.,6]) * Units.degrees
    #machs = np.array([0.4,1.4]) 
    
    #aoas  = np.array([0.,2.,4.,6.,8.,10,0.,2.,4.,6.,8.,10,0.,2.,4.,6.,8.,10,0.,2.,4.,6.,8.,10]) * Units.degrees
    #machs = np.array([1.4,1.4,1.4,1.4,1.4,1.4,1.6,1.6,1.6,1.6,1.6,1.6,1.8,1.8,1.8,1.8,1.8,1.8,2.0,2.0,2.0,2.0,2.0,2.0])    
    
    aoas  = np.array([-2.,0.,2.,4.,6.,-2.,0.,2.,4.,6.]) * Units.degrees
    machs = np.array([0.8,0.8,0.8,0.8,0.8,1.6,1.6,1.6,1.6,1.6])        
    
    
    #aoas  = xv.flatten()
    #machs = yv.flatten()
    
    conditions              = Data()
    conditions.aerodynamics = Data()
    conditions.freestream   = Data()
    conditions.freestream.velocity          = np.atleast_2d(100.*np.ones_like(aoas))
    conditions.aerodynamics.angle_of_attack = np.atleast_2d(aoas).T
    conditions.freestream.mach_number       = np.atleast_2d(machs).T

    return conditions

# ----------------------------------------------------------------------
#  analyze
# ----------------------------------------------------------------------



def analyze(config,conditions):
    
    
    results = Data()
    
    S                                  = config.reference_area
    settings                           = Data()
    settings.number_spanwise_vortices  = 50
    settings.number_chordwise_vortices = 25
    settings.propeller_wake_model      = None
    settings.spanwise_cosine_spacing   = True
    settings.model_fuselage            = True
    settings.initial_timestep_offset   = 0.0
    settings.wake_development_time     = 0.0
    settings.number_of_wake_timesteps  = 0.

    CL, CDi, CM, CL_wing, CDi_wing, cl_y, cdi_y, alpha_i, CP, Velocity_Profile = VLM(conditions, settings, config)

    results.CDi  = CDi
    results.CL   = CL
    results.CM   = CM
    results.mach = conditions.freestream.mach_number
    results.aoa  = conditions.aerodynamics.angle_of_attack
    
    print('CL')
    print(CL)
    
    print('CDi')
    print(CDi)

    return results, CP


def plot_CP(CP_in,conditions,vehicle,save_figure = True,file_type=".png"):
    
    VD         = vehicle.vortex_distribution	 
    n_cw       = VD.n_cw 	
    n_cw       = VD.n_cw 
    n_sw       = VD.n_sw 
    n_w        = VD.n_w 
    
    # Create a boolean for not plotting vertical wings
    idx        = 0
    plot_flag  = np.ones(n_w)
    for wing in vehicle.wings: 
        if wing.vertical: 
            plot_flag[idx] = 0 
            idx += 1    
        else:
            idx += 1 
        if wing.vertical and wing.symmetric:             
            plot_flag[idx] = 0 
            idx += 1
        else:
            idx += 1  
        
    for ii in range(len(conditions.aerodynamics.angle_of_attack)):
        aoa        = conditions.aerodynamics.angle_of_attack[ii] / Units.degrees
        mach       = conditions.freestream.mach_number[ii]
        CP         = CP_in[ii,:]
        
        save_filename = str(aoa) + '_' + str(mach)
        
        fig        = plt.figure()	
        axes       = fig.add_subplot(1, 1, 1)  
        x_max      = max(VD.XC) + 2
        y_max      = max(VD.YC) + 2
        axes.set_ylim(x_max, 0)
        axes.set_xlim(-y_max, y_max)            
        fig.set_size_inches(8,8)         	 
        for i in range(n_w):
            n_pts     = (n_sw + 1) * (n_cw + 1) 
            xc_pts    = VD.X[i*(n_pts):(i+1)*(n_pts)]
            x_pts     = np.reshape(np.atleast_2d(VD.XC[i*(n_sw*n_cw):(i+1)*(n_sw*n_cw)]).T, (n_sw,-1))
            y_pts     = np.reshape(np.atleast_2d(VD.YC[i*(n_sw*n_cw):(i+1)*(n_sw*n_cw)]).T, (n_sw,-1))
            z_pts     = np.reshape(np.atleast_2d(CP[i*(n_sw*n_cw):(i+1)*(n_sw*n_cw)]).T, (n_sw,-1))
            x_pts_p   = x_pts*((n_cw+1)/n_cw) - x_pts[0,0]*((n_cw+1)/n_cw)  +  xc_pts[0] 
            points    = np.linspace(0.001,1,50)
            A         = np.cumsum(np.sin(np.pi/2*points))
            levals    = -(np.concatenate([-A[::-1],A[1:]])/(2*A[-1])  + A[-1]/(2*A[-1]) )[::-1]*0.015  
            color_map = plt.cm.get_cmap('jet')
            rev_cm    = color_map.reversed()
            if plot_flag[i] == 1:
                CS  = axes.contourf(y_pts,x_pts_p, z_pts, cmap = rev_cm,extend='both')    
            
        # Set Color bar	
        cbar = fig.colorbar(CS, ax=axes)
        cbar.ax.set_ylabel('$C_{P}$', rotation =  0)  
        plt.axis('off')	
        plt.grid(None)            
        
        if save_figure: 
            plt.savefig( save_filename + file_type) 	    



if __name__ == '__main__': 
    main()    
    plt.show()

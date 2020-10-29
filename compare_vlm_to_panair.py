# compare_vlm_to_panair.py


# Created:  Oct 2020, E. Botero
# Modified: 

# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Units, Data
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import cm


from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift import VLM
from SUAVE.Methods.Aerodynamics.Supersonic_Zero.Drag.wave_drag_lift import wave_drag_lift
from SUAVE.Methods.Aerodynamics.Supersonic_Zero.Drag.wave_drag_volume import wave_drag_volume
from SUAVE.Methods.Geometry.Two_Dimensional.Planform.wing_planform import wing_planform
from SUAVE.Methods.Flight_Dynamics.Static_Stability.Approximations.Supporting_Functions.convert_sweep import convert_sweep
# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------

def main():
    
    # Straight biconvex
    length            = 5
    biconvex_file     = '/Users/emiliobotero/Dropbox/Postdoc/exo/Stanford-Exosonic_Aerodynamics/straight_biconvex.csv'
    panair_biconvex   = import_csv(biconvex_file)
    straight_biconvex = strt_biconvex()
    conditions        = setup_conditions(panair_biconvex)
    results_biconvex  = analyze(straight_biconvex, conditions)
    plot_results('biconvex',results_biconvex,panair_biconvex,length)
    
    ## Straight NACA
    #length            = 5
    #strt_NACA_file    = '/Users/emiliobotero/Dropbox/Postdoc/exo/Stanford-Exosonic_Aerodynamics/straight_NACA.csv'
    #panair_strt_NACA  = import_csv(strt_NACA_file)
    #straight_NACA     = strt_naca()
    #conditions        = setup_conditions(panair_strt_NACA)
    #results_strt_NACA = analyze(straight_NACA, conditions)
    #plot_results('Straight NACA',results_strt_NACA,panair_strt_NACA,length)    
    
    # Arrow NACA
    length             = 5
    arrow_NACA_file    = '/Users/emiliobotero/Dropbox/Postdoc/exo/Stanford-Exosonic_Aerodynamics/straight_biconvex.csv'
    panair_arrow_NACA  = import_csv(arrow_NACA_file)
    arrow_NACA         = arrw_naca()
    conditions         = setup_conditions(panair_arrow_NACA)
    results_arrow_NACA = analyze(arrow_NACA, conditions)
    plot_results('Arrow NACA',results_arrow_NACA,panair_arrow_NACA,length)
    

    
    plt.show()

    return
 

def import_csv(filename):
    
    my_data = np.genfromtxt(filename, delimiter=',',skip_header=1)
    
    results = Data()
    results.mach = np.atleast_2d(my_data[:,0]).T
    results.aoa  = np.atleast_2d(my_data[:,1]).T * Units.degrees
    results.CL   = np.atleast_2d(my_data[:,2]).T
    results.CD   = np.atleast_2d(my_data[:,3]).T
    
    return results

def plot_results(name,vlm_results,panair_results,length):
    
    
    mach  = panair_results.mach.reshape((-1,length))
    aoa   = panair_results.aoa.reshape((-1,length))
    v_CL  = vlm_results.CL.reshape((-1,length))
    v_CD  = vlm_results.CD.reshape((-1,length))
    v_CDi = vlm_results.CDi.reshape((-1,length))
    p_CL  = panair_results.CL.reshape((-1,length))
    p_CD  = panair_results.CD.reshape((-1,length))


    # CL
    fig = plt.figure(name+' CL')
    ax = fig.gca(projection='3d')    
    
    vlm_cl = ax.plot_surface(mach, aoa, v_CL, cmap=cm.Reds,
                           linewidth=0, antialiased=False)  
    
    pan_cl = ax.plot_surface(mach, aoa, p_CL, cmap=cm.Blues,
                           linewidth=0, antialiased=False)  
    
    ax.set_xlabel('Mach')
    ax.set_ylabel('AoA')
    ax.set_zlabel('CL')
    
    fig.colorbar(vlm_cl, shrink=1, aspect=1,label='VLM')
    fig.colorbar(pan_cl, shrink=1, aspect=1,label='Panair')
    
    
    # CD
    fig2 = plt.figure(name+' CD no Wave from VLM')
    ax2 = fig2.gca(projection='3d')    
    
    vlm_cd = ax2.plot_surface(mach, v_CL, v_CD, cmap=cm.Reds,
                           linewidth=0, antialiased=False)  
    
    pan_cd = ax2.plot_surface(mach, p_CL, p_CD, cmap=cm.Blues,
                           linewidth=0, antialiased=False)  
    
    ax2.set_xlabel('Mach')
    ax2.set_ylabel('CL')
    ax2.set_zlabel('CD')    
    
    fig2.colorbar(vlm_cd, shrink=0.5, aspect=5,label='VLM')
    fig2.colorbar(pan_cd, shrink=0.5, aspect=5,label='Panair')        
    
    
    # CD
    fig3 = plt.figure(name+' CD with Wave due to Lift')
    ax3  = fig3.gca(projection='3d')    
    
    vlm_cd = ax3.plot_surface(mach, v_CL, v_CD, cmap=cm.Reds,
                           linewidth=0, antialiased=False)  
    
    pan_cd = ax3.plot_surface(mach, p_CL, p_CD, cmap=cm.Blues,
                           linewidth=0, antialiased=False)  
    
    ax3.set_xlabel('Mach')
    ax3.set_ylabel('CL')
    ax3.set_zlabel('CD')    
    
    fig3.colorbar(vlm_cd, shrink=0.5, aspect=5,label='VLM')
    fig3.colorbar(pan_cd, shrink=0.5, aspect=5,label='Panair')    
    
    return


def analyze(config,conditions):
    
    
    results = Data()
    
    S                                  = config.reference_area
    settings                           = Data()
    settings.number_spanwise_vortices  = 20
    settings.number_chordwise_vortices = 5
    settings.propeller_wake_model      = None

    CL, CDi, CM, CL_wing, CDi_wing, cl_y , cdi_y , CP ,Velocity_Profile = VLM(conditions, settings, config)
    
    # Save the CL's to conditions
    conditions.aerodynamics.lift_coefficient = CL
    
    # Now do wave drag due to lift
    configuration = None
    wing = config.wings.main_wing
    wave_drag_lift_coefficient = wave_drag_lift(conditions, configuration, wing)
    
    CD  = CDi + wave_drag_lift_coefficient
    
    ## Wave drag due to volume
    #wave_drag_volume(config,mach,scaling_factor)
    

    results.CDi  = CDi
    results.CD   = CD
    results.CL   = CL
    results.mach = conditions.freestream.mach_number
    results.aoa  = conditions.aerodynamics.angle_of_attack
    
    
    return results


#def plots(conditions,results):
    

def setup_conditions(panair_results):
        
    aoas  = panair_results.aoa
    machs = panair_results.mach
    
    #aoas  = xv.flatten()
    #machs = yv.flatten()
    
    conditions              = Data()
    conditions.aerodynamics = Data()
    conditions.freestream   = Data()
    conditions.freestream.velocity          = np.atleast_2d(100.*np.ones_like(aoas))
    conditions.aerodynamics.angle_of_attack = np.atleast_2d(aoas)
    conditions.freestream.mach_number       = np.atleast_2d(machs)

    return conditions


def strt_biconvex():
    
    # ------------------------------------------------------------------
    #   Initialize the Vehicle
    # ------------------------------------------------------------------

    vehicle = SUAVE.Vehicle()
    vehicle.tag = 'strt_biconvex'   
    
    # basic parameters
    vehicle.reference_area = 200
    
    # ------------------------------------------------------------------
    #   Main Wing
    # ------------------------------------------------------------------

    wing = SUAVE.Components.Wings.Main_Wing()
    wing.tag = 'main_wing'    
    wing.aspect_ratio            = 8
    wing.sweeps.quarter_chord    = 0
    wing.thickness_to_chord      = 0.03
    wing.taper                   = 1.
    wing.spans.projected         = 40

    wing.chords.root             = 5. * Units.meter
    wing.chords.tip              = 5. * Units.meter
    wing.chords.mean_aerodynamic = 5. * Units.meter

    wing.areas.reference         = 200.

    wing.twists.root             = 0.0 * Units.degrees
    wing.twists.tip              = 0.0 * Units.degrees

    wing.origin                  = [[0.,0.,0.]]
    wing.aerodynamic_center      = [0,0,0]

    wing.vertical                = False
    wing.symmetric               = True
    wing.high_lift               = False

    wing.dynamic_pressure_ratio  = 1.0    
    
    wing_airfoil = SUAVE.Components.Wings.Airfoils.Airfoil()
    wing_airfoil.coordinate_file   = '/Users/emiliobotero/Dropbox/SUAVE/SUAVE/naca64203.dat' 
        
    wing =  wing_planform(wing)
    
    wing.append_airfoil(wing_airfoil)      
    
    vehicle.append_component(wing)
    
    vehicle.total_length = wing.total_length
    
    return vehicle

def strt_naca():
    
    # ------------------------------------------------------------------
    #   Initialize the Vehicle
    # ------------------------------------------------------------------

    vehicle = SUAVE.Vehicle()
    vehicle.tag = 'straight_naca'   
    
    # basic parameters
    vehicle.reference_area = 200
    
    # ------------------------------------------------------------------
    #   Main Wing
    # ------------------------------------------------------------------

    wing = SUAVE.Components.Wings.Main_Wing()
    wing.tag = 'main_wing'    
    wing.aspect_ratio            = 8
    wing.sweeps.quarter_chord    = 0
    wing.thickness_to_chord      = 0.01
    wing.taper                   = 1.
    wing.spans.projected         = 40

    wing.chords.root             = 5. * Units.meter
    wing.chords.tip              = 5. * Units.meter
    wing.chords.mean_aerodynamic = 5. * Units.meter

    wing.areas.reference         = 200.

    wing.twists.root             = 0.0 * Units.degrees
    wing.twists.tip              = 0.0 * Units.degrees

    wing.origin                  = [[0.,0.,0.]]
    wing.aerodynamic_center      = [0,0,0]

    wing.vertical                = False
    wing.symmetric               = True
    wing.high_lift               = False

    wing.dynamic_pressure_ratio  = 1.0    
    
    wing_airfoil = SUAVE.Components.Wings.Airfoils.Airfoil()
    wing_airfoil.coordinate_file = '/Users/emiliobotero/Dropbox/SUAVE/SUAVE/naca64203.dat' 
    
    wing =  wing_planform(wing)
    
    wing.append_airfoil(wing_airfoil)      
    
    vehicle.append_component(wing)
    
    vehicle.total_length = wing.total_length
    
    return vehicle


def arrw_naca():
    
    # ------------------------------------------------------------------
    #   Initialize the Vehicle
    # ------------------------------------------------------------------

    vehicle = SUAVE.Vehicle()
    vehicle.tag = 'arrow_naca'   
    
    # basic parameters
    vehicle.reference_area = 198
    
    # ------------------------------------------------------------------
    #   Main Wing
    # ------------------------------------------------------------------

    wing = SUAVE.Components.Wings.Main_Wing()
    wing.tag = 'main_wing'    
    wing.aspect_ratio            = 2.44444
    wing.sweeps.leading_edge     = 60. * Units.degrees
    wing.thickness_to_chord      = 0.03
    wing.taper                   = 1./17.
    wing.spans.projected         = 22.

    wing.chords.root             = 17. * Units.meter
    wing.chords.tip              = 1. * Units.meter

    wing.areas.reference         = 198.

    wing.twists.root             = 0.0 * Units.degrees
    wing.twists.tip              = 0.0 * Units.degrees

    wing.origin                  = [[0.,0.,0.]]
    wing.aerodynamic_center      = [0,0,0]

    wing.vertical                = False
    wing.symmetric               = True
    wing.high_lift               = False

    wing.dynamic_pressure_ratio  = 1.0    
    
    wing_airfoil = SUAVE.Components.Wings.Airfoils.Airfoil()
    wing_airfoil.coordinate_file = '/Users/emiliobotero/Dropbox/SUAVE/SUAVE/naca64203.dat' 
    
    wing.sweeps.quarter_chord = convert_sweep(wing,old_ref_chord_fraction = 0.0,new_ref_chord_fraction = 0.25)    
    
    wing =  wing_planform(wing)
    
    wing.append_airfoil(wing_airfoil)      
    
    vehicle.append_component(wing)
    
    vehicle.total_length = wing.total_length
    
    return vehicle




if __name__ == '__main__': 
    main()    
    plt.show()

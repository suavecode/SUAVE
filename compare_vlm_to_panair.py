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


from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift import VLM_supersonic as VLM
from SUAVE.Methods.Aerodynamics.Supersonic_Zero.Drag.wave_drag_lift import wave_drag_lift
from SUAVE.Methods.Geometry.Two_Dimensional.Planform.wing_planform import wing_planform
from SUAVE.Methods.Flight_Dynamics.Static_Stability.Approximations.Supporting_Functions.convert_sweep import convert_sweep
from SUAVE.Input_Output.OpenVSP.vsp_write import write
# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------

def main():
    
    ## Straight biconvex
    #length            = 5
    #biconvex_file     = '/Users/emiliobotero/Dropbox/Postdoc/exo/Stanford-Exosonic_Aerodynamics/straight_biconvex.csv'
    #panair_biconvex   = import_csv(biconvex_file)
    #straight_biconvex = strt_biconvex()
    #conditions        = setup_conditions(panair_biconvex)
    #results_biconvex  = analyze(straight_biconvex, conditions)
    #plot_results_2D('biconvex',results_biconvex,panair_biconvex,length)
    #plot_results('biconvex',results_biconvex,panair_biconvex,length)
    
    ## Straight NACA
    #length            = 5
    #strt_NACA_file    = '/Users/emiliobotero/Dropbox/Postdoc/exo/Stanford-Exosonic_Aerodynamics/straight_NACA.csv'
    #panair_strt_NACA  = import_csv(strt_NACA_file)
    #straight_NACA     = strt_naca()
    #conditions        = setup_conditions(panair_strt_NACA)
    #results_strt_NACA = analyze(straight_NACA, conditions)
    #plot_results('Straight NACA',results_strt_NACA,panair_strt_NACA,length)    
    
    ## Arrow NACA
    #length             = 5
    #arrow_NACA_file    = '/Users/emiliobotero/Dropbox/Postdoc/exo/Stanford-Exosonic_Aerodynamics/arrow_NACA.csv'
    #panair_arrow_NACA  = import_csv(arrow_NACA_file)
    #arrow_NACA         = arrw_naca()
    #conditions         = setup_conditions(panair_arrow_NACA)
    #results_arrow_NACA = analyze(arrow_NACA, conditions)
    #plot_results('Arrow NACA',results_arrow_NACA,panair_arrow_NACA,length)
    #plot_results_2D('Arrow NACA',results_arrow_NACA,panair_arrow_NACA,length)
    
    
    # Arrow biconvex
    #arrow_biconvex_file_vsp = '/Users/emiliobotero/Dropbox/Postdoc/exo/Stanford-Exosonic_Aerodynamics/arrow_biconvex_vspaero.csv'
    #arrow_biconvex_file_pan = '/Users/emiliobotero/Dropbox/Postdoc/exo/Stanford-Exosonic_Aerodynamics/arrow_biconvex.csv'
    #arrow_biconvex_file_su2 = '/Users/emiliobotero/Dropbox/Postdoc/exo/Stanford-Exosonic_Aerodynamics/arrow_biconvex_su2.csv'
    #vsp_arrow_biconvex     = import_csv(arrow_biconvex_file_vsp)
    #pan_arrow_biconvex     = import_csv(arrow_biconvex_file_pan)
    #su2_arrow_biconvex     = import_c8sv(arrow_biconvex_file_su2)
    #arrow_biconvex         = arrw_biconvex_vertical_dih() # Check if this is vertical
    arrow_biconvex           = arrw_biconvex_twist_dih()
    arrow_biconvex           = arrw_biconvex()
    #write(arrow_biconvex,'Check')
    conditions             = setup_conditions()
    results_arrow_biconvex = analyze(arrow_biconvex, conditions)
    print('stop')
    #plot_results('Arrow NACA',results_arrow_NACA,panair_arrow_NACA,length)
    #plot_results_2D('Arrow biconvex',results_arrow_biconvex,pan_arrow_biconvex,vsp_arrow_biconvex,length)    
    
    
    ## Arrow NACA Twist
    #length             = 5
    #arrow_NACA_twist_file    = '/Users/emiliobotero/Dropbox/Postdoc/exo/Stanford-Exosonic_Aerodynamics/arrow_NACA_twist.csv'
    #panair_arrow_NACA_twist  = import_csv(arrow_NACA_twist_file)
    #arrow_NACA_twist         = arrw_naca_twist()
    #conditions               = setup_conditions(panair_arrow_NACA_twist)
    #results_arrow_NACA_twist = analyze(arrow_NACA_twist, conditions)
    #plot_results('Arrow NACA Twist',results_arrow_NACA_twist,panair_arrow_NACA_twist,length)
    
    # Arrow NACA Twist Dihedral
    #length             = 5
    #arrow_NACA_twist_dih_file    = '/Users/emiliobotero/Dropbox/Postdoc/exo/Stanford-Exosonic_Aerodynamics/arrow_NACA_dihedral.csv'
    #panair_arrow_NACA_twist_dih  = import_csv(arrow_NACA_twist_dih_file)
    #arrow_NACA_twist_dih         = arrw_naca_twist_dih()
    #conditions                   = setup_conditions_input(panair_arrow_NACA_twist_dih)
    #results_arrow_NACA_twist_dih = analyze(arrow_NACA_twist_dih, conditions)
    ##plot_results('Arrow NACA Twist Dihedral',results_arrow_NACA_twist_dih,panair_arrow_NACA_twist_dih,length)
    #plot_results_2D('Arrow NACA Twist Dihedral',results_arrow_NACA_twist_dih,panair_arrow_NACA_twist_dih,length)    
        
        
    
    

    
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

def plot_results(name,vlm_results,panair_results,length,label_name='Panair'):
    
    
    mach  = panair_results.mach.reshape((-1,length))
    aoa   = panair_results.aoa.reshape((-1,length))
    s_CL  = vlm_results.CL.reshape((-1,length))
    s_CD  = vlm_results.CD.reshape((-1,length))
    s_CDi = vlm_results.CDi.reshape((-1,length))
    p_CL  = panair_results.CL.reshape((-1,length))
    p_CD  = panair_results.CD.reshape((-1,length))
    

    # CL
    fig = plt.figure(name+' CL')
    ax = fig.gca(projection='3d')    
    
    vlm_cl = ax.plot_surface(mach, aoa, s_CL, cmap=cm.Reds,
                           linewidth=0, antialiased=False)  
    
    pan_cl = ax.plot_surface(mach, aoa, p_CL, cmap=cm.Blues,
                           linewidth=0, antialiased=False)  
    
    ax.set_xlabel('Mach')
    ax.set_ylabel('AoA')
    ax.set_zlabel('CL')
    
    fig.colorbar(vlm_cl, shrink=0.5, aspect=5,label='VLM')
    fig.colorbar(pan_cl, shrink=0.5, aspect=5,label=label_name)
    
    
    # CD
    fig2 = plt.figure(name+' CD no Wave from VLM')
    ax2 = fig2.gca(projection='3d')    
    
    vlm_cd = ax2.plot_surface(mach, s_CL, s_CDi, cmap=cm.Reds,
                           linewidth=0, antialiased=False)  
    
    pan_cd = ax2.plot_surface(mach, p_CL, p_CD, cmap=cm.Blues,
                           linewidth=0, antialiased=False)  
    
    ax2.set_xlabel('Mach')
    ax2.set_ylabel('CL')
    ax2.set_zlabel('CD')    
    
    fig2.colorbar(vlm_cd, shrink=0.5, aspect=5,label='VLM')
    fig2.colorbar(pan_cd, shrink=0.5, aspect=5,label=label_name)        
    
    
    # CD
    fig3 = plt.figure(name+' CD with Wave due to Lift')
    ax3  = fig3.gca(projection='3d')    
    
    vlm_cd = ax3.plot_surface(mach, s_CL, s_CD, cmap=cm.Reds,
                           linewidth=0, antialiased=False)  
    
    pan_cd = ax3.plot_surface(mach, p_CL, p_CD, cmap=cm.Blues,
                           linewidth=0, antialiased=False)  
    
    ax3.set_xlabel('Mach')
    ax3.set_ylabel('CL')
    ax3.set_zlabel('CD')    
    
    fig3.colorbar(vlm_cd, shrink=0.5, aspect=5,label='VLM')
    fig3.colorbar(pan_cd, shrink=0.5, aspect=5,label=label_name)    
    
    
    return

def plot_results_2D(name,vlm_results,panair_results,length):
    
    mach  = panair_results.mach.reshape((-1,length))
    aoa   = panair_results.aoa.reshape((-1,length))
    s_CL  = vlm_results.CL.reshape((-1,length))
    s_CD  = vlm_results.CD.reshape((-1,length))
    s_CDi = vlm_results.CDi.reshape((-1,length))
    p_CL  = panair_results.CL.reshape((-1,length))
    p_CD  = panair_results.CD.reshape((-1,length))
    #v_CL  = vsp_results.CL.reshape((-1,length))
    #v_CD  = vsp_results.CD.reshape((-1,length))
    
    fig_CL  = plt.figure(name+' 2D CL')
    fig_CDi = plt.figure(name+' 2D CD no Wave from VLM')
    fig_CD  = plt.figure(name+' 2D CD with Wave due to Lift')
    fig_CL.set_size_inches(12, 8)
    fig_CD.set_size_inches(12, 8)
    fig_CDi.set_size_inches(12, 8)
    n_plots = np.shape(mach)[0]
    for ii in range(n_plots):
        a_mach = mach[ii,0]
        axes_CL = fig_CL.add_subplot(n_plots,1,ii+1)
        axes_CL.plot( aoa[ii,:] / Units.degrees, s_CL[ii,:] , 'ro-',label='VLM')
        axes_CL.plot( aoa[ii,:] / Units.degrees, p_CL[ii,:] , 'bo-',label='Panair')
        #axes_CL.plot( aoa[ii,:] / Units.degrees, v_CL[ii,:] , 'go-',label='VSPaero VLM')
        axes_CL.set_ylabel('CL Mach = ' + str(a_mach))
        axes_CL.set_xlabel('AoA')
        
        axes_CDi = fig_CDi.add_subplot(n_plots,1,ii+1)
        axes_CDi.plot( s_CL[ii,:], s_CDi[ii,:] , 'ro-',label='VLM')
        axes_CDi.plot( p_CL[ii,:], p_CD[ii,:] , 'bo-',label='Panair')
        #axes_CDi.plot( v_CL[ii,:], v_CD[ii,:] , 'go-',label='VSPaero VLM')
        axes_CDi.set_ylabel('CDi Mach = ' + str(a_mach))
        axes_CDi.set_xlabel('CL')
        
        axes_CD = fig_CD.add_subplot(n_plots,1,ii+1)
        axes_CD.plot( s_CL[ii,:], s_CD[ii,:] , 'ro-',label='VLM')
        axes_CD.plot( p_CL[ii,:], p_CD[ii,:] , 'bo-',label='Panair')
        #axes_CD.plot( v_CL[ii,:], v_CD[ii,:] , 'go-',label='VSPaero VLM')
        axes_CD.set_ylabel('CD Mach = ' + str(a_mach))
        axes_CD.set_xlabel('CL')     
        
        if ii == 0: 
            axes_CL.legend(loc='upper left')   
            axes_CD.legend(loc='upper left')    
            axes_CDi.legend(loc='upper left')  

    return



def analyze(config,conditions, use_MCM = False):
    
    
    results = Data()
    
    S                                  = config.reference_area
    settings                           = Data()
    settings.number_spanwise_vortices  = 2
    settings.number_chordwise_vortices = 2
    settings.propeller_wake_model      = None
    settings.spanwise_cosine_spacing   = False
    settings.model_fuselage            = True
    settings.initial_timestep_offset   = 0.0
    settings.wake_development_time     = 0.0 

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
    
    print('CL')
    print(CL)
    
    print('CDi')
    print(CDi)
    
    
    
    
    return results


#def plots(conditions,results):
    

def setup_conditions():
        
    #aoas  = np.array([-2,0,2,4,6,-2,0,2,4,6,-2,0,2,4,6,-2,0,2,4,6,-2,0,2,4,6,-2,0,2,4,6]) * Units.degrees
    #machs = np.array([0.4,0.4,0.4,0.4,0.4,0.8,0.8,0.8,0.8,0.8,1.4,1.4,1.4,1.4,1.4,1.6,1.6,1.6,1.6,1.6,1.8,1.8,1.8,1.8,1.8,2,2,2,2,2])
    
    
    #aoas  = np.array([6.,2.,2.,6.]) * Units.degrees
    #machs = np.array([0.4,1.,2.0,2.0])    
    
    aoas  = np.array([6.,6]) * Units.degrees
    machs = np.array([0.4,1.4])        
    
    #aoas  = xv.flatten()
    #machs = yv.flatten()
    
    conditions              = Data()
    conditions.aerodynamics = Data()
    conditions.freestream   = Data()
    conditions.freestream.velocity          = np.atleast_2d(100.*np.ones_like(aoas))
    conditions.aerodynamics.angle_of_attack = np.atleast_2d(aoas).T
    conditions.freestream.mach_number       = np.atleast_2d(machs).T

    return conditions

def setup_conditions_input(panair_results):
        
    
    
    #aoas  = np.array([6.,2.,2.,6.]) * Units.degrees
    #machs = np.array([0.4,1.,2.0,2.0])    
    
    #aoas  = np.array([6.,6]) * Units.degrees
    #machs = np.array([0.4,1.4])        
    

    
    aoas  = panair_results.aoa
    machs = panair_results.mach    
    
    aoas  = aoas.flatten()
    machs = machs.flatten()    
    
    conditions              = Data()
    conditions.aerodynamics = Data()
    conditions.freestream   = Data()
    conditions.freestream.velocity          = np.atleast_2d(100.*np.ones_like(aoas))
    conditions.aerodynamics.angle_of_attack = np.atleast_2d(aoas).T
    conditions.freestream.mach_number       = np.atleast_2d(machs).T

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

def arrw_biconvex_vertical():
    
    # ------------------------------------------------------------------
    #   Initialize the Vehicle
    # ------------------------------------------------------------------

    vehicle = SUAVE.Vehicle()
    vehicle.tag = 'arrow_biconvex'   
    
    # basic parameters
    vehicle.reference_area = 198
    
    # ------------------------------------------------------------------
    #   Main Wing
    # ------------------------------------------------------------------

    wing = SUAVE.Components.Wings.Main_Wing()
    wing.tag = 'main_wing'    
    wing.aspect_ratio            = 2.44444/2
    wing.sweeps.leading_edge     = 60. * Units.degrees
    wing.thickness_to_chord      = 0.01
    wing.taper                   = 1./17.
    #wing.spans.projected         = 22.
    wing.spans.projected         = 11.

    wing.chords.root             = 17. * Units.meter
    wing.chords.tip              = 1. * Units.meter

    wing.areas.reference         = 198./2

    wing.twists.root             = 0.0 * Units.degrees
    wing.twists.tip              = 0.0 * Units.degrees

    wing.origin                  = [[0.,0.,0.]]
    wing.aerodynamic_center      = [0,0,0]

    wing.vertical                = True
    wing.symmetric               = False
    wing.high_lift               = False

    wing.dynamic_pressure_ratio  = 1.0    
    
    
    wing.sweeps.quarter_chord = convert_sweep(wing,old_ref_chord_fraction = 0.0,new_ref_chord_fraction = 0.25)    
    
    wing =  wing_planform(wing)
        
    vehicle.append_component(wing)
    
    vehicle.total_length = wing.total_length
    
    return vehicle


def arrw_biconvex():
    
    # ------------------------------------------------------------------
    #   Initialize the Vehicle
    # ------------------------------------------------------------------

    vehicle = SUAVE.Vehicle()
    vehicle.tag = 'arrow_biconvex'   
    
    # basic parameters
    vehicle.reference_area = 198
    
    # ------------------------------------------------------------------
    #   Main Wing
    # ------------------------------------------------------------------

    wing = SUAVE.Components.Wings.Main_Wing()
    wing.tag = 'main_wing'    
    wing.aspect_ratio            = 2.44444
    wing.sweeps.leading_edge     = 60. * Units.degrees
    wing.thickness_to_chord      = 0.01
    wing.taper                   = 1./17.
    ##wing.spans.projected         = 22.

    wing.chords.root             = 17. * Units.meter
    wing.chords.tip              = 1. * Units.meter

    wing.areas.reference         = 198

    wing.twists.root             = 0.0 * Units.degrees
    wing.twists.tip              = 0.0 * Units.degrees

    wing.origin                  = [[0.,0.,0.]]
    wing.aerodynamic_center      = [0,0,0]

    wing.vertical                = False
    wing.symmetric               = True
    wing.high_lift               = False

    wing.dynamic_pressure_ratio  = 1.0    
    
    
    wing.sweeps.quarter_chord = convert_sweep(wing,old_ref_chord_fraction = 0.0,new_ref_chord_fraction = 0.25)    
    
    wing =  wing_planform(wing)
        
    vehicle.append_component(wing)
    
    vehicle.total_length = wing.total_length
    
    return vehicle

def arrw_biconvex_vertical_dih():
    
    # ------------------------------------------------------------------
    #   Initialize the Vehicle
    # ------------------------------------------------------------------

    vehicle = SUAVE.Vehicle()
    vehicle.tag = 'arrow_biconvex'   
    
    # basic parameters
    vehicle.reference_area = 198
    
    # ------------------------------------------------------------------
    #   Main Wing
    # ------------------------------------------------------------------

    wing = SUAVE.Components.Wings.Main_Wing()
    wing.tag = 'main_wing'    
    wing.aspect_ratio            = 2.44444/2
    wing.sweeps.leading_edge     = 60. * Units.degrees
    wing.thickness_to_chord      = 0.01
    wing.taper                   = 1./17.
    wing.spans.projected         = 22./2
    wing.dihedral                = 30. * Units.degrees

    wing.chords.root             = 17. * Units.meter
    wing.chords.tip              = 1. * Units.meter

    wing.areas.reference         = 198/2

    wing.twists.root             = 0.0 * Units.degrees
    wing.twists.tip              = 0.0 * Units.degrees

    wing.origin                  = [[0.,0.,0.]]
    wing.aerodynamic_center      = [0,0,0]

    wing.vertical                = True
    wing.symmetric               = False
    wing.high_lift               = False

    wing.dynamic_pressure_ratio  = 1.0    
    
    
    wing.sweeps.quarter_chord = convert_sweep(wing,old_ref_chord_fraction = 0.0,new_ref_chord_fraction = 0.25)    
    
    wing =  wing_planform(wing)
        
    vehicle.append_component(wing)
    
    vehicle.total_length = wing.total_length
    
    return vehicle


def arrw_naca_twist():
    
    # ------------------------------------------------------------------
    #   Initialize the Vehicle
    # ------------------------------------------------------------------

    vehicle = SUAVE.Vehicle()
    vehicle.tag = 'arrow_naca_twist'   
    
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


def arrw_naca_twist_dih():
    
    # ------------------------------------------------------------------
    #   Initialize the Vehicle
    # ------------------------------------------------------------------

    vehicle = SUAVE.Vehicle()
    vehicle.tag = 'arrow_naca_twist_dihedral'   
    
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
    wing.dihedral                = 10. * Units.degrees

    wing.chords.root             = 17. * Units.meter
    wing.chords.tip              = 1. * Units.meter

    wing.areas.reference         = 198.

    wing.twists.root             = 0.0 * Units.degrees
    wing.twists.tip              = -2.0 * Units.degrees

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


def arrw_biconvex_dih():
    
    # ------------------------------------------------------------------
    #   Initialize the Vehicle
    # ------------------------------------------------------------------

    vehicle = SUAVE.Vehicle()
    vehicle.tag = 'arrow_biconvex'   
    
    # basic parameters
    vehicle.reference_area = 198
    
    # ------------------------------------------------------------------
    #   Main Wing
    # ------------------------------------------------------------------

    wing = SUAVE.Components.Wings.Main_Wing()
    wing.tag = 'main_wing'    
    wing.aspect_ratio            = 2.44444
    wing.sweeps.leading_edge     = 60. * Units.degrees
    wing.thickness_to_chord      = 0.01
    wing.taper                   = 1./17.
    wing.spans.projected         = 22.

    wing.chords.root             = 17. * Units.meter
    wing.chords.tip              = 1. * Units.meter

    wing.areas.reference         = 198.
    wing.dihedral                = 30. * Units.degrees

    wing.twists.root             = 0.0 * Units.degrees
    wing.twists.tip              = 0.0 * Units.degrees

    wing.origin                  = [[0.,0.,0.]]
    wing.aerodynamic_center      = [0,0,0]

    wing.vertical                = False
    wing.symmetric               = True
    wing.high_lift               = False

    wing.dynamic_pressure_ratio  = 1.0    
    
    
    wing.sweeps.quarter_chord = convert_sweep(wing,old_ref_chord_fraction = 0.0,new_ref_chord_fraction = 0.25)    
    
    wing =  wing_planform(wing)
        
    vehicle.append_component(wing)
    
    vehicle.total_length = wing.total_length
    
    return vehicle

def arrw_biconvex_twist_dih():
    
    # ------------------------------------------------------------------
    #   Initialize the Vehicle
    # ------------------------------------------------------------------

    vehicle = SUAVE.Vehicle()
    vehicle.tag = 'arrow_biconvex'   
    
    # basic parameters
    vehicle.reference_area = 198
    
    # ------------------------------------------------------------------
    #   Main Wing
    # ------------------------------------------------------------------

    wing = SUAVE.Components.Wings.Main_Wing()
    wing.tag = 'main_wing'    
    wing.aspect_ratio            = 2.44444
    wing.sweeps.leading_edge     = 60. * Units.degrees
    wing.thickness_to_chord      = 0.01
    wing.taper                   = 1./17.
    wing.spans.projected         = 22.

    wing.chords.root             = 17. * Units.meter
    wing.chords.tip              = 1. * Units.meter

    wing.areas.reference         = 198.
    wing.dihedral                = 30. * Units.degrees

    wing.twists.root             =  0.0 * Units.degrees
    wing.twists.tip              =  0.0 * Units.degrees

    wing.origin                  = [[0.,0.,0.]]
    wing.aerodynamic_center      = [0,0,0]

    wing.vertical                = False
    wing.symmetric               = True
    wing.high_lift               = False

    wing.dynamic_pressure_ratio  = 1.0    
    
    
    wing.sweeps.quarter_chord = convert_sweep(wing,old_ref_chord_fraction = 0.0,new_ref_chord_fraction = 0.25)    
    
    wing =  wing_planform(wing)
        
    vehicle.append_component(wing)
    
    vehicle.total_length = wing.total_length
    
    return vehicle




if __name__ == '__main__': 
    main()    
    plt.show()

## @ingroup Methods-Propulsion-Rotor_Wake-Fidelity_One
# fidelity_one_wake_convergence.py
#
# Created:  Feb 2022, R. Erhard
# Modified:


from SUAVE.Methods.Propulsion.Rotor_Wake.Fidelity_One.compute_fidelity_one_inflow_velocities import compute_fidelity_one_inflow_velocities
from SUAVE.Methods.Propulsion.Rotor_Wake.Fidelity_One.generate_fidelity_one_wake_shape import generate_fidelity_one_wake_shape
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.BET_calculations import compute_inflow_and_tip_loss, compute_airfoil_aerodynamics
import numpy as np

from DCode.Common.generalFunctions import savitzky_golay

## @ingroup Methods-Propulsion-Rotor_Wake-Fidelity_One
def fidelity_one_wake_convergence(wake,rotor,wake_inputs):
    """
    This converges on the wake shape for the fidelity-one rotor wake.

    Assumptions:
    None

    Source:
    N/A

    Inputs:
    wake        - rotor wake
    rotor       - rotor
    wake_inputs - inputs passed from the BET rotor spin function

    Outputs:
    None

    Properties Used:
    None
    """
    # Unpack inputs
    Ua = wake_inputs.velocity_axial
    Ut = wake_inputs.velocity_tangential
    r  = wake_inputs.radius_distribution

    R  = rotor.tip_radius
    Rh = rotor.hub_radius
    B  = rotor.number_of_blades

    beta = wake_inputs.twist_distribution
    c = wake_inputs.chord_distribution
    a = wake_inputs.speed_of_sounds
    nu = wake_inputs.dynamic_viscosities
    ctrl_pts = wake_inputs.ctrl_pts
    a_geo = rotor.airfoil_geometry
    a_loc = rotor.airfoil_polar_stations
    cl_sur = rotor.airfoil_cl_surrogates
    cd_sur = rotor.airfoil_cd_surrogates
    Nr = len(rotor.radius_distribution)
    Na = rotor.number_azimuthal_stations
    tc = rotor.thickness_to_chord

    # converge on (gamma, va) for a semi-prescribed wake method
    if wake.semi_prescribed_converge:
        if wake.verbose:
            print("\tConverging on semi-prescribed wake shape...")
        ii_max_g = wake.maximum_convergence_iteration_gamma
        ii_max_va = wake.maximum_convergence_iteration_va
    else:
        if wake.verbose:
            print("\tGenerating fully-prescribed wake shape...")
        ii_max_g  = 1
        ii_max_va = 1

    # DEBUG OVERLAYING PLOTS;
    #import pylab as plt
    #from DCode.Common.plottingFunctions import plotRotorDistributions
    #fig, (ax1,ax2,ax3) = plt.subplots(1,3)
    #fig.set_size_inches(14,4)

    #for l in range(1):

    # -------------------------------------------------------------------------------------------------------
    # Converge on disc circulation
    # -------------------------------------------------------------------------------------------------------
    print("\tCirculation convergence:")
    g_tol = wake.circulation_convergence_tolerance
    g_diff, g_ii = 1, 0
    while g_diff>g_tol:
        ## debug: plot rotor distributions
        #plotRotorDistributions(rotor, fig, ax1, ax2, ax3, figF, ax1F, ax2F, g_ii, ii_max)

        # generate wake geometry for rotor
        wake, rotor  = generate_fidelity_one_wake_shape(wake,rotor)

        # compute axial wake-induced velocity (a byproduct of the circulation distribution which is an input to the wake geometry)
        va, vt = compute_fidelity_one_inflow_velocities(wake,rotor,ctrl_pts)

        # compute new blade velocities
        Wa   = va + Ua
        Wt   = Ut - vt

        # compute blade forces
        lamdaw, F, _ = compute_inflow_and_tip_loss(r,R,Rh,Wa,Wt,B)
        Cl, Cdval, alpha, Ma, W, Re = compute_airfoil_aerodynamics(beta,c,r,R,B,F,Wa,Wt,a,nu,a_loc,a_geo,cl_sur,cd_sur,ctrl_pts,Nr,Na,tc,use_2d_analysis=True)

        # compute circulation at the blade
        Gamma = 0.5*W*c*Cl*F

        gfit = Gamma
        #gfit = np.zeros_like(r)
        #for cpt in range(ctrl_pts):
            ## FILTER OUTLIER DATA
            #for a in range(Na):
                #gPoly = np.poly1d(np.polyfit(r[0,:,0], Gamma[cpt,:,a], 4))
                #gfit[cpt,:,a] = F[cpt,:,a]*gPoly(r[0,:,0])

        ## debug plot
        #import pylab as plt
        #from scipy.interpolate import RectBivariateSpline
        #from DCode.Common.plottingFunctions import colorFader
        #psi = rotor.outputs.disc_azimuthal_distribution
        #plt.figure()
        #g_spline = RectBivariateSpline(r[0,:,0], psi[0,0,:], Gamma[0,:,:])
        #for a in range(Na):
            #bcol = colorFader("darkblue", "lightblue", mix=a/Na)
            #rcol = colorFader("darkred", "lightpink", mix=a/Na)
            #gcol = colorFader("darkgreen", "lightgreen", mix=a/Na)
            #plt.plot(r[0,:,0], gfit[0,:,a],'k-')
            #plt.plot(r[0,:,0],rotor.outputs.disc_circulation[0,:,a],gcol)

        #gfit = Gamma # DEBUG
        g_diff = np.max(abs(gfit - rotor.outputs.disc_circulation))
        print("\t\t"+str(g_diff))

        rotor.outputs.disc_circulation = rotor.outputs.disc_circulation + 0.25*(gfit - rotor.outputs.disc_circulation)

        g_ii+=1
        if g_ii >= ii_max_g and g_diff>g_tol:
            if wake.semi_prescribed_converge and wake.verbose:
                print("Semi-prescribed vortex wake did not converge on disc circulation for fidelity one wake.")
            break

    ## -------------------------------------------------------------------------------------------------------
    ## Converge on axial inflow distribution
    ## -------------------------------------------------------------------------------------------------------
    ##print("\tAxial inflow convergence:")
    #print("\tAxial inflow and vortex strength convergence:")
    #va_diff, va_ii = 1, 0
    #va_tol = wake.axial_velocity_convergence_tolerance
    #while va_diff > va_tol:
        ## generate wake geometry for rotor
        #wake, rotor  = generate_fidelity_one_wake_shape(wake,rotor)

        ## compute axial wake-induced velocity (a byproduct of the circulation distribution which is an input to the wake geometry)
        #va, vt = compute_fidelity_one_inflow_velocities(wake,rotor,ctrl_pts)

        ## compute new blade velocities
        #Wa   = va + Ua
        #Wt   = Ut - vt

        ## compute blade forces
        #lamdaw, F, _ = compute_inflow_and_tip_loss(r,R,Rh,Wa,Wt,B)
        #Cl, Cdval, alpha, Ma, W = compute_airfoil_aerodynamics(beta,c,r,R,B,F,Wa,Wt,a,nu,a_loc,a_geo,cl_sur,cd_sur,ctrl_pts,Nr,Na,tc,use_2d_analysis=True)

        ## compute circulation at the blade
        #Gamma = 0.5*W*c*Cl*F

        #gfit = np.zeros_like(r)
        #for cpt in range(ctrl_pts):
            ## FILTER OUTLIER DATA
            #for a in range(Na):
                #gPoly = np.poly1d(np.polyfit(r[0,:,0], Gamma[cpt,:,a], 4))
                #gfit[cpt,:,a] = F[cpt,:,a]*gPoly(r[0,:,0])
        ## filter va
        #ws, order = 7, 2
        #for cpt in range(ctrl_pts):
            ## FILTER OUTLIER DATA
            #for a in range(Na):
                #va[cpt,:,a] = savitzky_golay(va[cpt,:,a], ws, order)

        #vafit = np.zeros_like(r)
        #for cpt in range(ctrl_pts):
            ## FILTER OUTLIER DATA
            #for a in range(Na):
                #vaPoly = np.poly1d(np.polyfit(r[0,:,0], va[cpt,:,a], 4))
                #vafit[cpt,:,a] = F[cpt,:,a]*vaPoly(r[0,:,0])

        ##vafit = va
        #va_diff = np.max(abs(rotor.outputs.disc_axial_induced_velocity - vafit))
        #print("\t\tva_diff: "+str(va_diff))
        #gamma_diff = np.max(abs(rotor.outputs.disc_circulation - gfit))
        #print("\t\tgamma_diff: "+str(gamma_diff))
        #rotor.outputs.disc_axial_induced_velocity = rotor.outputs.disc_axial_induced_velocity + 0.25*(vafit - rotor.outputs.disc_axial_induced_velocity) #F*va #
        #rotor.outputs.disc_circulation = rotor.outputs.disc_circulation + 0.25*(gfit - rotor.outputs.disc_circulation)

        ### DEBUG
        ##from DCode.Common.generalFunctions import save_single_prop_vehicle_vtk
        ##save_single_prop_vehicle_vtk(rotor, time_step=va_ii, save_loc="/Users/rerha/Desktop/Debug_F1_vtk/")

        #va_ii+=1
        #if va_ii>=ii_max_va and va_diff>va_tol:
            #if wake.semi_prescribed_converge and wake.verbose:
                #print("Semi-prescribed vortex wake did not converge on axial inflow used for wake shape.")
            #break




    ## save converged wake:
    #wake, rotor  = generate_fidelity_one_wake_shape(wake,rotor)

    return wake.vortex_distribution, va, vt

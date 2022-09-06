## @ingroup Methods-Propulsion-Rotor_Wake-Fidelity_One
# fidelity_one_wake_convergence.py
#
# Created:  Feb 2022, R. Erhard
# Modified: 


from SUAVE.Methods.Propulsion.Rotor_Wake.Fidelity_One.compute_fidelity_one_inflow_velocities import compute_fidelity_one_inflow_velocities
from SUAVE.Methods.Propulsion.Rotor_Wake.Fidelity_One.generate_fidelity_one_wake_shape import generate_fidelity_one_wake_shape
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.BET_calculations import compute_inflow_and_tip_loss, compute_airfoil_aerodynamics
import numpy as np

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
    
    # converge on va for a semi-prescribed wake method
    va_diff, va_ii = 1, 0
    tol = wake.axial_velocity_convergence_tolerance
    if wake.semi_prescribed_converge:
        if wake.verbose:
            print("\tConverging on semi-prescribed wake shape...")
        ii_max = wake.maximum_convergence_iteration
    else:
        if wake.verbose:
            print("\tGenerating fully-prescribed wake shape...")
        ii_max = 1
     
    import pylab as plt
    from DCode.Common.plottingFunctions import plotRotorDistributions
    fig, (ax1,ax2,ax3) = plt.subplots(1,3)
    fig.set_size_inches(14,4)

    #while va_diff > tol:     
    g_diff, g_ii = 1, 0

    while g_diff>tol:
        # debug: plot rotor distributions
        plotRotorDistributions(rotor, fig, ax1, ax2, ax3, g_ii, ii_max)
        
        # generate wake geometry for rotor
        wake, rotor  = generate_fidelity_one_wake_shape(wake,rotor)
        
        # compute axial wake-induced velocity (a byproduct of the circulation distribution which is an input to the wake geometry)
        va, vt = compute_fidelity_one_inflow_velocities(wake,rotor)
    
        # compute new blade velocities
        Wa   = va + Ua
        Wt   = Ut - vt
    
        
        # generate new wake with new circulation
        # update disc circulation
        # compute HFW circulation at the blade
        Cl, Cdval, alpha, Ma, W = compute_airfoil_aerodynamics(beta,c,r,R,B,Wa,Wt,a,nu,a_loc,a_geo,cl_sur,cd_sur,ctrl_pts,Nr,Na,tc,use_2d_analysis=True)

        lamdaw, F, _ = compute_inflow_and_tip_loss(r,R,Wa,Wt,B)
        
        Gamma = 0.5*W*c*Cl*F
        g_diff = np.max(abs(Gamma - rotor.outputs.disc_circulation))
        print(g_diff)
        
        
        rotor.outputs.disc_circulation = rotor.outputs.disc_circulation + 0.25*(Gamma - rotor.outputs.disc_circulation)
        rotor.outputs.disc_axial_induced_velocity = F*va #rotor.outputs.disc_axial_induced_velocity + 0.5*(F*va - rotor.outputs.disc_axial_induced_velocity)
        rotor.outputs.disc_tangential_induced_velocity = F*vt #rotor.outputs.disc_axial_induced_velocity + 0.5*(F*va - rotor.outputs.disc_axial_induced_velocity)

        g_ii+=1
        if g_ii >= ii_max and g_diff>tol:
            if wake.semi_prescribed_converge and wake.verbose:
                print("Semi-prescribed vortex wake did not converge on disc circulation for fidelity one wake.")
            break               
           

        ## update the axial disc velocity based on new va from HFW
        #va_diff = np.max(abs(F*va - rotor.outputs.disc_axial_induced_velocity))
        #rotor.outputs.disc_axial_induced_velocity = F*va #rotor.outputs.disc_axial_induced_velocity + 0.5*(F*va - rotor.outputs.disc_axial_induced_velocity)
        #rotor.outputs.disc_tangential_induced_velocity = F*vt #rotor.outputs.disc_axial_induced_velocity + 0.5*(F*va - rotor.outputs.disc_axial_induced_velocity)
        
        #va_ii+=1
        #if va_ii>=ii_max and va_diff>tol:
            #if wake.semi_prescribed_converge and wake.verbose:
                #print("Semi-prescribed vortex wake did not converge on axial inflow used for wake shape.")
            #break
        
             
            
    #while g_diff>tol:
        #va_diff, va_ii = 1, 0
        
        #while va_diff > tol:  
            ## generate wake geometry for rotor
            #wake, rotor  = generate_fidelity_one_wake_shape(wake,rotor)
            
            ## compute axial wake-induced velocity (a byproduct of the circulation distribution which is an input to the wake geometry)
            #va, vt = compute_fidelity_one_inflow_velocities(wake,rotor)
        
            ## compute new blade velocities
            #Wa   = va + Ua
            #Wt   = Ut - vt
        
            #lamdaw, F, _ = compute_inflow_and_tip_loss(r,R,Wa,Wt,B)
        
            #va_diff = np.max(abs(F*va - rotor.outputs.disc_axial_induced_velocity))
        
            ## update the axial disc velocity based on new va from HFW
            #rotor.outputs.disc_axial_induced_velocity = rotor.outputs.disc_axial_induced_velocity + 0.5*(F*va - rotor.outputs.disc_axial_induced_velocity)
            
            
            #va_ii+=1
            #if va_ii>=ii_max and va_diff>tol:
                #if wake.semi_prescribed_converge and wake.verbose:
                    #print("Semi-prescribed vortex wake did not converge on axial inflow used for wake shape.")
                #break
            
        ## generate new wake with new circulation
        ## update disc circulation
        ## compute HFW circulation at the blade
        #Cl, Cdval, alpha, Ma, W = compute_airfoil_aerodynamics(beta,c,r,R,B,Wa,Wt,a,nu,a_loc,a_geo,cl_sur,cd_sur,ctrl_pts,Nr,Na,tc,use_2d_analysis=True)
        
        #Gamma = 0.5*W*c*Cl        
        #g_diff = np.max(abs(Gamma - rotor.outputs.disc_circulation))
        #print(g_diff)
        
        #rotor.outputs.disc_circulation = rotor.outputs.disc_circulation + 0.5*(Gamma - rotor.outputs.disc_circulation)
        
        #g_ii+=1
        #if g_ii >= ii_max and g_diff>tol:
            #if wake.semi_prescribed_converge and wake.verbose:
                #print("Semi-prescribed vortex wake did not converge on disc circulation for fidelity one wake.")
            #break        
    
    # save converged wake:
    wake, rotor  = generate_fidelity_one_wake_shape(wake,rotor)
    
    return wake.vortex_distribution, va, vt
## @ingroup Methods-Aerodynamics-Fidelity_Zero-Lift
# wing_compressibility_correction.py
# 
# Created:  Aug 2017, E. Botero
# Modified: 
#           

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np

# ----------------------------------------------------------------------
#  The Function
# ----------------------------------------------------------------------

## @ingroup Methods-Aerodynamics-Fidelity_Zero-Lift
def lifting_line(state,settings,geometry):
    """

    Assumptions:
    subsonic and unswept

    Source:
    Traub, L. W., Botero, E., Waghela, R., Callahan, R., & Watson, A. (2015). Effect of Taper Ratio at Low Reynolds Number. Journal of Aircraft.
    
    Inputs:
    N/A

    Outputs:
    N/A

    Properties Used:
    N/A
    """  
    
    # Unpack first round:
    wing        = geometry
    freestream  = state.conditions.freestream
    orientation = wing.vertical    
    
    # Don't bother doing the calculation if it is a vertical tail
    if orientation == True:
        Cl = 0.0
        Cd = 0.0
        return Cl, Cd
    else:
        pass
        
    # Potentially useful unpacks
    #sweep       = wing.sweeps.quarter_chord
    #sym_para    = wing.symmetric
    #

    # Unpack fo'real
    b     = wing.spans.projected
    S     = wing.areas.reference
    AR    = wing.aspect_ratio
    MAC   = wing.chords.mean_aerodynamic
    r     = settings.number_of_stations # Number of divisions
    rho   = freestream.density # Freestream density
    V     = freestream.velocity # Freestream velocity
    mu    = freestream.dynamic_viscosity # Freestream viscosity
    alpha = state.conditions.aerodynamics.angle_of_attack
    
    repeats = np.size(alpha)
    
    # Need to set to something
    cla   = 2 * np.pi # 2-D lift curve slope
    azl   = 0. # 2-D 


    # Start doing calculations
    N      = r-1                  # number of spanwise divisions
    n      = np.linspace(1,N,N)   # vectorize
    thetan = n*np.pi/r            # angular stations
    yn     = -b*np.cos(thetan)/2. # y locations based on the angular spacing
    etan   = np.abs(2.*yn/b)       # normalized coordinates
    etam   = np.pi*np.sin(thetan)/(2*r) # Useful mulitplier

    
    # Project the spanwise y locations into the chords
    # If spanwise stations are setup
    if len(wing.Segments.keys())>0:
 
        c    = None
        ageo = None
    
    # Spanwise stations are not setup
    else:
        # Use the taper ratio to determine the chord distribution
        # Use the geometric twist applied to the ends to    
        
        # unpack
        taper       = wing.taper
        tip_twist   = wing.twists.root
        root_twist  = wing.twists.tip 
        root_chord  = wing.chords.root
        tip_chord   = wing.chords.tip  
        
        c    = root_chord+root_chord*(taper-1.)*etan
        ageo = (tip_twist-root_twist)*etan+root_twist

    
    
    k = c*cla/(4.*b) # Grouped term 
    
    n_trans = np.atleast_2d(n).T
        
    # Right hand side matrix
    RHS = (np.sin(n_trans*thetan)*(np.sin(thetan)+n_trans*k))
    
    # Expand out for all the angles of attack
    RHS2 =  np.tile(RHS, (repeats,1,1))

    # Left hand side vector    
    LHS = (k*np.sin(thetan)*(alpha+ageo-azl))
        
    # The Fourier Coefficients
    A = np.linalg.solve(RHS2,LHS)
    
    # The 3-D Coefficient of lift
    CL = A[:,0]*np.pi*AR
    
    # Find the sectional coefficients of lift
    Cl = b*np.cumsum(4*A*np.sin(n*thetan),axis=1)/c
    
    # induced alpha
    alpha_i = np.cumsum(n*A*np.sin(n*A)/np.sin(thetan),axis=1)
    
    # Sectional vortex drag
    Cdv = Cl*alpha_i
    
    # Total vortex drag
    CDv = np.sum(Cdv*AR*etam,axis=1)
    
    #############
    # Profile drag of a 2-D section
    Cdn = 0.01
    #############
    
    # Find the profile drag
    CDp = np.sum(Cdn*c*etam)/MAC
    
    CD  = CDv + CDp
   
    return CL, CD


# ----------------------------------------------------------------------
#   Unit Tests
# ----------------------------------------------------------------------
# this will run from command line, put simple tests for your code here
if __name__ == '__main__':
    
    import SUAVE
    from SUAVE.Core import Data, Units
    
    
    # ------------------------------------------------------------------        
    #   State
    # ------------------------------------------------------------------           
    
    state = Data()
    state.conditions = Data()
    state.conditions.freestream = Data()
    state.conditions.aerodynamics = Data()
    
    atmosphere = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    atmosphere_conditions = atmosphere.compute_values(0.)
    state.conditions.freestream.update(atmosphere_conditions)
    state.conditions.freestream.velocity = np.atleast_2d([100., 100.]).T
    state.conditions.aerodynamics.angle_of_attack = np.atleast_2d([2. * Units.deg, 2. * Units.deg]).T

    # ------------------------------------------------------------------        
    #   Main Wing
    # ------------------------------------------------------------------        
    
    wing = SUAVE.Components.Wings.Main_Wing()
    wing.tag = 'main_wing'
    
    wing.aspect_ratio            = 10.18
    wing.sweeps.quarter_chord    = 25 * Units.deg
    wing.thickness_to_chord      = 0.1
    wing.taper                   = 0.1
    wing.span_efficiency         = 0.9
    
    wing.spans.projected         = 34.32   
    
    wing.chords.root             = 7.760 * Units.meter
    wing.chords.tip              = 0.782 * Units.meter
    wing.chords.mean_aerodynamic = 4.235 * Units.meter
    
    wing.areas.reference         = 124.862 
    
    wing.twists.root             = 4.0 * Units.degrees
    wing.twists.tip              = 0.0 * Units.degrees
    
    wing.origin                  = [13.61,0,-1.27]
    wing.aerodynamic_center      = [0,0,0]  #[3,0,0]
    
    wing.vertical                = False
    wing.symmetric               = True
    wing.high_lift               = True
    
    wing.dynamic_pressure_ratio  = 1.0    
    
    geometry = wing
    
    # ------------------------------------------------------------------        
    #   Main Wing
    # ------------------------------------------------------------------      
    
    settings = Data()
    settings.number_of_stations = 4
    
    
    
    CL, CD = lifting_line(state, settings, geometry)
    
    print 'CL: '
    print CL
    print 'CD: '
    print CD
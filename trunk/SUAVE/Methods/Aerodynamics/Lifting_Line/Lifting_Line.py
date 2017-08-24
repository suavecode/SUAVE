## @ingroup Methods-Aerodynamics-Lifting_line
# Lifting_Line.py
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

## @ingroup Methods-Aerodynamics-Lifting_line
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
        CL = 0.0
        CD = 0.0
        return CL, CD
    else:
        pass
        
    # Potentially useful unpacks
    #sweep       = wing.sweeps.quarter_chord


    # Unpack fo'real
    b     = wing.spans.projected
    S     = wing.areas.reference
    AR    = wing.aspect_ratio
    MAC   = wing.chords.mean_aerodynamic
    sym   = wing.symmetric
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
    N      = r-1                        # number of spanwise divisions
    n      = np.linspace(1,N,N)         # vectorize
    thetan = n*np.pi/r                  # angular stations
    yn     = -b*np.cos(thetan)/2.       # y locations based on the angular spacing
    etan   = np.abs(2.*yn/b)            # normalized coordinates
    etam   = np.pi*np.sin(thetan)/(2*r) # Useful mulitplier
    
    # Project the spanwise y locations into the chords
    # If spanwise stations are setup
    if len(wing.Segments.keys())>0:
        c    = np.empty_like(etan)
        ageo = np.empty_like(etan)
        for i_segs in xrange(n_segments):
            # Figure out where the segment ends
            
            # For the range of etan's that exist add in the corresponding chord values
            
            # For the range of etan's that exist add in the corresponding twist values

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
        
        # Find the chords and twist profile
        c    = root_chord+root_chord*(taper-1.)*etan
        ageo = (tip_twist-root_twist)*etan+root_twist

    
    
    k = c*cla/(4.*b) # Grouped term 
    
    n_trans = np.atleast_2d(n).T
        
    # Right hand side matrix
    RHS = (np.sin(n_trans*thetan)*(np.sin(thetan)+n_trans*k))
    
    # Expand out for all the angles of attack
    RHS2 =  np.tile(RHS, (repeats,1,1))

    # Left hand side vector    
    LHS = k*np.sin(thetan)*(alpha+ageo-azl)
        
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

    wing.aspect_ratio            = 289.**2 / (7840. * 2)
    wing.thickness_to_chord      = 0.15
    wing.taper                   = 0.0138
    wing.span_efficiency         = 0.95
    
    wing.spans.projected         = 289.0 * Units.feet    

    wing.chords.root             = 145.0 * Units.feet
    wing.chords.tip              = 3.5  * Units.feet
    wing.chords.mean_aerodynamic = 80. * Units.feet

    wing.areas.reference         = 7840. * 2 * Units.feet**2
    wing.sweeps.quarter_chord    = 33. * Units.degrees

    wing.twists.root             = 0.0 * Units.degrees
    wing.twists.tip              = 0.0 * Units.degrees
    wing.dihedral                = 2.5 * Units.degrees

    wing.origin                  = [0.,0.,0]
    wing.aerodynamic_center      = [0,0,0] 

    wing.vertical                = False
    wing.symmetric               = True
    wing.high_lift               = True

    wing.dynamic_pressure_ratio  = 1.0

    segment = SUAVE.Components.Wings.Segment()
    segment.tag                   = 'section_1'
    segment.percent_span_location = 0.0
    segment.twist                 = 0. * Units.deg
    segment.root_chord_percent    = 1.
    segment.dihedral_outboard     = 0. * Units.degrees
    segment.sweeps.quarter_chord  = 30.0 * Units.degrees
    segment.thickness_to_chord    = 0.165 
    wing.Segments.append(segment)    
    
    segment = SUAVE.Components.Wings.Segment()
    segment.tag                   = 'section_2'
    segment.percent_span_location = 0.052
    segment.twist                 = 0. * Units.deg
    segment.root_chord_percent    = 0.921
    segment.dihedral_outboard     = 0.   * Units.degrees
    segment.sweeps.quarter_chord  = 52.5 * Units.degrees
    segment.thickness_to_chord    = 0.167    
    wing.Segments.append(segment)   

    segment = SUAVE.Components.Wings.Segment()
    segment.tag                   = 'section_3'
    segment.percent_span_location = 0.138
    segment.twist                 = 0. * Units.deg
    segment.root_chord_percent    = 0.76
    segment.dihedral_outboard     = 1.85 * Units.degrees
    segment.sweeps.quarter_chord  = 36.9 * Units.degrees  
    segment.thickness_to_chord    = 0.171    
    wing.Segments.append(segment)   
    
    segment = SUAVE.Components.Wings.Segment()
    segment.tag                   = 'section_4'
    segment.percent_span_location = 0.221
    segment.twist                 = 0. * Units.deg
    segment.root_chord_percent    = 0.624
    segment.dihedral_outboard     = 1.85 * Units.degrees
    segment.sweeps.quarter_chord  = 30.4 * Units.degrees    
    segment.thickness_to_chord    = 0.175  
    wing.Segments.append(segment)       
    
    segment = SUAVE.Components.Wings.Segment()
    segment.tag                   = 'section_5'
    segment.percent_span_location = 0.457
    segment.twist                 = 0. * Units.deg
    segment.root_chord_percent    = 0.313
    segment.dihedral_outboard     = 1.85  * Units.degrees
    segment.sweeps.quarter_chord  = 30.85 * Units.degrees
    segment.thickness_to_chord    = 0.118
    wing.Segments.append(segment)       
    
    segment = SUAVE.Components.Wings.Segment()
    segment.tag                   = 'section_6'
    segment.percent_span_location = 0.568
    segment.twist                 = 0. * Units.deg
    segment.root_chord_percent    = 0.197
    segment.dihedral_outboard     = 1.85 * Units.degrees
    segment.sweeps.quarter_chord  = 34.3 * Units.degrees
    segment.thickness_to_chord    = 0.10
    wing.Segments.append(segment)     
    
    segment = SUAVE.Components.Wings.Segment()
    segment.tag                   = 'section_7'
    segment.percent_span_location = 0.97
    segment.twist                 = 0. * Units.deg
    segment.root_chord_percent    = 0.086
    segment.dihedral_outboard     = 73. * Units.degrees
    segment.sweeps.quarter_chord  = 55. * Units.degrees
    segment.thickness_to_chord    = 0.10
    wing.Segments.append(segment)      
    

    ## ------------------------------------------------------------------        
    ##   Main Wing
    ## ------------------------------------------------------------------        
    
    #wing = SUAVE.Components.Wings.Main_Wing()
    #wing.tag = 'main_wing'
    
    #wing.aspect_ratio            = 10.18
    #wing.sweeps.quarter_chord    = 25 * Units.deg
    #wing.thickness_to_chord      = 0.1
    #wing.taper                   = 0.1
    #wing.span_efficiency         = 0.9
    
    #wing.spans.projected         = 34.32   
    
    #wing.chords.root             = 7.760 * Units.meter
    #wing.chords.tip              = 0.782 * Units.meter
    #wing.chords.mean_aerodynamic = 4.235 * Units.meter
    
    #wing.areas.reference         = 124.862 
    
    #wing.twists.root             = 4.0 * Units.degrees
    #wing.twists.tip              = 0.0 * Units.degrees
    
    #wing.origin                  = [13.61,0,-1.27]
    #wing.aerodynamic_center      = [0,0,0]  #[3,0,0]
    
    #wing.vertical                = False
    #wing.symmetric               = True
    #wing.high_lift               = True
    
    #wing.dynamic_pressure_ratio  = 1.0    
    
    geometry = wing
    
    # ------------------------------------------------------------------        
    #   Settings
    # ------------------------------------------------------------------      
    
    settings = Data()
    settings.number_of_stations = 4
    
    
    
    CL, CD = lifting_line(state, settings, geometry)
    
    print 'CL: '
    print CL
    print 'CD: '
    print CD
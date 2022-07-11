## @ingroup Methods-Weights-Buildups-Common

# wing.py
#
# Created:  Jun 2017, J. Smart
# Modified: Apr 2018, J. Smart
#           Mar 2020, M. Clarke
#           Mar 2020, J. Smart
#           May 2021, M. Clarke

#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------

from SUAVE.Core import Units
from SUAVE.Attributes.Solids import (
    Bidirectional_Carbon_Fiber, Carbon_Fiber_Honeycomb, Paint, Unidirectional_Carbon_Fiber, Aluminum, Epoxy, Aluminum_Rib)
import numpy as np
import copy as cp

#-------------------------------------------------------------------------------
# Wing
#-------------------------------------------------------------------------------

## @ingroup Methods-Weights-Buildups-Common
def wing(wing,
         config,
         max_thrust,
         num_analysis_points = 10,
         safety_factor = 1.5,
         max_g_load = 3.8,
         moment_to_lift_ratio = 0.02,
         lift_to_drag_ratio = 7,
         forward_web_locations = [0.25, 0.35],
         rear_web_locations = [0.65, 0.75],
         shear_center_location = 0.25,
         margin_factor = 1.2):
    
    """weight = SUAVE.Methods.Weights.Buildups.Common.wing(
            wing,
            config,
            maxThrust,
            numAnalysisPoints,
            safety_factor,
            max_g_load,
            moment_to_lift_ratio,
            lift_to_drag_ratio,
            forward_web_locations = [0.25, 0.35],
            rear_web_locations = [0.65, 0.75],
            shear_center = 0.25,
            margin_factor = 1.2)

        Calculates the structural mass of a wing for an eVTOL vehicle based on
        assumption of NACA airfoil wing, an assumed L/D, cm/cl, and structural
        geometry.

        Intended for use with the following SUAVE vehicle types, but may be used
        elsewhere:

            Electric Multicopter
            Electric Vectored_Thrust
            Electric Stopped Rotor

        Originally written as part of an AA 290 project intended for trade study
        of the above vehicle types plus an electric multicopter.

        If no materials are assigned to vehicle model, appropriate assumptions
        are made based on SUAVE's Solids Attributes library.
        
        Sources:
        Project Vahana Conceptual Trade Study

        Inputs:

            wing                          SUAVE Wing Data Structure           [None]
                 winglet_fraction         winglet fraction                    [Unitless]
                 motor_spanwise_locations spanwise fraction location of motor [Unitless]
            config                        SUAVE Config Data Structure         [None]
            maxThrust                     Maximum Thrust                      [N]
            numAnalysisPoints             Analysis Points for Sizing          [Unitless]
            safety_factor                 Design Safety Factor                [Unitless]
            max_g_load                    Maximum Accelerative Load           [Unitless]
            moment_to_lift_ratio          Coeff. of Moment to Coeff. of Lift  [Unitless]
            lift_to_drag_ratio            Coeff. of Lift to Coeff. of Drag    [Unitless]
            forward_web_locations         Location of Forward Spar Webbing    [m]
            rear_web_locations            Location of Rear Spar Webbing       [m]
            shear_center                  Location of Shear Center            [m]
            margin_factor                 Allowable Extra Mass Fraction       [Unitless]

        Outputs:

            weight:                       Wing Mass                           [kg]
    """

    #-------------------------------------------------------------------------------
    # Unpack Inputs
    #-------------------------------------------------------------------------------

    MTOW                        = config.mass_properties.max_takeoff
    wingspan                    = wing.spans.projected
    chord                       = wing.chords.mean_aerodynamic
    thicknessToChord            = wing.thickness_to_chord
    wingArea                    = wing.areas.reference

    try:
        wingletFraction = wing.winglet_fraction
    except AttributeError:
        wingletFraction = 0.0

    totalWingArea = 0
    for w in config.wings:
        totalWingArea += w.areas.reference
    liftFraction                    = wingArea/totalWingArea

    try:
        motor_spanwise_locations    = wing.motor_spanwise_locations
    except AttributeError:
        motor_spanwise_locations    = []

    N       = num_analysis_points                   # Number of spanwise points
    SF      = safety_factor                         # Safety Factor
    G_max   = max_g_load                            # Maximum G's experienced during climb
    cmocl   = moment_to_lift_ratio                  # Ratio of cm to cl
    LoD     = lift_to_drag_ratio                    # L/D
    fwdWeb  = cp.deepcopy(forward_web_locations)    # Locations of forward spars
    aftWeb  = cp.deepcopy(rear_web_locations)       # Locations of aft spars
    xShear  = shear_center_location                 # Approximate shear center
    grace   = margin_factor                         # Grace factor for estimation

    nRibs = len(motor_spanwise_locations) + 2
    motor_spanwise_locations = np.multiply(motor_spanwise_locations,wingspan/2)

    #-------------------------------------------------------------------------------
    # Unpack Material Properties
    #-------------------------------------------------------------------------------

    try:
        torsMat = wing.materials.skin_materials.torsion_carrier
    except AttributeError:
        torsMat = Bidirectional_Carbon_Fiber()
    torsUSS = torsMat.ultimate_shear_strength
    torsDen = torsMat.density
    torsMGT = torsMat.minimum_gage_thickness

    try:
        coreMat = wing.materials.skin_materials.core
    except AttributeError:
        coreMat = Carbon_Fiber_Honeycomb()
    coreDen = coreMat.density
    coreMGT = coreMat.minimum_gage_thickness

    try:
        bendMat = wing.materials.flap_materials.bending_carrier
    except AttributeError:
        bendMat = Unidirectional_Carbon_Fiber()
    bendUTS = bendMat.ultimate_tensile_strength
    bendDen = bendMat.density

    try:
        glueMat = wing.materials.skin_materials.adhesive
    except AttributeError:
        glueMat = Epoxy()
    glueMGT = glueMat.minimum_gage_thickness
    glueDen = glueMat.density

    try:
        coverMat = wing.materials.skin_materials.covering
    except AttributeError:
        coverMat = Paint()
    coverMGT = coverMat.minimum_gage_thickness
    coverDen = coverMat.density

    try:
        ribMat = wing.rib_materials.structural
    except AttributeError:
        ribMat = Aluminum_Rib()
    ribWid = ribMat.minimum_width
    ribMGT = ribMat.minimum_gage_thickness
    ribDen = ribMat.density

    try:
        shearMat = wing.spar_materials.shear_carrier
    except AttributeError:
        shearMat = Bidirectional_Carbon_Fiber()
    shearMGT = shearMat.minimum_gage_thickness
    shearDen = shearMat.density
    shearUSS = shearMat.ultimate_shear_strength


    #-------------------------------------------------------------------------------
    # Airfoil
    #-------------------------------------------------------------------------------

    NACA = np.multiply(5 * thicknessToChord, [0.2969, -0.1260, -0.3516, 0.2843, -0.1015])
    coord = np.unique(fwdWeb+aftWeb+np.linspace(0, 1, N).tolist())[:, np.newaxis]
    coordMAT = np.concatenate((coord**0.5, coord, coord**2, coord**3, coord**4), axis=1)
    nacaMAT = coordMAT.dot(NACA)[:, np.newaxis]
    coord = np.concatenate((coord, nacaMAT), axis=1)
    coord = np.concatenate((coord[-1:0:-1], coord.dot(np.array([[1., 0.], [0., -1.]]))), axis=0)
    coord[:, 0] = coord[:, 0] - xShear

    #-------------------------------------------------------------------------------
    # Beam Geometry
    #-------------------------------------------------------------------------------

    x = np.concatenate((np.linspace(0, 1, N), np.linspace(1, 1+wingletFraction, N)), axis=0)
    x = x * wingspan/2
    x = np.sort(np.concatenate((x,motor_spanwise_locations), axis=0))
    dx = x[1] - x[0]
    N = np.size(x)
    fwdWeb[:] = [round(locFwd - xShear, 2) for locFwd in fwdWeb]
    aftWeb[:] = [round(locAft - xShear, 2) for locAft in aftWeb]

    #-------------------------------------------------------------------------------
    # Loads
    #-------------------------------------------------------------------------------

    L = (1-(x/np.max(x))**2)**0.5           # Assumes Elliptic Lift Distribution
    L0 = 0.5*G_max*MTOW*9.8*liftFraction*SF # Total Design Lift Force
    L = L0/np.sum(L[0:-1]*np.diff(x))*L     # Net Lift Distribution

    T = L * chord * cmocl                # Torsion Distribution
    D = L/LoD                               # Drag Distribution

    #-------------------------------------------------------------------------------
    # Shear/Moments
    #-------------------------------------------------------------------------------

    Vx = np.append(np.cumsum((D[0:-1]*np.diff(x))[::-1])[::-1], 0)   # Drag Shear
    Vz = np.append(np.cumsum((L[0:-1]*np.diff(x))[::-1])[::-1], 0)   # Lift Shear
    Vt = 0 * Vz                                        # Initialize Thrust Shear

    # Calculate shear due to thrust by adding thrust from each motor to each
    # analysis point that's closer to the wing root than the motor. Accomplished
    # by indexing Vt according to a boolean mask of the design points that area
    # less than or aligned with the motor location under consideration in an
    # iterative loop

    for i in range(np.size(motor_spanwise_locations)):
        Vt[x<=motor_spanwise_locations[i]] = Vt[x<=motor_spanwise_locations[i]] + max_thrust

    Mx = np.append(np.cumsum((Vz[0:-1]*np.diff(x))[::-1])[::-1],0)  # Bending Moment
    My = np.append(np.cumsum(( T[0:-1]*np.diff(x))[::-1])[::-1],0)  # Torsion Moment
    Mz = np.append(np.cumsum((Vx[0:-1]*np.diff(x))[::-1])[::-1],0)  # Drag Moment
    Mt = np.append(np.cumsum((Vt[0:-1]*np.diff(x))[::-1])[::-1],0)  # Thrust Moment
    Mz = np.max((Mz, Mt))    # Worst Case of Drag vs. Thrust Moment

    #-------------------------------------------------------------------------------
    # General Structural Properties
    #-------------------------------------------------------------------------------

    seg = []                        # LIST of Structural Segments

    # Torsion

    box = coord                        # Box Initally Matches Airfoil
    box = box[box[:, 0] <= aftWeb[1]]  # Inlcude Only Parts Fwd of Aftmost Spar
    box = box[box[:, 0] >= fwdWeb[0]]  # Include Only Parts Aft of Fwdmost Spar
    box = box * chord                  # Scale by Chord Length

    # Use Shoelace Formula to calculate box area

    torsionArea = 0.5*np.abs(np.dot(box[:, 0], np.roll(box[:, 1], 1)) -
        np.dot(box[:, 1], np.roll(box[:, 0], 1)))

    torsionLength = np.sum(np.sqrt(np.sum(np.diff(box, axis=0)**2, axis=1)))

    # Bending

    box = coord                        # Box Initially Matches Airfoil
    box = box[box[:, 0] <= fwdWeb[1]]  # Include Only Parts Fwd of Aft Fwd Spar
    box = box[box[:, 0] >= fwdWeb[0]]  # Include Only Parts Aft of Fwdmost Spar
    seg.append(box[box[:, 1] > np.mean(box[:, 1])]*chord)   # Upper Fwd Segment
    seg.append(box[box[:, 1] < np.mean(box[:, 1])]*chord)   # Lower Fwd Segment

    # Drag

    box = coord                        # Box Initially Matches Airfoil
    box = box[box[:, 0] <= aftWeb[1]]  # Include Only Parts Fwd of Aftmost Spar
    box = box[box[:, 0] >= aftWeb[0]]  # Include Only Parts Aft of Fwd Aft Spar
    seg.append(box[box[:, 1] > np.mean(box[:, 1])]*chord)   # Upper Aft Segment
    seg.append(box[box[:, 1] < np.mean(box[:, 1])]*chord)   # Lower Aft Segment

    # Bending/Drag Inertia

    flapInertia = 0
    flapLength  = 0
    dragInertia = 0
    dragLength  = 0

    for i in range(0, 4):
        l = np.sqrt(np.sum(np.diff(seg[i], axis=0)**2, axis=1))    # Segment lengths
        c = (seg[i][1::]+seg[i][0:-1])/2                         # Segment centroids

        if i<2:
            flapInertia += np.abs(np.sum(l*c[:,1]**2))   # Bending Inertia per Unit Thickness
            flapLength  += np.sum(l)
        else:
            dragInertia += np.abs(np.sum(l*c[:,0]**2))   # Drag Inertia per Unit Thickness
            dragLength  += np.sum(l)


    # Shear

    box = coord                     # Box Initially Matches Airfoil
    box = box[box[:,0]<=fwdWeb[1]]  # Include Only Parts Fwd of Aft Fwd Spar
    z = np.zeros(2)
    z[0] = np.interp(fwdWeb[0], box[box[:, 1] > 0,0],box[box[:,1] > 0,1])*chord  # Upper Surf of Box at Fwdmost Spar
    z[1] = np.interp(fwdWeb[0], box[box[:, 1] < 0,0],box[box[:,1] < 0,1])*chord  # Lower Surf of Box at Fwdmost Spar
    h = np.abs(z[0] - z[1])         # Height of Box at Fwdmost Spar

    # Skin

    box = coord * chord             # Box Initially is Airfoil Scaled by Chord
    skinLength = np.sum(np.sqrt(np.sum(np.diff(box, axis=0)**2, axis=1)))
    A = 0.5*np.abs(np.dot(box[:,0],np.roll(box[:, 1], 1)) -
        np.dot(box[:, 1], np.roll(box[:, 0], 1)))   # Box Area via Shoelace Formula

    #---------------------------------------------------------------------------
    # Structural Calculations
    #---------------------------------------------------------------------------

    # Calculate Skin Weight Based on Torsion

    tTorsion = My*dx/(2*torsUSS*torsionArea)                # Torsion Skin Thickness
    tTorsion = np.maximum(tTorsion,torsMGT*np.ones(N))      # Gage Constraint
    mTorsion = tTorsion * torsionLength * torsDen           # Torsion Mass
    mCore = coreMGT*torsionLength*coreDen*np.ones(N)        # Core Mass
    mGlue = glueMGT*glueDen*torsionLength*np.ones(N)        # Epoxy Mass

    # Calculate Flap Mass Based on Bending

    tFlap = Mx*np.max(seg[0][:,1])/(flapInertia*bendUTS)    # Bending Flap Thickness
    mFlap = tFlap*flapLength*bendDen                        # Bending Flap Mass
    mGlue += glueMGT*glueDen*flapLength*np.ones(N)          # Updated Epoxy Mass

    # Calculate Drag Flap Mass

    tDrag = Mz*np.max(seg[2][:,0])/(dragInertia*bendUTS)    # Drag Flap Thickness
    mDrag = tDrag*dragLength*bendDen                        # Drag Flap Mass
    mGlue += glueMGT*glueDen*dragLength*np.ones(N)          # Updated Epoxy Mass

    # Calculate Shear Spar Mass

    tShear = 1.5*Vz/(shearUSS*h)                            # Shear Spar Thickness
    tShear = np.maximum(tShear, shearMGT*np.ones(N))        # Gage constraint
    mShear = tShear*h*shearDen                              # Shear Spar Mass

    # Paint

    mPaint = skinLength*coverMGT*coverDen*np.ones(N)        # Paint Mass

    # Section Mass Total

    m = mTorsion + mCore + mFlap + mDrag + mShear + mGlue + mPaint

    # Rib Mass

    mRib = (A+skinLength*ribWid)*ribMGT*ribDen

    # Total Mass

    mass = 2*(sum(m[0:-1]*np.diff(x))+nRibs*mRib)*grace

    return mass
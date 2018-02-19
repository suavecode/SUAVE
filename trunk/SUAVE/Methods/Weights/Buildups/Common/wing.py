# wing.py
#
# Created: Jun 2017, J. Smart

#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------

from SUAVE.Core import Units
from SUAVE.Attributes.Solids import (
    BiCF, Honeycomb, Paint, UniCF, Acrylic, Steel, Aluminum, Epoxy, Nickel, Rib)
import numpy as np
import copy as cp

#-------------------------------------------------------------------------------
# Wing
#-------------------------------------------------------------------------------

def wing(wing,
         config,
         maxThrust,
         numAnalysisPoints = 10,
         safetyFactor = 1.5,
         maxGLoad = 3.8,
         momentToLiftRatio = 0.02,
         liftToDragRatio = 7,
         forwardWeb = [0.25, 0.35],
         rearWeb = [0.65, 0.75],
         shearCenter = 0.25,
         marginFactor = 1.2):
    
    """weight = SUAVE.Methods.Weights.Correlations.eHelicopter.wing(
            MTOW,
            wingspan,
            chord,
            thicknessToChord,
            wingletFraction,
            liftFraction,
            xMotor,
            maxThrust
        )

        Calculates the structural mass of a wing for an eVTOL vehicle based on
        assumption of NACA airfoil wing, an assumed L/D, cm/cl, and structural
        geometry.

        Intended for use with the following SUAVE vehicle types, but may be used
        elsewhere:

            eTiltwing
            eTiltrotor
            eStopped_Rotor

        Originally written as part of an AA 290 project intended for trade study
        of the above vehicle types plus an eHelicopter.

        Inputs:

            MTOW:               Maximum TO Weight       [kg]
            wingspan:           Wingspan                [m]
            chord:              Wing Chord              [m]
            wingletFraction:    Winglet Length/Wingspan [Unitless]
            thicknessToChord:   Wing t/c Ratio          [Unitless]
            liftFraction:       Fraction of Total Lift  [Unitless]
            xMotor:             Motor Span Fractions    [Unitless]
            maxThrust:          Maximum Motor Thrust    [N]

        Outputs:

            weight:             Wing Mass               [kg]
    """

#-------------------------------------------------------------------------------
# Unpack Inputs
#-------------------------------------------------------------------------------

    MTOW                = config.mass_properties.max_takeoff
    wingspan            = wing.spans.projected
    chord               = wing.chords.mean_aerodynamic,
    thicknessToChord    = wing.thickness_to_chord, 
    wingletFraction     = wing.winglet_fraction, 
    wingArea            = wing.areas.reference
    totalWingArea       = (config.wings['main_wing'].areas.reference + 
                           config.wings['secondary_wing'].areas.reference)
    liftFraction        = wingArea/totalWingArea
    xMotor              = wing.xMotor

    N       = numAnalysisPoints             # Number of spanwise points
    SF      = safetyFactor                  # Safety Factor
    G_max   = maxGLoad                      # Maximum G's experienced during climb
    cmocl   = momentToLiftRatio             # Ratio of cm to cl
    LoD     = liftToDragRatio               # L/D
    fwdWeb  = cp.deepcopy(forwardWeb)       # Locations of forward spars
    aftWeb  = cp.deepcopy(rearWeb)          # Locations of aft spars
    xShear  = shearCenter                   # Approximate shear center
    grace   = marginFactor                  # Grace factor for estimation

    nRibs = len(xMotor) + 2
    xMotor = np.multiply(xMotor,wingspan/2)

#-------------------------------------------------------------------------------
# Airfoil
#-------------------------------------------------------------------------------

    NACA = np.multiply(5 * thicknessToChord, [0.2969, -0.1260, -0.3516, 0.2843, -0.1015])
    coord = np.unique(fwdWeb+aftWeb+np.linspace(0,1,N).tolist())[:,np.newaxis]
    coordMAT = np.concatenate((coord**0.5,coord,coord**2,coord**3,coord**4),axis=1)
    nacaMAT = coordMAT.dot(NACA)[:, np.newaxis]
    coord = np.concatenate((coord,nacaMAT),axis=1)
    coord = np.concatenate((coord[-1:0:-1],coord.dot(np.array([[1.,0.],[0.,-1.]]))),axis=0)
    coord[:,0] = coord[:,0] - xShear

#-------------------------------------------------------------------------------
# Beam Geometry
#-------------------------------------------------------------------------------

    x = np.concatenate((np.linspace(0,1,N),np.linspace(1,1+wingletFraction[0],N)),axis=0)
    x = x * wingspan/2
    x = np.sort(np.concatenate((x,xMotor),axis=0))
    dx = x[1] - x[0]
    N = np.size(x)
    fwdWeb[:] = [round(locFwd - xShear,2) for locFwd in fwdWeb]
    aftWeb[:] = [round(locAft - xShear,2) for locAft in aftWeb]

#-------------------------------------------------------------------------------
# Loads
#-------------------------------------------------------------------------------

    L = (1-(x/np.max(x))**2)**0.5       # Assumes Elliptic Lift Distribution
    L0 = 0.5*G_max*MTOW*9.8*liftFraction*SF # Total Design Lift Force
    L = L0/np.sum(L[0:-1]*np.diff(x))*L # Net Lift Distribution

    T = L * chord[0] * cmocl               # Torsion Distribution
    D = L/LoD                           # Drag Distribution

#-------------------------------------------------------------------------------
# Shear/Moments
#-------------------------------------------------------------------------------

    Vx = np.append(np.cumsum((D[0:-1]*np.diff(x))[::-1])[::-1],0)   # Drag Shear
    Vz = np.append(np.cumsum((L[0:-1]*np.diff(x))[::-1])[::-1],0)   # Lift Shear
    Vt = 0 * Vz                                        # Initialize Thrust Shear

    # Calculate shear due to thrust by adding thrust from each motor to each
    # analysis point that's closer to the wing root than the motor. Accomplished
    # by indexing Vt according to a boolean mask of the design points that area
    # less than or aligned with the motor location under consideration in an
    # iterative loop

    for i in range(np.size(xMotor)):
        Vt[x<=xMotor[i]] = Vt[x<=xMotor[i]] + maxThrust

    Mx = np.append(np.cumsum((Vz[0:-1]*np.diff(x))[::-1])[::-1],0)  # Bending Moment
    My = np.append(np.cumsum(( T[0:-1]*np.diff(x))[::-1])[::-1],0)  # Torsion Moment
    Mz = np.append(np.cumsum((Vx[0:-1]*np.diff(x))[::-1])[::-1],0)  # Drag Moment
    Mt = np.append(np.cumsum((Vt[0:-1]*np.diff(x))[::-1])[::-1],0)  # Thrust Moment
    Mz = np.max((Mz,Mt))    # Worst Case of Drag vs. Thrust Moment

#-------------------------------------------------------------------------------
# General Structural Properties
#-------------------------------------------------------------------------------

    seg = []                        # LIST of Structural Segments

    # Torsion

    box = coord                     # Box Initally Matches Airfoil
    box = box[box[:,0]<=aftWeb[1]]   # Inlcude Only Parts Fwd of Aftmost Spar
    box = box[box[:,0]>=fwdWeb[0]]   # Include Only Parts Aft of Fwdmost Spar
    box = box * chord[0]               # Scale by Chord Length

        # Use Shoelace Formula to calculate box area

    torsionArea = 0.5*np.abs(np.dot(box[:,0],np.roll(box[:,1],1))-
        np.dot(box[:,1],np.roll(box[:,0],1)))

    torsionLength = np.sum(np.sqrt(np.sum(np.diff(box,axis=0)**2,axis=1)))

    # Bending

    box = coord                     # Box Initally Matches Airfoil
    box = box[box[:,0]<=fwdWeb[1]]  # Inlcude Only Parts Fwd of Aft Fwd Spar
    box = box[box[:,0]>=fwdWeb[0]]  # Include Only Parts Aft of Fwdmost Spar
    seg.append(box[box[:,1]>np.mean(box[:,1])]*chord[0])   # Upper Fwd Segment
    seg.append(box[box[:,1]<np.mean(box[:,1])]*chord[0])   # Lower Fwd Segment

    # Drag

    box = coord                     # Box Initally Matches Airfoil
    box = box[box[:,0]<=aftWeb[1]]  # Inlcude Only Parts Fwd of Aftmost Spar
    box = box[box[:,0]>=aftWeb[0]]  # Include Only Parts Aft of Fwd Aft Spar
    seg.append(box[box[:,1]>np.mean(box[:,1])]*chord[0])   # Upper Aft Segment
    seg.append(box[box[:,1]<np.mean(box[:,1])]*chord[0])   # Lower Aft Segment

    # Bending/Drag Inertia

    flapInertia = 0
    flapLength  = 0
    dragInertia = 0
    dragLength  = 0

    for i in range(0,4):
        l = np.sqrt(np.sum(np.diff(seg[i],axis=0)**2,axis=1))    # Segment lengths
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
    z[0] = np.interp(fwdWeb[0],box[box[:,1]>0,0],box[box[:,1]>0,1])*chord[0]  # Upper Surface of Box at Fwdmost Spar
    z[1] = np.interp(fwdWeb[0],box[box[:,1]<0,0],box[box[:,1]<0,1])*chord[0]  # Lower Surface of Box at Fwdmost Spar
    h = np.abs(z[0] - z[1])                 # Height of Box at Fwdmost Spar

    # Skin

    box = coord * chord             # Box Initially is Airfoil Scaled by Chord
    skinLength = np.sum(np.sqrt(np.sum(np.diff(box,axis=0)**2,axis=1)))
    A = 0.5*np.abs(np.dot(box[:,0],np.roll(box[:,1],1))-
        np.dot(box[:,1],np.roll(box[:,0],1)))   # Box Area via Shoelace Formula

    #---------------------------------------------------------------------------
    # Structural Calculations
    #---------------------------------------------------------------------------

    # Calculate Skin Weight Based on Torsion

    tTorsion = My*dx/(2*BiCF().ultimateShearStrength*torsionArea)                                    # Torsion Skin Thickness
    tTorsion = np.maximum(tTorsion,BiCF().minimumGageThickness*np.ones(N))                       # Gage Constraint
    mTorsion = tTorsion * torsionLength * BiCF().density                           # Torsion Mass
    mCore = Honeycomb().minimumGageThickness*torsionLength*Honeycomb().density*np.ones(N) # Core Mass
    mGlue = Epoxy().minimumGageThickness*Epoxy().density*torsionLength*np.ones(N)         # Epoxy Mass

    # Calculate Flap Mass Based on Bending

    tFlap = Mx*np.max(seg[0][:,1])/(flapInertia*UniCF().ultimateTensileStrength)                       # Bending Flap Thickness
    mFlap = tFlap*flapLength*UniCF().density                                       # Bending Flap Mass
    mGlue += Epoxy().minimumGageThickness*Epoxy().density*flapLength*np.ones(N)           # Updated Epoxy Mass

    # Calculate Drag Flap Mass

    tDrag = Mz*np.max(seg[2][:,0])/(dragInertia*UniCF().ultimateTensileStrength)                       # Drag Flap Thickness
    mDrag = tDrag*dragLength*UniCF().density                                       # Drag Flap Mass
    mGlue += Epoxy().minimumGageThickness*Epoxy().density*dragLength*np.ones(N)           # Updated Epoxy Mass

    # Calculate Shear Spar Mass

    tShear = 1.5*Vz/(BiCF().ultimateShearStrength*h)                                                 # Shear Spar Thickness
    tShear = np.maximum(tShear, BiCF().minimumGageThickness*np.ones(N))                          # Gage constraint
    mShear = tShear*h*BiCF().density                                               # Shear Spar Mass

    # Paint

    mPaint = skinLength*Paint().minimumGageThickness*Paint().density*np.ones(N)           # Paint Mass

    # Section Mass Total

    m = mTorsion + mCore + mFlap + mDrag + mShear + mGlue + mPaint

    # Rib Mass

    mRib = (A+skinLength*Rib().minWidth)*Rib().minimumGageThickness*Aluminum().density

    # Total Mass

    mass = 2*(sum(m[0:-1]*np.diff(x))+nRibs*mRib)*grace

    return mass
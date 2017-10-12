# prop.py
#
# Created: Jun 2017, J. Smart

#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------

from SUAVE.Core import Units
from SUAVE.Attributes import Solids
import numpy as np

#-------------------------------------------------------------------------------
# Prop
#-------------------------------------------------------------------------------

def prop(rProp, maxThrust, nBlades):
    """weight = SUAVE.Methods.Weights.Correlations.eHelicopter.prop(
            rProp,
            maxThrust,
            nBlades
        )

        Calculates propeller blade pass for an eVTOL vehicle based on assumption
        of a NACA airfoil prop, an assumed cm/cl, tip Mach limit, and structural
        geometry.

        Intended for use with the following SUAVE vehicle types, but may be used
        elsewhere:

            eHelicopter
            eTiltwing
            eTiltrotor
            eStopped_Rotor

        Originally written as part of an AA 290 project inteded for trade study
        of the above vehicle types.

        Inputs:

            rProp:      Propeller Radius            [m]
            maxThrust:  Maximum Motor Thrust        [N]
            nBlades:    Number of Propeller Blades  [Unitless]

        Outputs:

            weight:     Propeller Mass              [kg]
    """

    chord       = rProp * 0.1    # Assumed Prop chord
    N           = 5              # Number of Analysis Points
    SF          = 1.5            # Saftey Factor
    toc         = 0.12           # Average Blade t/c
    fwdWeb      = [0.25, 0.35]   # Forward Web Chord Fraction locations
    xShear      = 0.25           # Approximate Shear Center
    rootLength  = rProp * 0.1    # Assumed Root Fitting Length
    grace       = 1.2            # Grace Factor for Estimation
    sound       = 340.294        # Assumed Speed of Sound
    tipMach     = 0.65           # Propeller Tip Mach Number Limit
    cmocl       = 0.02           # Assumed cm/cl for Sizing Torsion

#-------------------------------------------------------------------------------
# Airfoil
#-------------------------------------------------------------------------------

    NACA = np.multiply(5 * toc, [0.2969, -0.1260, -0.3516, 0.2843, -0.1015])
    coord = np.unique(fwdWeb+np.linspace(0,1,N).tolist())[:,np.newaxis]
    coordMAT = np.concatenate((coord**0.5,coord,coord**2,coord**3,coord**4),axis=1)
    nacaMAT = coordMAT.dot(NACA)[:, np.newaxis]
    coord = np.concatenate((coord,nacaMAT),axis=1)
    coord = np.concatenate((coord[-1:0:-1],coord.dot(np.array([[1.,0.],[0.,-1.]]))),axis=0)
    coord[:,0] = coord[:,0] - xShear

#-------------------------------------------------------------------------------
# Beam Geometry
#-------------------------------------------------------------------------------

    x = np.linspace(0,rProp,N)
    dx = x[1] - x[0]
    fwdWeb[:] = [round(loc - xShear,2) for loc in fwdWeb]

#-------------------------------------------------------------------------------
# Loads
#-------------------------------------------------------------------------------

    omega = sound*tipMach/rProp                   # Propeller Angular Velocity
    F = SF*3*(maxThrust/rProp**3)*(x**2)/nBlades  # Force Distribution
    Q = F * chord * cmocl                         # Torsion Distribution

#-------------------------------------------------------------------------------
# Initial Mass Estimates
#-------------------------------------------------------------------------------

    box = coord * chord
    skinLength = np.sum(np.sqrt(np.sum(np.diff(box,axis=0)**2,axis=1)))
    maxThickness = (np.amax(box[:,1])-np.amin(box[:,1]))/2
    rootBendingMoment = SF*maxThrust/nBlades*0.75*rProp
    m = (Solids.UniCF().density*dx*rootBendingMoment/
        (2*Solids.UniCF().USS*maxThickness))+ \
        skinLength*Solids.BiCF().minThk*dx*Solids.BiCF().density
    m = m*np.ones(N)
    error = 1               # Initialize Error
    tolerance = 1e-8        # Mass Tolerance
    massOld = np.sum(m)

#-------------------------------------------------------------------------------
# General Structural Properties
#-------------------------------------------------------------------------------

    seg = []                        # LIST of Structural Segments

    # Torsion

    enclosedArea = 0.5*np.abs(np.dot(box[:,0],np.roll(box[:,1],1))-
        np.dot(box[:,1],np.roll(box[:,0],1)))   # Shoelace Formula

    # Flap Properties

    box = coord                     # Box Initially Matches Airfoil
    box = box[box[:,0]<=fwdWeb[1]]   # Trim Coordinates Aft of Aft Web
    box = box[box[:,0]>=fwdWeb[0]]   # Trim Coordinates Fwd of Fwd Web
    seg.append(box[box[:,1]>np.mean(box[:,1])]*chord)   # Upper Fwd Segment
    seg.append(box[box[:,1]<np.mean(box[:,1])]*chord)   # Lower Fwd Segment

    # Flap & Drag Inertia

    capInertia = 0
    capLength = 0

    for i in range(0,2):
        l = np.sqrt(np.sum(np.diff(seg[i],axis=0)**2,axis=1))   # Segment Lengths
        c = (seg[i][1::]+seg[i][0::-1])/2                       # Segment Centroids

        capInertia += np.abs(np.sum(l*c[:,1] **2))
        capLength  += np.sum(l)

    # Shear Properties

    box = coord
    box = box[box[:,0]<=fwdWeb[1]]
    z = box[box[:,0]==fwdWeb[0],1]*chord
    shearHeight = np.abs(z[0] - z[1])

    # Core Properties

    box = coord
    box = box[box[:,0]>=fwdWeb[0]]
    box = box*chord
    coreArea = 0.5*np.abs(np.dot(box[:,0],np.roll(box[:,1],1))-
        np.dot(box[:,1],np.roll(box[:,0],1)))   # Shoelace Formula

    # Shear/Moment Calculations

    Vz = np.append(np.cumsum(( F[0:-1]*np.diff(x))[::-1])[::-1],0)  # Bending Moment
    Mx = np.append(np.cumsum((Vz[0:-1]*np.diff(x))[::-1])[::-1],0)  # Torsion Moment
    My = np.append(np.cumsum(( Q[0:-1]*np.diff(x))[::-1])[::-1],0)  # Drag Moment

#-------------------------------------------------------------------------------
# Mass Calculation
#-------------------------------------------------------------------------------

    while error > tolerance:
        CF = (SF*omega**2*
            np.append(np.cumsum(( m[0:-1]*np.diff(x)*x[0:-1])[::-1])[::-1],0))  # Centripetal Force

        # Calculate Skin Weight Based on Torsion

        tTorsion = My/(2*Solids.BiCF().USS*enclosedArea)                          # Torsion Skin Thickness
        tTorsion = np.maximum(tTorsion,Solids.BiCF().minThk*np.ones(N))           # Gage Constraint
        mTorsion = tTorsion * skinLength * Solids.BiCF().density                  # Torsion Mass

        # Calculate Flap Mass Based on Bending

        tFlap = CF/(capLength*Solids.UniCF().UTS) +   \
            Mx*np.amax(np.abs(box[:,1]))/(capInertia*Solids.UniCF().UTS)
        mFlap = tFlap*capLength*Solids.UniCF().density
        mGlue = Solids.Epoxy().minThk*Solids.Epoxy().density*capLength*np.ones(N)

        # Calculate Web Mass Based on Shear

        tShear = 1.5*Vz/(Solids.BiCF().USS*shearHeight)
        tShear = np.maximum(tShear,Solids.BiCF().minThk*np.ones(N))
        mShear = tShear*shearHeight*Solids.BiCF().density

        # Paint Weight

        mPaint = skinLength*Solids.Paint().minThk*Solids.Paint().density*np.ones(N)

        # Core Mass

        mCore = coreArea*Solids.Honeycomb().density*np.ones(N)
        mGlue += Solids.Epoxy().minThk*Solids.Epoxy().density*skinLength*np.ones(N)

        # Leading Edge Protection

        box = coord * chord
        box = box[box[:,0]<(0.1*chord)]
        leLength = np.sum(np.sqrt(np.sum(np.diff(box,axis=0)**2,axis=1)))
        mLE = leLength*420e-6*Solids.Nickel().density*np.ones(N)

        # Section Mass

        m = mTorsion + mCore + mFlap + mShear + mGlue + mPaint + mLE

        # Rib Weight

        mRib = (enclosedArea+skinLength*Solids.Rib().minWidth)*Solids.Rib().minThk*Solids.Aluminum().density

        # Root Fitting

        box = coord * chord
        rRoot = (np.amax(box[:,1])-np.amin(box[:,1]))/2
        t = np.amax(CF)/(2*np.pi*rRoot*Solids.Aluminum().UTS) +    \
            np.amax(Mx)/(3*np.pi*rRoot**2*Solids.Aluminum().UTS)
        mRoot = 2*np.pi*rRoot*t*rootLength*Solids.Aluminum().density

        # Total Weight

        mass = nBlades*(np.sum(m[0:-1]*np.diff(x))+2*mRib+mRoot)
        error = np.abs(mass-massOld)
        massOld = mass

    mass = mass * grace

    return mass
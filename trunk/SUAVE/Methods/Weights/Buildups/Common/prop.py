## @ingroup Methods-Weights-Buildups-Common

# prop.py
#
# Created:  Jun 2017, J. Smart
# Modified: Apr 2018, J. Smart
#           Mar 2020, M. Clarke

#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------

from SUAVE.Attributes.Solids import (
    Bidirectional_Carbon_Fiber, Carbon_Fiber_Honeycomb, Paint, Unidirectional_Carbon_Fiber, Aluminum, Epoxy, Nickel, Aluminum_Rib)
import numpy as np
import copy as cp

#-------------------------------------------------------------------------------
# Prop
#-------------------------------------------------------------------------------

## @ingroup Methods-Weights-Buildups-Common
def prop(prop,
         maximum_lifting_thrust,
         chord_to_radius_ratio = 0.1,
         thickness_to_chord = 0.12,
         root_to_radius_ratio = 0.1,
         moment_to_lift_ratio = 0.02,
         spanwise_analysis_points = 5,
         safety_factor = 1.5,
         margin_factor = 1.2,
         forward_web_locations = [0.25, 0.35],
         shear_center = 0.25,
         speed_of_sound = 340.294,
         tip_max_mach_number = 0.65):
    """weight = SUAVE.Methods.Weights.Buildups.Common.prop(
            prop,
            maximum_thrust,
            chord_to_radius_ratio = 0.1,
            thickness_to_chord = 0.12,
            root_to_radius_ratio = 0.1,
            moment_to_lift_ratio = 0.02,
            spanwise_analysis_points = 5,
            safety_factor = 1.5,
            margin_factor = 1.2,
            forward_web_locationss = [0.25, 0.35],
            shear_center = 0.25,
            speed_of_sound = 340.294,
            tip_max_mach_number = 0.65)

        Assumptions:
        Calculates propeller blade pass for an eVTOL vehicle based on assumption
        of a NACA airfoil prop, an assumed cm/cl, tip Mach limit, and structural
        geometry.

        Intended for use with the following SUAVE vehicle types, but may be used
        elsewhere:

            Electric Multicopter
            Electric Vectored_Thrust
            Electric Stopped Rotor

        Originally written as part of an AA 290 project inteded for trade study
        of the above vehicle types.

        If vehicle model does not have material properties assigned, appropriate
        assumptions are made based on SUAVE's Solids Attributes library.
        
        Sources:
        Project Vahana Conceptual Trade Study

        Inputs:

            prop                        SUAVE Propeller Data Structure
            maximum_thrust              Maximum Design Thrust               [N]
            chord_to_radius_ratio       Chord to Blade Radius               [Unitless]
            thickness_to_chord          Blade Thickness to Chord            [Unitless]
            root_to_radius_ratio        Root Structure to Blade Radius      [Unitless]
            moment_to_lift_ratio        Coeff. of Moment to Coeff. of Lift  [Unitless]
            spanwise_analysis_points    Analysis Points for Sizing          [Unitless]
            safety_factor               Design Safety Factor                [Unitless]
            margin_factor               Allowable Extra Mass Fraction       [Unitless]
            forward_web_locationss      Location of Forward Spar Webbing    [m]
            shear_center                Location of Shear Center            [m]
            speed_of_sound              Local Speed of Sound                [m/s]
            tip_max_mach_number         Allowable Tip Mach Number           [Unitless]

        Outputs:

            weight:                 Propeller Mass                      [kg]
            
        Properties Used:
        Material properties of imported SUAVE Solids
    """

    #-------------------------------------------------------------------------------
    # Unpack Inputs
    #-------------------------------------------------------------------------------
    rProp              = prop.tip_radius
    maxLiftingThrust   = maximum_lifting_thrust
    nBlades            = prop.number_of_blades
    chord              = rProp * chord_to_radius_ratio
    N                  = spanwise_analysis_points
    SF                 = safety_factor
    toc                = thickness_to_chord
    fwdWeb             = cp.deepcopy(forward_web_locations)
    xShear             = shear_center
    rootLength         = rProp * root_to_radius_ratio
    grace              = margin_factor
    sound              = speed_of_sound
    tipMach            = tip_max_mach_number
    cmocl              = moment_to_lift_ratio
    
    #-------------------------------------------------------------------------------
    # Unpack Material Properties
    #-------------------------------------------------------------------------------

    try:
        torsMat = prop.materials.skin_materials.torsion_carrier
    except AttributeError:
        torsMat = Bidirectional_Carbon_Fiber()
    torsUSS = torsMat.ultimate_shear_strength
    torsMGT = torsMat.minimum_gage_thickness
    torsDen = torsMat.density

    try:
        shearMat = prop.materials.spar_materials.shear_carrier
    except AttributeError:
        shearMat = Bidirectional_Carbon_Fiber()
    shearUSS = shearMat.ultimate_shear_strength
    shearMGT = shearMat.minimum_gage_thickness
    shearDen = shearMat.density

    try:
        bendMat = prop.materials.flap_materials.bending_carrier
    except AttributeError:
        bendMat = Unidirectional_Carbon_Fiber()
    bendDen = bendMat.density
    bendUTS = bendMat.ultimate_tensile_strength
    bendUSS = bendMat.ultimate_shear_strength

    try:
        coreMat = prop.materials.skin_materials.core
    except AttributeError:
        coreMat = Carbon_Fiber_Honeycomb()
    coreDen = coreMat.density

    try:
        ribMat = prop.materials.rib_materials.structural
    except AttributeError:
        ribMat = Aluminum_Rib()
    ribWid = ribMat.minimum_width
    ribMGT = ribMat.minimum_gage_thickness
    ribDen = ribMat.density

    try:
        rootMat = prop.materials.root_materials.structural
    except AttributeError:
        rootMat = Aluminum()
    rootDen = rootMat.density
    rootUTS = rootMat.ultimate_tensile_strength

    try:
        leMat = prop.materials.skin_materials.leading_edge
    except:
        leMat = Nickel()
    leDen = leMat.density
    
    try:
        glueMat = prop.materials.skin_materials.adhesive
    except AttributeError:
        glueMat = Epoxy()
    glueMGT = glueMat.minimum_gage_thickness
    glueDen = glueMat.density
    
    try:
        coverMat = prop.materials.skin_materials.cover
    except:
        coverMat = Paint()
    coverMGT = coverMat.minimum_gage_thickness
    coverDen = coverMat.density

    #-------------------------------------------------------------------------------
    # Airfoil
    #-------------------------------------------------------------------------------
    NACA       = np.multiply(5 * toc, [0.2969, -0.1260, -0.3516, 0.2843, -0.1015])
    coord      = np.unique(fwdWeb+np.linspace(0,1,N).tolist())[:,np.newaxis]
    coordMAT   = np.concatenate((coord**0.5,coord,coord**2,coord**3,coord**4),axis=1)
    nacaMAT    = coordMAT.dot(NACA)[:, np.newaxis]
    coord      = np.concatenate((coord,nacaMAT),axis=1)
    coord      = np.concatenate((coord[-1:0:-1],coord.dot(np.array([[1.,0.],[0.,-1.]]))),axis=0)
    coord[:,0] = coord[:,0] - xShear

    #-------------------------------------------------------------------------------
    # Beam Geometry
    #-------------------------------------------------------------------------------
    x         = np.linspace(0,rProp,N)
    dx        = x[1] - x[0]
    fwdWeb[:] = [round(loc - xShear,2) for loc in fwdWeb]

    #-------------------------------------------------------------------------------
    # Loads
    #-------------------------------------------------------------------------------
    omega = sound*tipMach/rProp                              # Propeller Angular Velocity
    F     = SF*3*(maxLiftingThrust/rProp**3)*(x**2)/nBlades  # Force Distribution
    Q     = F * chord * cmocl                                # Torsion Distribution

    #-------------------------------------------------------------------------------
    # Initial Mass Estimates
    #-------------------------------------------------------------------------------
    box               = coord * chord
    skinLength        = np.sum(np.sqrt(np.sum(np.diff(box,axis=0)**2,axis=1)))
    maxThickness      = (np.amax(box[:,1])-np.amin(box[:,1]))/2
    rootBendingMoment = SF*maxLiftingThrust/nBlades*0.75*rProp
    m                 = (bendDen*dx*rootBendingMoment/
                        (2*bendUSS*maxThickness))+ \
                        skinLength*shearMGT*dx*shearDen
    m                 = m*np.ones(N)
    error             = 1               # Initialize Error
    tolerance         = 1e-8        # Mass Tolerance
    massOld           = np.sum(m)

    #-------------------------------------------------------------------------------
    # General Structural Properties
    #-------------------------------------------------------------------------------
    seg = []                                            # List of Structural Segments

    # Torsion
    enclosedArea = 0.5*np.abs(np.dot(box[:,0],np.roll(box[:,1],1))-
        np.dot(box[:,1],np.roll(box[:,0],1)))           # Shoelace Formula

    # Flap Properties
    box = coord                                         # Box Initially Matches Airfoil
    box = box[box[:,0]<=fwdWeb[1]]                      # Trim Coordinates Aft of Aft Web
    box = box[box[:,0]>=fwdWeb[0]]                      # Trim Coordinates Fwd of Fwd Web
    seg.append(box[box[:,1]>np.mean(box[:,1])]*chord)   # Upper Fwd Segment
    seg.append(box[box[:,1]<np.mean(box[:,1])]*chord)   # Lower Fwd Segment

    # Flap & Drag Inertia
    capInertia = 0
    capLength  = 0

    for i in range(0,2):
        l = np.sqrt(np.sum(np.diff(seg[i],axis=0)**2,axis=1))   # Segment Lengths
        c = (seg[i][1::]+seg[i][0::-1])/2                       # Segment Centroids

        capInertia += np.abs(np.sum(l*c[:,1] **2))
        capLength  += np.sum(l)

    # Shear Properties
    box = coord
    box = box[box[:,0]<=fwdWeb[1]]
    z   = box[box[:,0]==fwdWeb[0],1]*chord
    shearHeight = np.abs(z[0] - z[1])

    # Core Properties
    box = coord
    box = box[box[:,0]>=fwdWeb[0]]
    box = box*chord
    coreArea = 0.5*np.abs(np.dot(box[:,0],np.roll(box[:,1],1))-
        np.dot(box[:,1],np.roll(box[:,0],1)))                       # Shoelace Formula

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
        tTorsion = My/(2*torsUSS*enclosedArea)                 # Torsion Skin Thickness
        tTorsion = np.maximum(tTorsion,torsMGT*np.ones(N))     # Gage Constraint
        mTorsion = tTorsion * skinLength * torsDen             # Torsion Mass

        # Calculate Flap Mass Based on Bending
        tFlap = CF/(capLength*bendUTS) +   \
            Mx*np.amax(np.abs(box[:,1]))/(capInertia*bendUTS)
        mFlap = tFlap*capLength*bendDen
        mGlue = glueMGT*glueDen*capLength*np.ones(N)

        # Calculate Web Mass Based on Shear
        tShear = 1.5*Vz/(shearUSS*shearHeight)
        tShear = np.maximum(tShear,shearMGT*np.ones(N))
        mShear = tShear*shearHeight*shearDen

        # Paint Weight
        mPaint = skinLength*coverMGT*coverDen*np.ones(N)

        # Core Mass
        mCore = coreArea*coreDen*np.ones(N)
        mGlue += glueMGT*glueDen*skinLength*np.ones(N)

        # Leading Edge Protection
        box      = coord * chord
        box      = box[box[:,0]<(0.1*chord)]
        leLength = np.sum(np.sqrt(np.sum(np.diff(box,axis=0)**2,axis=1)))
        mLE      = leLength*420e-6*leDen*np.ones(N)

        # Section Mass
        m = mTorsion + mCore + mFlap + mShear + mGlue + mPaint + mLE

        # Rib Weight
        mRib = (enclosedArea+skinLength*ribWid)*ribMGT*ribDen

        # Root Fitting
        box   = coord * chord
        rRoot = (np.amax(box[:,1])-np.amin(box[:,1]))/2
        t     = np.amax(CF)/(2*np.pi*rRoot*rootUTS) +    \
                np.amax(Mx)/(3*np.pi*rRoot**2*rootUTS)
        mRoot = 2*np.pi*rRoot*t*rootLength*rootDen

        # Total Weight
        mass    = nBlades*(np.sum(m[0:-1]*np.diff(x))+2*mRib+mRoot)
        error   = np.abs(mass-massOld)
        massOld = mass

    mass = mass * grace

    return mass
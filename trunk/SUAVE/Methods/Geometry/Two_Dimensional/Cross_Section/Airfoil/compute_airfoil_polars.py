## @ingroup Methods-Geometry-Two_Dimensional-Cross_Section-Airfoil
# compute_airfoil_polars.py
# 
# Created:  Mar 2019, M. Clarke
#           Mar 2020, M. Clarke
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
from SUAVE.Core               import Data , Units
from .import_airfoil_geometry import import_airfoil_geometry 
from .import_airfoil_polars   import import_airfoil_polars 
from scipy.interpolate        import RectBivariateSpline
import numpy as np

## @ingroup Methods-Geometry-Two_Dimensional-Cross_Section-Airfoil
def compute_airfoil_polars(propeller,a_geo,a_polar):
    """This computes the lift and drag coefficients of an airfoil in stall regimes using pre-stall
    characterstics and AERODAS formation for post stall characteristics. This is useful for 
    obtaining a more accurate prediction of wing and blade loading. Pre stall characteristics 
    are obtained in the from of a text file of airfoil polar data obtained from airfoiltools.com
    
    Assumptions:
    Uses AERODAS forumatuon for post stall characteristics 

    Source:
    Models of Lift and Drag Coefficients of Stalled and Unstalled Airfoils in Wind Turbines and Wind Tunnels
    by D Spera, 2008

    Inputs:
    propeller. 
        hub_radius         [m]
        tip_radius         [m]
        chord_distribution [unitless]
    airfoils                <string>
           

    Outputs:
    airfoil_data.
        cl_polars          [unitless]
        cd_polars          [unitless]      
        aoa_sweep          [unitless]
    
    Properties Used:
    N/A
    """  
    
    num_airfoils = len(a_geo)
    num_polars   = len(a_polar[0])
    if num_polars < 3:
        raise AttributeError('Provide three or more airfoil polars to compute surrogate')
        
    # unpack 
    Rh = propeller.hub_radius
    Rt = propeller.tip_radius
    n  = len(propeller.chord_distribution)
    cm = propeller.chord_distribution[round(n*0.5)] 

    # read airfoil geometry  
    airfoil_data = import_airfoil_geometry(a_geo)

    AR = 2*(Rt - Rh)/cm

    # Get all of the coefficients for AERODAS wings
    AoA_sweep_deg = np.linspace(-20,90,111)
    CL = np.zeros((num_airfoils,num_polars,len(AoA_sweep_deg)))
    CD = np.zeros((num_airfoils,num_polars,len(AoA_sweep_deg)))
    

    CL_surs = Data()
    CD_surs = Data()    


    # AERODAS 
    for i in range(num_airfoils):
        for j in range(num_polars):
            # read airfoil polars 
            airfoil_polar_data = import_airfoil_polars(a_polar)
            airfoil_cl         = airfoil_polar_data.lift_coefficients[i,j] 
            airfoil_cd         = airfoil_polar_data.drag_coefficients[i,j] 
            airfoil_aoa        = airfoil_polar_data.angle_of_attacks  
            
            # computing approximate zero lift aoa
            airfoil_cl_plus = airfoil_cl[airfoil_cl>0]
            idx_zero_lift = np.where(airfoil_cl == min(airfoil_cl_plus))[0][0]
            A0  = airfoil_aoa[idx_zero_lift]
            
            # computing approximate lift curve slope
            cl_range = airfoil_aoa[idx_zero_lift:idx_zero_lift+50]
            aoa_range = airfoil_cl[idx_zero_lift:idx_zero_lift+50]
            S1 = np.mean(np.diff(cl_range)/np.diff(aoa_range))  
            
            # max lift coefficent and associated aoa
            CL1max  = np.max(airfoil_cl) 
            idx_aoa_max_prestall_cl = np.where(airfoil_cl == CL1max)[0][0]
            ACL1  = airfoil_aoa[idx_aoa_max_prestall_cl]
            
            # max drag coefficent and associated aoa
            CD1max  = np.max(airfoil_cd) 
            idx_aoa_max_prestall_cd = np.where(airfoil_cd == CD1max)[0][0]
            ACD1   = airfoil_aoa[idx_aoa_max_prestall_cd]          
            
            CD0     = airfoil_cd[idx_zero_lift]       
            CL1maxp = CL1max
            ACL1p   = ACL1
            ACD1p   = ACD1
            CD1maxp = CD1max
            S1p     = S1
            
            for k in range(len(AoA_sweep_deg)):
                alpha  = AoA_sweep_deg[k] 
                t_c = airfoil_data.thickness_to_chord[i]
            
                # Equation 5a
                ACL1   = ACL1p + 18.2*CL1maxp*(AR**(-0.9)) 
            
                # From McCormick
                S1 = S1p*AR/(2+np.sqrt(4+AR**2)) 
            
                # Equation 5c
                ACD1   =  ACD1p + 18.2*CL1maxp*(AR**(-0.9)) 
            
                # Equation 5d
                CD1max = CD1maxp + 0.280*(CL1maxp*CL1maxp)*(AR**(-0.9))
            
                # Equation 5e
                CL1max = CL1maxp*(0.67+0.33*np.exp(-(4.0/AR)**2.))
            
                # ------------------------------------------------------
                # Equations for coefficients in pre-stall regime 
                # ------------------------------------------------------
                # Equation 6c
                RCL1   = S1*(ACL1-A0)-CL1max
            
                # Equation 6d
                N1     = 1 + CL1max/RCL1
            
                # Equation 6a or 6b depending on the alpha                  
                if alpha == A0:
                    CL[i,j,k] = 0.0        
                elif alpha > A0: 
                    CL[i,j,k] = S1*(alpha - A0)-RCL1*((alpha-A0)/(ACL1-A0))**N1        
                else:
                    CL[i,j,k] = S1*(alpha - A0)+RCL1 *((A0-alpha )/(ACL1 -A0))**N1 
            
                # Equation 7a or 7b depending on alpha
                M    = 2.0  
                con  = np.logical_and((2*A0-ACD1)<=alpha,alpha<=ACD1)
                if con == True:
                    CD[i,j,k] = CD0  + (CD1max -CD0)*((alpha  -A0)/(ACD1 -A0))**M   
                else:
                    CD[i,j,k] = 0.  
                # ------------------------------------------------------
                # Equations for coefficients in post-stall regime 
                # ------------------------------------------------------               
                # Equation 9a and b
                F1        = 1.190*(1.0-(t_c**2))
                F2        = 0.65 + 0.35*np.exp(-(9.0/AR)**2.3)
            
                # Equation 10b and c
                G1        = 2.3*np.exp(-(0.65*t_c)**0.9)
                G2        = 0.52 + 0.48*np.exp(-(6.5/AR)**1.1)
            
                # Equation 8a and b
                CL2max    = F1*F2
                CD2max    = G1*G2
            
                # Equation 11d
                RCL2      = 1.632-CL2max
            
                # Equation 11e
                N2        = 1 + CL2max/RCL2
            
                # LIFT COEFFICIENT
                # Equation 11a,b,c
                if alpha > ACL1:
                    con2      = np.logical_and(ACL1<=alpha,alpha<=(92.0))
                    con3      = [alpha>=(92.0)]                   
                    if con2 == True:
                        CL[i,j,k] = -0.032*(alpha-92.0) - RCL2*((92.-alpha)/(51.0))**N2
                    elif con3 == True:
                        CL[i,j,k] = -0.032*(alpha-92.0) + RCL2*((alpha-92.)/(51.0))**N2
            
                # If alpha is negative flip things for lift
                elif alpha < 0.:  
                    alphan    = - alpha+2*A0
                    con2      = np.logical_and(ACL1<=alpha, alpha<=(92.0))
                    con3      = alpha>=(92.0)                    
                    if con2 == True:
                        CL[i,j,k] = 0.032*(alphan-92.0) + RCL2*((92.-alpha)/(51.0))**N2
                    elif con3 == True:
                        CL[i,j,k] = 0.032*(alphan-92.0) - RCL2*((alphan-92.)/(51.0))**N2
            
                # DRAG COEFFICIENT
                # Equation 12a 
                if  alpha > ACD1:
                    CD[i,j,k]  = CD1max + (CD2max - CD1max) * np.sin(((alpha-ACD1)/(90.-ACD1))*90.*Units.degrees)
            
                # If alpha is negative flip things for drag
                elif alpha < 0.:
                    alphan    = -alpha + 2*A0
                    if alphan>=ACD1:
                        CD[i,j,k]  = CD1max + (CD2max - CD1max) * np.sin(((alphan-ACD1)/(90.-ACD1))*Units.degrees)  
        
        AoA_sweep_radians = AoA_sweep_deg*Units.degrees       
        CL_sur = RectBivariateSpline(airfoil_polar_data.reynolds_number[i],AoA_sweep_radians, CL[i,:,:])  
        CD_sur = RectBivariateSpline(airfoil_polar_data.reynolds_number[i],AoA_sweep_radians, CD[i,:,:])   
        
        CL_surs[a_geo[i]]  = CL_sur
        CD_surs[a_geo[i]]  = CD_sur       
      
    airfoil_data.angle_of_attacks              = AoA_sweep_radians
    airfoil_data.lift_coefficient_surrogates   = CL_surs
    airfoil_data.drag_coefficient_surrogates   = CD_surs 
    
    return airfoil_data

 
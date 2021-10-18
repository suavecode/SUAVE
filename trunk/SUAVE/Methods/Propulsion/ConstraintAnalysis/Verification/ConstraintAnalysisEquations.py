import SUAVE
import numpy as np
import matplotlib as plt
from SUAVE.Core import Data
from SUAVE.Methods.Propulsion.ConstraintAnalysis.Verification.Parameters import *
from SUAVE.Methods.Propulsion.ConstraintAnalysis.Verification.intersect import intersection

WS=np.linspace(20,20000,1)

def LandingConstraint():
    WS_Landing = clmax_landing * rho_SL * LandingDistance / (2 * 0.5847 * FractionWeightLanding)  # Check if clmax of landing is the same as this
    i=2
    return WS_Landing

def StallConstraint():
    WS_Stall = 0.5 * rho_SL * V_stall ** 2 * clmax
    return WS_Stall

def VmaxConstraint():
    TW_Vmax = []
    for x in WS:
        TW_Vmax_i = (rho_SL * V_max ** 2 * cd0) / (2 * x) + 2 * K_aerodynamics * x / (rho_cruise * sigma_cruise * V_max ** 2)
        TW_Vmax.append(TW_Vmax_i)
    return TW_Vmax

def TakeoffConstraint():
    TW_Takeoff = []
    for x in WS:
        TW_Takeoff_i=(mu - (mu + cdG/clR)*np.exp(0.6*rho_SL*g*cdG*S_TO/x))/(1-np.exp(0.6*rho_SL*g*cdG*S_TO/x))
        TW_Takeoff.append(TW_Takeoff_i)
    return TW_Takeoff

def ROCConstraint():
    TW_ROC = []
    for x in WS:
        TW_ROC_i=ROC/np.sqrt(2*x/(rho_SL*np.sqrt(cd0/K_aerodynamics))) + 1/LD_max
        TW_ROC.append(TW_ROC_i)
    return TW_ROC

def CeilingConstraint():
    TW_Ceiling = []
    for x in WS:
        TW_Ceiling_i=ROC_service_ceiling/(sigma_service_ceiling*np.sqrt(2*x/(rho_service_ceiling*np.sqrt(cd0/K_aerodynamics)))) + 1/(sigma_service_ceiling*LD_max)
        TW_Ceiling.append(TW_Ceiling_i)
    return TW_Ceiling

def ClimbGradientConstraint():
    TW_ClimbGradient = []
    for x in WS:
        TW_ClimbGradient_i = NumberEngines/(NumberEngines-1)*(cV_ClimbGradient+2*np.sqrt(cd0/(np.pi*AspectRatio*Oswald)))
        TW_ClimbGradient.append(TW_ClimbGradient_i)
    return TW_ClimbGradient

def ManoueuvringConstraint():
    TW_Manoeuvring = []
    for x in WS:
        TW_Manoeuvring_i = 0.5*rho_cruise*(V_cruise**2)*cd0/x + 2*x*(nmax**2)/(np.pi*AspectRatio*Oswald*rho_cruise*(V_cruise**2))
        TW_Manoeuvring.append(TW_Manoeuvring_i)
    return TW_Manoeuvring
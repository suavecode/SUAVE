import SUAVE
import numpy as np
import matplotlib as plt
from SUAVE.Core import Data
from SUAVE.Methods.Propulsion.ConstraintAnalysis.Verification.Parameters import *
from SUAVE.Methods.Propulsion.ConstraintAnalysis.Verification.ConstraintAnalysisEquations import *

def plot_constraint_analysis():
    # Run constraint analysis equations

    WS_Landing = LandingConstraint()
    WS_Stall = StallConstraint()
    TW_Vmax = VmaxConstraint()
    TW_Takeoff = TakeoffConstraint()
    TW_ROC = ROCConstraint()
    TW_Ceiling = CeilingConstraint()
    TW_ClimbGradient = ClimbGradientConstraint()
    TW_Manoeuvring = ManoueuvringConstraint()

    # ----------------------------------------------------------------------
    #   Plot of the constraint analysis in a W/S-T/W chart
    # ----------------------------------------------------------------------

    WS = np.linspace(20, 20000, 1)

    plt.pyplot.plot([WS_Stall, WS_Stall], [0, 1], label="Stall")
    plt.pyplot.plot([WS_Landing, WS_Landing], [0, 1], label="Landing")
    plt.pyplot.plot(WS, TW_Vmax, label="Max speed")
    plt.pyplot.plot(WS, TW_Takeoff, label="Takeoff")
    plt.pyplot.plot(WS, TW_ROC, label="ROC")
    plt.pyplot.plot(WS, TW_Ceiling, label="Service ceiling")
    plt.pyplot.plot(WS, TW_ClimbGradient, label="Climb gradient")
    plt.pyplot.plot(WS, TW_Manoeuvring, label="Manoeuvring")
    plt.pyplot.xlabel('W/S (N/m^2)')
    plt.pyplot.ylabel('T/W')
    plt.pyplot.title('Constraint analysis')
    plt.pyplot.legend()
    plt.pyplot.show()
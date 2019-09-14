## @ingroup Methods-Aerodynamics-Common-Gas_Dynamics
# Conical_Shock.py
#
# Created:  June 2019, M. Dethy
# Modified:  
#           

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np

# ----------------------------------------------------------------------
#  Conical Shock Relations
# ----------------------------------------------------------------------

## @ingroup Methods-Aerodynamics-Common-Gas_Dynamics
def get_invisc_press_recov(theta_c, M):
    """The throat inviscid total pressure recovery from the cone angle and local mach number

    Assumptions:
    None

    Source:
    Appendix E of Nicolai's Fundamentals of Aircraft and Airship Design, Volume 1 – Aircraft Design

    
    Inputs:
    Cone Angle, theta_c                                              [deg]
    Local Mach Number, M                                             [-]

    Outputs:
    Throat Inviscid Total Pressure Recovery, P_ratio_invis           [-]

    
    Properties Used:
    N/A
    """

    # Coefficients for the polynomial fits of polynomial fits
    c1_list = [4.350905049075176e-06, -0.0001199212091541324, 0.001473049009170714, -0.01061532945451371, 0.049681690145056495, -0.15772878493519066, 0.34387262132063395, -0.5081431701643666, 0.48688821582891917, -0.27305357246492956, 0.06803659988236214]
    c2_list = [-0.00042292871212270763, 0.011655926857554016, -0.1431687224047205, 1.031738549961427, -4.829171321263979, 15.334629037105937, -33.44291699272302, 49.44331627490299, -47.4076453531451, 26.61077828800693, -6.638113715636017]
    c3_list = [0.012647719847822116, -0.3487216084353584, 4.285186990097771, -30.894910117326123, 144.67591011833733, -459.641924049883, 1002.9894005317848, -1483.8018102044036, 1423.7382324736022, -799.8272984646909, 199.70469529244235]
    c4_list = [-0.11541313837056255, 3.186624105936058, -39.21277215049977, 283.103908188231, -1327.5498175102687, 4223.419345261226, -9228.380957679474, 13670.52065785141, -13134.566698345967, 7388.503361115493, -1847.2407423340148]
    c5_list = [0.10829137392949904, -2.9653220670445553, 36.186528135856356, -259.082157552149, 1204.8384237047933, -3801.5740149913267, 8239.467642627327, -12108.64304013928, 11543.009121748704, -6443.360961773748, 1599.90131353175]

    # Get the coefficients for the specified mach number
    c1 = np.polyval(c1_list, M)
    c2 = np.polyval(c2_list, M)
    c3 = np.polyval(c3_list, M)
    c4 = np.polyval(c4_list, M)
    c5 = np.polyval(c5_list, M)

    # Use coefficients on theta_c to get the pressure recovery
    fit = [c1, c2, c3, c4, c5]
    P_ratio_invis = np.polyval(fit, theta_c)
    return P_ratio_invis

def get_beta(M, theta_c):
    """The shock wave angle from the mach number and cone angle

    Assumptions:
    None

    Source:
    Appendix E of Nicolai's Fundamentals of Aircraft and Airship Design, Volume 1 – Aircraft Design

    
    Inputs:
    Mach Number, M                                             [-]
    Cone Angle, theta_c                                        [deg]

    Outputs:
    Shock Wave Angle, beta                                     [deg]

    
    Properties Used:
    N/A
    """

    # Coefficients for the polynomial fits of polynomial fits
    c1_list = [-3.3710257888720006e-08, 6.318887120059392e-06, -0.00047455191593887756, 0.01815417956016697, -0.3699753459684698, 3.7681915034391524, -14.952470590772341, -7.9583913934684425]
    c2_list = [2.2749170338090888e-07, -4.315628975558707e-05, 0.003283652250788974, -0.12738271573255136, 2.633507830976124, -27.177354336068767, 108.20918451074687, 41.58871923133814]
    c3_list = [-5.003028967237037e-07, 9.570576860333848e-05, -0.007346757820245642, 0.28761922507297405, -5.99985286896233, 62.39209150818428, -248.34622630654033, -82.21288470271705]
    c4_list = [3.607199072659998e-07, -6.940582479966812e-05, 0.005359793812773401, -0.211071766025574, 4.426615196635283, -46.19240039325365, 183.7888623501148, 69.38622081708557]

    # Get the coefficients for the cone angle
    c1 = np.polyval(c1_list, theta_c)
    c2 = np.polyval(c2_list, theta_c)
    c3 = np.polyval(c3_list, theta_c)
    c4 = np.polyval(c4_list, theta_c)

    # Use coefficients on the log of M to get beta
    fit = [c1, c2, c3, c4]
    beta = np.polyval(fit, np.log(M))
    return beta

def get_Cp(M, theta_c):
    """Coefficient of pressure from the mach number and cone angle

    Assumptions:
    None

    Source:
    Appendix E of Nicolai's Fundamentals of Aircraft and Airship Design, Volume 1 – Aircraft Design

    
    Inputs:
    Mach Number, M                                             [-]
    Cone Angle, theta_c                                        [deg]

    Outputs:
    Pressure Coefficient, Cp                                    [-]

    
    Properties Used:
    N/A
    """

    # Coefficients for the polynomial fits of polynomial fits
    c1_list = [1.7633653834372048e-07, -3.392210961953478e-05, 0.00262012956168885, -0.10324002446973837, 2.1666379723702778, -22.617773106981787, 89.79208824725325]
    c2_list = [-1.4792619222164575e-06, 0.0002852272997941191, -0.0220775539162887, 0.8715014869612325, -18.316276605865934, 191.44208863528544, -760.6693894869614]
    c3_list = [4.625717539273876e-06, -0.000893775419313979, 0.06931389450605625, -2.74071166272147, 57.67937689894854, -603.5470923685374, 2400.0371032744915]
    c4_list = [-6.389584227158195e-06, 0.0012368677751048007, -0.09608591779245888, 3.8050653595080104, -80.17955291673923, 839.8520792556219, -3342.2378161803285]
    c5_list = [3.288730809067659e-06, -0.0006376112918285232, 0.049605040472884054, -1.9669646303065382, 41.49345002409283, -435.0096824087901, 1732.3035121144699]

    # Get the coefficients for the cone angle
    c1 = np.polyval(c1_list, theta_c)
    c2 = np.polyval(c2_list, theta_c)
    c3 = np.polyval(c3_list, theta_c)
    c4 = np.polyval(c4_list, theta_c)
    c5 = np.polyval(c5_list, theta_c)

    # Use coefficients on the log of M to get Cp
    fit = [c1, c2, c3, c4, c5]
    Cp = np.polyval(fit, np.log(M))
    return Cp

def get_Ms(M, theta):
    """Surface mach number from freestream mach number and cone semi-angle

    Assumptions:
    None

    Source:
    Appendix E of Nicolai's Fundamentals of Aircraft and Airship Design, Volume 1 – Aircraft Design

    
    Inputs:
    Freestream Mach Number, M                                  [-]
    Cone Semi-Angle, theta                                     [deg]

    Outputs:
    Surface Mach Number, Ms                                    [-]

    
    Properties Used:
    N/A
    """

    # Coefficients for the polynomial fits of polynomial fits
    c1_list = [-1.7556199804252492e-12, 3.164276000297232e-10, -2.515288017609384e-08, 1.1603850517055833e-06, -3.439582014310394e-05, 0.0006849232110711314, -0.009302373552285605, 0.08564832707138871, -0.519952514670386, 1.9628525535690642, -4.095378578672224, 3.504927254074139]
    c2_list = [1.106849367478925e-11, -1.997605893834249e-09, 1.5901579200595063e-07, -7.346819845757375e-06, 0.00021810208460545957, -0.0043495741413903, 0.05915932650875921, -0.5454149558295122, 3.3150485080115306, -12.528034407482547, 26.151681199336323, -21.404701959442452]
    c3_list = [-1.0915325579359898e-11, 1.971035027784013e-09, -1.5694001038613308e-07, 7.249975935107071e-06, -0.0002150981566919854, 0.004284731639405878, -0.0581748016147024, 0.5350588940874457, -3.242492155257401, 12.212931732413136, -25.425221833334305, 21.742419045519174]

    # Get the coefficients for the cone semi-angle
    c1 = np.polyval(c1_list, theta)
    c2 = np.polyval(c2_list, theta)
    c3 = np.polyval(c3_list, theta)

    # Use coefficients on the incoming mach to get surface mach
    fit = [c1, c2, c3]
    Ms = np.polyval(fit, M)
    return Ms

if __name__ == "__main__":
    new = get_Ms(4,30)


    






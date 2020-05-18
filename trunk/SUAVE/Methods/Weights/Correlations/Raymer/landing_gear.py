from SUAVE.Core import Units, Data


def landing_gear_Raymer(vehicle):
    Kmp = 1
    if vehicle.systems.accessories == "sst":
        RFACT = 0.00009
    else:
        RFACT = 0.00004
    DESRNG = vehicle.design_range / Units.nmi  # Design range in nautical miles
    WLDG = vehicle.mass_properties.max_takeoff / Units.lbs * (1 - RFACT * DESRNG)
    Ngear = 3
    Nl = Ngear * 1.5
    Lm = vehicle.landing_gear.main_strut_length / Units.inch
    Nmss = 2
    Nmw = vehicle.landing_gear.main_wheels * Nmss
    Vstall = 51 * Units.kts
    Knp = 1
    Ln = vehicle.landing_gear.nose_strut_length / Units.inch
    Nnw = vehicle.landing_gear.nose_wheels
    WLGM = 0.0106 * Kmp * WLDG ** 0.888 * Nl ** 0.25 * Lm ** 0.4 * Nmw ** 0.321 * Nmss ** (-0.5) * Vstall ** 0.1
    WLGN = 0.032 * Knp * WLDG ** 0.646 * Nl ** 0.2 * Ln ** 0.5 * Nnw ** 0.45

    output = Data()
    output.main = WLGM * Units.lbs
    output.nose = WLGN * Units.lbs
    return output

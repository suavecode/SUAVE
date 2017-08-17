## @ingroup Methods-Costs-Industrial_Costs
# estimate_escalation_factor.py
#
# Created:  Sep 2016, T. Orra

import numpy as np
# ----------------------------------------------------------------------
#  Estimate escalation factor according to United States Consumer Price Index
# ----------------------------------------------------------------------

## @ingroup Methods-Costs-Industrial_Costs
def estimate_escalation_factor(reference_year):
    """Estimates the escalation factor for a given year. Escalation is similar 
    to inflation but for a specific good.

    Assumptions:
    None

    Source:
    Historical data from United States Consumer Price Index

    Inputs:
    reference_year    [-]

    Outputs:
    escalation_factor [-]

    Properties Used:
    N/A
    """         

    # Unpack

    reference_year_table    = np.array( [1915,	1920,	1925,	1930,	1935,	1940,	1945,	1950,	1955,	1960,	1965,	1970,	1975,	1980,	1985,	1990,	1998,	2000,	2005,	2010,	2015,	2020,	2025,	2030,	2035,	2040,	2100])
    escalation_factor_table = np.array( [0.066,	0.116,	0.104,	0.101,	0.083,	0.093,	0.113,	0.153,	0.158,	0.182,	0.194,	0.235,	0.315,	0.518,	0.651,	0.795,	1.000,	1.048,	1.189,	1.328,	1.451,	1.516,	1.581,	1.647,	1.712,	1.777,	2.5])

    escalation_factor = np.interp(reference_year,reference_year_table,escalation_factor_table)

    return escalation_factor
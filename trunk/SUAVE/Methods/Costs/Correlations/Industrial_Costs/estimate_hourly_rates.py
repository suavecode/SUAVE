## @ingroup Methods-Costs-Industrial_Costs
# estimate_hourly_rates.py
#
# Created:  Sep 2016, T. Orra

# Suave imports
from SUAVE.Core import Data

# ----------------------------------------------------------------------
#  Estimate hourly rates according to a trend line
# ----------------------------------------------------------------------

## @ingroup Methods-Costs-Industrial_Costs
def estimate_hourly_rates(year):
    """Estimates the hourly rate according to a trend line.

    Assumptions:
    None

    Source:
    Trends in hourly rates according to "Fundamentals of Aircraft Design", 
      vol 1, Nicolai Figure 24.4.

    Inputs:
    year              [-]

    Outputs:
    hourly_rates.
      engineering     [$/hr]
      tooling         [$/hr]
      manufacturing   [$/hr]
      quality_control [$/hr]

    Properties Used:
    N/A
    """
    # Unpack
    reference_year = year

    hourly_rates = Data()
    hourly_rates.engineering     = 2.576 * reference_year - 5058.
    hourly_rates.tooling         = 2.883 * reference_year - 5666.
    hourly_rates.manufacturing   = 2.316 * reference_year - 4552.
    hourly_rates.quality_control = 2.600 * reference_year - 5112.

    return hourly_rates
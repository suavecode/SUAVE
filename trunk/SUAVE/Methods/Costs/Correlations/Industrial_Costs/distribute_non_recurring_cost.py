## @ingroup Methods-Costs-Industrial_Costs
# distribute_non_recurring_cost.py
#
# Created:  Sep 2016, T. Orra

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
from SUAVE.Core import Data

# general packages
import numpy as np

# ----------------------------------------------------------------------
#  Distribute non recurring costs, for cash flow proposes
# ----------------------------------------------------------------------
## @ingroup Methods-Costs-Industrial_Costs
def distribute_non_recurring_cost(costs):
    """Distributes the non-recurring costs over the appropriate time period

    Assumptions:
    None

    Source:
    Markish, J., 'Valuation Techniques for Commercial Aircraft Program Design'

    Inputs:
    cost.industrial.
      development_total_years        [-]        
      non_recurring.breakdown.
        airframe_engineering         [$]
        development_support          [$]
        flight_test                  [$]
        engines                      [$]
        avionics                     [$]
        tooling_development          [$]
        tooling_production           [$]
        manufacturing_labor          [$]
        manufacturing_material       [$]
        quality_control              [$]
        test_facilities              [$]
        manufacturing_facilities     [$]

    Outputs:
    costs.industrial.non_recurring.cash_flow.:
      breakdown.engineering          [$]
      breakdown.manufacturing        [$]
      breakdown.tooling_design       [$]
      breakdown.tooling_production   [$]
      breakdown.support              [$]
      breakdown.facilities           [$]
      total                          [$]

    Properties Used:
    N/A
    """      
    # unpack
    nrec                    = costs.industrial.non_recurring.breakdown
    nrec_engineering        = nrec.airframe_engineering + nrec.flight_test + nrec.engines + nrec.avionics
    nrec_manufacturing      = nrec.manufacturing_labor + nrec.manufacturing_material
    nrec_tooling_design     = nrec.tooling_development * 1.
    nrec_tooling_production = nrec.tooling_production  * 1.
    nrec_support            = nrec.development_support + nrec.quality_control
    nrec_facilities         = nrec.test_facilities + nrec.manufacturing_facilities
    development_duration    = int(costs.industrial.development_total_years)

    # non-dimentional distribution of cost on time, according to Markish, J., 'Valuation Techniques for Commercial Aircraft Program Design'
    ndim_time               = np.array([ 0.000000 , 0.025000 , 0.075000 , 0.125000 , 0.175000 , 0.225000 , 0.275000 , 0.325000 , 0.375000 , 0.425000 , 0.475000 , 0.525000 , 0.575000 , 0.625000 , 0.675000 , 0.725000 , 0.775000 , 0.825000 , 0.875000 , 0.925000 , 0.975000 , 1.000000 ])
    ndim_engineering        = np.array([ 0.000000 , 0.010436 , 0.029984 , 0.048012 , 0.063051 , 0.074632 , 0.082648 , 0.087182 , 0.088442 , 0.086718 , 0.082367 , 0.075797 , 0.067455 , 0.057826 , 0.047833 , 0.037841 , 0.027470 , 0.017950 , 0.009882 , 0.003884 , 0.000588 , 0.000000 ])
    ndim_manuf_engineering  = np.array([ 0.000000 , 0.007591 , 0.027399 , 0.049416 , 0.069671 , 0.086167 , 0.097835 , 0.104193 , 0.105194 , 0.101133 , 0.092586 , 0.080375 , 0.065537 , 0.049302 , 0.034542 , 0.019781 , 0.008029 , 0.001249 , 0.000000 , 0.000000 , 0.000000 , 0.000000 ])
    ndim_tooling_design     = np.array([ 0.000000 , 0.000000 , 0.000000 , 0.000000 , 0.000000 , 0.001431 , 0.024631 , 0.080421 , 0.147639 , 0.198853 , 0.212278 , 0.178952 , 0.108137 , 0.031772 , 0.015886 , 0.000000 , 0.000000 , 0.000000 , 0.000000 , 0.000000 , 0.000000 , 0.000000 ])
    ndim_tooling_production = np.array([ 0.000000 , 0.000000 , 0.000000 , 0.000000 , 0.000000 , 0.000000 , 0.003078 , 0.034057 , 0.087407 , 0.139896 , 0.175125 , 0.183825 , 0.163857 , 0.120213 , 0.070714 , 0.021215 , 0.000614 , 0.000000 , 0.000000 , 0.000000 , 0.000000 , 0.000000 ])
    ndim_support            = np.array([ 0.000000 , 0.020808 , 0.034207 , 0.042527 , 0.048625 , 0.053283 , 0.056860 , 0.059553 , 0.061479 , 0.062708 , 0.063282 , 0.063219 , 0.062516 , 0.061151 , 0.058848 , 0.056544 , 0.052870 , 0.048088 , 0.041815 , 0.033189 , 0.018429 , 0.000000 ])

    # allocating variables
    engineering         = np.zeros(development_duration)
    manuf_engineering   = np.zeros(development_duration)
    tooling_design      = np.zeros(development_duration)
    tooling_production  = np.zeros(development_duration)
    support             = np.zeros(development_duration)

    # compute non-dimentional costs distributions for the development duration
    for t in range(development_duration):
        t_ref                   = 1./development_duration*(t+1)
        engineering[t]          = sum(ndim_engineering[       ndim_time<=t_ref]) - sum(engineering)
        manuf_engineering[t]    = sum(ndim_manuf_engineering[ ndim_time<=t_ref]) - sum(manuf_engineering)
        tooling_design[t]       = sum(ndim_tooling_design[    ndim_time<=t_ref]) - sum(tooling_design)
        tooling_production[t]   = sum(ndim_tooling_production[ndim_time<=t_ref]) - sum(tooling_production)
        support[t]              = sum(ndim_support[           ndim_time<=t_ref]) - sum(support)

    # compute dimensional cost data, applying non-dimentional costs to each cost fraction
    engineering         *= nrec_engineering
    manuf_engineering   *= nrec_manufacturing
    tooling_design      *= nrec_tooling_design
    tooling_production  *= nrec_tooling_production
    support             *= nrec_support
    facilities           = nrec_facilities * tooling_design

    # pack output
    costs.industrial.non_recurring.cash_flow           = Data()
    costs.industrial.non_recurring.cash_flow.breakdown = Data()
    cash_flow = costs.industrial.non_recurring.cash_flow
    cash_flow.breakdown.engineering         = engineering
    cash_flow.breakdown.manufacturing       = manuf_engineering
    cash_flow.breakdown.tooling_design      = tooling_design
    cash_flow.breakdown.tooling_production  = tooling_production
    cash_flow.breakdown.support             = support
    cash_flow.breakdown.facilities          = facilities
    cash_flow.total = engineering+manuf_engineering+tooling_design+tooling_production+support+facilities

    return

if __name__ == '__main__':

    import SUAVE

    # unpack
    costs = Data()
    costs.industrial = Data()
    costs.industrial.non_recurring = Data()
    costs.industrial.non_recurring.breakdown = Data()

    nrec = costs.industrial.non_recurring.breakdown
    nrec.airframe_engineering      = 1.0e6
    nrec.flight_test               = 1.1e6
    nrec.engines                   = 1.2e6
    nrec.avionics                  = 1.3e6
    nrec.manufacturing_labor       = 1.4e6
    nrec.manufacturing_material    = 1.5e6
    nrec.tooling_development       = 1.6e6
    nrec.tooling_production        = 1.7e6
    nrec.development_support       = 1.8e6
    nrec.quality_control           = 1.9e6
    nrec.test_facilities           = 1.95e6
    nrec.manufacturing_facilities  = 1.97e6
    costs.industrial.development_total_years   = 5

    distribute_non_recurring_cost(costs)

    pass

## @defgroup Methods-Costs-Industrial_Costs Industrial Costs
# These functions provide cost estimates for an aircraft program.
# @ingroup Methods-Costs
from .estimate_hourly_rates          import estimate_hourly_rates
from .estimate_escalation_factor     import estimate_escalation_factor
from .distribute_non_recurring_cost  import distribute_non_recurring_cost
from .compute_industrial_costs       import compute_industrial_costs
## @ingroup Methods-Costs-Industrial_Costs
# compute_industrial_costs.py
#
# Created:  Sep 2016, T. Orra
# Modified: Aug 2019, T. MacDonald
#
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
from SUAVE.Core import Units,Data
from SUAVE.Methods.Costs.Correlations.Industrial_Costs import ( estimate_escalation_factor, \
                                                                estimate_hourly_rates, \
                                                                distribute_non_recurring_cost )

lb_to_kg = 1*Units.lb
lbf_to_N = 1*Units.lbf

# ----------------------------------------------------------------------
#  Compute costs to develop and produce the vehicle (only airplanes)
# ----------------------------------------------------------------------
## @ingroup Methods-Costs-Industrial_Costs
def compute_industrial_costs(vehicle,determine_cash_flow=True):
    """Computes costs for design, development, test, and manufacturing of an airplane program

    Assumptions:
    Production tooling is considered a non-recurring cost

    Source:
    "Airplane Design, Part VIII - Airplane Cost Estimation", Roskam

    Inputs:
    vehicle.costs.industrial.        data dictionary with inputs for costs estimations:
      reference_year                 [-]        reference date for calculations
      production_total_units         [-]        total units to be produced
      units_to_amortize              [-]        number of units to amortize development costs
      prototypes_units               [-]        number of prototypes used in flight test campaign
      avionics_cost                  [$]        user-informed avionics costs
      test_facilities_cost           [$]        user-informed test facilities costs
      manufacturing_facilities_cost  [$]        user-informed manufact. facilities costs
      development_total_years        [-]        total years of development, for cash flow
      aircraft_type                  <string>   for interior costs: 'military' or 'general aviation' or 'regional' or 'commercial' or 'business'.
      difficulty_factor              [-]        1.0 for conventional tech., 1.5 for moderately advanc. tech., 2 for agressive use of adv. tech.
      cad_factor                     [-]        1.2 for learning CAD, 1.0 for manual, 0.8 for experienced
      stealth                        [-]        0 for non-stealth, 1 for stealth
      material_factor                [-]        1.0 for conventional Al, 1.5 for stainless steel, 2~2.5 for composites, 3 for carbon fiber
    vehicle.mass_properties.empty    [kg]
    vehicle.networks.turbofan.
      number_of_engines              [-]
      sealevel_static_thrust         [N]
    vehicle.passengers               [-]

    Outputs:
    vehicle.costs.industrial.
      unit_cost                      [$] total cost of each airplane produced
      non_recurring.total            [$]
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
        total                        [$]
      recurring.total                [$]
      recurring.breakdown.           
        airframe_engineering         [$]
        interior                     [$]
        manufacturing_labor          [$]
        manufacturing_material       [$]
        quality_control              [$]
        engines                      [$]
        avionics                     [$]
        total                        [$]

    Properties Used:
    N/A
    """          
    # Unpack
    costs                   = vehicle.costs.industrial
    reference_year          = costs.reference_year
    total_production        = costs.production_total_units
    n_prototypes            = costs.prototypes_units
    development_total_years = vehicle.costs.industrial.development_total_years
    ac_type 		    = costs.aircraft_type.lower()

    # define number of airplanes to amortize non-recurring costs
    if costs.units_to_amortize:
        amortize_units = costs.units_to_amortize
    else:
        amortize_units = costs.production_total_units

    # user-defined costs
    avionics_costs        = costs.avionics_cost
    test_facilities_cost  = costs.test_facilities_cost
    manuf_facilities_cost = costs.manufacturing_facilities_cost


        # factors to account for design especific characteristics
    F_diff  = costs.difficulty_factor # (1.0 for conventional, 1.5 for moderately advanc. tech., 2 for agressive use of adv. tech.)
    F_CAD   = costs.cad_factor #(1.2 for learning, 1.0 for manual, 0.8 for experienced)
    F_obs   = 1 + 3. * costs.stealth #(0 for non-stealth, 1 for stealth)
    F_mat   = costs.material_factor #1 for conventional Al, 1.5 for stainless steel, 2~2.5 for composites, 3 for carbon fiber

    # general airplane data
    weight            = 0.62 * vehicle.mass_properties.empty / lb_to_kg # correlation for AMPR weight, typical 62% * Empty weight
    n_engines         = vehicle.networks.turbofan.number_of_engines
    sls_thrust        = vehicle.networks.turbofan.sealevel_static_thrust / lbf_to_N
    n_pax             = vehicle.passengers

    # estimate escalation factor
    method_reference_year = 1970
    escalation_factor     = estimate_escalation_factor(reference_year) / estimate_escalation_factor(method_reference_year)

    # estimate hourly rates
    hourly_rates          = estimate_hourly_rates(reference_year)
    rates_engineering     = hourly_rates.engineering
    rates_tooling         = hourly_rates.tooling
    rates_manufacturing   = hourly_rates.manufacturing
    rates_quality_control = hourly_rates.quality_control
    costs.hourly_rates    = hourly_rates

    # determine equivalent airspeed from MMO (assuming HP=35kft)
    MMO   = vehicle.envelope.maximum_mach_operational
    speed = MMO * 321.32 # KEAS

    # =============================================
    # Non-recurring costs estimation - DT&E costs
    # =============================================

    # Airframe Engineering (DT&E)
    AENGHD = 0.0396 * weight ** 0.791 * speed ** 1.526 * n_prototypes ** 0.183 * F_diff * F_CAD
    AENGCD = AENGHD * rates_engineering # airframe eng costs development

    # Development Support (DT&E)
    DSC = 0.008325 * weight ** 0.873 * speed ** 1.89 * n_prototypes ** 0.346 * F_diff * escalation_factor

    # Engine Cost for prototypes
    eng_unit_cost = 520. * sls_thrust ** 0.8356 * escalation_factor * 0.235 # 0.235 to account for cost difference between 1970 and 1998 (roskam vs nicolai method)
    ECD = eng_unit_cost * n_prototypes * n_engines * 1.10 #10% to account for spare engine

    # Avionics cost for prototypes
    AVCOST = avionics_costs * n_prototypes

    # Manufacturing Labor (DT&E)
    MLHD = 28.984 * weight ** 0.74 * speed ** 0.543 * n_prototypes ** 0.524 * F_diff
    MLCD = MLHD * rates_manufacturing

    # Manufacturing materials (DT&E)
    MMED = 2.0 * 37.632 * F_mat * weight ** 0.689 * speed ** 0.624 * n_prototypes ** 0.792 * escalation_factor

    # Tooling (DT&E)
    THD = 4.0127 * weight**0.764 * speed ** 0.899 * n_prototypes**0.178*(0.33)**0.066 * F_diff
    TCD = THD * rates_tooling # tooling costs for prototypes

    # Tooling (Production)
    THP = 4.0127 * weight**0.764 * speed ** 0.899 * total_production**0.178*(0.33)**0.066 * F_diff
    TCP = THP * rates_tooling - TCD # tooling costs for total production

    # Quality Control (DT&E)
    QCHD = 0.130 * MLHD
    QCCD = QCHD * rates_quality_control

    # Flight Test Operations (DT&E)
    FTC = 0.001244 * weight ** 1.16 * speed ** 1.371 * n_prototypes ** 1.281 * F_diff * F_obs * escalation_factor

    # TOTAL NON-RECURRING COSTS
    TNRCE  = AENGCD         # airframe engineering
    TNRCDS = DSC            # development support engineering
    TNRCFT = FTC            # flight test operation costs
    TNRCEN = ECD            # engine cost for prototypes
    TNRCAV = AVCOST         # avionics cost for prototypes
    TNRCTD = TCD            # tooling for development
    TNRCTP = TCP            # tooling for production
    TNRCMN = MLCD           # manufacturing labor
    TNRCMT = MMED           # manufacturing materials
    TNRCQA = QCCD           # quality and control
    TNRTF  = test_facilities_cost # test facilities cost
    TNRMF  = manuf_facilities_cost # manufacturing facilities costs

    # sum all components above
    TNRC = TNRCE + TNRCDS + TNRCFT + TNRCEN + TNRCAV + TNRCTD + TNRCTP + TNRCMN + TNRCMT + TNRCQA + TNRTF + TNRMF

    # append in breakdown structure
    cost = Data()
    cost.non_recurring = Data()
    nrec = cost.non_recurring
    nrec.airframe_engineering   = TNRCE
    nrec.development_support    = TNRCDS
    nrec.flight_test            = TNRCFT
    nrec.engines                = TNRCEN
    nrec.avionics               = TNRCAV
    nrec.tooling_development    = TNRCTD
    nrec.tooling_production     = TNRCTP
    nrec.manufacturing_labor    = TNRCMN
    nrec.manufacturing_material = TNRCMT
    nrec.quality_control        = TNRCQA
    nrec.test_facilities          = TNRTF
    nrec.manufacturing_facilities = TNRMF
    nrec.total                    = TNRC

    # ================================
    # Recurring costs estimation
    # ================================

    # Airframe Engineering (Production)
    AENGHP = 2.0 * (0.0396 * weight ** 0.791 * speed ** 1.526 *(n_prototypes + total_production)**0.183 * F_diff * F_CAD)
    AENGCP = AENGHP * rates_engineering - AENGCD

    # Engine Cost
    ECP = eng_unit_cost * total_production * n_engines

    # Avionics cost
    AVCOSTR = avionics_costs * total_production

    # Interiors cost
    if ac_type == 'military':
        interior_index =    0.
    if ac_type == 'general aviation':
        interior_index =  500.
    if ac_type == 'regional':
        interior_index = 1000.
    if ac_type == 'commercial':
        interior_index = 2000.
    if ac_type == 'business':
        interior_index = 3000.

    INTRC = interior_index * n_pax * total_production * escalation_factor * 0.296

    # Manufacturing Labor (Production)
    MLHP = 1.3 * 28.984 * weight ** 0.74 * speed ** 0.543 * total_production ** 0.524 * F_diff
    MLCP = MLHP * rates_manufacturing - MLCD

    # Manufacturing materials and equipment (Production)
    MMEP = 2.0 * 37.632 * F_mat * weight ** 0.689 * speed ** 0.624 * total_production ** 0.792 * escalation_factor
    MMEP = MMEP - MMED

    # Quality Control (Production)
    QCHP = 0.130 * MLHP
    QCCP = QCHP * rates_quality_control

    # TOTAL RECURRING COSTS
    TRCE  = AENGCP      # airframe engineering
    TRCI  = INTRC       # interior costs
    TRCMN = MLCP        # manufacturing labor
    TRCMT = MMEP        # manufacturing materials
    TRCQA = QCCP        # quality and control
    TRCEN = ECP         # engine cost for production
    TRCAV = AVCOSTR     # avionics cost for production

    # sum all components above
    TRC = TRCE + TRCI + TRCMN + TRCMT + TRCQA + TRCEN + TRCAV

    # store rec breakdown
    cost.recurring = Data()
    rec = cost.recurring
    rec.airframe_engineering        = TRCE
    rec.interior                    = TRCI
    rec.manufacturing_labor         = TRCMN
    rec.manufacturing_material      = TRCMT
    rec.quality_control             = TRCQA
    rec.engines                     = TRCEN
    rec.avionics                    = TRCAV
    rec.total                       = TRC

    # Total cost per unit
    unit_cost = TRC / total_production + TNRC / amortize_units
    vehicle.costs.industrial.unit_cost  = unit_cost

    # packing results
    vehicle.costs.industrial.non_recurring = Data()
    vehicle.costs.industrial.non_recurring.total = TNRC
    vehicle.costs.industrial.non_recurring.breakdown = nrec

    vehicle.costs.industrial.recurring = Data()
    vehicle.costs.industrial.recurring.total     = TRC
    vehicle.costs.industrial.recurring.breakdown = rec

    # distribute non-recurring costs on time
    if not development_total_years:
        vehicle.costs.industrial.development_total_years = 5.
    if determine_cash_flow:
        distribute_non_recurring_cost(vehicle.costs) # returns costs.industrial.non_recurring.cash_flow

    return

# ----------------------------------------------------------------------
#   Module Tests
# ----------------------------------------------------------------------
# this will run from command line, put simple tests for your code here

## @ingroup Methods-Costs-Industrial_Costs
def call_print(config):
    """Prints precalculated costs for an airplane program.

    Assumptions:
    N/A

    Source:
    N/A

    Inputs:
    config.tag                <string>
    config.costs.industrial.
      non_recurring.total     [$]
      unit_cost               [$]
      recurring.total         [$]
      production_total_units  [$]

    Outputs:
    None

    Properties Used:
    N/A
    """      

    nrec = config.costs.industrial.non_recurring.total / 1e6
    unit = config.costs.industrial.unit_cost / 1e6
    abc = config.costs.industrial.recurring.total / config.costs.industrial.production_total_units / 1e6

    print('{:10s} Total non reccuring USM: {:7.2f} , Unit: {:7.2f} , Materials:{:7.2f}'.format(config.tag,nrec,unit,abc))

if __name__ == '__main__':

    import SUAVE

#==================================
    config = SUAVE.Vehicle()
    config.tag = 'B777-200'
#==================================
    manufact_costs = config.costs.industrial

    gt_engine       = SUAVE.Components.Energy.Networks.Turbofan()
    gt_engine.tag   = 'turbofan'
    gt_engine.number_of_engines                 = 2
    gt_engine.sealevel_static_thrust            = 110000 * Units.lbf
    config.append_component(gt_engine)
    config.mass_properties.empty                 = 326000 * Units.lb
    config.envelope.maximum_mach_operational     = 0.89
    config.passengers                            = 250

    manufact_costs.avionics_cost                 = 2500000.
    manufact_costs.production_total_units        = 500
    manufact_costs.units_to_amortize             = 500
    manufact_costs.prototypes_units              = 9
    manufact_costs.reference_year                = 2004
    manufact_costs.first_flight_year             = 1993

    manufact_costs.difficulty_factor = 1.75         # (1.0 for conventional, 1.5 for moderately advanc. tech., 2 for agressive use of adv. tech.)
    manufact_costs.cad_factor        = 1.2          # (1.2 for learning, 1.0 for manual, 0.8 for experienced)
    manufact_costs.stealth           = 0.0          # (0 for non-stealth, 1 for stealth)
    manufact_costs.material_factor   = 1.0          # (1 for conventional Al, 1.5 for stainless steel, 2~2.5 for composites, 3 for carbon fiber)
    manufact_costs.aircraft_type     = 'commercial' # ('military','general aviation','regional','commercial','business')

    compute_industrial_costs(config)
    call_print(config)

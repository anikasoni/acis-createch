def compute_cycle_cost(
    demould_time_h, curing_method, automation_level,
    n_molds, target_cycle_h,
    labor_cost_per_h=450, energy_cost_per_kwh=8.0,
    mold_holding_per_h=200, capex_amort=150,
    throughput_value_per_h=800
):
    energy_kwh = {
        'ambient': 2.0, 'polythene_wrap': 3.5,
        'wet_burlap': 3.0, 'steam': 22.0, 'heated_enclosure': 18.0
    }
    energy       = energy_kwh.get(curing_method, 3.0) * energy_cost_per_kwh
    labor_factor = [1.0, 0.7, 0.4][automation_level]
    labor        = demould_time_h * labor_cost_per_h * labor_factor
    mold_hold    = demould_time_h * mold_holding_per_h
    capex        = capex_amort * [1.0, 1.3, 2.0][automation_level]
    delay_ratio  = max(0, (demould_time_h - target_cycle_h) / target_cycle_h)
    schedule_penalty = delay_ratio * n_molds * throughput_value_per_h
    total = energy + labor + mold_hold + capex + schedule_penalty
    return {
        'energy': round(energy, 2),
        'labor': round(labor, 2),
        'mold_holding': round(mold_hold, 2),
        'capex': round(capex, 2),
        'schedule_penalty': round(schedule_penalty, 2),
        'total': round(total, 2)
    }
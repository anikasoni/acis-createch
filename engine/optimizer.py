import numpy as np
import pyswarms as ps
from engine.maturity import compute_maturity, strength_at_maturity
from engine.cost_model import compute_cycle_cost

def build_fitness_function(cfg, weather_temps, resources, w_time=0.5, w_cost=0.5):
    threshold = cfg['strength_threshold_mpa']
    target_h  = cfg['target_cycle_hours']
    S28       = 35.0
    TIME_MAX  = 30.0
    COST_MAX  = 50000
    INFEASIBLE= 1e6

    def fitness(particles):
        scores = []
        for p in particles:
            demould_h     = float(np.clip(p[0], 6, 30))
            curing_idx    = int(np.clip(round(p[1]), 0, len(resources)-1))
            auto_level    = int(np.clip(round(p[2]), 0, 2))
            curing_method = resources[curing_idx]

            # Hard constraint
            hours = int(demould_h)
            temps = weather_temps[:hours] if len(weather_temps) >= hours else weather_temps
            maturity = compute_maturity(temps, [1.0]*len(temps))
            pred_strength = strength_at_maturity(maturity, S28)

            if pred_strength < threshold:
                scores.append(INFEASIBLE)
                continue

            cost = compute_cycle_cost(
                demould_time_h=demould_h,
                curing_method=curing_method,
                automation_level=auto_level,
                n_molds=cfg['mold_count'],
                target_cycle_h=target_h,
                labor_cost_per_h=cfg.get('labor_cost_per_hour', 450),
                mold_holding_per_h=cfg.get('mold_holding_cost_per_hour', 200),
                throughput_value_per_h=cfg.get('throughput_value_per_hour', 800)
            )['total']

            f = w_time * (demould_h / TIME_MAX) + w_cost * (cost / COST_MAX)
            scores.append(f)
        return np.array(scores)

    return fitness


def run_pso(cfg, weather_temps, resources, w_time=0.5, w_cost=0.5,
            n_particles=30, iters=50):
    fitness_fn = build_fitness_function(
        cfg, weather_temps, resources, w_time, w_cost
    )
    bounds = (np.array([6.0, 0.0, 0.0]),
              np.array([30.0, float(len(resources)-1), 2.0]))
    optimizer = ps.single.GlobalBestPSO(
        n_particles=n_particles, dimensions=3,
        options={'c1': 0.5, 'c2': 0.3, 'w': 0.9},
        bounds=bounds
    )
    best_cost, best_pos = optimizer.optimize(fitness_fn, iters=iters, verbose=False)
    return best_pos, best_cost, optimizer.pos_history
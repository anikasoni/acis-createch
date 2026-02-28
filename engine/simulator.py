import simpy
import numpy as np
import pandas as pd

def run_yard_simulation(n_molds, n_beds, demould_time_h,
                         reset_time_h=1.5, sim_duration_h=72,
                         temps_C=None, avg_temp=28, wc_ratio=0.45):
    results = {'cycle_times': [], 'elements_produced': 0}
    env  = simpy.Environment()
    beds = simpy.Resource(env, capacity=n_beds)

    def casting_process(env, beds, demould_h, reset_h):
        while True:
            with beds.request() as req:
                yield req
                start = env.now
                yield env.timeout(demould_h)
                yield env.timeout(reset_h)
                results['cycle_times'].append(env.now - start)
                results['elements_produced'] += 1

    for i in range(n_molds):
        env.process(casting_process(env, beds, demould_time_h, reset_time_h))
    env.run(until=sim_duration_h)

    results['avg_cycle_h'] = (
        np.mean(results['cycle_times']) if results['cycle_times']
        else demould_time_h
    )
    return results


def monte_carlo_simulation(base_params, n_runs=300):
    records = []
    for _ in range(n_runs):
        # Source 1: Weather variability
        t1 = run_yard_simulation(**{
            **base_params,
            'demould_time_h': base_params['demould_time_h'] *
                              (1 + np.random.normal(0, 0.08))
        })
        records.append({'source': 'Weather Variability',
                        'cycle_time_h': t1['avg_cycle_h']})

        # Source 2: Mix design tolerance
        wc_var = base_params.get('wc_ratio', 0.45) + np.random.uniform(-0.02, 0.02)
        t2 = run_yard_simulation(**{
            **base_params,
            'demould_time_h': base_params['demould_time_h'] *
                              (1 + (wc_var - 0.45) * 0.5)
        })
        records.append({'source': 'Mix Design Tolerance',
                        'cycle_time_h': t2['avg_cycle_h']})

        # Source 3: Curing efficiency
        eff = np.random.uniform(0.90, 1.10)
        t3 = run_yard_simulation(**{
            **base_params,
            'demould_time_h': base_params['demould_time_h'] / eff
        })
        records.append({'source': 'Curing Efficiency',
                        'cycle_time_h': t3['avg_cycle_h']})

    return pd.DataFrame(records)
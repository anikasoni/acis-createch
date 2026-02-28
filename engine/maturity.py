import numpy as np

T0 = -10.0  # Datum temperature (°C) — standard per IS:456 / BS EN 13670

def compute_maturity(temps_C: list, times_h: list) -> float:
    """
    Nurse-Saul Maturity Index
    M = Σ (T - T0) × Δt   [°C·hours]
    temps_C : hourly temperatures during curing
    times_h : time intervals (all 1.0 for hourly data)
    """
    return sum(
        (T - T0) * dt
        for T, dt in zip(temps_C, times_h)
        if T > T0
    )


def strength_at_maturity(maturity_index: float, S28: float,
                          ku: float = 140.0) -> float:
    """
    Strength-Maturity relationship (exponential form)
    S(M) = S28 × exp( -sqrt( ku / M ) )
    ku : rate constant — depends on curing method
         140 = ambient, 110 = polythene, 80 = steam
    """
    return S28 * np.exp(-np.sqrt(ku / max(maturity_index, 0.01)))


def strength_curve(temps_C: list, S28: float,
                   hours: int = 48, ku: float = 140.0) -> dict:
    """
    Returns hourly strength predictions for plotting.
    temps_C : hourly temps over the curing period
    """
    curve = {'hour': [], 'maturity': [], 'strength_mpa': []}
    cumulative_maturity = 0.0

    for h in range(min(hours, len(temps_C))):
        cumulative_maturity += max(0, temps_C[h] - T0) * 1.0
        s = strength_at_maturity(cumulative_maturity, S28, ku)
        curve['hour'].append(h + 1)
        curve['maturity'].append(round(cumulative_maturity, 2))
        curve['strength_mpa'].append(round(s, 3))

    return curve


def earliest_safe_demould(temps_C: list, S28: float,
                           threshold_mpa: float,
                           ku: float = 140.0) -> int | None:
    """
    Returns the earliest hour at which strength >= threshold.
    Returns None if threshold not reached within temps_C window.
    """
    curve = strength_curve(temps_C, S28, hours=len(temps_C), ku=ku)
    for h, s in zip(curve['hour'], curve['strength_mpa']):
        if s >= threshold_mpa:
            return h
    return None
import numpy as np
import pandas as pd

np.random.seed(42)
N = 10000

# ── Mix parameters (IS:456 ranges) ──
wc_ratio       = np.random.uniform(0.35, 0.55, N)
cement_content = np.random.uniform(300, 450, N)
admixture_pct  = np.random.uniform(0.3, 1.2, N)
curing_method  = np.random.choice([0, 1, 2], N)
avg_temp_C     = np.random.uniform(5, 45, N)
humidity_pct   = np.random.uniform(30, 95, N)

# ── Force risky scenarios into 35% of dataset ──
n_risky = int(N * 0.20)
demould_time                = np.random.uniform(8, 24, N)
demould_time[:n_risky]      = np.random.uniform(6, 12, n_risky)
avg_temp_C[:n_risky]        = np.random.uniform(5, 18, n_risky)
wc_ratio[:n_risky]          = np.random.uniform(0.46, 0.55, n_risky)

# ── Shuffle ──
idx = np.random.permutation(N)
wc_ratio       = wc_ratio[idx]
cement_content = cement_content[idx]
admixture_pct  = admixture_pct[idx]
curing_method  = curing_method[idx]
avg_temp_C     = avg_temp_C[idx]
humidity_pct   = humidity_pct[idx]
demould_time   = demould_time[idx]

# ── 28-day strength (Abrams' Law) ──
S28 = 96.5 / (8.2 ** wc_ratio) + (cement_content - 350) * 0.04
S28 = np.clip(S28 + np.random.normal(0, 2, N), 15, 65)

# ── Nurse-Saul maturity ──
T0 = -10
maturity = (avg_temp_C - T0) * demould_time

# ── Strength at demould ──
ku = np.where(curing_method == 2,  80,    # steam — very fast
     np.where(curing_method == 1, 110,    # polythene — fast
                                  140))   # ambient — moderate
strength_at_demould = S28 * np.exp(-np.sqrt(ku / np.maximum(maturity, 0.01)))
strength_at_demould = np.clip(
    strength_at_demould + np.random.normal(0, 1.5, N), 2, 60
)

# ── Risk labels ──
threshold = 20.0
margin = strength_at_demould - threshold

risk = np.where(margin < 0,   2,
       np.where(margin < 3,   1,
                               0))

# ── Build dataframe ──
df = pd.DataFrame({
    'wc_ratio':            wc_ratio,
    'cement_content':      cement_content,
    'admixture_pct':       admixture_pct,
    'curing_method':       curing_method,
    'avg_temp_C':          avg_temp_C,
    'humidity_pct':        humidity_pct,
    'demould_time_h':      demould_time,
    'maturity_index':      maturity,
    'S28_predicted':       S28,
    'strength_at_demould': strength_at_demould,
    'risk_label':          risk
})

df.to_csv('data/concrete_data.csv', index=False)
print(f'Generated {N} rows. Risk distribution:')
print(df['risk_label'].value_counts().sort_index())
print(f'\nStrength stats:')
print(df['strength_at_demould'].describe().round(2))
print(f'\nSample HIGH risk rows:')
print(df[df['risk_label']==2][['wc_ratio','avg_temp_C','demould_time_h','strength_at_demould']].head(3).round(2))
import numpy as np
import pandas as pd

SEED = 42
N = 200
rng = np.random.default_rng(SEED)

# ---- Feature generation (realistic, mildly correlated lifestyle habits) ----
stress = np.clip(rng.normal(5.5, 2.0, N), 1, 10)

hydration = np.clip(rng.normal(2.4, 0.6, N), 1.0, 4.0)
sleep = np.clip(rng.normal(7.0, 1.1, N) - 0.10 * (stress - 5.5), 4.0, 10.0)   # stress erodes sleep
sugar = np.clip(rng.normal(2.5, 1.4, N) + 0.15 * (stress - 5.5), 0, 6)        # stress-eating
spf = np.clip(rng.normal(6.0, 2.2, N), 1, 10)
routine = np.clip(rng.normal(6.0, 2.0, N) + 0.3 * (spf - 6.0), 1, 10)         # habits cluster

# ---- Base linear equation ----
INTERCEPT = 30.0
COEF = {
    "hydration_liters":       +4.5,
    "sleep_hours":            +3.0,
    "spf_consistency":        +2.0,
    "stress_level":           -2.8,
    "refined_sugar_servings": -2.6,
    "routine_discipline":     +2.4,
}

score = (INTERCEPT
         + COEF["hydration_liters"]       * hydration
         + COEF["sleep_hours"]            * sleep
         + COEF["spf_consistency"]        * spf
         + COEF["stress_level"]           * stress
         + COEF["refined_sugar_servings"] * sugar
         + COEF["routine_discipline"]     * routine)

# ---- Gaussian human variance (std in 3-5 range) ----
noise = rng.normal(0, 3.5, N)
score = np.clip(score + noise, 0, 100)

df = pd.DataFrame({
    "hydration_liters":       hydration.round(1),
    "sleep_hours":            sleep.round(1),
    "spf_consistency":        spf.round(1),
    "stress_level":           stress.round(1),
    "refined_sugar_servings": sugar.round(1),
    "routine_discipline":     routine.round(1),
    "skin_health_score":      score.round(2),
})

df.to_csv("skin_barrier_dataset.csv", index=False)

# ---- Sanity check: fit OLS with numpy and report R^2 ----
X = np.column_stack([np.ones(N), df.iloc[:, :6].values])
y = df["skin_health_score"].values
beta, *_ = np.linalg.lstsq(X, y, rcond=None)
r2 = 1 - ((y - X @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()
print(f"Saved {len(df)} rows -> skin_barrier_dataset.csv | R^2 = {r2:.3f}")
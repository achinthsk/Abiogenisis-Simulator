"""
model.py — ML layer on top of engine.py.
Trains GradientBoostingRegressors (one per output) to interpolate the rule engine
across the full parameter space, enabling fast predictions and generalisation to
unseen planet configurations.

Dependencies: pip install scikit-learn numpy pandas joblib
"""

import json
import random
import joblib
import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split

from engine import run_simulation, PLANET, validate

MODEL_FILE        = "model.pkl"
FEATURE_NAMES_FILE = "feature_names.json"

# Categorical → ordinal encoding. Ordinal encoding is intentional here:
# these categories have a natural ordering (e.g. low < medium < high)
# that GBR can exploit without exploding feature dimensionality.
FEATURE_ENCODING = {
    'uv':               {'low': 0, 'medium': 1, 'high': 2},
    'clay':             {'none': 0, 'kaolinite': 1, 'montmorillonite': 2},
    'metal_ions':       {'low': 0, 'medium': 1, 'high': 2},
    'sulfur_phosphate': {'low': 0, 'medium': 1, 'high': 2},
    'wet_dry':          {'none': 0, 'tidal': 1, 'volcanic': 2},
    'solvent':          {'methane': 0, 'ammonia': 1, 'water': 2},
    'hydrothermal':     {True: 1, False: 0},
}

# Canonical feature order — must be identical at train and predict time.
FEATURE_ORDER = [
    'atm_CH4', 'atm_NH3', 'atm_H2', 'atm_CO2', 'atm_N2', 'atm_H2O',
    'pressure', 'uv', 'lightning', 'geothermal', 'temperature',
    'ocean_pH', 'salinity', 'hydrothermal', 'clay',
    'metal_ions', 'sulfur_phosphate', 'gravity', 'wet_dry', 'solvent',
]

# Target outputs — match run_simulation() result keys exactly.
OUTPUT_KEYS = ['s1', 's2', 's3', 's4', 's5', 'overall_score']

# Module-level model cache (loaded once, reused across predict() calls).
_model_cache = None


# =============================================================================
# COMPONENT 1 — Training data generator
# =============================================================================

def generate_training_data(n_samples: int = 50000):
    """Randomly sample the planet parameter space and label with the rule engine.

    Design notes:
    - atm_N2 is derived (not sampled) so the atmosphere always sums to 100.
      This mirrors how real planets work and avoids the model learning to
      route around the atmosphere constraint.
    - Solvent is weighted toward water (60 %) because habitable-zone planets
      are the primary use case; ammonia and methane worlds are rarer targets.
    - We run the real engine for every sample, so the ML model is learning a
      smooth approximation of the rule engine, not inventing new science.
    """
    categorical = {
        'uv':               ['low', 'medium', 'high'],
        'clay':             ['none', 'kaolinite', 'montmorillonite'],
        'metal_ions':       ['low', 'medium', 'high'],
        'sulfur_phosphate': ['low', 'medium', 'high'],
        'wet_dry':          ['none', 'tidal', 'volcanic'],
    }
    solvent_choices = ['water'] * 60 + ['ammonia'] * 25 + ['methane'] * 15

    X_rows, y_rows = [], []

    for _ in range(n_samples):
        # Sample individual gases; clamp to valid range before deriving N2.
        ch4  = random.uniform(0, 40)
        nh3  = random.uniform(0, 20)
        h2   = random.uniform(0, 40)
        co2  = random.uniform(0, 80)
        h2o  = random.uniform(0, 10)
        used = ch4 + nh3 + h2 + co2 + h2o
        n2   = max(0, 100 - used)

        # If the five gases exceed 100, scale them down proportionally.
        if used > 100:
            scale = 100 / used
            ch4, nh3, h2, co2, h2o = (v * scale for v in (ch4, nh3, h2, co2, h2o))
            n2 = 0.0

        p = {
            'atm_CH4':   ch4,
            'atm_NH3':   nh3,
            'atm_H2':    h2,
            'atm_CO2':   co2,
            'atm_N2':    n2,
            'atm_H2O':   h2o,
            'pressure':  random.uniform(0.1, 10),
            'uv':        random.choice(categorical['uv']),
            'lightning': random.uniform(0, 5000),
            'geothermal':random.uniform(0, 500),
            'temperature':random.uniform(-100, 200),
            'ocean_pH':  random.uniform(2, 12),
            'salinity':  random.uniform(0, 400),
            'hydrothermal': random.choice([True, False]),
            'clay':      random.choice(categorical['clay']),
            'metal_ions':random.choice(categorical['metal_ions']),
            'sulfur_phosphate': random.choice(categorical['sulfur_phosphate']),
            'gravity':   random.uniform(0.3, 3),
            'wet_dry':   random.choice(categorical['wet_dry']),
            'solvent':   random.choice(solvent_choices),
        }

        result = run_simulation(p)
        stages = result['stages']
        y_rows.append([stages['s1'], stages['s2'], stages['s3'],
                        stages['s4'], stages['s5'], result['overall_score']])
        X_rows.append(_encode(p))

    return np.array(X_rows, dtype=np.float32), np.array(y_rows, dtype=np.float32)


# =============================================================================
# COMPONENT 2 — Model trainer
# =============================================================================

def train_model(X: np.ndarray, y: np.ndarray):
    """Train one GradientBoostingRegressor per output via MultiOutputRegressor.

    GBR is chosen over linear models because the engine has hard thresholds
    (stage N blocked if stage N-1 < 0.4), creating non-linear discontinuities
    that tree-based ensembles handle well.

    MultiOutputRegressor trains independent estimators per column — appropriate
    here because each stage score depends on the previous stage in a chain, so
    sharing estimators (e.g. via a single vectorised model) would conflate
    relationships the rule engine keeps separate.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    base = GradientBoostingRegressor(
        n_estimators=200,    # 200 trees balances accuracy vs training time
        max_depth=5,         # deep enough to learn threshold interactions
        learning_rate=0.05,  # conservative LR with more trees → lower variance
        subsample=0.8,       # stochastic GBR reduces overfitting
        random_state=42,
    )
    model = MultiOutputRegressor(base, n_jobs=-1)
    model.fit(X_train, y_train)

    # Per-output R² on held-out test set.
    y_pred = model.predict(X_test)
    scores = {k: round(float(np.corrcoef(y_test[:, i], y_pred[:, i])[0, 1] ** 2), 4)
              for i, k in enumerate(OUTPUT_KEYS)}

    joblib.dump(model, MODEL_FILE)
    with open(FEATURE_NAMES_FILE, 'w') as f:
        json.dump(FEATURE_ORDER, f)

    print(f"Model saved to {MODEL_FILE}")
    return model, scores


# =============================================================================
# COMPONENT 3 — Predictor
# =============================================================================

def predict(planet_params: dict) -> dict:
    """Return engine-format predictions for a planet dict.

    Falls back to the rule engine if model.pkl is missing — this lets the rest
    of the app call predict() without caring whether training has run yet.
    """
    global _model_cache

    # Merge supplied params over PLANET defaults so partial dicts are valid.
    p = {**PLANET, **planet_params}

    if _model_cache is None:
        try:
            _model_cache = joblib.load(MODEL_FILE)
        except FileNotFoundError:
            result = run_simulation(p)
            result['prediction_source'] = 'rule_engine'
            return result

    x = np.array([_encode(p)], dtype=np.float32)
    y_pred = _model_cache.predict(x)[0]

    # Clamp all outputs to [0, 1] — GBR can extrapolate slightly outside range.
    s1, s2, s3, s4, s5, overall = (float(np.clip(v, 0, 1)) for v in y_pred)

    # Reconstruct verdict using the same thresholds as engine._build_result.
    if overall >= 0.75:   verdict = "HIGHLY FAVOURABLE"
    elif overall >= 0.55: verdict = "FAVOURABLE"
    elif overall >= 0.35: verdict = "MARGINAL"
    else:                 verdict = "UNLIKELY"

    return {
        'stages':           {'s1': round(s1, 4), 's2': round(s2, 4),
                              's3': round(s3, 4), 's4': round(s4, 4), 's5': round(s5, 4)},
        'reasons':          {k: '(ML prediction)' for k in ['s1','s2','s3','s4','s5']},
        'biochemistry':     '(ML prediction)',
        'proto_life_score': round(s5, 4),
        'overall_score':    round(overall, 4),
        'verdict':          verdict,
        'prediction_source': 'ml_model',
    }


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _encode(p: dict) -> list:
    """Convert a planet dict to a fixed-length numeric feature vector.
    Categorical fields are replaced with their ordinal integer values.
    """
    row = []
    for key in FEATURE_ORDER:
        val = p[key]
        if key in FEATURE_ENCODING:
            val = FEATURE_ENCODING[key].get(val, 0)
        row.append(float(val))
    return row


# =============================================================================
# VALIDATION
# =============================================================================

def validate_against_known_planets():
    """Compare ML and rule-engine outputs against scientific consensus ranking.

    Consensus order (highest → lowest life potential, per astrobiological literature):
    Early Earth > TRAPPIST-1e > K2-18b > TOI-270d > Titan-like > Mars-like > Venus-like

    We compute Spearman rank correlation between our scores and this ordering.
    Spearman ρ = 1.0 means perfect rank agreement; < 0 means inverted.

    All six atm_ keys are set explicitly on every planet so the atmosphere
    always sums to exactly 100 with no inheritance from PLANET defaults and
    no silent rescaling fallback.
    """
    VALIDATION_PLANETS = {
        # Hadean/Archean Earth ~3.8 Gya — matches PLANET defaults exactly,
        # but written out in full so no key is inherited implicitly.
        'Early Earth': {
            'atm_CH4': 10.0, 'atm_NH3': 5.0, 'atm_H2': 15.0,
            'atm_CO2': 20.0, 'atm_N2': 45.0, 'atm_H2O': 5.0,
            'pressure': 1.0,    'uv': 'medium',  'lightning': 500,
            'geothermal': 120,  'temperature': 40.0, 'ocean_pH': 6.5,
            'salinity': 35.0,   'hydrothermal': True,
            'clay': 'montmorillonite', 'metal_ions': 'medium',
            'sulfur_phosphate': 'medium', 'gravity': 1.0,
            'wet_dry': 'tidal', 'solvent': 'water',
        },
        # Mars-like: thin CO2 atmosphere, frozen, no liquid water.
        # CO2 85 + N2 8 + CH4 2 + H2 2 + H2O 3 + NH3 0 = 100.
        'Mars-like': {
            'atm_CH4': 2.0,  'atm_NH3': 0.0,  'atm_H2': 2.0,
            'atm_CO2': 85.0, 'atm_N2': 8.0,   'atm_H2O': 3.0,
            'pressure': 0.006,  'uv': 'high',    'lightning': 10,
            'geothermal': 30,   'temperature': -40.0, 'ocean_pH': 7.0,
            'salinity': 0.0,    'hydrothermal': False,
            'clay': 'none',     'metal_ions': 'low',
            'sulfur_phosphate': 'low', 'gravity': 0.38,
            'wet_dry': 'none',  'solvent': 'methane',
        },
        # Venus-like: crushing CO2 atmosphere, extreme heat, no liquid solvent.
        # CO2 96 + N2 3.5 + H2O 0.5 + CH4 0 + NH3 0 + H2 0 = 100.
        'Venus-like': {
            'atm_CH4': 0.0,  'atm_NH3': 0.0,  'atm_H2': 0.0,
            'atm_CO2': 96.0, 'atm_N2': 3.5,   'atm_H2O': 0.5,
            'pressure': 92.0,   'uv': 'medium',  'lightning': 50,
            'geothermal': 80,   'temperature': 460.0, 'ocean_pH': 7.0,
            'salinity': 0.0,    'hydrothermal': False,
            'clay': 'none',     'metal_ions': 'low',
            'sulfur_phosphate': 'low', 'gravity': 0.9,
            'wet_dry': 'none',  'solvent': 'methane',
        },
        # TRAPPIST-1e: temperate rocky planet, tidal locking, liquid water plausible.
        # CH4 5 + NH3 3 + H2 7 + CO2 15 + N2 65 + H2O 5 = 100.
        'TRAPPIST-1e': {
            'atm_CH4': 5.0,  'atm_NH3': 3.0,  'atm_H2': 7.0,
            'atm_CO2': 15.0, 'atm_N2': 65.0,  'atm_H2O': 5.0,
            'pressure': 1.0,    'uv': 'low',     'lightning': 300,
            'geothermal': 120,  'temperature': -40.0, 'ocean_pH': 6.5,
            'salinity': 35.0,   'hydrothermal': True,
            'clay': 'montmorillonite', 'metal_ions': 'medium',
            'sulfur_phosphate': 'medium', 'gravity': 0.93,
            'wet_dry': 'tidal', 'solvent': 'water',
        },
        # K2-18b: sub-Neptune with JWST-detected CH4/CO2, possible Hycean world.
        # CH4 3 + NH3 2 + H2 8 + CO2 1 + N2 85 + H2O 1 = 100.
        'K2-18b': {
            'atm_CH4': 3.0,  'atm_NH3': 2.0,  'atm_H2': 8.0,
            'atm_CO2': 1.0,  'atm_N2': 85.0,  'atm_H2O': 1.0,
            'pressure': 1.0,    'uv': 'low',     'lightning': 200,
            'geothermal': 120,  'temperature': -8.0, 'ocean_pH': 6.5,
            'salinity': 35.0,   'hydrothermal': True,
            'clay': 'montmorillonite', 'metal_ions': 'medium',
            'sulfur_phosphate': 'medium', 'gravity': 1.1,
            'wet_dry': 'tidal', 'solvent': 'water',
        },
        # TOI-270d: warm sub-Neptune, greenhouse-pushed surface, possible steam world.
        # CH4 2 + NH3 1 + H2 22 + CO2 2 + N2 70 + H2O 3 = 100.
        'TOI-270d': {
            'atm_CH4': 2.0,  'atm_NH3': 1.0,  'atm_H2': 22.0,
            'atm_CO2': 2.0,  'atm_N2': 70.0,  'atm_H2O': 3.0,
            'pressure': 1.0,    'uv': 'low',     'lightning': 300,
            'geothermal': 120,  'temperature': 107.0, 'ocean_pH': 6.5,
            'salinity': 35.0,   'hydrothermal': True,
            'clay': 'montmorillonite', 'metal_ions': 'medium',
            'sulfur_phosphate': 'medium', 'gravity': 0.9,
            'wet_dry': 'tidal', 'solvent': 'water',
        },
        # Titan-like: nitrogen-dominated, liquid ammonia/methane lakes, very cold.
        # CH4 5 + NH3 10 + H2 1 + CO2 0 + N2 84 + H2O 0 = 100.
        'Titan-like': {
            'atm_CH4': 5.0,  'atm_NH3': 10.0, 'atm_H2': 1.0,
            'atm_CO2': 0.0,  'atm_N2': 84.0,  'atm_H2O': 0.0,
            'pressure': 1.5,    'uv': 'low',     'lightning': 5,
            'geothermal': 40,   'temperature': -80.0, 'ocean_pH': 7.0,
            'salinity': 0.0,    'hydrothermal': False,
            'clay': 'none',     'metal_ions': 'low',
            'sulfur_phosphate': 'low', 'gravity': 0.14,
            'wet_dry': 'none',  'solvent': 'ammonia',
        },
    }

    CONSENSUS_ORDER = ['Early Earth', 'TRAPPIST-1e', 'K2-18b',
                        'TOI-270d', 'Titan-like', 'Mars-like', 'Venus-like']
    consensus_ranks = {name: i + 1 for i, name in enumerate(CONSENSUS_ORDER)}

    # Pre-flight check: assert every planet sums to exactly 100 before any
    # simulation runs. Raises immediately with the offending planet name and
    # actual total so the error is never hidden by silent rescaling.
    atm_keys = ['atm_CH4', 'atm_NH3', 'atm_H2', 'atm_CO2', 'atm_N2', 'atm_H2O']
    for name, params in VALIDATION_PLANETS.items():
        total = sum(params[k] for k in atm_keys)
        if abs(total - 100.0) > 0.01:
            raise ValueError(
                f"Atmosphere for '{name}' sums to {total:.4f}%, not 100%. "
                "Fix the explicit values above — do not rely on rescaling."
            )

    print(f"\n{'Planet':<15} {'Engine':>8} {'ML':>8} {'Consensus':>10}")
    print("-" * 45)

    engine_scores, ml_scores, consensus_scores = [], [], []
    for name in CONSENSUS_ORDER:
        params = VALIDATION_PLANETS[name]
        eng  = run_simulation(params)['overall_score']
        ml   = predict(params)['overall_score']
        rank = consensus_ranks[name]

        engine_scores.append(eng)
        ml_scores.append(ml)
        consensus_scores.append(rank)

        print(f"{name:<15} {eng:>8.3f} {ml:>8.3f} {rank:>10}")

    # Spearman ρ: negate engine/ml scores so rank 1 = highest score.
    neg_engine = [-s for s in engine_scores]
    neg_ml     = [-s for s in ml_scores]
    rho_engine, _ = spearmanr(neg_engine, consensus_scores)
    rho_ml,     _ = spearmanr(neg_ml,     consensus_scores)

    print(f"\nSpearman ρ — Rule engine vs consensus: {rho_engine:+.3f}")
    print(f"Spearman ρ — ML model vs consensus:    {rho_ml:+.3f}")
    if rho_engine > 0.7:
        print("✓ Rule engine ranking broadly agrees with scientific consensus.")
    if rho_ml > 0.7:
        print("✓ ML model ranking broadly agrees with scientific consensus.")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    print("Generating training data...")
    X, y = generate_training_data(50000)
    print(f"Generated {len(X)} samples")

    print("Training model...")
    model, scores = train_model(X, y)
    print("R² scores per stage:", scores)

    print("\nValidation against known planets:")
    validate_against_known_planets()

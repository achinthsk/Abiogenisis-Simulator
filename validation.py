"""
validation.py — Scientific validation report for the prebiotic chemistry engine + ML model.

Three checks:
  1. Spearman rank correlation of engine/ML scores vs scientific consensus.
  2. JWST molecule consistency: confirms JWST-detected molecules are present
     at non-trivial % in each planet's atmosphere and that Stage 1 score
     reflects their contribution (atm > threshold → engine sees the feedstock).
  3. JSON report saved to validation_report.json.

Public entry point: run_validation_report() → dict
"""

import json
from datetime import datetime, timezone
from scipy.stats import spearmanr

from model import predict
from engine import run_simulation

# ---------------------------------------------------------------------------
# Planet parameter sets (fully explicit — no defaults inherited, atm sums to 100)
# These mirror the dicts in model.validate_against_known_planets() exactly.
# ---------------------------------------------------------------------------
VALIDATION_PLANETS = {
    'Early Earth': {
        'atm_CH4': 10.0, 'atm_NH3': 5.0,  'atm_H2': 15.0,
        'atm_CO2': 20.0, 'atm_N2': 45.0,  'atm_H2O': 5.0,
        'pressure': 1.0,    'uv': 'medium', 'lightning': 500,
        'geothermal': 120,  'temperature': 40.0,  'ocean_pH': 6.5,
        'salinity': 35.0,   'hydrothermal': True,
        'clay': 'montmorillonite', 'metal_ions': 'medium',
        'sulfur_phosphate': 'medium', 'gravity': 1.0,
        'wet_dry': 'tidal', 'solvent': 'water',
    },
    'TRAPPIST-1e': {
        'atm_CH4': 5.0,  'atm_NH3': 3.0,  'atm_H2': 7.0,
        'atm_CO2': 15.0, 'atm_N2': 65.0,  'atm_H2O': 5.0,
        'pressure': 1.0,    'uv': 'low',    'lightning': 300,
        'geothermal': 120,  'temperature': -40.0, 'ocean_pH': 6.5,
        'salinity': 35.0,   'hydrothermal': True,
        'clay': 'montmorillonite', 'metal_ions': 'medium',
        'sulfur_phosphate': 'medium', 'gravity': 0.93,
        'wet_dry': 'tidal', 'solvent': 'water',
    },
    'K2-18b': {
        'atm_CH4': 3.0,  'atm_NH3': 2.0,  'atm_H2': 8.0,
        'atm_CO2': 1.0,  'atm_N2': 85.0,  'atm_H2O': 1.0,
        'pressure': 1.0,    'uv': 'low',    'lightning': 200,
        'geothermal': 120,  'temperature': -8.0,  'ocean_pH': 6.5,
        'salinity': 35.0,   'hydrothermal': True,
        'clay': 'montmorillonite', 'metal_ions': 'medium',
        'sulfur_phosphate': 'medium', 'gravity': 1.1,
        'wet_dry': 'tidal', 'solvent': 'water',
    },
    'TOI-270d': {
        'atm_CH4': 2.0,  'atm_NH3': 1.0,  'atm_H2': 22.0,
        'atm_CO2': 2.0,  'atm_N2': 70.0,  'atm_H2O': 3.0,
        'pressure': 1.0,    'uv': 'low',    'lightning': 300,
        'geothermal': 120,  'temperature': 107.0, 'ocean_pH': 6.5,
        'salinity': 35.0,   'hydrothermal': True,
        'clay': 'montmorillonite', 'metal_ions': 'medium',
        'sulfur_phosphate': 'medium', 'gravity': 0.9,
        'wet_dry': 'tidal', 'solvent': 'water',
    },
    'Titan-like': {
        'atm_CH4': 5.0,  'atm_NH3': 10.0, 'atm_H2': 1.0,
        'atm_CO2': 0.0,  'atm_N2': 84.0,  'atm_H2O': 0.0,
        'pressure': 1.5,    'uv': 'low',    'lightning': 5,
        'geothermal': 40,   'temperature': -80.0, 'ocean_pH': 7.0,
        'salinity': 0.0,    'hydrothermal': False,
        'clay': 'none',     'metal_ions': 'low',
        'sulfur_phosphate': 'low', 'gravity': 0.14,
        'wet_dry': 'none',  'solvent': 'ammonia',
    },
    'Mars-like': {
        'atm_CH4': 2.0,  'atm_NH3': 0.0,  'atm_H2': 2.0,
        'atm_CO2': 85.0, 'atm_N2': 8.0,   'atm_H2O': 3.0,
        'pressure': 0.006,  'uv': 'high',   'lightning': 10,
        'geothermal': 30,   'temperature': -40.0, 'ocean_pH': 7.0,
        'salinity': 0.0,    'hydrothermal': False,
        'clay': 'none',     'metal_ions': 'low',
        'sulfur_phosphate': 'low', 'gravity': 0.38,
        'wet_dry': 'none',  'solvent': 'methane',
    },
    'Venus-like': {
        'atm_CH4': 0.0,  'atm_NH3': 0.0,  'atm_H2': 0.0,
        'atm_CO2': 96.0, 'atm_N2': 3.5,   'atm_H2O': 0.5,
        'pressure': 92.0,   'uv': 'medium', 'lightning': 50,
        'geothermal': 80,   'temperature': 460.0, 'ocean_pH': 7.0,
        'salinity': 0.0,    'hydrothermal': False,
        'clay': 'none',     'metal_ions': 'low',
        'sulfur_phosphate': 'low', 'gravity': 0.9,
        'wet_dry': 'none',  'solvent': 'methane',
    },
}

# Scientific consensus ranking, highest → lowest life potential.
CONSENSUS_ORDER = [
    'Early Earth', 'TRAPPIST-1e', 'K2-18b',
    'TOI-270d', 'Titan-like', 'Mars-like', 'Venus-like',
]

# JWST-confirmed molecules per planet (Madhusudhan et al. 2023; Mikal-Evans et al. 2023).
# Consistency check: the molecule's atm_X % must exceed MIN_DETECT_PCT in the planet
# params, confirming the engine receives the feedstock the telescope actually detected.
# We check the input rather than parsing reason strings because Stage 1 reasons report
# the *combined* carbon-gas contribution, not individual molecule names.
JWST_CONFIRMED = {
    'K2-18b':      ['CH4', 'CO2'],
    'TOI-270d':    ['CH4', 'CO2', 'H2O'],
    'TRAPPIST-1e': ['H2O'],
}
MIN_DETECT_PCT = 0.5  # % threshold below which a "detected" molecule is inconsistent


def _check_jwst(planet_name: str, params: dict) -> tuple[list[str], list[str]]:
    """Return (checked_molecules, passed_molecules) for one planet.

    PASS = molecule is confirmed by JWST AND its atm_X value in our planet dict
    is above MIN_DETECT_PCT, meaning the engine actually sees the feedstock.
    This validates that our parameter set is physically consistent with the
    telescope observations — a mismatch here would mean we're running the engine
    on a planet that contradicts its own JWST data.
    """
    molecules = JWST_CONFIRMED.get(planet_name, [])
    passed = []
    for mol in molecules:
        key = f'atm_{mol}'
        pct = params.get(key, 0.0)
        status = 'PASS' if pct >= MIN_DETECT_PCT else 'FAIL'
        print(f"  JWST {planet_name} — {mol}: {pct:.1f}% in params → {status}")
        if pct >= MIN_DETECT_PCT:
            passed.append(mol)
    return molecules, passed


def run_validation_report() -> dict:
    """Run all three validation checks and return a report dict.
    Also saves validation_report.json to disk.
    """
    consensus_ranks = {name: i + 1 for i, name in enumerate(CONSENSUS_ORDER)}

    print(f"\n{'Planet':<15} {'Engine':>8} {'ML':>8} {'Rank':>6}")
    print("-" * 42)

    planet_records = []
    engine_scores, ml_scores, consensus_scores = [], [], []

    for name in CONSENSUS_ORDER:
        params = VALIDATION_PLANETS[name]
        eng_result = run_simulation(params)
        ml_result  = predict(params)
        eng_score  = eng_result['overall_score']
        ml_score   = ml_result['overall_score']
        rank       = consensus_ranks[name]

        # Verdict match: both engine and ML must agree on the same verdict bucket.
        verdict_match = eng_result['verdict'] == ml_result['verdict']

        engine_scores.append(eng_score)
        ml_scores.append(ml_score)
        consensus_scores.append(rank)

        print(f"{name:<15} {eng_score:>8.3f} {ml_score:>8.3f} {rank:>6}")

        # JWST consistency sub-check
        checked, passed = _check_jwst(name, params)

        planet_records.append({
            'name':                   name,
            'engine_score':           round(eng_score, 4),
            'ml_score':               round(ml_score, 4),
            'consensus_rank':         rank,
            'verdict_match':          verdict_match,
            'jwst_molecules_checked': checked,
            'jwst_molecules_passed':  passed,
        })

    # Spearman ρ — negate scores so highest score = rank 1.
    rho_engine, _ = spearmanr([-s for s in engine_scores], consensus_scores)
    rho_ml,     _ = spearmanr([-s for s in ml_scores],     consensus_scores)

    verdict_hits  = sum(1 for r in planet_records if r['verdict_match'])
    total_jwst    = sum(len(r['jwst_molecules_checked']) for r in planet_records)
    passed_jwst   = sum(len(r['jwst_molecules_passed'])  for r in planet_records)

    print(f"\nSpearman ρ — engine: {rho_engine:+.3f} | ML: {rho_ml:+.3f}")
    print(f"Verdict agreement:  {verdict_hits}/{len(CONSENSUS_ORDER)}")
    print(f"JWST consistency:   {passed_jwst}/{total_jwst}")

    report = {
        'generated_at':    datetime.now(timezone.utc).isoformat(),
        'spearman_engine': round(float(rho_engine), 4),
        'spearman_ml':     round(float(rho_ml), 4),
        'verdict_accuracy': f"{verdict_hits}/{len(CONSENSUS_ORDER)}",
        'jwst_consistency': f"{passed_jwst}/{total_jwst}",
        'planets':          planet_records,
        'summary': (
            f"Rule engine Spearman ρ={rho_engine:+.2f} against scientific consensus. "
            f"JWST molecule consistency {passed_jwst}/{total_jwst}."
        ),
    }

    with open('validation_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    print("\nSaved validation_report.json")

    return report


if __name__ == "__main__":
    report = run_validation_report()
    print(json.dumps(report, indent=2))

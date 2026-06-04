"""
Prebiotic Chemistry Simulation Engine — Multiplicative Gating v2
=================================================================
Scores the likelihood of prebiotic chemistry progressing through 5 stages
toward proto-life, based on real planetary input parameters.

Scoring logic uses multiplicative gating: every factor is multiplied together.
A single critical factor at zero collapses the entire stage score toward zero.
This is physically correct — glycine cannot form at 500°C regardless of how
good every other condition is.

Scientific basis:
- Miller-Urey experiment (reducing atmosphere + energy → organics)
- Strecker synthesis (HCN + aldehydes → amino acids)
- Montmorillonite-catalyzed polymerization (Ferris et al., 1996)
- Hydrothermal vent chemistry (Fox, 1965; Wächtershäuser, 1990)
- Autocatalytic sets (Kauffman, 1986)
"""

import math

# =============================================================================
# DEFAULT PLANET: Earth-like (Hadean/Archean conditions, ~3.8 Gya)
# =============================================================================
PLANET = {
    # Atmosphere composition (must sum to 100)
    "atm_CH4":   10.0,   # % methane
    "atm_NH3":    5.0,   # % ammonia
    "atm_H2":    15.0,   # % hydrogen
    "atm_CO2":   20.0,   # % carbon dioxide
    "atm_N2":    45.0,   # % nitrogen
    "atm_H2O":    5.0,   # % water vapour

    "pressure":       1.0,    # bar
    "uv":             "medium",  # low / medium / high
    "lightning":      500,    # strikes/day
    "geothermal":     120,    # mW/m²
    "temperature":    40.0,   # °C surface

    "ocean_pH":       6.5,    # slightly acidic early ocean
    "salinity":       35.0,   # g/L
    "hydrothermal":   True,   # vents present?
    "clay":           "montmorillonite",  # none / montmorillonite / kaolinite
    "metal_ions":     "medium",   # low / medium / high
    "sulfur_phosphate": "medium", # low / medium / high
    "gravity":        1.0,    # relative to Earth
    "wet_dry":        "tidal",    # none / tidal / volcanic
    "solvent":        "water",    # water / ammonia / methane
}

# =============================================================================
# HELPER UTILITIES
# =============================================================================

def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))

def level(val, low_val, med_val):
    """Convert low/medium/high string to numeric."""
    if val == "low":    return low_val
    if val == "medium": return med_val
    return 1.0

def _gaussian(value, peak, width):
    """Returns 0–1 score peaked at `peak`, falling off with `width`.
    At extreme values (e.g. 500°C when peak is 40°C) this returns near zero.
    """
    return math.exp(-((value - peak) ** 2) / (2 * width ** 2))

def _eff_chem_temp(p):
    """When hydrothermal vents are active and the surface is cold (<20°C),
    chemistry occurs at vent temperatures (~50°C) rather than at the cold
    surface. Returns the effective temperature for reaction-rate Gaussians.
    This is why ice-covered ocean worlds (TRAPPIST-1e) still rank highly —
    their vent chemistry is warm even if the surface is frozen solid.
    """
    t = p["temperature"]
    has_vents = (p.get("hydrothermal") == True or p.get("hydrothermal") == "yes")
    if has_vents and t < 20:
        return max(t, 50)  # hydrothermal vents run at 50–100°C
    return t

def _vent_solvent_boost(p, base_solvent_f):
    """When hydrothermal vents are active with a water solvent, locally liquid
    water is guaranteed at the vent interface even if the surface is frozen.
    Boosts the solvent factor floor to 0.75 for vent-bearing water worlds.
    """
    if (p.get("hydrothermal") == True or p.get("hydrothermal") == "yes") and \
       p.get("solvent", "water") == "water":
        return max(base_solvent_f, 0.75)
    return base_solvent_f

def _liquid_solvent_factor(p):
    """Hard gate — returns near zero if solvent is not liquid at given temperature.
    This single factor collapses all downstream scores for impossible conditions.
    """
    t = p["temperature"]
    s = p.get("solvent", "water")
    if s == "water":
        if -5 <= t <= 105:   return 1.0
        if -40 <= t <= 150:  return 0.55  # subsurface/pressurised liquid possible
        if -60 <= t <= 200:  return 0.25  # marginal
        return 0.0
    if s == "ammonia":
        if -78 <= t <= -33:  return 1.0
        if -100 <= t <= -10: return 0.4
        return 0.0
    if s == "methane":
        if -182 <= t <= -161: return 1.0
        if -200 <= t <= -100: return 0.5
        return 0.0
    return 0.0

# =============================================================================
# STAGE SCORING FUNCTIONS
# =============================================================================

def stage1_organic_formation(p):
    """
    Stage 1 — Organic molecule formation (aldehydes, HCN, fatty acids).

    Multiplicative gate: liquid solvent × temperature window × energy × carbon.
    At 460°C (Venus) or 500°C, _liquid_solvent_factor returns 0.0, collapsing
    the entire score regardless of how abundant carbon feedstocks are.
    """
    # Energy factor — best available source wins
    lightning_f = min(1.0, p["lightning"] / 500)
    geo_f       = min(1.0, p["geothermal"] / 120)
    uv_f        = {"low": 0.2, "medium": 0.5, "high": 0.9}.get(p["uv"], 0.3)
    energy      = max(lightning_f, geo_f, uv_f)

    # Carbon source factor — needs reducing gases
    ch4_f  = min(1.0, p["atm_CH4"] / 15)
    co2_f  = min(1.0, p["atm_CO2"] / 20)
    h2_f   = min(1.0, p["atm_H2"]  / 15)
    carbon = max(ch4_f, co2_f) * (1 + 0.3 * h2_f)
    carbon = min(1.0, carbon)

    # Temperature window — Gaussian peaked at 60°C for gas-phase reactions
    # At 500°C this returns near zero — effectively kills score
    temp_f    = _gaussian(p["temperature"], 60, 120)

    # Liquid solvent hard gate
    solvent_f = _liquid_solvent_factor(p)

    score = round(min(1.0, energy * carbon * temp_f * solvent_f * 1.4), 4)
    reason = (
        f"Energy={energy:.2f} (best source), carbon={carbon:.2f}, "
        f"temp_f={temp_f:.3f} (@{p['temperature']}°C), solvent_f={solvent_f:.2f}."
    )
    return score, reason


def stage2_amino_nucleotide(p, s1, mols_s1=None):
    """
    Stage 2 — Amino acid and nucleotide formation.

    Requires Stage 1 ≥ 0.3. HCN and formaldehyde presence boosted by
    atmosphere composition when no molecule list is available (pure engine mode).
    pH, temperature, and solvent gate are all multiplicative hard requirements.
    """
    if mols_s1 is None:
        mols_s1 = []

    if s1 < 0.3:
        return 0.0, "Blocked: Stage 1 score too low."

    # HCN and formaldehyde — use molecule list if available, else estimate
    if mols_s1:
        hcn  = 1.0 if any("HCN" in m.get("name", "") for m in mols_s1) else 0.1
        form = 1.0 if any("Formaldehyde" in m.get("name", "") for m in mols_s1) else 0.15
    else:
        # Estimate from atmospheric chemistry (Miller-Urey / HCN synthesis)
        hcn  = max(0.1, min(1.0,
            (p.get("atm_NH3", 0) / 8) *
            min(1.0, (p.get("atm_CH4", 0) + p.get("atm_H2", 0)) / 20)
        ))
        form = max(0.15, min(1.0,
            (p.get("atm_CO2", 0) + p.get("atm_CH4", 0)) / 30
        ))

    # pH window — Strecker synthesis optimal 5–8
    ph_f      = _gaussian(p["ocean_pH"], 6.5, 1.8)

    # Temperature window — use vent temperature for cold hydrothermal worlds
    eff_t  = _eff_chem_temp(p)
    temp_f = _gaussian(eff_t, 40, 60)

    # Liquid solvent gate; boosted for hydrothermal vent worlds
    solvent_f = _vent_solvent_boost(p, _liquid_solvent_factor(p))

    # Phosphate for nucleotide branch
    phos_f    = {"low": 0.4, "medium": 0.75, "high": 1.0}.get(
        p.get("sulfur_phosphate", "low"), 0.4)

    # Stage 1 organics directly seed amino acid precursors — additive s1 boost
    s1_boost = 0.3 * s1

    base  = hcn * form * ph_f * temp_f * solvent_f + s1_boost
    score = round(min(1.0, base * (0.7 + 0.3 * phos_f) * 1.6), 4)
    reason = (
        f"HCN={hcn:.2f}, form={form:.2f}, pH_f={ph_f:.2f}, "
        f"eff_t={eff_t:.0f}°C, temp_f={temp_f:.3f}, solvent_f={solvent_f:.2f}, "
        f"s1_boost={s1_boost:.2f}, phos={phos_f:.2f}."
    )
    return score, reason


def stage3_polymer_formation(p, s2, mols_s2=None):
    """
    Stage 3 — Polymer formation (peptides, proto-RNA, lipid vesicles).

    Requires Stage 2 ≥ 0.3. Clay, wet-dry cycling, and liquid solvent
    are all multiplicative. Missing any one seriously degrades polymer yield.
    """
    if mols_s2 is None:
        mols_s2 = []

    if s2 < 0.3:
        return 0.0, "Blocked: Stage 2 score too low."

    # Amino acid availability — from molecule list or estimated from s2
    if mols_s2:
        amino_f = min(1.0, sum(
            1 for m in mols_s2
            if "acid" in m.get("name", "").lower()
            or "Glycine" in m.get("name", "")
            or "Alanine" in m.get("name", "")
        ) * 0.4)
        amino_f = max(0.05, amino_f)
    else:
        amino_f = min(1.0, s2 * 0.85)

    # Clay mineral catalysis (Ferris et al., 1996)
    clay_f = {"none": 0.15, "kaolinite": 0.5, "montmorillonite": 1.0}.get(
        p.get("clay", "none"), 0.15)

    # Wet-dry cycling for condensation reactions (Deamer, 2017)
    wetdry_f = {"none": 0.2, "tidal": 0.9, "volcanic": 0.8}.get(
        p.get("wet_dry", "none"), 0.2)

    # Vents add thermal energy for longer chains
    vent_f = 1.3 if (p.get("hydrothermal") == True or p.get("hydrothermal") == "yes") else 1.0

    # Temperature — use vent temperature for cold hydrothermal worlds
    eff_t     = _eff_chem_temp(p)
    temp_f    = _gaussian(eff_t, 50, 70)
    solvent_f = _vent_solvent_boost(p, _liquid_solvent_factor(p))

    score = round(min(1.0, amino_f * clay_f * wetdry_f * temp_f * solvent_f * vent_f * 2.0), 4)
    reason = (
        f"Amino_f={amino_f:.2f}, clay={p.get('clay','none')} ({clay_f:.2f}), "
        f"wet-dry={p.get('wet_dry','none')} ({wetdry_f:.2f}), "
        f"eff_t={eff_t:.0f}°C, temp_f={temp_f:.3f}, vent_f={vent_f:.1f}."
    )
    return score, reason


def stage4_autocatalytic_network(p, s3, mols_s3=None):
    """
    Stage 4 — Autocatalytic network likelihood.

    Requires Stage 3 ≥ 0.3. Metal ions and liquid solvent are hard gates.
    Either RNA-world or metabolism-first (Fe-S) path can succeed.
    """
    if mols_s3 is None:
        mols_s3 = []

    if s3 < 0.3:
        return 0.0, "Blocked: Stage 3 score too low."

    # Metal ions critical for catalysis
    metal_f = {"low": 0.2, "medium": 0.65, "high": 1.0}.get(
        p.get("metal_ions", "low"), 0.2)

    # Vents essential for Fe-S chemistry
    has_vents = (p.get("hydrothermal") == True or p.get("hydrothermal") == "yes")
    vent_f    = 1.0 if has_vents else 0.4

    if mols_s3:
        has_rna   = any("RNA" in m.get("name", "") or "oligomer" in m.get("name", "").lower()
                        for m in mols_s3)
        has_lipid = any("Lipid" in m.get("name", "") or "vesicle" in m.get("name", "").lower()
                        for m in mols_s3)
        has_fes   = any("Fe" in m.get("name", "") or "iron" in m.get("name", "").lower()
                        or "sulfur" in m.get("name", "").lower() for m in mols_s3)
        rna_path   = (0.8 if has_rna else 0.1) * metal_f
        meta_path  = (0.8 if has_fes else 0.2) * vent_f * metal_f
        lipid_bonus = 0.3 if has_lipid else 0.0
    else:
        # Estimate from conditions and s3 score
        rna_prob   = min(1.0, s3 * 0.5)
        fes_prob   = 0.6 if has_vents else 0.25
        lipid_prob = min(1.0, s3 * 0.4)
        rna_path   = rna_prob * metal_f
        meta_path  = fes_prob * vent_f * metal_f
        lipid_bonus = lipid_prob * 0.3

    best_path = max(rna_path, meta_path)
    solvent_f = _vent_solvent_boost(p, _liquid_solvent_factor(p))

    score = round(min(1.0, (best_path + lipid_bonus) * solvent_f * 1.2), 4)
    reason = (
        f"RNA_path={rna_path:.2f}, metal_path={meta_path:.2f}, "
        f"lipid_bonus={lipid_bonus:.2f}, solvent_f={solvent_f:.2f}, metal_f={metal_f:.2f}."
    )
    return score, reason


def stage5_proto_life(p, s4, mols_s4=None):
    """
    Stage 5 — Proto-life potential + predicted biochemistry type.

    Requires Stage 4 ≥ 0.3. Combines compartment/replicator structure,
    temporal duration, temperature window, and liquid solvent gate.
    Also determines which biochemistry branch (carbon-water, -ammonia, -exotic).
    """
    if mols_s4 is None:
        mols_s4 = []

    if s4 < 0.3:
        return 0.0, "unknown", "Blocked: Stage 4 score too low."

    # Determine biochemistry type
    solvent = p.get("solvent", "water")
    if solvent == "water":
        biochem    = "carbon-water"
        biochem_f  = 1.0
    elif solvent == "ammonia":
        biochem    = "carbon-ammonia"
        biochem_f  = 0.75
    else:
        biochem    = "carbon-exotic"
        biochem_f  = 0.45

    # Duration of stable conditions
    duration_f = min(1.0, p.get("duration", 100) / 200)

    # Temperature uses surface temp for long-term persistence (not vent override)
    temp_f    = _gaussian(p["temperature"], 40, 80)
    # Solvent: vent boost still applies — liquid water exists at vents
    solvent_f = _vent_solvent_boost(p, _liquid_solvent_factor(p))

    if mols_s4:
        has_proto = any("proto" in m.get("name", "").lower()
                        or "compartment" in m.get("name", "").lower() for m in mols_s4)
        has_ribo  = any("ribozyme" in m.get("name", "").lower()
                        or "Ribozyme" in m.get("name", "") for m in mols_s4)
        has_auto  = any("autocatalytic" in m.get("name", "").lower()
                        or "Autocatalytic" in m.get("name", "") for m in mols_s4)
        structure = ((0.5 if has_proto else 0.1) +
                     (0.3 if has_ribo  else 0.0) +
                     (0.2 if has_auto  else 0.0))
    else:
        # Estimate from s4 and conditions
        phos_ok    = level(p.get("sulfur_phosphate", "medium"), 0.3, 0.65) > 0.5
        proto_prob = min(0.5, s4 * 0.5)
        ribo_prob  = min(0.3, s4 * 0.3) if phos_ok else 0.0
        auto_prob  = min(0.2, s4 * 0.35)
        structure  = proto_prob + ribo_prob + auto_prob

    structure = min(1.0, structure)
    score = round(min(1.0, structure * duration_f * temp_f * solvent_f * 1.5) * biochem_f, 4)
    reason = (
        f"Structure={structure:.2f}, duration_f={duration_f:.2f}, "
        f"temp_f={temp_f:.3f}, solvent_f={solvent_f:.2f}, biochem={biochem} (×{biochem_f})."
    )
    return score, biochem, reason

# =============================================================================
# MAIN RUNNER
# =============================================================================

def validate(p):
    atm_total = p["atm_CH4"] + p["atm_NH3"] + p["atm_H2"] + p["atm_CO2"] + p["atm_N2"] + p["atm_H2O"]
    if abs(atm_total - 100.0) > 0.5:
        raise ValueError(f"Atmosphere percentages must sum to 100 (got {atm_total:.1f}%).")

def _build_result(proto_score, biochem, s1, s2, s3, s4, s5,
                  r1="", r2="", r3="", r4="", r5=""):
    """Assemble and return the standardised results dictionary."""
    scores  = [s for s in [s1, s2, s3, s4, s5] if s > 0]
    overall = round(sum(scores) / 5.0, 4) if scores else 0.0

    if overall >= 0.75:
        verdict = "HIGHLY FAVOURABLE"
    elif overall >= 0.55:
        verdict = "FAVOURABLE"
    elif overall >= 0.35:
        verdict = "MARGINAL"
    else:
        verdict = "UNLIKELY"

    return {
        "stages": {
            "s1": round(s1, 4),
            "s2": round(s2, 4),
            "s3": round(s3, 4),
            "s4": round(s4, 4),
            "s5": round(s5, 4),
        },
        "reasons": {
            "s1": r1,
            "s2": r2,
            "s3": r3,
            "s4": r4,
            "s5": r5,
        },
        "biochemistry":     biochem if biochem != "none" else "—",
        "proto_life_score": round(proto_score, 4),
        "overall_score":    overall,
        "verdict":          verdict,
    }


def run_simulation(p=None):
    """Run all five stages and return a results dict. Never raises on a blocked
    stage — blocked stages simply score 0.0 with a reason string."""
    if p is None:
        p = PLANET
    validate(p)

    # Stage 1 — always runs; liquid solvent gate kills impossible planets here
    s1, r1 = stage1_organic_formation(p)
    if s1 < 0.05:
        r1 += " ⛔ Pipeline blocked at Stage 1."
        return _build_result(0.0, "none", s1, 0.0, 0.0, 0.0, 0.0, r1)

    # Stage 2 — molecules from reaction_engine passed in when available
    s2, r2 = stage2_amino_nucleotide(p, s1)
    if s2 < 0.05:
        r2 += " ⛔ Threshold not met."
        return _build_result(0.0, "none", s1, s2, 0.0, 0.0, 0.0, r1, r2)

    # Stage 3
    s3, r3 = stage3_polymer_formation(p, s2)
    if s3 < 0.05:
        r3 += " ⛔ Threshold not met."
        return _build_result(0.0, "none", s1, s2, s3, 0.0, 0.0, r1, r2, r3)

    # Stage 4
    s4, r4 = stage4_autocatalytic_network(p, s3)
    if s4 < 0.05:
        r4 += " ⛔ Threshold not met."
        return _build_result(0.0, "none", s1, s2, s3, s4, 0.0, r1, r2, r3, r4)

    # Stage 5
    s5, biochem, r5 = stage5_proto_life(p, s4)
    return _build_result(s5, biochem, s1, s2, s3, s4, s5, r1, r2, r3, r4, r5)


def _print_results(result):
    """Pretty-print a results dict to stdout."""
    stages  = result["stages"]
    reasons = result["reasons"]
    labels  = {
        "s1": "Organic molecule formation",
        "s2": "Amino acid & nucleotide form",
        "s3": "Polymer formation",
        "s4": "Autocatalytic network",
        "s5": "Proto-life potential",
    }
    print("=" * 60)
    print("  PREBIOTIC CHEMISTRY SIMULATION ENGINE")
    print("=" * 60)
    for key, label in labels.items():
        score  = stages[key]
        reason = reasons[key]
        print(f"\n[{key.upper()}] {label:<30} →  score: {score:.3f}")
        if reason:
            print(f"     {reason}")
    print("\n" + "=" * 60)
    print("  FINAL RESULTS")
    print("=" * 60)
    print(f"  Biochemistry type      : {result['biochemistry']}")
    print(f"  Proto-life score (S5)  : {result['proto_life_score']:.3f}")
    print(f"  Overall likelihood     : {result['overall_score']:.3f}  [{result['verdict']}]")
    print("=" * 60)


if __name__ == "__main__":
    _print_results(run_simulation())

    print()

    # ── Validation tests ──────────────────────────────────────────
    venus = {**PLANET, "temperature": 460, "atm_CO2": 96, "atm_N2": 3.5,
             "atm_H2O": 0.5, "atm_CH4": 0, "atm_NH3": 0, "atm_H2": 0,
             "pressure": 92, "solvent": "water"}
    r = run_simulation(venus)
    tag = "✓ PASS" if r["overall_score"] < 0.05 else "✗ FAIL"
    print(f"Venus score:   {r['overall_score']:.3f} — should be < 0.05  {tag}")

    r2 = run_simulation(PLANET)
    tag2 = "✓ PASS" if r2["overall_score"] > 0.55 else "✗ FAIL"
    print(f"Early Earth:   {r2['overall_score']:.3f} — should be > 0.55  {tag2}")

    hot = {**PLANET, "temperature": 500}
    r3 = run_simulation(hot)
    tag3 = "✓ PASS" if r3["overall_score"] < 0.05 else "✗ FAIL"
    print(f"500°C score:   {r3['overall_score']:.3f} — should be < 0.05  {tag3}")

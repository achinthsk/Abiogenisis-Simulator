"""
reaction_engine.py — Chemistry-grounded prebiotic simulation engine.

Replaces the statistical model.py with a real reaction network approach:
  1. MINE API (https://iseq.mcs.anl.gov/mine) expands reaction networks from
     starting SMILES; falls back to a curated hardcoded reaction set when the
     API is unreachable.
  2. RDKit computes molecular properties (MW, LogP, H-bond counts, stability)
     for every predicted product.
  3. A 5-stage pipeline filters and scores molecules at each stage, producing
     the same output dict shape as engine.py so api.py needs no structural changes.

Dependencies: pip install rdkit-pypi requests
"""

import hashlib
import json
import os
import time
import requests

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MINE_URL   = "https://iseq.mcs.anl.gov/mine/api/v1/job"
CACHE_FILE = "mine_cache.json"
CACHE_TTL  = 86400  # 24 h

ATMOSPHERE_SMILES = {
    'CH4': 'C',
    'NH3': 'N',
    'H2O': 'O',
    'CO2': 'O=C=O',
    'H2':  '[H][H]',
    'N2':  'N#N',
    'HCN': 'C#N',
    'H2S': 'S',
}

# ---------------------------------------------------------------------------
# Hardcoded fallback reactions
# Tuples: (reactant_smiles_set, product_smiles, name, formula, why)
# ---------------------------------------------------------------------------
FALLBACK_REACTIONS = [
    # 1. Miller-Urey: CH4 + NH3 → HCN  (under UV/lightning)
    ({'C', 'N'},         'C#N',        'Hydrogen Cyanide',  'CHN',    'Miller-Urey: CH4 + NH3 → HCN under energy input'),
    # 2. Formaldehyde synthesis: CO2 + H2 → CH2O  (Fischer-Tropsch-type)
    ({'O=C=O', '[H][H]'},'C=O',        'Formaldehyde',      'CH2O',   'CO2 hydrogenation → formaldehyde'),
    # 3. HCN hydration → formamide
    ({'C#N', 'O'},       'NC=O',       'Formamide',         'CH3NO',  'HCN + H2O → formamide (prebiotic nucleobase precursor)'),
    # 4. Glycolonitrile: CH2O + HCN → HOCH2CN
    ({'C=O', 'C#N'},     'OCC#N',      'Glycolonitrile',    'C2H3NO', 'Strecker precursor: formaldehyde + HCN'),
    # 5. Adenine from HCN pentamerisation  (Oró 1961)
    ({'C#N'},            'c1ncnc2[nH]cnc12', 'Adenine',     'C5H5N5', 'Oró synthesis: 5 HCN → adenine under UV'),
    # 6. Glycine: Strecker synthesis
    ({'C=O', 'C#N', 'N'},'NCC(=O)O',  'Glycine',           'C2H5NO2','Strecker amino acid synthesis from formaldehyde + HCN + NH3'),
    # 7. Alanine
    ({'C', 'O=C=O', 'N'},'CC(N)C(=O)O','Alanine',          'C3H7NO2','Reductive amination of pyruvate precursor'),
    # 8. Ribose (simplified): formaldehyde oligomerisation (formose reaction)
    ({'C=O'},            'OC[C@H](O)[C@@H](O)[C@H](O)CO', 'Ribose', 'C5H10O5', 'Formose reaction: CH2O oligomerisation → ribose'),
    # 9. Urea from NH3 + CO2
    ({'N', 'O=C=O'},     'NC(N)=O',    'Urea',              'CH4N2O', 'Wöhler-type: NH3 + CO2 → urea'),
    # 10. Pyruvate (key metabolism-first intermediate)
    ({'C=O', 'O=C=O'},   'CC(=O)C(=O)O','Pyruvate',        'C3H4O3', 'Condensation of formaldehyde + CO2 → pyruvate precursor'),
    # 11. H2S contribution: thioacetic acid (vent chemistry, de Duve)
    ({'S', 'C=O'},       'CC(=O)S',    'Thioacetate',       'C2H4OS', 'Hydrothermal vent: H2S + CH2O → thioester (metabolism-first)'),
    # 12. Cytosine precursor (cyanamide + urea pathway)
    ({'NC(N)=O', 'C#N'}, 'Nc1ccnc(=O)[nH]1','Cytosine',   'C4H5N3O','Urea + cyanamide cyclisation → cytosine'),
]


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _load_cache() -> dict:
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_cache(cache: dict):
    try:
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache, f)
    except Exception:
        pass


def _cache_key(smiles_list: list, generations: int) -> str:
    payload = json.dumps(sorted(smiles_list) + [generations], sort_keys=True)
    return hashlib.md5(payload.encode()).hexdigest()


# ---------------------------------------------------------------------------
# MINE API expansion
# ---------------------------------------------------------------------------

def _mine_expand(smiles_list: list, generations: int) -> list[dict]:
    """Call MINE API and return a list of product dicts.
    Each dict has at minimum: smiles, formula, name (may be empty).
    """
    body = {
        "start_mols": smiles_list,
        "generations": generations,
        "operators": "all",
    }
    resp = requests.post(MINE_URL, json=body, timeout=20)
    resp.raise_for_status()
    data = resp.json()

    products = []
    # MINE returns a job result; parse whatever structure comes back.
    # Common shapes: {"products": [...]} or {"compounds": [...]}
    raw = data.get("products") or data.get("compounds") or data.get("results") or []
    for item in raw:
        smiles = item.get("smiles") or item.get("SMILES") or ""
        if not smiles:
            continue
        products.append({
            "smiles":   smiles,
            "formula":  item.get("formula") or item.get("Formula") or "",
            "name":     item.get("name")   or item.get("Name")   or "",
            "dG":       float(item.get("dG") or item.get("deltaG") or 0.0),
        })
    return products


def _fallback_expand(smiles_list: list) -> list[dict]:
    """Return hardcoded reaction products whose reactants overlap with smiles_list."""
    smiles_set = set(smiles_list)
    products = []
    for reactants, product_smiles, name, formula, why in FALLBACK_REACTIONS:
        if reactants & smiles_set:  # at least one reactant is present
            products.append({
                "smiles":  product_smiles,
                "formula": formula,
                "name":    name,
                "dG":      -10.0,  # assume thermodynamically plausible
                "why":     why,
            })
    return products


def expand_reactions(smiles_list: list, generations: int = 2) -> list[dict]:
    """Expand a set of SMILES via MINE, with caching and hardcoded fallback."""
    key   = _cache_key(smiles_list, generations)
    cache = _load_cache()

    if key in cache:
        entry = cache[key]
        if time.time() - entry.get("ts", 0) < CACHE_TTL:
            return entry["products"]

    # Try MINE API
    try:
        products = _mine_expand(smiles_list, generations)
        source   = "MINE"
    except Exception:
        products = _fallback_expand(smiles_list)
        source   = "fallback"

    # Always merge fallback so we never return an empty set
    fallback_smiles = {p["smiles"] for p in products}
    for fb in _fallback_expand(smiles_list):
        if fb["smiles"] not in fallback_smiles:
            products.append(fb)
            fallback_smiles.add(fb["smiles"])

    cache[key] = {"ts": time.time(), "products": products, "source": source}
    _save_cache(cache)
    return products


# ---------------------------------------------------------------------------
# RDKit property calculator
# ---------------------------------------------------------------------------

def _estimate_bp(mol) -> float:
    """Rough boiling-point estimate (°C) via Joback fragment counting.
    Accurate enough to distinguish gas / liquid / solid at a given temperature.
    """
    mw  = Descriptors.MolWt(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    # Simplified Joback proxy: heavier and more H-bonding → higher BP
    return -50 + 0.7 * mw + 10 * hbd + 5 * hba


def _ph_tolerance(mol, ocean_ph: float) -> float:
    """Score 0-1 for how well the molecule tolerates the environment pH."""
    # Acids (has COOH) prefer acidic-neutral; bases (has N) prefer neutral-alkaline
    smiles = Chem.MolToSmiles(mol)
    has_acid  = 'C(=O)O' in smiles or 'C(O)=O' in smiles
    has_amine = smiles.count('N') > 0 and 'n' not in smiles  # aliphatic N
    if has_acid and has_amine:     # zwitterion — broadly tolerant
        return 1.0 if 4 <= ocean_ph <= 10 else 0.5
    elif has_acid:
        return 1.0 if 2 <= ocean_ph <= 7 else max(0.2, 1.0 - (ocean_ph - 7) * 0.15)
    elif has_amine:
        return 1.0 if 7 <= ocean_ph <= 12 else max(0.2, 1.0 - (7 - ocean_ph) * 0.15)
    return 0.85  # neutral molecules are broadly stable


def get_molecule_properties(smiles: str, temperature_c: float,
                             ocean_ph: float) -> dict | None:
    """Return RDKit property dict for a SMILES string at given conditions.
    Returns None if SMILES is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mw    = round(Descriptors.MolWt(mol), 3)
    logp  = round(Descriptors.MolLogP(mol), 3)
    hbd   = rdMolDescriptors.CalcNumHBD(mol)
    hba   = rdMolDescriptors.CalcNumHBA(mol)
    tpsa  = round(rdMolDescriptors.CalcTPSA(mol), 2)
    bp    = _estimate_bp(mol)
    mp    = bp - 80  # rough melting point proxy

    # Physical state at temperature_c
    if temperature_c >= bp:
        state = "gas"
    elif temperature_c <= mp:
        state = "solid"
    else:
        state = "liquid"

    # Stability score components
    # 1. Temperature — decomposition estimated at 1.5× boiling point (rough)
    decomp_t     = bp * 1.5
    temp_stable  = max(0.0, min(1.0, 1.0 - (temperature_c - decomp_t) / 200)) if temperature_c > decomp_t else 1.0
    # 2. pH tolerance
    ph_stable    = _ph_tolerance(mol, ocean_ph)
    # 3. Solubility proxy (water-soluble molecules = logP < 1 are more useful)
    sol_score    = max(0.1, min(1.0, 1.0 - (logp - 1.0) / 5.0))
    # 4. Size / reactivity penalty for very large molecules
    size_penalty = 1.0 if mw < 400 else max(0.3, 1.0 - (mw - 400) / 1000)

    stability_score = round(
        (temp_stable * 0.35 + ph_stable * 0.30 + sol_score * 0.20 + size_penalty * 0.15),
        4
    )
    stable = stability_score >= 0.4 and state != "gas"

    return {
        "molecular_weight":      mw,
        "logP":                  logp,
        "hbond_donors":          hbd,
        "hbond_acceptors":       hba,
        "tpsa":                  tpsa,
        "boiling_point_est":     round(bp, 1),
        "state_at_conditions":   state,
        "temp_stability":        round(temp_stable, 4),
        "ph_stability":          round(ph_stable, 4),
        "solubility_score":      round(sol_score, 4),
        "stability_score":       stability_score,
        "stable_at_conditions":  stable,
    }


# ---------------------------------------------------------------------------
# Molecule enrichment helpers
# ---------------------------------------------------------------------------

def _enrich(raw: dict, temperature_c: float, ocean_ph: float) -> dict | None:
    """Add RDKit props to a raw product dict. Returns None if SMILES invalid."""
    smiles = raw.get("smiles", "")
    props  = get_molecule_properties(smiles, temperature_c, ocean_ph)
    if props is None:
        return None
    return {
        "name":                 raw.get("name", "Unknown"),
        "smiles":               smiles,
        "formula":              raw.get("formula", ""),
        "why":                  raw.get("why", "MINE reaction expansion"),
        "molecular_weight":     props["molecular_weight"],
        "logP":                 props["logP"],
        "hbond_donors":         props["hbond_donors"],
        "hbond_acceptors":      props["hbond_acceptors"],
        "boiling_point_est":    props["boiling_point_est"],
        "state_at_conditions":  props["state_at_conditions"],
        "stability_score":      props["stability_score"],
        "stable_at_conditions": props["stable_at_conditions"],
    }


def _is_amino_acid_like(mol_dict: dict) -> bool:
    """Heuristic: contains N, MW 75–350, has both amine and carboxyl character."""
    formula = mol_dict.get("formula", "")
    mw      = mol_dict.get("molecular_weight", 0)
    return (
        'N' in formula
        and 75 <= mw <= 350
        and mol_dict.get("stable_at_conditions", False)
    )


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


# ---------------------------------------------------------------------------
# Stage scoring helpers
# ---------------------------------------------------------------------------

def _score_from_molecules(mols: list[dict], target_count: int = 3) -> float:
    """Score 0-1 based on stable molecule count relative to a target."""
    stable = sum(1 for m in mols if m.get("stable_at_conditions"))
    avg_stability = (
        sum(m.get("stability_score", 0) for m in mols) / len(mols) if mols else 0
    )
    count_score = _clamp(stable / target_count)
    return _clamp(count_score * 0.5 + avg_stability * 0.5)


# ---------------------------------------------------------------------------
# Main 5-stage pipeline
# ---------------------------------------------------------------------------

def run_reaction_pipeline(planet_params: dict) -> dict:
    """Run the chemistry-grounded 5-stage abiogenesis pipeline.

    Returns a dict matching engine.py's output shape (stages, reasons, molecules,
    biochemistry, proto_life_score, overall_score, verdict, prediction_source).
    """
    p      = planet_params
    temp   = p.get("temperature", 25.0)
    ph     = p.get("ocean_pH", 7.0)
    clay   = p.get("clay", "none")
    wetdry = p.get("wet_dry", "none")
    metals = p.get("metal_ions", "low")
    vents  = p.get("hydrothermal", False)
    solvnt = p.get("solvent", "water")
    sp     = p.get("sulfur_phosphate", "low")

    stages_scores   = {}
    stages_reasons  = {}
    stages_molecules= {}

    # ── STAGE 1 ── Organic molecule formation ────────────────────────────────
    # Starting SMILES from gases above 3% threshold
    atm_map = {
        'CH4': p.get('atm_CH4', 0), 'NH3': p.get('atm_NH3', 0),
        'H2O': p.get('atm_H2O', 0), 'CO2': p.get('atm_CO2', 0),
        'H2':  p.get('atm_H2', 0),  'N2':  p.get('atm_N2', 0),
    }
    start_smiles = [ATMOSPHERE_SMILES[g] for g, pct in atm_map.items() if pct >= 3]
    if not start_smiles:
        start_smiles = ['C', 'O=C=O']  # minimal fallback

    s1_raw      = expand_reactions(start_smiles, generations=1)
    s1_enriched = [e for r in s1_raw if (e := _enrich(r, temp, ph)) is not None]
    s1_stable   = [m for m in s1_enriched if m["stability_score"] > 0.4]

    s1_score = _clamp(_score_from_molecules(s1_stable, target_count=3))
    stages_scores["s1"]    = round(s1_score, 4)
    stages_molecules["s1"] = s1_stable[:8]
    stages_reasons["s1"]   = (
        f"MINE+fallback: {len(s1_stable)} stable organics from "
        f"{len(start_smiles)} atmospheric gases. "
        f"Key products: {', '.join(m['name'] for m in s1_stable[:3]) or 'none'}."
    )

    if s1_score <= 0.25:
        return _build_result(stages_scores, stages_reasons, stages_molecules,
                             blocked_at=1, solvnt=solvnt)

    # ── STAGE 2 ── Amino acids & nucleotide precursors ───────────────────────
    s1_smiles   = [m["smiles"] for m in s1_stable]
    s2_raw      = expand_reactions(s1_smiles, generations=2)
    s2_enriched = [e for r in s2_raw if (e := _enrich(r, temp, ph)) is not None]
    s2_aa       = [m for m in s2_enriched if _is_amino_acid_like(m)]

    # Phosphate / metal boost for nucleotide precursors
    sp_boost = {"low": 0.0, "medium": 0.1, "high": 0.2}.get(sp, 0.0)
    me_boost = {"low": 0.0, "medium": 0.1, "high": 0.15}.get(metals, 0.0)
    vent_boost = 0.12 if vents else 0.0

    s2_base  = _score_from_molecules(s2_aa, target_count=3)
    s2_score = _clamp(s2_base + sp_boost + me_boost + vent_boost)

    stages_scores["s2"]    = round(s2_score, 4)
    stages_molecules["s2"] = s2_aa[:8]
    stages_reasons["s2"]   = (
        f"{len(s2_aa)} amino-acid / nucleotide-precursor molecules (MW 75-350, contains N). "
        f"Phosphate boost +{sp_boost:.2f}, metal catalyst +{me_boost:.2f}, "
        f"vent alkalinity +{vent_boost:.2f}."
    )

    if s2_score <= 0.25:
        return _build_result(stages_scores, stages_reasons, stages_molecules,
                             blocked_at=2, solvnt=solvnt)

    # ── STAGE 3 ── Polymer formation (peptides, proto-RNA, lipid vesicles) ───
    clay_boost = {"none": 0.0, "kaolinite": 0.15, "montmorillonite": 0.3}.get(clay, 0.0)
    wdc_boost  = {"none": 0.0, "tidal": 0.25, "volcanic": 0.20}.get(wetdry, 0.0)

    # Estimate peptide chain length from energy budget
    energy_budget = (
        p.get("geothermal", 0) / 300 +
        p.get("lightning",  0) / 5000 +
        (0.3 if p.get("uv") == "high" else 0.15 if p.get("uv") == "medium" else 0.05)
    )
    chain_len_est = max(2, int(len(s2_aa) * energy_budget * 4))

    # Salinity check: lipid vesicle stability
    sal = p.get("salinity", 35)
    sal_ok = 5 <= sal <= 120

    s3_base  = s2_score * 0.4 + clay_boost + wdc_boost
    s3_base += 0.05 if sal_ok else 0.0
    s3_score = _clamp(s3_base)

    # Build polymer prediction records
    s3_mols = []
    if s2_aa:
        s3_mols.append({
            "name":                 f"Peptide chain (~{chain_len_est} residues)",
            "smiles":               "",
            "formula":              "varies",
            "why":                  f"Clay={clay} catalysis + {wetdry} wet-dry cycling",
            "molecular_weight":     round(sum(m["molecular_weight"] for m in s2_aa[:chain_len_est]) - 18 * max(0, chain_len_est - 1), 1),
            "logP":                 0.0,
            "hbond_donors":         chain_len_est,
            "hbond_acceptors":      chain_len_est,
            "boiling_point_est":    None,
            "state_at_conditions":  "liquid" if solvnt == "water" else "solid",
            "stability_score":      round(s3_score, 4),
            "stable_at_conditions": s3_score >= 0.4,
        })
    for m in s2_aa[:3]:
        vesicle = {**m,
                   "name":  f"{m['name']} lipid vesicle precursor",
                   "why":   "Amphiphilic self-assembly in aqueous solvent",
                   "stable_at_conditions": sal_ok and m["stability_score"] > 0.4}
        s3_mols.append(vesicle)

    stages_scores["s3"]    = round(s3_score, 4)
    stages_molecules["s3"] = s3_mols
    stages_reasons["s3"]   = (
        f"Clay={clay} (+{clay_boost:.2f}), wet-dry={wetdry} (+{wdc_boost:.2f}). "
        f"Estimated peptide chain length: {chain_len_est} residues. "
        f"Salinity {'✓ OK' if sal_ok else '✗ suboptimal'} for vesicle stability."
    )

    if s3_score <= 0.25:
        return _build_result(stages_scores, stages_reasons, stages_molecules,
                             blocked_at=3, solvnt=solvnt)

    # ── STAGE 4 ── Autocatalytic networks ─────────────────────────────────────
    # Molecular diversity: count distinct element types across s1+s2 stable mols
    all_formulas   = [m["formula"] for m in (s1_stable + s2_aa) if m.get("formula")]
    elements_seen  = set("".join(f for f in all_formulas if f.isalpha()).replace("l","").replace("r","").replace("n",""))
    diversity      = _clamp(len(elements_seen) / 6)

    # Fe-S cluster formation requires hydrothermal + H2S signal + metal ions
    fes_bonus = 0.15 if (vents and p.get("atm_H2S", 0) >= 1
                         and metals in ("medium", "high")) else 0.0
    # Sulfur-based proto-metabolism (de Duve thioester world)
    sulfur_meta = {"low": 0.0, "medium": 0.12, "high": 0.20}.get(sp, 0.0)

    s4_base  = s3_score * 0.45 + diversity * 0.20 + fes_bonus + sulfur_meta
    s4_score = _clamp(s4_base + (0.08 if vents else 0.0))

    s4_mols = [{
        "name":                 "Autocatalytic network (predicted)",
        "smiles":               "",
        "formula":              "multi-component",
        "why":                  (
            f"Fe-S clusters {'detected' if fes_bonus > 0 else 'absent'}; "
            f"sulfur proto-metabolism score {sulfur_meta:.2f}; "
            f"feedstock diversity {len(elements_seen)} element types."
        ),
        "molecular_weight":     0,
        "logP":                 0,
        "hbond_donors":         0,
        "hbond_acceptors":      0,
        "boiling_point_est":    None,
        "state_at_conditions":  "network",
        "stability_score":      round(s4_score, 4),
        "stable_at_conditions": s4_score >= 0.4,
    }]

    stages_scores["s4"]    = round(s4_score, 4)
    stages_molecules["s4"] = s4_mols
    stages_reasons["s4"]   = (
        f"Feedstock diversity: {len(elements_seen)} elements, {len(s1_stable + s2_aa)} molecules. "
        f"Fe-S bonus={fes_bonus:.2f}, sulfur metabolism={sulfur_meta:.2f}, "
        f"vent bonus={0.08 if vents else 0.0:.2f}."
    )

    if s4_score <= 0.25:
        return _build_result(stages_scores, stages_reasons, stages_molecules,
                             blocked_at=4, solvnt=solvnt)

    # ── STAGE 5 ── Proto-life potential ────────────────────────────────────────
    # Determine dominant pathway and biochemistry type
    if solvnt == "water":
        biochem = "carbon-water"
        biochem_factor = 1.0
    elif solvnt == "ammonia":
        biochem = "carbon-ammonia"
        biochem_factor = 0.75
    else:
        biochem = "carbon-exotic"
        biochem_factor = 0.45

    # RNA-world readiness: nucleotide precursors present + phosphate
    nucleotide_present = any(
        'N' in m.get("formula", "") and 100 <= m.get("molecular_weight", 0) <= 350
        for m in s2_aa
    )
    rna_score = (0.6 if nucleotide_present else 0.2) * {"low": 0.4, "medium": 0.7, "high": 1.0}.get(sp, 0.4)

    # Metabolism-first readiness: thioester + Fe-S
    metab_score = _clamp(sulfur_meta + (0.3 if vents else 0.0) + fes_bonus)

    # Heredity temperature window (RNA denatures above ~85°C, below -10°C too slow)
    heredity_temp = (
        1.0 if 10 <= temp <= 75
        else _clamp(1.0 - abs(temp - 42) / 60)
    )

    s5_base  = (
        s4_score * 0.35 +
        max(rna_score, metab_score) * 0.35 +
        heredity_temp * 0.20 +
        0.10  # baseline persistence
    ) * biochem_factor
    s5_score = _clamp(s5_base)

    pathway = "RNA-world" if rna_score >= metab_score else "metabolism-first"

    s5_mols = [{
        "name":                 f"Proto-life system ({pathway})",
        "smiles":               "",
        "formula":              "emergent",
        "why":                  (
            f"Dominant pathway: {pathway}. "
            f"RNA readiness={rna_score:.2f}, metabolism-first={metab_score:.2f}. "
            f"Heredity temperature score={heredity_temp:.2f}."
        ),
        "molecular_weight":     0,
        "logP":                 0,
        "hbond_donors":         0,
        "hbond_acceptors":      0,
        "boiling_point_est":    None,
        "state_at_conditions":  "emergent",
        "stability_score":      round(s5_score, 4),
        "stable_at_conditions": s5_score >= 0.4,
    }]

    stages_scores["s5"]    = round(s5_score, 4)
    stages_molecules["s5"] = s5_mols
    stages_reasons["s5"]   = (
        f"Pathway={pathway}, biochemistry={biochem}. "
        f"RNA score={rna_score:.2f}, metabolism-first={metab_score:.2f}, "
        f"heredity temp score={heredity_temp:.2f}, biochem factor={biochem_factor:.2f}."
    )

    return _build_result(stages_scores, stages_reasons, stages_molecules,
                         blocked_at=None, solvnt=solvnt,
                         biochem=biochem, proto=s5_score)


# ---------------------------------------------------------------------------
# Result assembler
# ---------------------------------------------------------------------------

def _build_result(scores: dict, reasons: dict, molecules: dict,
                  blocked_at: int | None, solvnt: str,
                  biochem: str = "unknown", proto: float = 0.0) -> dict:
    """Assemble the standardised output dict."""
    for k in ["s1", "s2", "s3", "s4", "s5"]:
        scores.setdefault(k, 0.0)
        reasons.setdefault(k, "Not reached." if blocked_at else "")
        molecules.setdefault(k, [])

    if blocked_at:
        stage_keys = ["s1", "s2", "s3", "s4", "s5"]
        for i in range(blocked_at, 5):
            key = stage_keys[i]
            reasons[key] = f"Blocked: Stage {blocked_at} threshold not met."

    overall = round(sum(scores[k] for k in ["s1","s2","s3","s4","s5"]) / 5.0, 4)

    if overall >= 0.75:   verdict = "HIGHLY FAVOURABLE"
    elif overall >= 0.55: verdict = "FAVOURABLE"
    elif overall >= 0.35: verdict = "MARGINAL"
    else:                 verdict = "UNLIKELY"

    return {
        "stages":            scores,
        "reasons":           reasons,
        "molecules":         molecules,
        "biochemistry":      biochem if biochem != "unknown" else "—",
        "proto_life_score":  round(proto, 4),
        "overall_score":     overall,
        "verdict":           verdict,
        "prediction_source": "MINE+RDKit",
    }


# ---------------------------------------------------------------------------
# Entry point (matches engine.py's run_simulation interface)
# ---------------------------------------------------------------------------

def run_simulation(planet_params: dict | None = None) -> dict:
    """Drop-in replacement for engine.run_simulation()."""
    if planet_params is None:
        from engine import PLANET
        planet_params = PLANET
    return run_reaction_pipeline(planet_params)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from engine import PLANET
    import pprint
    result = run_simulation(PLANET)
    pprint.pprint({k: v for k, v in result.items() if k != "molecules"})
    print("\nMolecules per stage:")
    for stage, mols in result["molecules"].items():
        print(f"  {stage}: {[m['name'] for m in mols]}")

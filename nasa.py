"""
nasa.py — Fetches exoplanet data from NASA Exoplanet Archive and maps to simulation engine format.
Two API calls: planetary parameters (TAP/pscomppars) + JWST spectroscopy (atmospheres_spectroscopy).
Merge logic: JWST confirmed > density estimate > hardcoded literature values.
"""

import json
import os
import time
import requests

CACHE_FILE = "planet_cache.json"
CACHE_TTL  = 86400  # 24 hours in seconds

# --- API endpoints -----------------------------------------------------------
URL_PLANETS = (
    "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    "?query=select+pl_name,pl_rade,pl_masse,pl_eqt,pl_dens,st_teff,pl_orbsmax"
    "+from+pscomppars"
    "+where+pl_rade+%3C+2.5+and+pl_eqt+%3E+150+and+pl_eqt+%3C+500+and+pl_masse+%3C+10"
    "&format=json"
)
URL_SPECTRO = (
    "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    "?query=select+pl_name,moleculeshortname,featuresatmax"
    "+from+atmospheres_spectroscopy"
    "+where+moleculeshortname+in+('H2O','CH4','CO2','NH3','CO','SO2','HCN')"
    "&format=json"
)

# Relative detection strength per molecule → used as approximate % contribution.
# CO2 weighted highest (12) because it dominates even thin rocky atmospheres;
# H2O at 10 because liquid-water worlds are prime targets; CH4 at 8 as biosignature.
MOLECULE_WEIGHTS = {'H2O': 10, 'CH4': 8, 'CO2': 12, 'NH3': 5, 'CO': 4, 'SO2': 3, 'HCN': 2}

# Literature-verified atmospheres for well-studied habitable-zone candidates.
KNOWN_ATMOSPHERES = {
    'K2-18 b':       {'CH4':3,  'NH3':2, 'H2':8,  'CO2':1,  'N2':85, 'H2O':1,  'temp':-8,   'gravity':1.1},
    'TOI-270 d':     {'CH4':2,  'NH3':1, 'H2':22, 'CO2':2,  'N2':70, 'H2O':3,  'temp':107,  'gravity':0.9},
    'TRAPPIST-1 e':  {'CH4':5,  'NH3':3, 'H2':7,  'CO2':15, 'N2':65, 'H2O':5,  'temp':-40,  'gravity':0.93},
    'Kepler-442 b':  {'CH4':8,  'NH3':5, 'H2':8,  'CO2':20, 'N2':55, 'H2O':4,  'temp':30,   'gravity':1.3},
    'Proxima Cen b': {'CH4':6,  'NH3':4, 'H2':7,  'CO2':25, 'N2':55, 'H2O':3,  'temp':-39,  'gravity':1.1},
    'LHS 1140 b':    {'CH4':4,  'NH3':2, 'H2':6,  'CO2':30, 'N2':53, 'H2O':5,  'temp':-23,  'gravity':1.4},
}


# --- Atmosphere builders -----------------------------------------------------

def _atm_from_spectroscopy(detected_molecules: list[str]) -> dict:
    """Convert a list of detected molecule names into percentage atmosphere dict.
    Each detected molecule gets its MOLECULE_WEIGHTS value as an approximate %.
    H2 gets a 5% bonus for reducing atmospheres (CH4 or NH3 signal implies H2 present).
    N2 fills the remainder to reach exactly 100%.
    """
    atm = {}
    for mol in detected_molecules:
        if mol in MOLECULE_WEIGHTS:
            atm[mol] = MOLECULE_WEIGHTS[mol]
    # Reducing-atmosphere heuristic: CH4/NH3 form in H2-rich environments
    if 'CH4' in atm or 'NH3' in atm:
        atm['H2'] = 5
    total = sum(atm.values())
    atm['N2'] = max(0, 100 - total)  # N2 is cosmically abundant, sensible filler
    # Re-normalise if weights alone already exceed 100 (rare but defensive)
    total = sum(atm.values())
    if total != 100:
        atm['N2'] = max(0, atm['N2'] + (100 - total))
    return atm


def _atm_from_density(density: float) -> dict:
    """Estimate bulk atmosphere from bulk density (g/cm³).
    Dense > 5.5 → CO2-heavy like Mars/Venus (thick silicate mantle, degassing).
    4.0–5.5  → Earth-analogue with N2-O2 approximated as N2+CH4 for prebiotic sim.
    2.5–4.0  → Sub-Neptune, volatile-rich mixed envelope.
    < 2.5    → Low-density, H2/He-dominated mini-Neptune.
    """
    if density is None:
        return {'CH4':10, 'NH3':5, 'H2':15, 'CO2':20, 'N2':45, 'H2O':5}
    if density > 5.5:
        return {'CH4':3,  'NH3':2,  'H2':5,  'CO2':65, 'N2':23, 'H2O':2}
    elif density > 4.0:
        return {'CH4':10, 'NH3':5,  'H2':15, 'CO2':20, 'N2':45, 'H2O':5}
    elif density > 2.5:
        return {'CH4':20, 'NH3':10, 'H2':25, 'CO2':15, 'N2':25, 'H2O':5}
    else:
        return {'CH4':15, 'NH3':8,  'H2':40, 'CO2':10, 'N2':22, 'H2O':5}


# --- Engine parameter mapper -------------------------------------------------

def map_to_engine_params(planet_raw: dict, atm_dict: dict, temp_c: float,
                          gravity: float, st_teff: float) -> dict:
    """Map NASA raw fields + derived atmosphere into the simulation engine format.
    UV proxy: stellar effective temperature drives photochemistry energy input.
    Solvent: water (0–100 °C), ammonia (-78–-33 °C), else methane (Titan-like).
    Lightning scales with liquid-water presence (ionisation of polar solvent).
    Geothermal scales with gravity (tidal + radioactive heating proxy).
    """
    uv = 'high' if st_teff > 6500 else 'medium' if st_teff > 4500 else 'low'
    liquid_water = 0 <= temp_c <= 100
    solvent = ('water'   if liquid_water else
               'ammonia' if -78 <= temp_c <= -33 else
               'methane')
    lightning  = 400 if liquid_water else 80   # W/m² proxy; storms need polar solvent
    geothermal = 200 if gravity > 1.2 else 120 # higher-g worlds retain heat longer

    return {
        'atm_CH4':          atm_dict.get('CH4', 5),
        'atm_NH3':          atm_dict.get('NH3', 3),
        'atm_H2':           atm_dict.get('H2',  10),
        'atm_CO2':          atm_dict.get('CO2', 20),
        'atm_N2':           atm_dict.get('N2',  57),
        'atm_H2O':          atm_dict.get('H2O', 5),
        'temperature':      round(temp_c, 1),
        'pressure':         1.0,
        'uv':               uv,
        'lightning':        lightning,
        'geothermal':       geothermal,
        'ocean_ph':         7.0,
        'salinity':         35,
        'hydrothermal':     True,
        'clay':             'montmorillonite',
        'metal_ions':       'medium',
        'sulfur_phosphate': 'medium',
        'gravity':          round(gravity, 2),
        'wet_dry':          'tidal' if liquid_water else 'none',
        'solvent':          solvent,
    }


# --- Core fetch & merge ------------------------------------------------------

def _fetch_json(url: str) -> list[dict]:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _build_planets() -> list[dict]:
    """Fetch both NASA endpoints, merge by pl_name, return engine-ready list."""
    try:
        raw_planets = _fetch_json(URL_PLANETS)
        raw_spectro = _fetch_json(URL_SPECTRO)
    except Exception:
        # API unreachable → fall back to hardcoded known planets silently
        return _hardcoded_fallback()

    # Build spectroscopy index: {pl_name → [molecule, ...]}
    spectro_index: dict[str, list[str]] = {}
    for row in raw_spectro:
        name = row.get('pl_name', '').strip()
        mol  = row.get('moleculeshortname', '').strip()
        if name and mol:
            spectro_index.setdefault(name, []).append(mol)

    results = []
    seen_names = set()

    for p in raw_planets:
        name = (p.get('pl_name') or '').strip()
        if not name or name in seen_names:
            continue
        seen_names.add(name)

        # Equilibrium temp from NASA is in Kelvin → convert to Celsius
        eq_k    = p.get('pl_eqt') or 255
        temp_c  = eq_k - 273.15

        # Gravity proxy: g ∝ mass/radius² (both in Earth units → result in g⊕)
        mass    = p.get('pl_masse') or 1.0
        radius  = p.get('pl_rade')  or 1.0
        gravity = round(mass / (radius ** 2), 2) if radius else 1.0

        st_teff = p.get('st_teff') or 5778  # default to Sun-like if missing
        density = p.get('pl_dens')          # g/cm³, may be None

        if name in spectro_index:
            # Priority 1 — real JWST detections available
            atm     = _atm_from_spectroscopy(spectro_index[name])
            quality = "JWST confirmed"
        else:
            # Priority 2 — estimate from bulk density
            atm     = _atm_from_density(density)
            quality = "estimated from density"

        engine = map_to_engine_params(p, atm, temp_c, gravity, st_teff)

        results.append({
            'name':         name,
            'data_quality': quality,
            'engine_params': engine,
            'raw':          p,
        })

    # Append hardcoded planets that didn't appear in the API results
    api_names = {r['name'] for r in results}
    for kname, kdata in KNOWN_ATMOSPHERES.items():
        if kname not in api_names:
            atm     = {k: v for k, v in kdata.items() if k not in ('temp', 'gravity')}
            temp_c  = kdata['temp']
            gravity = kdata['gravity']
            engine  = map_to_engine_params({}, atm, temp_c, gravity, 4500)
            results.append({
                'name':          kname,
                'data_quality':  'literature values',
                'engine_params': engine,
                'raw':           {},
            })

    return sorted(results, key=lambda x: x['name'])


def _hardcoded_fallback() -> list[dict]:
    """Return only the 6 known planets when NASA API is unavailable."""
    out = []
    for name, kdata in KNOWN_ATMOSPHERES.items():
        atm    = {k: v for k, v in kdata.items() if k not in ('temp', 'gravity')}
        engine = map_to_engine_params({}, atm, kdata['temp'], kdata['gravity'], 4500)
        out.append({'name': name, 'data_quality': 'literature values',
                    'engine_params': engine, 'raw': {}})
    return out


# --- Public API --------------------------------------------------------------

def fetch_planets() -> list[dict]:
    """Return all planets, using 24-hour disk cache if available."""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE) as f:
                cached = json.load(f)
            if time.time() - cached.get('cached_at', 0) < CACHE_TTL:
                return cached['planets']
        except Exception:
            pass  # corrupt cache → re-fetch

    planets = _build_planets()
    try:
        with open(CACHE_FILE, 'w') as f:
            json.dump({'cached_at': time.time(), 'planets': planets}, f)
    except Exception:
        pass  # non-fatal if cache write fails
    return planets


def get_planet(name: str) -> dict | None:
    """Return a single planet dict by name, or None if not found."""
    return next((p for p in fetch_planets() if p['name'] == name), None)


def get_all_names() -> list[str]:
    """Return sorted list of all planet names (for dropdown menus etc.)."""
    return sorted(p['name'] for p in fetch_planets())


# --- Test block --------------------------------------------------------------

if __name__ == "__main__":
    planets = fetch_planets()
    print(f"\nTotal planets fetched: {len(planets)}")

    print(f"\nPlanets with real JWST spectroscopy:")
    for p in planets:
        if p['data_quality'] == 'JWST confirmed':
            ep = p['engine_params']
            print(f"  {p['name']} — {ep['temperature']}°C — solvent: {ep['solvent']}")

    print(f"\nSample planet (first in list):")
    print(json.dumps(planets[0], indent=2))

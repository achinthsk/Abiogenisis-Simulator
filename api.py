from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response, StreamingResponse, RedirectResponse
import asyncio
import time

_START_TS = str(int(time.time()))
from pydantic import BaseModel, Field
from typing import Optional
from engine import PLANET, validate
from validation import run_validation_report
from nasa import fetch_planets, get_planet

try:
    from reaction_engine import run_simulation as _re_simulate
    _USE_REACTION_ENGINE = True
except Exception:
    from engine import run_simulation as _re_simulate
    _USE_REACTION_ENGINE = False

def run_simulation(planet: dict) -> dict:
    """Try reaction_engine first; fall back to rule engine on any error."""
    if _USE_REACTION_ENGINE:
        try:
            return _re_simulate(planet)
        except Exception:
            from engine import run_simulation as _fallback
            result = _fallback(planet)
            result["prediction_source"] = "rule_engine_fallback"
            return result
    return _re_simulate(planet)

app = FastAPI(title="Prebiotic Chemistry Simulation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


class PlanetParams(BaseModel):
    atm_CH4:          Optional[float] = Field(None, ge=0, le=100)
    atm_NH3:          Optional[float] = Field(None, ge=0, le=100)
    atm_H2:           Optional[float] = Field(None, ge=0, le=100)
    atm_CO2:          Optional[float] = Field(None, ge=0, le=100)
    atm_N2:           Optional[float] = Field(None, ge=0, le=100)
    atm_H2O:          Optional[float] = Field(None, ge=0, le=100)
    pressure:         Optional[float] = Field(None, ge=0.01, le=100)
    uv:               Optional[str]   = Field(None, pattern="^(low|medium|high)$")
    lightning:        Optional[float] = Field(None, ge=0, le=10000)
    geothermal:       Optional[float] = Field(None, ge=0, le=500)
    temperature:      Optional[float] = Field(None, ge=-200, le=500)
    ocean_pH:         Optional[float] = Field(None, ge=0, le=14)
    salinity:         Optional[float] = Field(None, ge=0, le=400)
    hydrothermal:     Optional[bool]  = None
    clay:             Optional[str]   = Field(None, pattern="^(none|montmorillonite|kaolinite)$")
    metal_ions:       Optional[str]   = Field(None, pattern="^(low|medium|high)$")
    sulfur_phosphate: Optional[str]   = Field(None, pattern="^(low|medium|high)$")
    gravity:          Optional[float] = Field(None, ge=0.1, le=5)
    wet_dry:          Optional[str]   = Field(None, pattern="^(none|tidal|volcanic)$")
    solvent:          Optional[str]   = Field(None, pattern="^(water|ammonia|methane)$")


@app.post("/simulate")
def simulate(params: PlanetParams):
    planet = {**PLANET, **{k: v for k, v in params.model_dump().items() if v is not None}}
    try:
        validate(planet)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return run_simulation(planet)


@app.get("/validation")
def validation_endpoint():
    return run_validation_report()


@app.get("/planets")
def planets_list():
    planets = fetch_planets()
    return [{"name": p["name"], "data_quality": p.get("data_quality", "estimated")} for p in planets]


@app.get("/planets/{name}")
def planet_detail(name: str):
    p = get_planet(name)
    if p is None:
        raise HTTPException(status_code=404, detail=f"Planet '{name}' not found.")
    ep = p.get("engine_params", {})
    return {**ep, "data_quality": p.get("data_quality", "estimated")}


_SVG_CACHE: dict[str, str] = {}

@app.get("/mol/svg")
def mol_svg(smiles: str = Query(..., description="SMILES string to render")):
    """Return a dark-themed 2D structure SVG for a SMILES string.
    Rendered server-side via pure-Python RDKit coordinate generation (no X11).
    Results are cached in memory for the lifetime of the process.
    """
    if smiles in _SVG_CACHE:
        return Response(content=_SVG_CACHE[smiles], media_type="image/svg+xml",
                        headers={"Cache-Control": "public, max-age=86400"})
    try:
        from mol_svg import smiles_to_svg
        svg = smiles_to_svg(smiles)
        if svg is None:
            raise HTTPException(status_code=404, detail="Invalid SMILES")
        _SVG_CACHE[smiles] = svg
        return Response(content=svg, media_type="image/svg+xml",
                        headers={"Cache-Control": "public, max-age=86400"})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


_NO_CACHE = {"Cache-Control": "no-store, no-cache, must-revalidate", "Pragma": "no-cache"}

@app.get("/livereload")
async def livereload():
    async def stream():
        try:
            while True:
                yield "data: ping\n\n"
                await asyncio.sleep(10)
        except asyncio.CancelledError:
            pass
    return StreamingResponse(stream(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

@app.get("/app")
def frontend():
    return FileResponse("index.html", headers=_NO_CACHE)


@app.get("/")
def root(request: Request):
    t = request.query_params.get("_t")
    if t != _START_TS:
        return RedirectResponse(url=f"/?_t={_START_TS}", status_code=302,
            headers={"Cache-Control": "no-store, no-cache", "Pragma": "no-cache"})
    return FileResponse("index.html", headers=_NO_CACHE)


from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

@app.get("/planets")
def list_planets():
    from nasa import fetch_planets
    return [{"name": p["name"], "data_quality": p["data_quality"]} for p in fetch_planets()]

@app.get("/planets/{name}")
def planet_detail(name: str):
    from nasa import get_planet
    p = get_planet(name)
    if not p:
        raise HTTPException(404, "Planet not found")
    return p

@app.get("/mol/svg")
def mol_svg(smiles: str):
    from mol_svg import render_svg
    svg = render_svg(smiles)
    if not svg:
        raise HTTPException(404, "Invalid SMILES")
    from fastapi.responses import Response
    return Response(content=svg, media_type="image/svg+xml")

@app.get("/app")
def frontend():
    return FileResponse("index.html")

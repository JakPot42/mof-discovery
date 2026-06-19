"""
Deterministic scoring and ranking of MOF structures by application.

Scoring philosophy:
  - Published experimental data (water uptake, CO2 uptake, etc.) → primary score.
  - Structural proxies (surface area, pore size, functional groups) → fallback when
    no direct experimental data is available for the target gas.
  - data_source field in result distinguishes the two cases.

Score scale: 0–100. Higher = better fit for the stated use case.
"""
from __future__ import annotations
import math
from enum import Enum


class UseCase(str, Enum):
    WATER_HARVESTING = "water_harvesting"
    CO2_CAPTURE = "co2_capture"
    H2_STORAGE = "h2_storage"
    CH4_STORAGE = "ch4_storage"
    VOC_REMOVAL = "voc_removal"


USE_CASE_LABELS: dict[UseCase, str] = {
    UseCase.WATER_HARVESTING: "Atmospheric Water Harvesting",
    UseCase.CO2_CAPTURE: "CO₂ Capture (Flue Gas, ~15% CO₂)",
    UseCase.H2_STORAGE: "Hydrogen Storage (Cryogenic, 77 K)",
    UseCase.CH4_STORAGE: "Natural Gas / Methane Storage",
    UseCase.VOC_REMOVAL: "VOC Removal / Air Purification",
}

USE_CASE_KEY_METRIC: dict[UseCase, str] = {
    UseCase.WATER_HARVESTING: "Water uptake (g/g, 20% RH)",
    UseCase.CO2_CAPTURE: "CO₂ uptake (mmol/g, 0.15 bar)",
    UseCase.H2_STORAGE: "H₂ uptake (wt%, 1 bar / 77 K)",
    UseCase.CH4_STORAGE: "CH₄ uptake (cm³/cm³, 35 bar)",
    UseCase.VOC_REMOVAL: "BET surface area (m²/g)",
}

USE_CASE_DESCRIPTIONS: dict[UseCase, str] = {
    UseCase.WATER_HARVESTING: (
        "Harvest drinking water from atmospheric humidity — including desert air at "
        "20–30% RH. Nobel Prize 2025 application (Yaghi group: MOF-801, Science 2017)."
    ),
    UseCase.CO2_CAPTURE: (
        "Capture CO₂ from power plant or industrial flue gas (~15% CO₂ in N₂) "
        "at 298 K. Targets both total uptake at 0.15 bar and CO₂/N₂ selectivity."
    ),
    UseCase.H2_STORAGE: (
        "Physisorptive H₂ storage at cryogenic temperature (77 K). Relevant to "
        "high-density hydrogen transport and fuel-cell vehicle range."
    ),
    UseCase.CH4_STORAGE: (
        "Volumetric methane storage at 35–65 bar, 298 K for natural gas vehicles. "
        "DoE target: 263 cm³(STP)/cm³ at 65 bar."
    ),
    UseCase.VOC_REMOVAL: (
        "Adsorptive removal of aromatic VOCs (benzene, toluene, xylene) and aldehydes "
        "from indoor/industrial air streams."
    ),
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _normalize(value: float | None, max_val: float) -> float:
    """Linear 0–1 normalisation capped at max_val. Returns 0 if value is None."""
    if value is None or max_val == 0:
        return 0.0
    return min(1.0, max(0.0, value / max_val))


def _pore_score(pld: float | None, ideal_min: float, ideal_max: float) -> float:
    """Score 0–1 based on how close PLD is to the ideal range.
    Returns 0.5 (neutral) if pld is unknown."""
    if pld is None:
        return 0.5
    if ideal_min <= pld <= ideal_max:
        return 1.0
    elif pld < ideal_min:
        penalty_range = ideal_min
        return max(0.0, 1.0 - (ideal_min - pld) / penalty_range)
    else:
        penalty_range = ideal_max
        return max(0.0, 1.0 - (pld - ideal_max) / penalty_range)


def _hydrophilicity_score(functional_groups: str, metal_node: str, water_stability: str) -> float:
    """0–1 score for hydrophilic character, relevant to water harvesting."""
    score = 0.0
    fg = (functional_groups or "").lower()
    metal = (metal_node or "").lower()

    if "-oh" in fg or fg == "oh":
        score += 0.30
    if "-nh2" in fg or "nh2" in fg:
        score += 0.25
    if "-cooh" in fg:
        score += 0.10
    if "-f" in fg:
        score -= 0.10  # fluorinated pores are hydrophobic

    if metal in ("al", "zr", "fe", "mg", "ti"):
        score += 0.30
    elif metal in ("cu", "co", "ni", "mn"):
        score += 0.10

    if water_stability == "high":
        score += 0.10

    return min(1.0, max(0.0, score))


def _water_stability_score(water_stability: str | None) -> float:
    return {"high": 1.0, "moderate": 0.5, "low": 0.0}.get(water_stability or "low", 0.0)


# ---------------------------------------------------------------------------
# Per-use-case scoring
# ---------------------------------------------------------------------------

def _score_water(mof: dict) -> tuple[float, dict, str]:
    score = 0.0
    breakdown: dict[str, float] = {}
    data_source = "structural_proxy"

    # Primary: published water uptake at 20% RH (45 pts max)
    if mof.get("water_uptake_g_g_20rh") is not None:
        wu = mof["water_uptake_g_g_20rh"]
        s = _normalize(wu, 0.55) * 45.0
        score += s
        breakdown["water_uptake_20pct_RH"] = round(s, 1)
        data_source = "published"
    elif mof.get("water_uptake_g_g_40rh") is not None:
        # 40% RH data available — discount because step happens at higher RH
        wu = mof["water_uptake_g_g_40rh"] * 0.55
        s = _normalize(wu, 0.55) * 32.0
        score += s
        breakdown["water_uptake_40pct_RH_discounted"] = round(s, 1)
        data_source = "published_40rh"
    else:
        # Void fraction proxy: higher void fraction → more potential pore volume for water
        s = _normalize(mof.get("void_fraction"), 0.75) * 20.0
        score += s
        breakdown["void_fraction_proxy"] = round(s, 1)

    # Pore size: ideal 5–15 Å for cooperative water uptake (20 pts max)
    ps = _pore_score(mof.get("pore_limiting_diameter_A"), 5.0, 15.0) * 20.0
    score += ps
    breakdown["pore_size_score"] = round(ps, 1)

    # Hydrophilicity: drives low-RH step onset (20 pts max)
    hy = _hydrophilicity_score(
        mof.get("functional_groups", ""),
        mof.get("metal_node", ""),
        mof.get("water_stability", ""),
    ) * 20.0
    score += hy
    breakdown["hydrophilicity"] = round(hy, 1)

    # Water stability: must survive humid cycling (15 pts max)
    ws = _water_stability_score(mof.get("water_stability")) * 15.0
    score += ws
    breakdown["water_stability"] = round(ws, 1)

    return round(min(score, 100.0), 1), breakdown, data_source


def _score_co2(mof: dict) -> tuple[float, dict, str]:
    score = 0.0
    breakdown: dict[str, float] = {}
    data_source = "structural_proxy"

    # Primary: CO2 uptake at 0.15 bar, 298 K (50 pts max)
    if mof.get("co2_uptake_mmol_g_015bar") is not None:
        co2 = mof["co2_uptake_mmol_g_015bar"]
        s = _normalize(co2, 5.0) * 50.0
        score += s
        breakdown["co2_uptake_0.15bar"] = round(s, 1)
        data_source = "published"
    elif mof.get("co2_uptake_mmol_g_1bar") is not None:
        # 1 bar data — discount (uptake at 0.15 bar is ~40-60% of 1 bar)
        co2 = mof["co2_uptake_mmol_g_1bar"] * 0.45
        s = _normalize(co2, 5.0) * 38.0
        score += s
        breakdown["co2_uptake_1bar_discounted"] = round(s, 1)
        data_source = "published_1bar"
    else:
        # Structural proxy: open metal sites + amine groups → high CO2 affinity
        oms_bonus = 18.0 if mof.get("has_open_metal_sites") else 0.0
        fg = (mof.get("functional_groups", "") or "").lower()
        amine_bonus = 15.0 if ("-nh2" in fg or "nh2" in fg) else 0.0
        sa_s = _normalize(mof.get("surface_area_m2_g"), 4500.0) * 17.0
        score += oms_bonus + amine_bonus + sa_s
        breakdown["open_metal_sites_proxy"] = oms_bonus
        breakdown["amine_groups_proxy"] = amine_bonus
        breakdown["surface_area_proxy"] = round(sa_s, 1)

    # Selectivity: CO2/N2 IAST (35 pts max, log-scaled so 10→16 pts, 100→24 pts, 10000→35 pts)
    if mof.get("co2_n2_selectivity") is not None:
        sel = mof["co2_n2_selectivity"]
        log_score = _normalize(math.log10(max(sel, 1)), math.log10(10001)) * 35.0
        score += log_score
        breakdown["co2_n2_selectivity"] = round(log_score, 1)
    else:
        fg = (mof.get("functional_groups", "") or "").lower()
        oms = mof.get("has_open_metal_sites", False)
        if oms or "-nh2" in fg or "-f" in fg:
            s = 12.0  # estimated moderate-high selectivity
            score += s
            breakdown["selectivity_proxy"] = s

    # Surface area bonus: total capacity (15 pts max)
    sa_s = _normalize(mof.get("surface_area_m2_g"), 4500.0) * 15.0
    score += sa_s
    breakdown["surface_area"] = round(sa_s, 1)

    return round(min(score, 100.0), 1), breakdown, data_source


def _score_h2(mof: dict) -> tuple[float, dict, str]:
    score = 0.0
    breakdown: dict[str, float] = {}
    data_source = "structural_proxy"

    # Primary: published H2 wt% at 1 bar, 77 K (55 pts max)
    if mof.get("h2_uptake_wt_1bar_77k") is not None:
        h2 = mof["h2_uptake_wt_1bar_77k"]
        s = _normalize(h2, 2.3) * 55.0
        score += s
        breakdown["h2_uptake_1bar_77K"] = round(s, 1)
        data_source = "published"
    else:
        # Surface area proxy (H2 uptake at 77 K ~ linearly correlated with SA)
        sa_s = _normalize(mof.get("surface_area_m2_g"), 6500.0) * 40.0
        score += sa_s
        breakdown["surface_area_proxy"] = round(sa_s, 1)

    # Surface area: primary driver at 77 K (35 pts max)
    sa_s = _normalize(mof.get("surface_area_m2_g"), 6500.0) * 35.0
    score += sa_s
    breakdown["surface_area"] = round(sa_s, 1)

    # Open metal sites: slight benefit at 77 K (10 pts max)
    if mof.get("has_open_metal_sites"):
        score += 10.0
        breakdown["open_metal_sites"] = 10.0

    return round(min(score, 100.0), 1), breakdown, data_source


def _score_ch4(mof: dict) -> tuple[float, dict, str]:
    score = 0.0
    breakdown: dict[str, float] = {}
    data_source = "structural_proxy"

    # Primary: volumetric CH4 at 35 bar, 298 K (cm3/cm3) — DoE target 263 at 65 bar (55 pts max)
    if mof.get("ch4_uptake_cm3_cm3_35bar") is not None:
        ch4 = mof["ch4_uptake_cm3_cm3_35bar"]
        s = _normalize(ch4, 230.0) * 55.0
        score += s
        breakdown["ch4_uptake_volumetric"] = round(s, 1)
        data_source = "published"
    elif mof.get("ch4_uptake_cc_g_35bar") is not None:
        # Convert gravimetric to approximate volumetric
        density = mof.get("density_g_cm3") or 0.8
        ch4_vol = mof["ch4_uptake_cc_g_35bar"] * density
        s = _normalize(ch4_vol, 230.0) * 45.0
        score += s
        breakdown["ch4_uptake_gravimetric_converted"] = round(s, 1)
        data_source = "published_gravimetric"
    else:
        # Structural proxy
        oms_bonus = 20.0 if mof.get("has_open_metal_sites") else 0.0
        sa_s = _normalize(mof.get("surface_area_m2_g"), 5000.0) * 20.0
        vf_s = _normalize(mof.get("void_fraction"), 0.85) * 15.0
        score += oms_bonus + sa_s + vf_s
        breakdown["open_metal_sites_proxy"] = oms_bonus
        breakdown["surface_area_proxy"] = round(sa_s, 1)
        breakdown["void_fraction_proxy"] = round(vf_s, 1)

    # Open metal sites: Cu paddle-wheels strongly adsorb CH4 (20 pts max)
    if mof.get("has_open_metal_sites"):
        score += 20.0
        breakdown["open_metal_sites"] = 20.0

    # Surface area: secondary contributor (20 pts max)
    sa_s = _normalize(mof.get("surface_area_m2_g"), 5000.0) * 20.0
    score += sa_s
    breakdown["surface_area"] = round(sa_s, 1)

    return round(min(score, 100.0), 1), breakdown, data_source


def _score_voc(mof: dict) -> tuple[float, dict, str]:
    score = 0.0
    breakdown: dict[str, float] = {}

    # Surface area: primary driver for VOC capacity (40 pts max)
    sa_s = _normalize(mof.get("surface_area_m2_g"), 4500.0) * 40.0
    score += sa_s
    breakdown["surface_area"] = round(sa_s, 1)

    # Pore size: benzene kd ~5.9 Å, toluene ~6.1 Å → optimal PLD 5.5–14 Å (30 pts max)
    ps = _pore_score(mof.get("pore_limiting_diameter_A"), 5.5, 14.0) * 30.0
    score += ps
    breakdown["pore_size_score"] = round(ps, 1)

    # Hydrophobicity (aromatic VOCs prefer hydrophobic surfaces) (10 pts max)
    fg = (mof.get("functional_groups", "") or "").lower()
    hydro = 10.0
    if "-oh" in fg or "-nh2" in fg or "-cooh" in fg:
        hydro = 0.0   # hydrophilic groups reduce VOC uptake
    elif "-f" in fg:
        hydro = 5.0   # moderate
    score += hydro
    breakdown["hydrophobicity"] = hydro

    # Void fraction: more pore volume → more VOC loading (20 pts max)
    vf_s = _normalize(mof.get("void_fraction"), 0.88) * 20.0
    score += vf_s
    breakdown["void_fraction"] = round(vf_s, 1)

    return round(min(score, 100.0), 1), breakdown, "structural_proxy"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_SCORERS = {
    UseCase.WATER_HARVESTING: _score_water,
    UseCase.CO2_CAPTURE: _score_co2,
    UseCase.H2_STORAGE: _score_h2,
    UseCase.CH4_STORAGE: _score_ch4,
    UseCase.VOC_REMOVAL: _score_voc,
}


def score_mof(mof: dict, use_case: UseCase) -> dict:
    """Return a dict with score, breakdown, and data_source for one MOF."""
    scorer = _SCORERS[use_case]
    score, breakdown, data_source = scorer(mof)
    return {"score": score, "breakdown": breakdown, "data_source": data_source}


def rank(mofs: list[dict], use_case: UseCase, top_k: int = 10) -> list[dict]:
    """Score all MOFs for the given use case and return the top_k sorted by score."""
    scored = []
    for mof in mofs:
        result = score_mof(mof, use_case)
        entry = dict(mof)
        entry["score"] = result["score"]
        entry["breakdown"] = result["breakdown"]
        entry["data_source"] = result["data_source"]
        scored.append(entry)
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]

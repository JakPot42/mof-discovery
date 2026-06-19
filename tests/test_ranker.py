"""Tests for the deterministic MOF ranking engine."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from ranker import (
    UseCase, rank, score_mof,
    _normalize, _pore_score, _hydrophilicity_score, _water_stability_score,
)

# ---------------------------------------------------------------------------
# Fixtures — minimal MOF dicts for testing
# ---------------------------------------------------------------------------

def _water_mof(name="TEST-W", water_20rh=0.30):
    return {
        "name": name, "metal_node": "Al", "functional_groups": "-OH",
        "water_stability": "high", "void_fraction": 0.45,
        "pore_limiting_diameter_A": 6.5, "surface_area_m2_g": 1000.0,
        "has_open_metal_sites": False,
        "water_uptake_g_g_20rh": water_20rh, "water_uptake_g_g_40rh": None,
        "co2_uptake_mmol_g_015bar": None, "co2_uptake_mmol_g_1bar": None,
        "co2_n2_selectivity": None, "ch4_uptake_cm3_cm3_35bar": None,
        "ch4_uptake_cc_g_35bar": None, "h2_uptake_wt_1bar_77k": None,
        "h2_uptake_wt_100bar_77k": None, "density_g_cm3": 0.90,
    }


def _co2_mof(name="TEST-CO2", uptake_015=4.0, selectivity=100.0):
    return {
        "name": name, "metal_node": "Mg", "functional_groups": "",
        "water_stability": "low", "void_fraction": 0.72,
        "pore_limiting_diameter_A": 11.0, "surface_area_m2_g": 1500.0,
        "has_open_metal_sites": True,
        "water_uptake_g_g_20rh": None, "water_uptake_g_g_40rh": None,
        "co2_uptake_mmol_g_015bar": uptake_015, "co2_uptake_mmol_g_1bar": None,
        "co2_n2_selectivity": selectivity, "ch4_uptake_cm3_cm3_35bar": None,
        "ch4_uptake_cc_g_35bar": None, "h2_uptake_wt_1bar_77k": None,
        "h2_uptake_wt_100bar_77k": None, "density_g_cm3": 0.90,
    }


def _h2_mof(name="TEST-H2", h2_wt=1.5, sa=4000.0):
    return {
        "name": name, "metal_node": "Zn", "functional_groups": "",
        "water_stability": "low", "void_fraction": 0.80,
        "pore_limiting_diameter_A": 12.0, "surface_area_m2_g": sa,
        "has_open_metal_sites": False,
        "water_uptake_g_g_20rh": None, "water_uptake_g_g_40rh": None,
        "co2_uptake_mmol_g_015bar": None, "co2_uptake_mmol_g_1bar": None,
        "co2_n2_selectivity": None, "ch4_uptake_cm3_cm3_35bar": None,
        "ch4_uptake_cc_g_35bar": None, "h2_uptake_wt_1bar_77k": h2_wt,
        "h2_uptake_wt_100bar_77k": None, "density_g_cm3": 0.50,
    }


def _ch4_mof(name="TEST-CH4", ch4_vol=180.0):
    return {
        "name": name, "metal_node": "Cu", "functional_groups": "",
        "water_stability": "low", "void_fraction": 0.75,
        "pore_limiting_diameter_A": 9.0, "surface_area_m2_g": 1800.0,
        "has_open_metal_sites": True,
        "water_uptake_g_g_20rh": None, "water_uptake_g_g_40rh": None,
        "co2_uptake_mmol_g_015bar": None, "co2_uptake_mmol_g_1bar": None,
        "co2_n2_selectivity": None, "ch4_uptake_cm3_cm3_35bar": ch4_vol,
        "ch4_uptake_cc_g_35bar": None, "h2_uptake_wt_1bar_77k": None,
        "h2_uptake_wt_100bar_77k": None, "density_g_cm3": 0.88,
    }


def _voc_mof(name="TEST-VOC", sa=3500.0, pld=10.0, fg="", vf=0.80):
    return {
        "name": name, "metal_node": "Cr", "functional_groups": fg,
        "water_stability": "high", "void_fraction": vf,
        "pore_limiting_diameter_A": pld, "surface_area_m2_g": sa,
        "has_open_metal_sites": False,
        "water_uptake_g_g_20rh": None, "water_uptake_g_g_40rh": None,
        "co2_uptake_mmol_g_015bar": None, "co2_uptake_mmol_g_1bar": None,
        "co2_n2_selectivity": None, "ch4_uptake_cm3_cm3_35bar": None,
        "ch4_uptake_cc_g_35bar": None, "h2_uptake_wt_1bar_77k": None,
        "h2_uptake_wt_100bar_77k": None, "density_g_cm3": 0.44,
    }


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_zero_value(self):
        assert _normalize(0.0, 100.0) == 0.0

    def test_max_value(self):
        assert _normalize(100.0, 100.0) == 1.0

    def test_above_max_clamped(self):
        assert _normalize(200.0, 100.0) == 1.0

    def test_none_returns_zero(self):
        assert _normalize(None, 100.0) == 0.0

    def test_half(self):
        assert _normalize(50.0, 100.0) == pytest.approx(0.5)


class TestPoreScore:
    def test_ideal_range_returns_one(self):
        assert _pore_score(8.0, 5.0, 15.0) == 1.0

    def test_at_min_boundary(self):
        assert _pore_score(5.0, 5.0, 15.0) == 1.0

    def test_at_max_boundary(self):
        assert _pore_score(15.0, 5.0, 15.0) == 1.0

    def test_below_range_penalised(self):
        score = _pore_score(2.5, 5.0, 15.0)
        assert 0.0 <= score < 1.0

    def test_above_range_penalised(self):
        score = _pore_score(30.0, 5.0, 15.0)
        assert 0.0 <= score < 1.0

    def test_none_returns_neutral(self):
        assert _pore_score(None, 5.0, 15.0) == 0.5

    def test_larger_above_penalty_is_smaller_score(self):
        score_40 = _pore_score(40.0, 5.0, 15.0)
        score_20 = _pore_score(20.0, 5.0, 15.0)
        assert score_40 < score_20


class TestHydrophilicityScore:
    def test_al_metal_boosts_score(self):
        s = _hydrophilicity_score("", "Al", "high")
        assert s > 0.3

    def test_nh2_boosts_score(self):
        s_with = _hydrophilicity_score("-NH2", "Cu", "moderate")
        s_without = _hydrophilicity_score("", "Cu", "moderate")
        assert s_with > s_without

    def test_fluorinated_reduces_score(self):
        s_f = _hydrophilicity_score("-F", "Cu", "moderate")
        s_plain = _hydrophilicity_score("", "Cu", "moderate")
        assert s_f < s_plain

    def test_score_bounded_0_1(self):
        s = _hydrophilicity_score("-OH,-NH2", "Al", "high")
        assert 0.0 <= s <= 1.0

    def test_no_groups_low_score(self):
        s = _hydrophilicity_score("", "Zn", "low")
        assert s < 0.3


class TestWaterStabilityScore:
    def test_high(self):
        assert _water_stability_score("high") == 1.0

    def test_moderate(self):
        assert _water_stability_score("moderate") == 0.5

    def test_low(self):
        assert _water_stability_score("low") == 0.0

    def test_none_or_unknown(self):
        assert _water_stability_score(None) == 0.0
        assert _water_stability_score("unknown") == 0.0


# ---------------------------------------------------------------------------
# Scoring function tests
# ---------------------------------------------------------------------------

class TestScoreWater:
    def test_published_20rh_gives_high_score(self):
        result = score_mof(_water_mof(water_20rh=0.48), UseCase.WATER_HARVESTING)
        assert result["score"] > 60.0

    def test_high_uptake_beats_low_uptake(self):
        high = score_mof(_water_mof(water_20rh=0.48), UseCase.WATER_HARVESTING)
        low = score_mof(_water_mof(water_20rh=0.10), UseCase.WATER_HARVESTING)
        assert high["score"] > low["score"]

    def test_data_source_published(self):
        result = score_mof(_water_mof(water_20rh=0.30), UseCase.WATER_HARVESTING)
        assert result["data_source"] == "published"

    def test_proxy_when_no_published_data(self):
        mof = _water_mof()
        mof["water_uptake_g_g_20rh"] = None
        result = score_mof(mof, UseCase.WATER_HARVESTING)
        assert result["data_source"] == "structural_proxy"

    def test_score_in_range(self):
        result = score_mof(_water_mof(water_20rh=0.22), UseCase.WATER_HARVESTING)
        assert 0.0 <= result["score"] <= 100.0


class TestScoreCO2:
    def test_high_uptake_and_selectivity_top_score(self):
        result = score_mof(_co2_mof(uptake_015=4.0, selectivity=148.0), UseCase.CO2_CAPTURE)
        assert result["score"] > 60.0

    def test_selectivity_matters(self):
        hi_sel = score_mof(_co2_mof(uptake_015=2.0, selectivity=300.0), UseCase.CO2_CAPTURE)
        lo_sel = score_mof(_co2_mof(uptake_015=2.0, selectivity=5.0), UseCase.CO2_CAPTURE)
        assert hi_sel["score"] > lo_sel["score"]

    def test_data_source_published(self):
        result = score_mof(_co2_mof(), UseCase.CO2_CAPTURE)
        assert result["data_source"] == "published"

    def test_score_in_range(self):
        result = score_mof(_co2_mof(), UseCase.CO2_CAPTURE)
        assert 0.0 <= result["score"] <= 100.0


class TestScoreH2:
    def test_high_sa_and_uptake_gives_high_score(self):
        result = score_mof(_h2_mof(h2_wt=2.2, sa=6100.0), UseCase.H2_STORAGE)
        assert result["score"] > 70.0

    def test_data_source_published(self):
        result = score_mof(_h2_mof(), UseCase.H2_STORAGE)
        assert result["data_source"] == "published"

    def test_proxy_fallback_when_no_h2_data(self):
        mof = _h2_mof()
        mof["h2_uptake_wt_1bar_77k"] = None
        result = score_mof(mof, UseCase.H2_STORAGE)
        assert result["data_source"] == "structural_proxy"

    def test_score_in_range(self):
        result = score_mof(_h2_mof(), UseCase.H2_STORAGE)
        assert 0.0 <= result["score"] <= 100.0


class TestScoreCH4:
    def test_high_volumetric_gives_high_score(self):
        result = score_mof(_ch4_mof(ch4_vol=230.0), UseCase.CH4_STORAGE)
        assert result["score"] > 60.0

    def test_data_source_published(self):
        result = score_mof(_ch4_mof(), UseCase.CH4_STORAGE)
        assert result["data_source"] == "published"

    def test_score_in_range(self):
        result = score_mof(_ch4_mof(), UseCase.CH4_STORAGE)
        assert 0.0 <= result["score"] <= 100.0


class TestScoreVOC:
    def test_high_sa_gives_high_score(self):
        result = score_mof(_voc_mof(sa=4000.0), UseCase.VOC_REMOVAL)
        assert result["score"] > 50.0

    def test_hydrophilic_fg_reduces_score(self):
        hydro = score_mof(_voc_mof(fg="-OH"), UseCase.VOC_REMOVAL)
        plain = score_mof(_voc_mof(fg=""), UseCase.VOC_REMOVAL)
        assert plain["score"] > hydro["score"]

    def test_always_proxy(self):
        result = score_mof(_voc_mof(), UseCase.VOC_REMOVAL)
        assert result["data_source"] == "structural_proxy"

    def test_score_in_range(self):
        result = score_mof(_voc_mof(), UseCase.VOC_REMOVAL)
        assert 0.0 <= result["score"] <= 100.0


# ---------------------------------------------------------------------------
# Rank function tests
# ---------------------------------------------------------------------------

class TestRank:
    def _build_mofs(self, n=5):
        return [_water_mof(name=f"M{i}", water_20rh=i * 0.05 + 0.10) for i in range(n)]

    def test_returns_top_k(self):
        mofs = self._build_mofs(10)
        result = rank(mofs, UseCase.WATER_HARVESTING, top_k=3)
        assert len(result) == 3

    def test_sorted_descending(self):
        mofs = self._build_mofs(5)
        result = rank(mofs, UseCase.WATER_HARVESTING, top_k=5)
        scores = [r["score"] for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_best_mof_is_first(self):
        best = _water_mof(name="BEST", water_20rh=0.55)
        others = self._build_mofs(5)
        result = rank([best] + others, UseCase.WATER_HARVESTING, top_k=1)
        assert result[0]["name"] == "BEST"

    def test_result_includes_score_field(self):
        mofs = self._build_mofs(3)
        result = rank(mofs, UseCase.WATER_HARVESTING)
        for r in result:
            assert "score" in r
            assert "breakdown" in r
            assert "data_source" in r

    def test_all_use_cases_run_without_error(self):
        mof = _water_mof()
        for uc in UseCase:
            result = rank([mof], uc, top_k=1)
            assert len(result) == 1
            assert 0.0 <= result[0]["score"] <= 100.0

    def test_top_k_larger_than_list_returns_all(self):
        mofs = self._build_mofs(3)
        result = rank(mofs, UseCase.WATER_HARVESTING, top_k=10)
        assert len(result) == 3

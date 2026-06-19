"""Tests for seed data integrity and database seeding."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["DEMO_MODE"] = "True"

import pytest
from sqlalchemy import create_engine, select, func
from sqlalchemy.orm import sessionmaker
from models import Base, MOFStructure
from seed_data import SEED_MOFS, load_seed_data


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()
    engine.dispose()


class TestSeedDataConstants:
    def test_at_least_40_mofs_defined(self):
        assert len(SEED_MOFS) >= 40

    def test_all_entries_have_name(self):
        for m in SEED_MOFS:
            assert "name" in m and m["name"], f"Missing name in {m}"

    def test_all_entries_have_metal_node(self):
        for m in SEED_MOFS:
            assert "metal_node" in m and m["metal_node"], f"Missing metal_node in {m['name']}"

    def test_water_stability_values_are_valid(self):
        valid = {"low", "moderate", "high"}
        for m in SEED_MOFS:
            ws = m.get("water_stability")
            assert ws in valid, f"{m['name']} has invalid water_stability: {ws}"

    def test_no_duplicate_names(self):
        names = [m["name"] for m in SEED_MOFS]
        assert len(names) == len(set(names)), "Duplicate MOF names in SEED_MOFS"

    def test_mof801_water_uptake_at_20rh(self):
        entry = next(m for m in SEED_MOFS if m["name"] == "MOF-801")
        assert entry["water_uptake_g_g_20rh"] == pytest.approx(0.22, abs=0.01)

    def test_mof303_water_uptake_at_20rh(self):
        entry = next(m for m in SEED_MOFS if m["name"] == "MOF-303")
        assert entry["water_uptake_g_g_20rh"] == pytest.approx(0.48, abs=0.01)

    def test_mgmof74_co2_uptake(self):
        entry = next(m for m in SEED_MOFS if m["name"] == "Mg-MOF-74")
        assert entry["co2_uptake_mmol_g_015bar"] is not None
        assert entry["co2_uptake_mmol_g_015bar"] > 3.0

    def test_mof5_h2_uptake(self):
        entry = next(m for m in SEED_MOFS if m["name"] == "MOF-5")
        assert entry["h2_uptake_wt_1bar_77k"] is not None
        assert entry["h2_uptake_wt_1bar_77k"] > 1.0

    def test_nu100_h2_uptake_exceeds_mof5(self):
        nu100 = next(m for m in SEED_MOFS if m["name"] == "NU-100")
        mof5 = next(m for m in SEED_MOFS if m["name"] == "MOF-5")
        assert nu100["h2_uptake_wt_1bar_77k"] > mof5["h2_uptake_wt_1bar_77k"]

    def test_hkust1_in_seed(self):
        names = {m["name"] for m in SEED_MOFS}
        assert "HKUST-1" in names

    def test_void_fractions_in_valid_range(self):
        for m in SEED_MOFS:
            vf = m.get("void_fraction")
            if vf is not None:
                assert 0.0 < vf < 1.0, f"{m['name']} void_fraction out of range: {vf}"

    def test_surface_areas_positive(self):
        for m in SEED_MOFS:
            sa = m.get("surface_area_m2_g")
            if sa is not None:
                assert sa > 0, f"{m['name']} has non-positive surface area"

    def test_pore_diameters_positive(self):
        for m in SEED_MOFS:
            pld = m.get("pore_limiting_diameter_A")
            if pld is not None:
                assert pld > 0, f"{m['name']} PLD not positive"


class TestLoadSeedData:
    def test_loads_mofs_into_db(self, db):
        load_seed_data(db)
        count = db.execute(select(func.count()).select_from(MOFStructure)).scalar()
        assert count == len(SEED_MOFS)

    def test_idempotent_double_load(self, db):
        load_seed_data(db)
        load_seed_data(db)
        count = db.execute(select(func.count()).select_from(MOFStructure)).scalar()
        assert count == len(SEED_MOFS)

    def test_mof801_retrievable(self, db):
        load_seed_data(db)
        row = db.execute(
            select(MOFStructure).where(MOFStructure.name == "MOF-801")
        ).scalar_one_or_none()
        assert row is not None
        assert row.water_uptake_g_g_20rh == pytest.approx(0.22, abs=0.01)

    def test_to_dict_returns_name(self, db):
        load_seed_data(db)
        row = db.execute(select(MOFStructure).limit(1)).scalar_one()
        d = row.to_dict()
        assert "name" in d and d["name"]

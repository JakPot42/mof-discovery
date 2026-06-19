from __future__ import annotations
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, Text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from config import DATABASE_URL


class Base(DeclarativeBase):
    pass


class MOFStructure(Base):
    __tablename__ = "mof_structures"

    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    common_name = Column(String)
    formula = Column(String)
    topology = Column(String)
    metal_node = Column(String)
    linker = Column(String)

    # Geometric properties (CoRE MOF 2019 / published crystallography)
    pore_limiting_diameter_A = Column(Float)    # PLD, Angstroms
    largest_cavity_diameter_A = Column(Float)   # LCD, Angstroms
    void_fraction = Column(Float)               # 0-1
    surface_area_m2_g = Column(Float)           # gravimetric BET m²/g
    surface_area_m2_cm3 = Column(Float)         # volumetric m²/cm³
    density_g_cm3 = Column(Float)

    # Chemical properties
    has_open_metal_sites = Column(Boolean, default=False)
    functional_groups = Column(String)          # comma-separated: "-OH,-NH2"
    water_stability = Column(String)            # "low" | "moderate" | "high"

    # Published adsorption data (null = not characterized for that gas)
    water_uptake_g_g_20rh = Column(Float)       # g/g at 20% RH, ~25°C
    water_uptake_g_g_40rh = Column(Float)       # g/g at 40% RH, ~25°C
    co2_uptake_mmol_g_015bar = Column(Float)    # mmol/g at 0.15 bar, 298 K
    co2_uptake_mmol_g_1bar = Column(Float)      # mmol/g at 1 bar, 298 K
    co2_n2_selectivity = Column(Float)          # IAST CO2/N2 selectivity
    ch4_uptake_cm3_cm3_35bar = Column(Float)    # cm³(STP)/cm³ at 35 bar, 298 K
    ch4_uptake_cc_g_35bar = Column(Float)       # cm³(STP)/g at 35 bar, 298 K
    h2_uptake_wt_1bar_77k = Column(Float)       # wt% at 1 bar, 77 K
    h2_uptake_wt_100bar_77k = Column(Float)     # wt% at 100 bar, 77 K

    key_paper = Column(String)
    core_mof_id = Column(String)
    notes = Column(Text)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "common_name": self.common_name,
            "formula": self.formula,
            "topology": self.topology,
            "metal_node": self.metal_node,
            "linker": self.linker,
            "pore_limiting_diameter_A": self.pore_limiting_diameter_A,
            "largest_cavity_diameter_A": self.largest_cavity_diameter_A,
            "void_fraction": self.void_fraction,
            "surface_area_m2_g": self.surface_area_m2_g,
            "surface_area_m2_cm3": self.surface_area_m2_cm3,
            "density_g_cm3": self.density_g_cm3,
            "has_open_metal_sites": bool(self.has_open_metal_sites),
            "functional_groups": self.functional_groups or "",
            "water_stability": self.water_stability or "unknown",
            "water_uptake_g_g_20rh": self.water_uptake_g_g_20rh,
            "water_uptake_g_g_40rh": self.water_uptake_g_g_40rh,
            "co2_uptake_mmol_g_015bar": self.co2_uptake_mmol_g_015bar,
            "co2_uptake_mmol_g_1bar": self.co2_uptake_mmol_g_1bar,
            "co2_n2_selectivity": self.co2_n2_selectivity,
            "ch4_uptake_cm3_cm3_35bar": self.ch4_uptake_cm3_cm3_35bar,
            "ch4_uptake_cc_g_35bar": self.ch4_uptake_cc_g_35bar,
            "h2_uptake_wt_1bar_77k": self.h2_uptake_wt_1bar_77k,
            "h2_uptake_wt_100bar_77k": self.h2_uptake_wt_100bar_77k,
            "key_paper": self.key_paper,
            "core_mof_id": self.core_mof_id,
            "notes": self.notes,
        }


engine = create_engine(
    f"sqlite:///{DATABASE_URL}",
    connect_args={"check_same_thread": False},
)
SessionLocal = sessionmaker(bind=engine)


def init_db() -> None:
    Base.metadata.create_all(engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

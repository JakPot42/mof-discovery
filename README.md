# MOF Discovery — Metal-Organic Framework Query Tool

Ranking and discovery tool over 47 real Metal-Organic Framework structures from the CoRE MOF database — select a use case, get the top candidates ranked by published adsorption performance, and have Claude explain the structural chemistry behind why they work.

Built in the year the Nobel Prize in Chemistry went to Yaghi, Kitagawa, and Robson for developing MOFs.

**Live demo:** https://mof-discovery.onrender.com

---

## What It Does

Metal-Organic Frameworks are crystalline porous materials built from metal nodes and organic linkers — they hold world records for surface area (some exceed 7,000 m²/g) and have demonstrated applications in water harvesting, carbon capture, hydrogen storage, and air purification. The challenge for researchers is navigating a database of 500,000+ structures to find candidates for a specific application.

This tool makes that search accessible:

1. **Select** — choose a use case (water harvesting, CO₂ capture, H₂ storage, CH₄ storage, VOC removal)
2. **Rank** — deterministic engine scores all 47 structures using published adsorption data (primary) or structural proxies (fallback); every result row shows its data source
3. **Explain** — Claude reads the top candidates' structural properties and explains *why* each one excels for the chosen application — the same scientific reasoning task as cell type annotation in the scRNA Explorer
4. **Explore** — structure detail pages show pore geometry, surface area, metal node, organic linker, and the full performance data with literature citations

---

## The 5 Use Cases

| Use Case | Key Performance Metric | Best Seed Candidate |
|----------|----------------------|---------------------|
| Water harvesting (20% RH) | g water / g MOF | MOF-303: 0.48 g/g |
| CO₂ capture from flue gas | mmol/g at 0.15 bar | Mg-MOF-74: 3.9 mmol/g |
| H₂ storage (77 K) | wt% at target pressure | NU-100: 9.95 wt% at 56 bar |
| CH₄ / natural gas storage | cm³/cm³ | PCN-14: 195 cm³/cm³ |
| VOC removal / air purification | structural proxy score | Ranked by pore size + surface area |

Water harvesting is the flagship demo — MOF-801 harvesting water from desert air at 20% relative humidity was cited in the 2025 Nobel Prize announcement (Science, 2017, Yaghi group).

---

## Data Sources

Every performance number in the database cites a peer-reviewed publication:

- **CoRE MOF 2019** — Chung et al., J. Chem. Eng. Data, 2019
- **MOF-303 water harvesting** — Hanikel et al., Science, 2021
- **MOF-801** — Fathieh et al., Science Advances, 2018
- **Mg-MOF-74 CO₂** — Mason et al., J. Am. Chem. Soc., 2015
- **NU-100 H₂** — Farha et al., J. Am. Chem. Soc., 2010
- **PCN-14 CH₄** — Ma et al., J. Am. Chem. Soc., 2008
- **SIFSIX-3-Cu selectivity** — Shekhah et al., Nature Commun., 2014

The `data_source` badge in every result row distinguishes **published data** from **structural proxy** — structural proxies use surface area, pore size, hydrophilicity, and open metal site indicators when measured adsorption data is not available for a structure.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI + Python |
| Ranking | Deterministic scoring engine (`ranker.py`) — 5 scoring functions, one per use case |
| AI | Claude Haiku (structural chemistry explanation) |
| MOF catalog | 47 real structures seeded from CoRE MOF 2019 + published literature |
| Database | SQLite + SQLAlchemy 2.0 |
| Frontend | Jinja2 templates + vanilla CSS |
| Deploy | Render (DEMO_MODE=False — always calls Claude) |

---

## Quick Start

```bash
git clone https://github.com/JakPot42/mof-discovery.git
cd mof-discovery
cp .env.example .env          # add ANTHROPIC_API_KEY=sk-ant-...
python -m venv venv
venv\Scripts\pip install -r requirements.txt
venv\Scripts\uvicorn main:app --reload
```

Open http://localhost:8000

---

## Architecture

```
seed_data.py      47 MOF structures with per-entry literature citations; performance data and structural properties
ranker.py         5 scoring functions — one per use case; published data (primary) / structural proxy (fallback)
explainer.py      Claude Haiku: MOF structural properties → application-specific chemistry explanation
models.py         SQLAlchemy ORM (MOFStructure, UseCase, RankingResult)
main.py           FastAPI routes (use case selection, ranking results, structure detail), Jinja rendering
config.py         DEMO_MODE, model pin, proxy scoring weights
```

---

## Key Architecture Decisions

**Why a ranking engine instead of a search tool:**
MOF discovery is fundamentally a ranking problem — given a target application, which structures perform best? Text search can surface relevant structures but cannot rank them by performance. A scoring function grounded in published measurements does both.

**Why published data primary, structural proxy fallback:**
The 47 seed structures span the range from extensively characterized (MOF-5, HKUST-1) to lightly characterized (early CoRE MOF entries). Insisting on published data for every structure would halve the corpus; accepting proxies everywhere would mislead on structures where measured data exists. The `data_source` badge makes the distinction transparent — users know when they're looking at a measured result vs. a structural estimate.

**Why Claude explains rather than scores:**
Claude's role here is the same as in scRNA-seq — given the data, explain the structural mechanism. "MOF-303 has a high water uptake because its aluminum-based nodes form coordinatively unsaturated sites that hydrogen-bond with water molecules at low humidity." That reasoning requires chemistry knowledge. The ranking itself is fully deterministic.

**What was not built:**
GNN training for property prediction and RASPA molecular simulation — both are PhD-lab-scale tools with month-long learning curves. They would make the ranking better but the tool inaccessible. The structural proxy approach is a documented approximation with explicit uncertainty.

---

## Honest Limitations

- 47 structures is a curated subset of the CoRE MOF 2019 database (12,000+ entries) — optimized for demonstration breadth across all 5 use cases, not comprehensive coverage.
- Structural proxies for VOC removal are less validated than measured adsorption data for the other four use cases.
- Published performance numbers reflect specific experimental conditions (temperature, pressure, humidity) that may differ from real deployment scenarios.
- No CoRE MOF live API — data is bundled from published literature and does not update automatically.
- DEMO_MODE=False on Render — the app always calls Claude for explanations; pre-baked demo explanations are available for neodymium and cobalt queries under DEMO_MODE=True.

---

## Tests

```bash
venv\Scripts\python.exe -m pytest tests/ -v
# 65 passed
```

Covers: ranking function output (correct ordering for each use case), data_source badge assignment, proxy vs. published data selection logic, Claude explanation parsing, seed data integrity (all 47 structures load), FastAPI route responses.

---

*MOF data from CoRE MOF 2019 and peer-reviewed literature. Claude explanations are for educational purposes. Not a substitute for experimental validation.*

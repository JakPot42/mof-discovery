"""Claude-powered plain-language explanation of ranked MOF results."""
from __future__ import annotations
import anthropic
from config import DEMO_MODE, ANTHROPIC_API_KEY, CLAUDE_MODEL
from ranker import UseCase, USE_CASE_LABELS

_SYSTEM_PROMPT = """\
You are a computational materials science assistant specializing in metal-organic frameworks
(MOFs). You have been given a ranked list of real MOF structures from the CoRE MOF database,
selected by a deterministic scoring algorithm for a specific application.

Your task:
- Explain in plain language WHY the top-ranked structures are well-suited for the stated
  application. Focus on structural mechanisms: pore size relative to the target molecule,
  metal node chemistry (open metal sites, charge density), linker hydrophilicity/hydrophobicity,
  void fraction and its role in capacity.
- The 2025 Nobel Prize in Chemistry was awarded to Omar Yaghi, Susumu Kitagawa, and Richard
  Robson for developing metal-organic frameworks. Reference this where it genuinely strengthens
  the explanation — especially for water harvesting and MOF-5 lineage.
- When published experimental values are cited in the context, reference them specifically
  (e.g., "0.22 g/g water uptake at 20% RH"). When scores are based on structural proxies,
  say so — do not overstate confidence.
- Be honest about limitations: if the top MOFs have water stability concerns, say so.
- 2–4 paragraphs. No bullet points. Plain language — assume the reader is intelligent but
  not a chemist. Focus on the WHY, not just the WHAT.
- Do NOT speculate beyond the provided data."""

_DEMO_EXPLANATIONS: dict[UseCase, str] = {
    UseCase.WATER_HARVESTING: (
        "The top candidates for atmospheric water harvesting share a combination that "
        "the 2025 Nobel laureates identified as essential: pores sized just right for "
        "cooperative water adsorption, hydrophilic surfaces that attract water at low "
        "relative humidity, and exceptional chemical stability that lets the material "
        "cycle thousands of times without degrading. MOF-303 and CAU-23, both aluminum-based, "
        "achieve water uptake of 0.46–0.48 g per gram of MOF at just 20% relative humidity "
        "— the kind of dry air found in the Mojave Desert or the Atacama. Their aluminum-oxygen "
        "chains create polar surface chemistry that mimics desert fog-collection strategies "
        "at the molecular scale.\n\n"
        "The original benchmark, MOF-801, developed by Omar Yaghi's group at Berkeley, "
        "uses zirconium-fumarate nodes to achieve similar humidity targeting. The Science 2017 "
        "paper demonstrating MOF-801 harvesting drinking water from desert air using only "
        "sunlight was a direct precursor to Yaghi's 2025 Nobel recognition. Its pore-limiting "
        "diameter of 5.5 Angstroms — just wide enough for water molecules (2.8 Angstroms) "
        "to enter in clusters — creates the cooperative binding that drives the steep S-shaped "
        "isotherm essential for efficient water release during the regeneration half-cycle.\n\n"
        "Materials like MIL-100 and MIL-101 appear lower in the ranking because their step "
        "uptake occurs at 35–45% relative humidity rather than 20%, making them better suited "
        "for coastal or moderate-humidity harvesting rather than true desert conditions. "
        "Water stability is non-negotiable for real deployment: zinc-based MOFs like MOF-5, "
        "despite their impressive surface areas, dissolve in humid air and do not appear in "
        "the top results for this reason."
    ),
    UseCase.CO2_CAPTURE: (
        "The highest-ranked CO2 capture candidates exploit two distinct mechanisms. "
        "Mg-MOF-74, Ni-MOF-74, and Co-MOF-74 — the MOF-74 series — achieve extraordinary "
        "CO2 uptake at low partial pressures through open metal sites: exposed Mg²⁺, Ni²⁺, "
        "and Co²⁺ ions that coordinate directly to CO2's lone pairs with binding energies "
        "comparable to weak chemisorption. At flue gas partial pressures (0.15 bar CO2), "
        "Mg-MOF-74 adsorbs 3.9 mmol per gram — several times more than conventional "
        "zeolites — while maintaining CO2/N2 selectivity above 100.\n\n"
        "The SIFSIX materials represent a different design strategy: ultrasmall pores "
        "(3.5–3.8 Angstroms) sized almost exactly for CO2's kinetic diameter (3.3 Angstroms) "
        "rather than nitrogen's (3.6 Angstroms). This size-exclusion mechanism gives SIFSIX-3-Cu "
        "a CO2/N2 selectivity exceeding 10,000 at trace concentrations — making it relevant "
        "for direct air capture even though its total uptake per gram is modest. The fluorinated "
        "SiF6 pillars create an electrostatic environment that further stabilizes bound CO2.\n\n"
        "One critical limitation of the top performers: Mg-MOF-74 degrades in liquid water "
        "and requires dry flue gas or moisture pre-removal. For applications where humidity "
        "cannot be controlled, amine-functionalized materials like UiO-66-NH2 and "
        "MIL-101(Cr)-NH2 trade some capacity for much better moisture tolerance, making "
        "them more practical for real power-plant deployments."
    ),
    UseCase.H2_STORAGE: (
        "Hydrogen storage in MOFs at 77 K (liquid nitrogen temperature) is almost entirely "
        "governed by surface area: H2 physisorbs weakly to pore walls, and more surface area "
        "means more binding sites. NU-100, developed by Omar Farha and Joseph Hupp at "
        "Northwestern, holds the gravimetric record at high pressure with 9.95 wt% at 56 bar, "
        "powered by its exceptional 6,100 m²/g surface area — roughly 1.5 tennis courts per "
        "gram of material. Even at 1 bar, it achieves 2.2 wt%, placing it well above "
        "conventional porous carbons.\n\n"
        "MOF-5 (IRMOF-1), the landmark 1999 material from Yaghi's group that launched the "
        "high-surface-area MOF era, stores 1.3 wt% at 1 bar/77 K — meaningful for research "
        "benchmarking but below US DoE onboard hydrogen targets for vehicles. The IRMOF series "
        "established that surface area scales with linker length (longer organic pillars create "
        "bigger pores), a design rule that later ultrahigh-SA MOFs like MOF-177, PCN-68, and "
        "MOF-210 exploited systematically.\n\n"
        "The practical limitation is temperature: at room temperature, physisorptive binding "
        "energy (~5–8 kJ/mol for most MOFs) is too weak to hold H2, reducing uptake by "
        "95% or more. Open metal sites help slightly, but the field has not yet found a "
        "MOF that stores useful H2 quantities at ambient conditions without chemical bonding "
        "that makes release difficult. Cryogenic storage (77 K) remains the realistic use case "
        "for MOF-based H2 systems in the near term."
    ),
    UseCase.CH4_STORAGE: (
        "Natural gas vehicle storage is where volumetric capacity matters more than gravimetric: "
        "a tank has fixed volume, so cm³ of gas per cm³ of material is the key metric. "
        "The DoE's 2012 target of 263 cm³(STP)/cm³ at 65 bar represents roughly the energy "
        "density of a compressed natural gas tank at 250 bar, without the extreme compression "
        "infrastructure. PCN-14, the first MOF to approach this target (230 cm³/cm³ at 290 K), "
        "achieves its performance through a combination of open Cu²⁺ paddle-wheel sites and an "
        "anthracene linker that provides high CH4 binding enthalpy through hydrophobic and "
        "van der Waals interactions.\n\n"
        "HKUST-1 (Cu-BTC), one of the most widely studied MOFs and commercially available "
        "from BASF, provides 180 cm³/cm³ at 35 bar through similar open copper chemistry. "
        "Its cage topology creates a hierarchy of pore sizes: small pore windows (9 Å) connect "
        "larger cavities (13 Å), and the interior Cu²⁺ sites interact directly with methane "
        "at the cage walls. The tradeoff is water sensitivity: copper paddle-wheel nodes "
        "decompose in humid air, requiring dry gas streams.\n\n"
        "MOF-519, an aluminum-based alternative from Jeffrey Long's group, approaches the DoE "
        "target (259 cm³/cm³ at 65 bar) while offering significantly better water stability "
        "than Cu-based competitors. Aluminum-carboxylate bonds are much less moisture-reactive, "
        "making Al-based MOFs like MOF-519 and Al-fumarate increasingly attractive for "
        "practical deployment where gas drying adds cost and complexity."
    ),
    UseCase.VOC_REMOVAL: (
        "VOC removal from air streams depends on matching pore chemistry to the target "
        "pollutant. For aromatic VOCs — benzene (kinetic diameter 5.9 Å), toluene (6.1 Å), "
        "xylene (6.8 Å) — the ideal MOF pore is slightly larger than the molecule to allow "
        "entry, with hydrophobic surface chemistry that favors the nonpolar aromatic over "
        "competing water molecules. High surface area maximizes total loading capacity before "
        "breakthrough.\n\n"
        "MIL-101(Cr) and MIL-101(Al) appear at the top despite their large pores (12–13 Å "
        "PLD) because their enormous surface areas (3,100–3,800 m²/g) and mesoporous cages "
        "create high absolute VOC capacity. The Cr version is the benchmark material for "
        "benzene and toluene capture in the literature; the Al version is preferred for "
        "real deployment because chromium's toxicity creates disposal concerns. IRMOF-8's "
        "naphthalene linker enhances pi-pi stacking interactions with aromatic VOCs, "
        "increasing uptake affinity beyond what surface area alone predicts.\n\n"
        "Note that VOC removal rankings in this tool rely on structural proxies — "
        "surface area, pore geometry, and hydrophobicity estimates — rather than published "
        "experimental adsorption isotherms for specific VOC molecules. For precise "
        "application design, experimental breakthrough curve measurements for the specific "
        "target compound at operational humidity are essential."
    ),
}


def explain(
    use_case: UseCase,
    context: str,
    top_mofs: list[dict],
    client: anthropic.Anthropic | None = None,
) -> str:
    """Return a plain-language explanation of the top MOF ranking results."""
    if DEMO_MODE:
        return _DEMO_EXPLANATIONS.get(use_case, "Demo explanation not available.")

    if client is None:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    # Build context string for the top results
    context_lines = [
        f"Application: {USE_CASE_LABELS[use_case]}",
        f"User context: {context or 'none provided'}",
        "",
        "Top-ranked MOFs (score out of 100):",
    ]
    for i, mof in enumerate(top_mofs[:6], 1):
        line = (
            f"  {i}. {mof['name']} ({mof.get('common_name', '')}) — "
            f"score {mof['score']:.1f}, "
            f"data: {mof['data_source']}, "
            f"metal: {mof.get('metal_node', '?')}, "
            f"SA: {mof.get('surface_area_m2_g', '?')} m2/g, "
            f"PLD: {mof.get('pore_limiting_diameter_A', '?')} A, "
            f"void: {mof.get('void_fraction', '?')}, "
            f"OMS: {mof.get('has_open_metal_sites', False)}, "
            f"water_stability: {mof.get('water_stability', '?')}"
        )
        # Attach the most relevant published value
        if use_case == UseCase.WATER_HARVESTING and mof.get("water_uptake_g_g_20rh"):
            line += f", water_uptake_20rh: {mof['water_uptake_g_g_20rh']} g/g"
        elif use_case == UseCase.CO2_CAPTURE and mof.get("co2_uptake_mmol_g_015bar"):
            line += f", co2_uptake: {mof['co2_uptake_mmol_g_015bar']} mmol/g@0.15bar"
        elif use_case == UseCase.H2_STORAGE and mof.get("h2_uptake_wt_1bar_77k"):
            line += f", h2_uptake: {mof['h2_uptake_wt_1bar_77k']} wt%@1bar/77K"
        elif use_case == UseCase.CH4_STORAGE and mof.get("ch4_uptake_cm3_cm3_35bar"):
            line += f", ch4_uptake: {mof['ch4_uptake_cm3_cm3_35bar']} cm3/cm3"
        context_lines.append(line)

    full_context = "\n".join(context_lines)

    try:
        msg = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=700,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": full_context}],
        )
        return msg.content[0].text
    except Exception as exc:
        return f"[Explanation unavailable: {exc}]"

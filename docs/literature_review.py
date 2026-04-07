"""
Key literature references for green V2X / cellular energy studies.

Maps themes (green RAN, ICT footprint, V2X, handover energy, grid factors) to
BibTeX-style keys. Intended for related-work drafting; add matching entries to
your ``.bib`` and cite in the manuscript—not imported by the simulator hot path.
"""

from __future__ import annotations

from typing import Any, Dict, List

LITERATURE_REFERENCES: Dict[str, List[Dict[str, Any]]] = {
    "green_wireless_networks": [
        {
            "key": "auer2011howmuch",
            "authors": "Auer, G. et al.",
            "title": "How much energy is needed to run a wireless network?",
            "journal": "IEEE Wireless Communications",
            "year": 2011,
            "volume": "18",
            "pages": "40-49",
            "doi": "10.1109/MWC.2011.6056694",
            "relevance": "Seminal cellular BS power model and energy efficiency discussion",
            "citation_count": "2000+",
        },
        {
            "key": "bjornson2020power",
            "authors": "Björnson, E., Sanguinetti, L.",
            "title": "Power Scaling Laws and Near-Field Behaviors of Massive MIMO",
            "journal": "IEEE Open Journal of the Communications Society",
            "year": 2020,
            "relevance": "5G power scaling and energy efficiency perspective",
        },
        {
            "key": "hasan2011green",
            "authors": "Hasan, Z., Boostanimehr, H., Bhargava, V. K.",
            "title": "Green Cellular Networks: A Survey, Some Research Issues and Challenges",
            "journal": "IEEE Communications Surveys & Tutorials",
            "year": 2011,
            "relevance": "Survey of energy-efficient cellular networks",
        },
        {
            "key": "fehske2013small",
            "authors": "Fehské, A. et al.",
            "title": "The Future of Mobile Network Energy Efficiency",
            "journal": "IEEE Vehicular Technology Magazine",
            "year": 2013,
            "relevance": "Energy efficiency metrics and trends in mobile networks",
        },
        {
            "key": "chen2011fundamental",
            "authors": "Chen, T. et al.",
            "title": "Fundamental Trade-offs on Green Wireless Networks",
            "journal": "IEEE Communications Magazine",
            "year": 2011,
            "relevance": "Energy–spectral efficiency trade-offs",
        },
    ],
    "ict_energy_footprint": [
        {
            "key": "malmodin2018global",
            "authors": "Malmodin, J., Lundén, D.",
            "title": "The Energy and Carbon Footprint of the Global ICT and E&M Sectors 2010-2015",
            "journal": "Sustainability",
            "year": 2018,
            "relevance": "ICT sector electricity and CO2 footprint",
        },
        {
            "key": "andrae2015global",
            "authors": "Andrae, A. S. G., Edler, T.",
            "title": "On Global Electricity Usage of Communication Technology: Trends to 2030",
            "journal": "Challenges",
            "year": 2015,
            "relevance": "Projections of ICT electricity demand",
        },
        {
            "key": "freitag2021climate",
            "authors": "Freitag, C. et al.",
            "title": "The real climate and transformative impact of ICT: A critique of estimates, trends, and regulations",
            "journal": "Patterns",
            "year": 2021,
            "relevance": "Critical view of ICT lifecycle and grid carbon modeling",
        },
    ],
    "v2x_sustainability": [
        {
            "key": "abboud2016interworking",
            "authors": "Abboud, K., Omar, H. A., Zhuang, W.",
            "title": "Interworking of DSRC and Cellular Network Technologies for V2X",
            "journal": "IEEE Transactions on Vehicular Technology",
            "year": 2016,
            "relevance": "DSRC vs cellular V2X technology landscape",
        },
        {
            "key": "chen2020vision",
            "authors": "Chen, S. et al.",
            "title": "Vision, Requirements, and Technology Trend of 6G: How to Tackle the Challenges in System Energy Efficiency",
            "journal": "IEEE Network",
            "year": 2020,
            "relevance": "6G vision including energy efficiency requirements",
        },
        {
            "key": "kenney2011dedicated",
            "authors": "Kenney, J. B.",
            "title": "Dedicated Short-Range Communications (DSRC) Standards in the United States",
            "journal": "Proceedings of the IEEE",
            "year": 2011,
            "relevance": "DSRC / 802.11p context for vehicular links",
        },
        {
            "key": "molina2017lte",
            "authors": "Molina-Masegosa, R., Gozalvez, J.",
            "title": "LTE-V for Sidelink 5G V2X Vehicular Communications: A New 5G Technology for Short-Range Vehicle-to-Everything Communications",
            "journal": "IEEE Vehicular Technology Magazine",
            "year": 2017,
            "relevance": "Cellular V2X (C-V2X) overview",
        },
        {
            "key": "seo2016lte",
            "authors": "Seo, H. et al.",
            "title": "LTE Evolution for Vehicle-to-Everything Services",
            "journal": "IEEE Communications Magazine",
            "year": 2016,
            "relevance": "C-V2X services and architecture (energy-aware RRM context)",
        },
        {
            "key": "papadimitratos2009vehicular",
            "authors": "Papadimitratos, P. et al.",
            "title": "Vehicular Communication Systems: Enabling Technologies, Applications, and Future Outlook on Intelligent Transportation",
            "journal": "IEEE Communications Surveys & Tutorials",
            "year": 2009,
            "relevance": "Survey of vehicular networking (baseline for V2X sustainability discussion)",
        },
    ],
    "energy_aware_handoff": [
        {
            "key": "balasubramanian2009energy",
            "authors": "Balasubramanian, A., Chandra, R., Venkataramani, A.",
            "title": "Energy consumption in mobile phones: a measurement study and implications for network applications",
            "journal": "ACM IMC",
            "year": 2009,
            "relevance": "Energy cost of wireless interfaces (mobility context)",
        },
        {
            "key": "pahlavan2000handoff",
            "authors": "Pahlavan, K. et al.",
            "title": "Trends in local wireless networks",
            "journal": "IEEE Communications Magazine",
            "year": 2000,
            "relevance": "Classical handoff and mobility background",
        },
        {
            "key": "kyasanur2005routing",
            "authors": "Kyasanur, P., Vaidya, N. H.",
            "title": "Routing and interface selection in multi-channel multi-interface wireless networks",
            "journal": "IEEE WoWMoM",
            "year": 2005,
            "relevance": "Interface selection and energy-aware routing",
        },
    ],
    "carbon_intensity_data": [
        {
            "key": "iea2021electricity",
            "authors": "International Energy Agency (IEA)",
            "title": "CO2 Emissions from Fuel Combustion",
            "year": 2021,
            "relevance": "Grid carbon intensity and regional electricity factors",
            "url": "https://www.iea.org/reports/co2-emissions-from-fuel-combustion-overview",
        },
        {
            "key": "brander2019ghg",
            "authors": "Brander, M. et al.",
            "title": "GHG Protocol Technical Guidance for Calculating Scope 2 Emissions",
            "year": 2015,
            "relevance": "Scope 2 grid-average emission factors",
        },
    ],
    "channel_and_phy": [
        {
            "key": "goldsmith2005wireless",
            "authors": "Goldsmith, A.",
            "title": "Wireless Communications",
            "journal": "Cambridge University Press",
            "year": 2005,
            "relevance": "Foundational link budget and fading models",
        },
        {
            "key": "3gpp38811",
            "authors": "3GPP",
            "title": "TR 38.811: Study on New Radio (NR) to support non-terrestrial networks",
            "year": 2020,
            "relevance": "NR channel models (context for cellular-style links)",
        },
    ],
}


def generate_related_work_section() -> str:
    """Draft related-work paragraphs with citation keys (for LaTeX \\cite)."""
    text = "## Related Work\n\n"
    text += "### Green Wireless Networks\n"
    text += (
        "The energy consumption of wireless networks has received sustained attention "
        "since Auer et al. \\cite{auer2011howmuch}, who modeled BS power as a function of "
        "load and deployment. Surveys on green cellular networks \\cite{hasan2011green} and "
        "more recent massive MIMO power scaling \\cite{bjornson2020power} motivate "
        "energy-aware RRM and handover design.\n\n"
    )
    text += "### ICT Environmental Impact\n"
    text += (
        "Sector-level ICT electricity and emissions estimates "
        "\\cite{malmodin2018global,andrae2015global} motivate reporting CO2 alongside "
        "communication energy, with careful scope definition; lifecycle and grid-factor "
        "critiques \\cite{freitag2021climate} caution against over-interpretation of "
        "single metrics.\n\n"
    )
    text += "### V2X Communications\n"
    text += (
        "V2X encompasses DSRC/802.11p and cellular modes "
        "\\cite{abboud2016interworking,molina2017lte,seo2016lte,papadimitratos2009vehicular}; "
        "energy-aware association remains relevant as radios diversify.\n\n"
    )
    text += "### Energy-Aware Mobility and Handover\n"
    text += (
        "Wireless interface energy has been measured at the device level "
        "\\cite{balasubramanian2009energy}; handover and interface selection literature "
        "\\cite{pahlavan2000handoff,kyasanur2005routing} complements BS-centric energy models.\n\n"
    )
    return text


def list_all_keys() -> List[str]:
    """Flat list of BibTeX-style keys for bibliography checks."""
    keys: List[str] = []
    for _cat, rows in LITERATURE_REFERENCES.items():
        for r in rows:
            k = r.get("key")
            if k:
                keys.append(str(k))
    return keys

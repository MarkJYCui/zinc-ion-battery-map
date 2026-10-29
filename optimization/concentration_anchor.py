"""
Electrolyte-Aware Concentration Anchoring Engine for Zinc-Ion Batteries (ZIB).
Module 2: Autonomous Design of Experiments (DoE) & Robotic Dispensing.

Dynamically shifts additive anchor concentrations, establishes realistic exploration
windows, and enforces aqueous solubility ceilings (Delaney ESOL) when transferring
literature-reported additives to alternative electrolyte chemistries.
"""

import math
import logging
import re
import time
from functools import lru_cache
from typing import Optional, Dict, Any, List, Tuple

import requests
from rdkit import Chem
from rdkit.Chem import Descriptors


from models import (
    ConcentrationAnchor,
    ExtractedAdditive,
    CandidateMolecule,
    ExtractionResult,
)
from discovery.formula_utils import convert_concentration_to_mM

logger = logging.getLogger(__name__)

CANONICAL_ELECTROLYTE_PROFILES: Dict[str, Dict[str, Any]] = {
    'ZnSO4': {
        'salt_name': 'ZnSO4',
        'display_name': 'Zinc Sulfate (Standard Baseline)',
        'default_molarity': 2.0,
        'molarity_range': [1.0, 3.0],
        'default_stock_molarity': 3.5,
        'anion_type': 'sulfate',
        'description': 'Conventional baseline chemistry; prone to basic zinc sulfate (BZS) byproduct passivation.',
    },
    'Zn(OTf)2': {
        'salt_name': 'Zn(OTf)2',
        'display_name': 'Zinc Trifluoromethanesulfonate / Triflate',
        'default_molarity': 1.0,
        'molarity_range': [0.5, 2.5],
        'default_stock_molarity': 3.0,
        'anion_type': 'triflate',
        'description': 'Fluorinated interphase chemistry; bulky OTf⁻ anions decompose to form ZnF₂-rich protective SEI.',
    },
    'Zn(ClO4)2': {
        'salt_name': 'Zn(ClO4)2',
        'display_name': 'Zinc Perchlorate',
        'default_molarity': 2.0,
        'molarity_range': [1.0, 3.5],
        'default_stock_molarity': 4.0,
        'anion_type': 'perchlorate',
        'description': 'Weakly coordinating anion enabling ultrafast Zn²⁺ desolvation kinetics and high rate capability.',
    },
    'Zn(TFSI)2': {
        'salt_name': 'Zn(TFSI)2',
        'display_name': 'Zinc Bis(trifluoromethanesulfonyl)imide (WiSE)',
        'default_molarity': 4.0,
        'molarity_range': [2.0, 7.5],
        'default_stock_molarity': 8.0,
        'anion_type': 'tfsi',
        'description': 'Water-in-salt / high-concentration fluorinated electrolyte; wide electrochemical stability window.',
    },
    'ZnCl2': {
        'salt_name': 'ZnCl2',
        'display_name': 'Zinc Chloride (Concentrated / Complexation)',
        'default_molarity': 5.0,
        'molarity_range': [2.0, 10.0],
        'default_stock_molarity': 12.0,
        'anion_type': 'chloride',
        'description': 'High ionic strength, chlorozinc complexation ([ZnCl₄]²⁻); prone to EDL compaction and high viscosity.',
    },
    'Zn(CH3COO)2': {
        'salt_name': 'Zn(CH3COO)2',
        'display_name': 'Zinc Acetate (Buffer)',
        'default_molarity': 1.0,
        'molarity_range': [0.5, 2.5],
        'default_stock_molarity': 2.0,
        'anion_type': 'acetate',
        'description': 'Mild pH-buffered electrolyte (~5-6); suppresses HER while maintaining stable complexation equilibrium.',
    },
}

DEMAND_RATIOS: Dict[str, float] = {
    'ZnSO4': 1.0,
    'Zn(OTf)2': 0.7,
    'Zn(ClO4)2': 0.9,
    'Zn(TFSI)2': 0.6,
    'ZnCl2': 1.2,
    'Zn(CH3COO)2': 0.9,
}

ELECTROLYTE_SCALING_RULES: Dict[str, Dict[str, Any]] = {

    'polyol': {
        'Zn(OTf)2': {
            'factor': 0.7,
            'rationale': (
                'Synergistic OTf⁻ solvation participation: Bulky triflate anions reconstruct the inner Zn²⁺ '
                'solvation sheath and promote in situ ZnF₂ interphase formation, reducing the required polyol additive dosage by ~30%.'
            ),
        },
        'Zn(TFSI)2': {
            'trace_factor': 0.6,
            'cosolvent_factor': 0.8,
            'rationale_trace': (
                'Water-in-salt (WiSE) Zn(TFSI)₂ regime: Free water activity is severely constrained (~1.8 H₂O:Zn²⁺), '
                'drastically decreasing the quantity of bulk water molecules needing additive solvation coordination.'
            ),
            'rationale_cosolvent': (
                'Co-solvent WiSE regime: Formulates hybrid organic-aqueous solvate; moderate concentration '
                'reduction to maintain broad electrochemical stability window.'
            ),
        },
        'ZnCl2': {
            'factor': 1.2,
            'rationale': (
                'Rich chlorozinc complex anions ([ZnCl₄]²⁻) increase solution viscosity; moderate polyol dosage '
                'elevation promotes uniform fluidic transport and suppresses dendritic growth.'
            ),
        },
        'Zn(ClO4)2': {
            'factor': 0.9,
            'rationale': (
                'Weakly coordinating perchlorate anions induce ultra-fast desolvation kinetics; polyol regulates '
                'water activity without impeding charge-transfer dynamics.'
            ),
        },
        'Zn(CH3COO)2': {
            'factor': 0.9,
            'rationale': (
                'Acetate buffer stabilizes electrolyte pH (~5–6); polyol coordinates residual free water while '
                'respecting buffer equilibrium.'
            ),
        },
    },
    'quaternary_ammonium': {
        'ZnCl2': {
            'factor': 1.75,
            'rationale': (
                'High ionic strength in concentrated ZnCl₂ (5.0 M) drastically compacts the electrical double '
                'layer (EDL). A 1.75x multiplier is applied to maintain effective dynamic electrostatic shielding against dendrites.'
            ),
        },
        'Zn(TFSI)2': {
            'factor': 0.5,
            'rationale': (
                'Precipitation risk alert: Bulky organic quaternary ammonium cations readily form insoluble '
                'hydrophobic [NR₄][TFSI] ion pairs with fluorinated TFSI⁻ anions; enforcing strict solubility ceiling and conservative dosage.'
            ),
        },
        'Zn(OTf)2': {
            'factor': 0.85,
            'rationale': (
                'Synergistic electrostatic shielding and OTf⁻ decomposition to form a robust organic-fluoride hybrid protective layer.'
            ),
        },
        'Zn(ClO4)2': {
            'factor': 1.0,
            'rationale': 'Dynamic electrostatic shielding balances non-coordinating perchlorate rapid tip growth.',
        },
        'Zn(CH3COO)2': {
            'factor': 1.0,
            'rationale': 'Electrostatic shielding functions stably within acetate buffer matrix.',
        },
    },
    'thiocarbonyl': {
        'ZnCl2': {
            'factor': 1.5,
            'rationale': (
                'Competitive adsorption compensation: High nucleophilic Cl⁻ anion concentration competitively '
                'adsorbs on Zn (002) facets. A 1.5x additive multiplier is required to maintain preferential thiocarbonyl surface anchoring.'
            ),
        },
        'Zn(OTf)2': {
            'factor': 0.8,
            'rationale': (
                'Dual interfacial protection: Thiocarbonyl sulfur adsorption directs planar (002) plating while '
                'OTf⁻ reduction generates a thin fluorinated interphase.'
            ),
        },
        'Zn(TFSI)2': {
            'factor': 0.7,
            'rationale': 'Interfacial leveling in fluorinated WiSE; lower free water allows reduced dosage.',
        },
        'Zn(ClO4)2': {
            'factor': 1.0,
            'rationale': 'Thiocarbonyl moderates rapid deposition flux induced by non-coordinating ClO₄⁻ anions.',
        },
        'Zn(CH3COO)2': {
            'factor': 1.0,
            'rationale': 'Thiocarbonyl operates harmoniously with acetate buffer without altering surface coordination.',
        },
    },
    'amide_urea': {
        'ZnCl2': {
            'factor': 1.2,
            'rationale': (
                'Intercalates into chlorozinc speciation network; forms deep eutectic solvates that reduce viscosity '
                'and eliminate dendrite initiation.'
            ),
        },
        'Zn(TFSI)2': {
            'factor': 0.75,
            'rationale': (
                'Deep eutectic coordination with TFSI⁻ further suppresses free water activity, widening the cathodic '
                'electrochemical stability window.'
            ),
        },
        'Zn(OTf)2': {
            'factor': 0.85,
            'rationale': 'Amide carbonyl coordination works synergistically with OTf⁻ decomposition into a dense hybrid SEI.',
        },
        'Zn(ClO4)2': {
            'factor': 0.9,
            'rationale': 'Carbonyl dipole regulates electric field and coordinates Zn²⁺, balancing perchlorate fast desolvation.',
        },
        'Zn(CH3COO)2': {
            'factor': 1.0,
            'rationale': 'Amide co-exists stably with acetate buffer equilibrium.',
        },
    },
    'sulfoxide': {
        'ZnCl2': {
            'factor': 1.2,
            'rationale': 'Sulfoxide coordinates Zn²⁺ in chlorozinc network to reduce viscosity.',
        },
        'Zn(OTf)2': {
            'factor': 0.8,
            'rationale': 'Sulfoxide modifies solvation sheath synergistically with OTf⁻ anions.',
        },
        'Zn(TFSI)2': {
            'factor': 0.7,
            'rationale': 'Sulfoxide binds residual free water in WiSE electrolyte.',
        },
        'Zn(ClO4)2': {
            'factor': 1.0,
            'rationale': 'Standard solvation coordination alongside ClO₄⁻.',
        },
        'Zn(CH3COO)2': {
            'factor': 1.0,
            'rationale': 'Functions stably with acetate buffer.',
        },
    },
}

from discovery.similarity import (
    FG_PATTERNS,
    calculate_delaney_esol,
    is_esol_applicable,
    detect_functional_groups,
)


def parse_salt_string(electrolyte_str: str) -> Tuple[str, float]:
    """Parse electrolyte string into salt system and molarity."""
    if not electrolyte_str:
        return ('ZnSO4', 2.0)

    sub_map = str.maketrans("₀₁₂₃₄₅₆₇₈₉⁺⁻", "0123456789+-")
    s = electrolyte_str.strip()
    s_norm = s.translate(sub_map)
    s_upper = s_norm.upper()

    if 'ZNSO4' in s_upper or 'SULFATE' in s_upper:
        salt = 'ZnSO4'
    elif 'ZN(OTF)2' in s_upper or 'TRIFLATE' in s_upper or 'OTF' in s_upper:
        salt = 'Zn(OTf)2'
    elif 'ZN(CLO4)2' in s_upper or 'PERCHLORATE' in s_upper or 'CLO4' in s_upper:
        salt = 'Zn(ClO4)2'
    elif 'ZN(TFSI)2' in s_upper or 'TFSI' in s_upper:
        salt = 'Zn(TFSI)2'
    elif 'ZNCL2' in s_upper or 'CHLORIDE' in s_upper:
        salt = 'ZnCl2'
    elif 'ZN(CH3COO)2' in s_upper or 'ACETATE' in s_upper or 'OAC' in s_upper or 'ZN(OAC)2' in s_upper:
        salt = 'Zn(CH3COO)2'
    else:
        salt = 'ZnSO4'

    match = re.search(r'(\d+(?:\.\d+)?)\s*(?:M|molar)\b', s_norm, re.IGNORECASE)
    if match:
        molarity = float(match.group(1))
    else:
        molarity = CANONICAL_ELECTROLYTE_PROFILES[salt]['default_molarity']

    return (salt, molarity)


@lru_cache(maxsize=128)
def fetch_pubchem_data(additive_name: str, params: Any = None) -> Dict[str, Any]:
    """
    Fetch compound data from PubChem PUG-REST API with caching and exponential backoff.
    """
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{additive_name}/JSON"

    max_retries = 4
    base_delay = 1.0

    for attempt in range(max_retries):
        # Replace with the actual URL and params used in the original code
        response = requests.get(url, params=params, timeout=10)

        if response.status_code == 200:
            break
        elif response.status_code in (429, 502, 503, 504):
            if attempt == max_retries - 1:
                raise ConnectionError(f"PubChem API rate limit/server error persistent after {max_retries} retries.")
            time.sleep(base_delay * (2 ** attempt))  # Exponential backoff: 1s, 2s, 4s...
        else:
            response.raise_for_status()

    return response.json()


@lru_cache(maxsize=128)
def get_solubility(additive_name: str, params: Any = None) -> Dict[str, Any]:
    """Query compound solubility or properties from PubChem with LRU cache."""
    return fetch_pubchem_data(additive_name, params=params)


def calculate_target_concentration(
    parent_concentration: float,
    parent_salt: str,
    target_salt: str,
    functional_group_multiplier: float = 1.0,
) -> float:
    """
    Calculate target additive concentration based on relative cross-salt demand ratios.
    """
    # Extract demand constants (default to 1.0 for standard ZnSO4 baseline)
    parent_demand = DEMAND_RATIOS.get(parent_salt, 1.0)
    target_demand = DEMAND_RATIOS.get(target_salt, 1.0)

    # Calculate relative cross-salt scaling ratio
    relative_ratio = target_demand / parent_demand

    # Apply scaling and functional group multipliers
    target_concentration = parent_concentration * relative_ratio * functional_group_multiplier

    # Ensure we round to standard precision
    target_concentration = round(target_concentration, 3)
    return target_concentration


scale_additive = calculate_target_concentration
identify_functional_groups = detect_functional_groups


class ConcentrationAnchorEngine:
    """Dynamically calculates concentration anchors for alternative electrolyte chemistries."""

    def __init__(self):
        self.profiles = CANONICAL_ELECTROLYTE_PROFILES
        self.scaling_rules = ELECTROLYTE_SCALING_RULES

    def get_canonical_profiles(self) -> Dict[str, Any]:
        """Return canonical electrolyte profiles dictionary."""
        return self.profiles

    @staticmethod
    def fetch_pubchem_data(additive_name: str, params: Any = None) -> Dict[str, Any]:
        """Fetch compound data from PubChem with LRU cache and exponential backoff."""
        return fetch_pubchem_data(additive_name, params=params)

    @staticmethod
    def get_solubility(additive_name: str, params: Any = None) -> Dict[str, Any]:
        """Query compound solubility/properties with LRU cache and exponential backoff."""
        return get_solubility(additive_name, params=params)

    @staticmethod
    def calculate_target_concentration(
        parent_concentration: float,
        parent_salt: str,
        target_salt: str,
        functional_group_multiplier: float = 1.0,
    ) -> float:
        """Calculate target additive concentration based on relative cross-salt demand ratios."""
        return calculate_target_concentration(
            parent_concentration=parent_concentration,
            parent_salt=parent_salt,
            target_salt=target_salt,
            functional_group_multiplier=functional_group_multiplier,
        )

    @staticmethod
    def scale_additive(
        parent_concentration: float,
        parent_salt: str,
        target_salt: str,
        functional_group_multiplier: float = 1.0,
    ) -> float:
        """Alias for calculate_target_concentration."""
        return calculate_target_concentration(
            parent_concentration=parent_concentration,
            parent_salt=parent_salt,
            target_salt=target_salt,
            functional_group_multiplier=functional_group_multiplier,
        )



    def calculate_anchor(
        self,
        additive_name: str,
        published_conc_mM: float,
        parent_electrolyte: str = "2 M ZnSO4",
        target_electrolyte: str = "1 M Zn(OTf)2",
        smiles: Optional[str] = None,
        ratio_min: float = 0.2,
        ratio_max: float = 3.0,
        esol_logs_override: Optional[float] = None,
        is_baseline_unspecified: bool = False,
        reported_in_file: bool = True,
    ) -> ConcentrationAnchor:
        """Calculate electrolyte-adjusted concentration anchor and bounds."""
        parent_salt, parent_molarity = parse_salt_string(parent_electrolyte)
        target_salt, target_molarity = parse_salt_string(target_electrolyte)

        mol = None
        if smiles:
            try:
                mol = Chem.MolFromSmiles(smiles)
            except Exception as e:
                logger.warning(f"Failed to parse SMILES '{smiles}': {e}")

        functional_groups = identify_functional_groups(mol, name=additive_name)

        primary_fg = 'general'
        for candidate in ('polyol', 'quaternary_ammonium', 'thiocarbonyl', 'amide_urea', 'sulfoxide'):
            if candidate in functional_groups:
                primary_fg = candidate
                break

        def _get_salt_demand(salt: str, conc: float) -> Tuple[float, Optional[str]]:
            if salt == 'ZnSO4':
                return 1.0, None
            if primary_fg in self.scaling_rules and salt in self.scaling_rules[primary_fg]:
                rule = self.scaling_rules[primary_fg][salt]
                if 'factor' in rule:
                    return float(rule['factor']), rule.get('rationale')
                elif primary_fg == 'polyol' and salt == 'Zn(TFSI)2':
                    if conc < 100.0:
                        return float(rule['trace_factor']), rule.get('rationale_trace')
                    else:
                        return float(rule['cosolvent_factor']), rule.get('rationale_cosolvent')
            return DEMAND_RATIOS.get(salt, 1.0), None

        # Extract demand constants (default to 1.0 for standard ZnSO4 baseline)
        parent_demand, _ = _get_salt_demand(parent_salt, published_conc_mM)
        target_demand, r_target = _get_salt_demand(target_salt, published_conc_mM)

        # Calculate relative cross-salt scaling ratio
        relative_ratio = target_demand / parent_demand
        scaling_factor = round(relative_ratio, 4)
        rationale = ''

        if parent_salt == target_salt and abs(parent_molarity - target_molarity) < 0.1:
            scaling_factor = 1.0
            rationale = f"Target electrolyte matches published parent baseline ({parent_electrolyte}); anchor maintained at 1.0x baseline."
        elif parent_salt == 'ZnSO4':
            if r_target:
                rationale = r_target
            else:
                rationale = (
                    f"General functional group behavior: Transitioning from {parent_salt} to {target_salt}. "
                    "Anchor centered on literature baseline with standard exploration margins."
                )
        else:
            if target_salt == 'ZnSO4':
                if parent_salt == 'Zn(OTf)2':
                    rationale = (
                        f"Cross-salt transition from Zn(OTf)2 to baseline ZnSO4: Rescaled by {scaling_factor:.3f}x "
                        f"without Zn(OTf)2 co-ion assistance; higher additive concentration required in baseline "
                        f"ZnSO4 to effectively displace active water from the solvation sheath and suppress BZS passivation."
                    )
                elif parent_salt == 'ZnCl2':
                    rationale = (
                        f"Cross-salt transition from ZnCl2 to baseline ZnSO4: Rescaled by {scaling_factor:.3f}x; "
                        f"ZnSO4 lacks the intense competitive Cl- adsorption and severe EDL compaction of concentrated "
                        f"chloride electrolytes, allowing lower additive dosage."
                    )
                elif parent_salt == 'Zn(TFSI)2':
                    rationale = (
                        f"Cross-salt transition from Zn(TFSI)2 to baseline ZnSO4: Rescaled by {scaling_factor:.3f}x; "
                        f"baseline ZnSO4 features abundant free water requiring increased additive solvation participation."
                    )
                else:
                    rationale = (
                        f"Cross-salt transition from {parent_salt} to baseline ZnSO4: Rescaled anchor by {scaling_factor:.3f}x "
                        f"(demand ratio {target_demand:.2f}/{parent_demand:.2f}) to match baseline ZnSO4 solvation demands."
                    )
            else:
                rationale = (
                    f"Cross-salt transition from {parent_salt} to {target_salt}: Relative electrolyte demand ratio "
                    f"({target_demand:.2f} in {target_salt} vs {parent_demand:.2f} in {parent_salt}) scales the anchor "
                    f"concentration by {scaling_factor:.3f}x for {primary_fg} additives."
                )


        shifted_anchor_conc = round(published_conc_mM * scaling_factor, 4)

        solubility_limit_mM = None
        solubility_ceiling_mM = None
        is_capped = False

        if mol is not None:
            esol_ok, _ = is_esol_applicable(mol, esol_logs_override)
            if esol_ok:
                logs = esol_logs_override if esol_logs_override is not None else calculate_delaney_esol(mol)
                saturation_mM = round(math.pow(10, logs + 3.0), 3)
                solubility_limit_mM = saturation_mM
                ceiling_mM = round(0.8 * saturation_mM, 3)
                solubility_ceiling_mM = ceiling_mM
        elif smiles is None and esol_logs_override is not None and -10.0 < esol_logs_override <= 2.0:
            logs = esol_logs_override
            saturation_mM = round(math.pow(10, logs + 3.0), 3)
            solubility_limit_mM = saturation_mM
            ceiling_mM = round(0.8 * saturation_mM, 3)
            solubility_ceiling_mM = ceiling_mM

        min_conc = round(ratio_min * shifted_anchor_conc, 4)
        max_conc = round(ratio_max * shifted_anchor_conc, 4)

        if solubility_ceiling_mM is not None and (max_conc > solubility_ceiling_mM or shifted_anchor_conc > solubility_ceiling_mM):
            is_capped = True
            rationale += (
                f" Exploration upper bound capped at Delaney ESOL 80% saturation solubility ceiling "
                f"({solubility_ceiling_mM:.2f} mM) to avoid precipitation."
            )
            if shifted_anchor_conc > solubility_ceiling_mM:
                shifted_anchor_conc = solubility_ceiling_mM
            if max_conc > solubility_ceiling_mM:
                max_conc = solubility_ceiling_mM
            if min_conc > max_conc:
                min_conc = round(max_conc * (ratio_min / ratio_max), 4)

        if 'quaternary_ammonium' in functional_groups and target_salt == 'Zn(TFSI)2':
            rationale += " [CRITICAL ALERT: Quaternary ammonium cation paired with Zn(TFSI)2 risks hydrophobic [NR4][TFSI] ion-pair precipitation.]"

        return ConcentrationAnchor(
            additive_name=additive_name,
            smiles=smiles,
            published_concentration=published_conc_mM,
            published_unit='mM',
            parent_electrolyte=parent_electrolyte,
            parent_salt=parent_salt,
            parent_molarity=parent_molarity,
            target_electrolyte=target_electrolyte,
            target_electrolyte_salt=target_salt,
            target_salt_molarity=target_molarity,
            scaling_factor=scaling_factor,
            shifted_anchor_conc=shifted_anchor_conc,
            anchor_unit='mM',
            min_conc=min_conc,
            max_conc=max_conc,
            solubility_limit_mM=solubility_limit_mM,
            solubility_ceiling_mM=solubility_ceiling_mM,
            is_capped_by_solubility=is_capped,
            electrolyte_adjustment_rationale=rationale.strip(),
            functional_groups=functional_groups,
            is_baseline_unspecified=is_baseline_unspecified,
            reported_in_file=reported_in_file,
        )

    def anchor_from_extracted_additive(
        self,
        additive: ExtractedAdditive,
        target_electrolyte: str = "1 M Zn(OTf)2",
        fallback_conc_mM: float = 10.0,
        ratio_min: float = 0.2,
        ratio_max: float = 3.0,
    ) -> ConcentrationAnchor:
        """
        Direct Phase 1 -> Phase 2 handoff: calculate ConcentrationAnchor from an ExtractedAdditive.
        Extracts optimal published concentration, baseline electrolyte context, and SMILES.
        """
        conc_mM = fallback_conc_mM
        has_reported_conc = False
        mw_val = None
        if additive.smiles and Chem is not None:
            try:
                _mol = Chem.MolFromSmiles(additive.smiles)
                if _mol is not None:
                    mw_val = float(Descriptors.MolWt(_mol))
            except Exception:
                mw_val = None
        if additive.optimal_concentration and additive.optimal_concentration.value is not None:
            conv_val, _ = convert_concentration_to_mM(
                additive.optimal_concentration.value,
                additive.optimal_concentration.unit,
                mw=mw_val,
                raw_text=additive.optimal_concentration.raw_text,
            )
            conc_mM = float(conv_val) if conv_val is not None else float(additive.optimal_concentration.value)
            has_reported_conc = True
        elif additive.concentration_range_tested and len(additive.concentration_range_tested) > 0:
            c0 = additive.concentration_range_tested[0]
            conv_val, _ = convert_concentration_to_mM(
                c0.value,
                c0.unit,
                mw=mw_val,
                raw_text=c0.raw_text,
            )
            conc_mM = float(conv_val) if conv_val is not None else float(c0.value)
            has_reported_conc = True

        parent_electrolyte = "2 M ZnSO4"
        has_reported_salt = False
        if additive.battery_context and additive.battery_context.baseline_electrolyte:
            raw_salt = additive.battery_context.baseline_electrolyte
            if raw_salt and "unspecified" not in str(raw_salt).lower() and "not reported" not in str(raw_salt).lower():
                parent_electrolyte = raw_salt
                has_reported_salt = True

        is_unspec = not (has_reported_conc and has_reported_salt)

        return self.calculate_anchor(
            additive_name=additive.name,
            published_conc_mM=conc_mM,
            parent_electrolyte=parent_electrolyte,
            target_electrolyte=target_electrolyte,
            smiles=additive.smiles,
            ratio_min=ratio_min,
            ratio_max=ratio_max,
            is_baseline_unspecified=is_unspec,
            reported_in_file=not is_unspec,
        )

    def anchor_from_candidate_molecule(
        self,
        candidate: CandidateMolecule,
        target_electrolyte: str = "1 M Zn(OTf)2",
        parent_conc_mM: float = 10.0,
        parent_electrolyte: Optional[str] = None,
        ratio_min: float = 0.2,
        ratio_max: float = 3.0,
    ) -> ConcentrationAnchor:
        """
        Direct Phase 1 -> Phase 2 handoff: calculate ConcentrationAnchor for an AI-recommended CandidateMolecule.
        Uses candidate.logs (Delaney ESOL) as esol_logs_override for rigorous 80% saturation cap enforcement.
        """
        parent_elyte = parent_electrolyte or candidate.parent_baseline_electrolyte or "2 M ZnSO4"
        is_unspec = (
            candidate.reported_concentration is None or
            not candidate.parent_baseline_electrolyte or
            "unspecified" in str(candidate.parent_baseline_electrolyte).lower() or
            "not reported" in str(candidate.parent_baseline_electrolyte).lower()
        )
        return self.calculate_anchor(
            additive_name=candidate.name,
            published_conc_mM=parent_conc_mM,
            parent_electrolyte=parent_elyte,
            target_electrolyte=target_electrolyte,
            smiles=candidate.smiles,
            ratio_min=ratio_min,
            ratio_max=ratio_max,
            esol_logs_override=candidate.logs,
            is_baseline_unspecified=is_unspec,
            reported_in_file=not is_unspec,
        )

    def anchor_from_extraction_result(
        self,
        extraction: ExtractionResult,
        target_electrolyte: str = "1 M Zn(OTf)2",
        additive_index: int = 0,
        candidate_index: Optional[int] = None,
        ratio_min: float = 0.2,
        ratio_max: float = 3.0,
    ) -> ConcentrationAnchor:
        """
        Complete Phase 1 -> Phase 2 handoff: calculate ConcentrationAnchor from an ExtractionResult object.
        If candidate_index is specified, anchors for the AI-discovered candidate molecule, inheriting
        the literature parent additive's published concentration and baseline electrolyte context.
        """
        if not extraction.additives:
            raise ValueError("ExtractionResult contains no extracted additives.")

        parent_add = extraction.additives[additive_index]
        parent_conc = 10.0
        if parent_add.optimal_concentration and parent_add.optimal_concentration.value is not None:
            mw_val = None
            if parent_add.smiles and Chem is not None:
                try:
                    _mol = Chem.MolFromSmiles(parent_add.smiles)
                    if _mol is not None:
                        mw_val = float(Descriptors.MolWt(_mol))
                except Exception:
                    mw_val = None
            conv_val, _ = convert_concentration_to_mM(
                parent_add.optimal_concentration.value,
                parent_add.optimal_concentration.unit,
                mw=mw_val,
                raw_text=parent_add.optimal_concentration.raw_text,
            )
            parent_conc = float(conv_val) if conv_val is not None else float(parent_add.optimal_concentration.value)
        parent_elyte = parent_add.battery_context.baseline_electrolyte if (parent_add.battery_context and parent_add.battery_context.baseline_electrolyte) else "2 M ZnSO4"

        if candidate_index is not None:
            if candidate_index < 0 or candidate_index >= len(extraction.candidate_molecules):
                raise IndexError(f"candidate_index {candidate_index} out of range (found {len(extraction.candidate_molecules)} candidates).")
            cand = extraction.candidate_molecules[candidate_index]
            return self.anchor_from_candidate_molecule(
                candidate=cand,
                target_electrolyte=target_electrolyte,
                parent_conc_mM=parent_conc,
                parent_electrolyte=parent_elyte,
                ratio_min=ratio_min,
                ratio_max=ratio_max,
            )

        return self.anchor_from_extracted_additive(
            additive=parent_add,
            target_electrolyte=target_electrolyte,
            fallback_conc_mM=parent_conc,
            ratio_min=ratio_min,
            ratio_max=ratio_max,
        )

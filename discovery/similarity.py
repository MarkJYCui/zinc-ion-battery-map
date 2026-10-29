# ----------------------------------------------------------------------
# Portions of this file incorporate ideas, algorithms, or code adapted
# from ChemCrow (https://github.com/ur-whitelab/chemcrow-public)
# Copyright (c) ChemCrow Contributors. Licensed under the MIT License.
# See third_party_licenses/ChemCrow-LICENSE.txt for the full license text.
# ----------------------------------------------------------------------

"""
RDKit Similarity & Electrolyte Compatibility Reasoner.
Module 1: Materials Acceleration Platform (MAP) for Zinc-Ion Batteries (ZIB).

Finds structurally similar candidate molecules using Morgan fingerprints and Tanimoto similarity.
Evaluates estimated aqueous solubility limits (ESOL) for automated liquid handling.
Provides multi-electrolyte electrochemical compatibility reasoning across 6 major aqueous ZIB salts:
1. ZnSO4 (standard baseline)
2. Zn(OTf)2 (fluorinated interphase)
3. Zn(ClO4)2 (weak coordination / fast kinetics)
4. Zn(TFSI)2 (water-in-salt / wide potential window)
5. ZnCl2 (concentrated / complexation)
6. Zn(CH3COO)2 (acetate buffer)
"""

import logging
from typing import List, Optional, Dict, Any, Tuple
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, rdMolDescriptors
from rdkit.DataStructs import TanimotoSimilarity
from models import CandidateMolecule, ElectrolyteCompatibilityDetail

try:
    from discovery.pubchem_client import PubChemClient
except ImportError:
    try:
        from pubchem_client import PubChemClient
    except ImportError:
        PubChemClient = None

logger = logging.getLogger("RDKitSimilarityEngine")

# Curated reference library of validated ZIB additives for SDL candidate exploration
DEFAULT_POOL = [
    {"name": "D-Sorbitol", "smiles": "O[C@H]([C@@H](O)[C@H](O)[C@H](O)CO)CO"},
    {"name": "D-Mannitol", "smiles": "O[C@@H]([C@@H](O)[C@H](O)[C@H](O)CO)CO"}, 
    {"name": "Glucose", "smiles": "C(C1C(C(C(C(O1)O)O)O)O)O"},
    {"name": "Polyacrylamide", "smiles": "CC(C(=O)N)C"}, 
    {"name": "Polyacrylamide (PAM)", "smiles": "CC(C(=O)N)C"},
    {"name": "Polyethylene Glycol", "smiles": "OCCOCCO"},
    {"name": "Polyethylene Glycol (PEG)", "smiles": "OCCOCCO"},
    {"name": "PEG", "smiles": "OCCOCCO"}, 
    {"name": "Thiourea", "smiles": "C(=S)(N)N"},
    {"name": "Urea", "smiles": "C(=O)(N)N"},
    {"name": "Ethylene Glycol", "smiles": "OCCO"},
    {"name": "Glycerol", "smiles": "OCC(O)CO"},
    {"name": "Ascorbic Acid", "smiles": "C(C(C1C(=C(C(=O)O1)O)O)O)O"},
    {"name": "Tetramethylammonium bromide", "smiles": "C[N+](C)(C)C.[Br-]"},
    {"name": "Tetrapropylammonium bromide", "smiles": "CCC[N+](CCC)(CCC)CCC.[Br-]"},
    {"name": "Tetrabutylammonium bromide", "smiles": "CCCC[N+](CCCC)(CCCC)CCCC.[Br-]"},
    {"name": "Trehalose", "smiles": "C(C1C(C(C(C(O1)OC2C(C(C(C(O2)CO)O)O)O)O)O)O)O"},
    {"name": "Tetraethylammonium bromide", "smiles": "CC[N+](CC)(CC)CC.[Br-]"},
    {"name": "Choline chloride", "smiles": "C[N+](C)(C)CCO.[Cl-]"},
    {"name": "Dimethyl sulfoxide", "smiles": "CS(=O)C"},
    {"name": "Disodium EDTA", "smiles": "C(CN(CC(=O)O)CC(=O)[O-])N(CC(=O)O)CC(=O)[O-].[Na+].[Na+]"},
    {"name": "Copper disodium ethylenediaminetetraacetate", "smiles": "[Na+].[Na+].[Cu+2].[O-]C(=O)CN(CCN(CC(=O)[O-])CC(=O)[O-])CC(=O)[O-]"},
    {"name": "N,N′-dimethylpropyleneurea", "smiles": "CN1CCCN(C)C1=O"},
    {"name": "1,3-Dimethyl-2-imidazolidinone", "smiles": "CN1CCN(C)C1=O"},
    {"name": "Copper(II) sulfate", "smiles": "[Cu+2].[O-]S(=O)(=O)[O-]"},
    {"name": "Manganese(II) sulfate", "smiles": "[Mn+2].[O-]S(=O)(=O)[O-]"},
    {"name": "Sodium citrate", "smiles": "[Na+].[Na+].[Na+].C(C(=O)[O-])C(CC(=O)[O-])(C(=O)[O-])O"},
    {"name": "Citric Acid", "smiles": "C(C(=O)O)C(CC(=O)O)(C(=O)O)O"},
    {"name": "Succinic Acid", "smiles": "C(CC(=O)O)C(=O)O"},
    {"name": "Cetyltrimethylammonium bromide", "smiles": "CCCCCCCCCCCCCCCC[N+](C)(C)C.[Br-]"},
    {"name": "Sodium dodecyl sulfate", "smiles": "CCCCCCCCCCCCOS(=O)(=O)[O-].[Na+]"},
    {"name": "Betaine", "smiles": "C[N+](C)(C)CC(=O)[O-]"},
]

CURATED_ZIB_ADDITIVES = DEFAULT_POOL

SUPPORTED_SALTS = [
    "ZnSO4",
    "Zn(OTf)2",
    "Zn(ClO4)2",
    "Zn(TFSI)2",
    "ZnCl2",
    "Zn(CH3COO)2",
]

# RDKit SMARTS patterns for key electrochemical functional groups
FG_PATTERNS = {
    "polyol": Chem.MolFromSmarts("[OX2H]"),
    "urea": Chem.MolFromSmarts("[NX3][CX3](=[OX1])[NX3]"),
    "amide": Chem.MolFromSmarts("[CX3](=[OX1])[NX3]"),
    "thiocarbonyl": Chem.MolFromSmarts("[CX3]=[SX1]"),
    "thioamide": Chem.MolFromSmarts("[NX3][CX3](=[SX1])"),
    "quaternary_ammonium": Chem.MolFromSmarts("[NX4+]"),
    "sulfoxide": Chem.MolFromSmarts("[$([#6][SX3](=[OX1])[#6]),$([#6][S+;X3][O-])]"),
    "ether": Chem.MolFromSmarts("[OD2]([#6])[#6]"),
    "carboxylate": Chem.MolFromSmarts("[CX3](=O)[OX2H1,O-]"),
}


def calculate_esol(logp: float, mw: float, n_rot: int, aromatic_prop: float) -> float:
    """
    Calculate estimated aqueous solubility (LogS) using the Delaney equation.
    Includes boundary clipping to prevent math overflows with extreme/invalid molecules.
    """
    # 1. Clip inputs to realistic physical boundaries
    clamped_logp = max(min(logp, 50.0), -50.0)
    clamped_mw = max(min(mw, 5000.0), 1.0)
    
    # 2. Standard Delaney Equation coefficients
    c_intercept = 0.16
    c_logp = -0.63
    c_mw = -0.0062
    c_rot = 0.066
    c_ap = -0.74

    try:
        log_s = (c_intercept 
                 + (c_logp * clamped_logp) 
                 + (c_mw * clamped_mw) 
                 + (c_rot * n_rot) 
                 + (c_ap * aromatic_prop))
        
        # 3. Clamp final LogS output to prevent math.exp() overflows downstream
        # Limits correspond roughly to between 10^20 and 10^-20 mol/L
        return max(min(log_s, 20.0), -20.0)
        
    except (OverflowError, ValueError):
        # Safe fallback for catastrophic calculation failure
        return -20.0


compute_delaney_logs = calculate_esol


def get_aromatic_proportion(mol: Chem.Mol) -> float:
    """
    Safely calculate the proportion of heavy atoms that are aromatic.
    Prevents division-by-zero for molecules with 0 heavy atoms.
    """
    if mol is None:
        return 0.0
        
    num_heavy_atoms = mol.GetNumHeavyAtoms()
    if num_heavy_atoms == 0:
        return 0.0
        
    num_aromatic_atoms = sum(
        1 for atom in mol.GetAtoms() if atom.GetIsAromatic()
    )
    
    return num_aromatic_atoms / num_heavy_atoms


def calculate_delaney_esol(
    mol: Any,
    mw: Optional[float] = None,
    n_rot: Optional[int] = None,
    aromatic_prop: Optional[float] = None,
) -> float:
    """
    Calculate Estimated Solubility (ESOL) using the Delaney equation.
    Supports either passing an RDKit Mol object, a SMILES string, or direct descriptors (logp, mw, n_rot, aromatic_prop).
    """
    if mw is not None and n_rot is not None and aromatic_prop is not None:
        return calculate_esol(mol, mw, n_rot, aromatic_prop)
    try:
        if isinstance(mol, str):
            mol = Chem.MolFromSmiles(mol.strip())
            if mol is None:
                return -99.9
        clogp = Descriptors.MolLogP(mol)
        mw_val = Descriptors.MolWt(mol)
        rb = Descriptors.NumRotatableBonds(mol)
        ap = get_aromatic_proportion(mol)
        
        logs = calculate_esol(clogp, mw_val, rb, ap)
        return round(logs, 3)
    except Exception as e:
        logger.warning(f"Failed to calculate ESOL: {e}")
        return -99.9


_ESOL_ALLOWED_ATOMIC_NUMS = {1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53}


def is_esol_applicable(mol: Any, logs: Optional[float] = None) -> Tuple[bool, str]:
    """
    Check whether a molecule falls strictly within Delaney ESOL's scientific applicability domain:
    1. Valid, RDKit-parseable structure with > 0 heavy atoms.
    2. Single-component (no disconnected salt counterions / mixtures).
    3. Neutral, metal-free organic molecule (contains carbon, net formal charge 0,
       no spatially separated ionic/zwitterionic formal charges).
    4. Estimated LogS within physical bounds (-10.0 < LogS <= 2.0).

    Returns:
        (is_applicable, reason_if_not_applicable)
    """
    if mol is None:
        return False, "Valid SMILES required"
    if isinstance(mol, str):
        s = mol.strip()
        if not s:
            return False, "Valid SMILES required"
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            return False, "Valid SMILES required"

    try:
        if mol.GetNumHeavyAtoms() == 0:
            return False, "Outside ESOL domain"

        if len(Chem.GetMolFrags(mol)) > 1:
            return False, "Ionic salt (outside ESOL domain)"

        atoms = list(mol.GetAtoms())
        has_carbon = any(a.GetAtomicNum() == 6 for a in atoms)
        if not has_carbon:
            return False, "Inorganic species (outside ESOL domain)"

        if any(a.GetAtomicNum() not in _ESOL_ALLOWED_ATOMIC_NUMS for a in atoms):
            return False, "Metal / inorganic species (outside ESOL domain)"

        if Chem.GetFormalCharge(mol) != 0:
            return False, "Charged ion (outside ESOL domain)"

        for atom in atoms:
            fc = atom.GetFormalCharge()
            if fc != 0:
                has_bonded_countercharge = any((nbr.GetFormalCharge() * fc) < 0 for nbr in atom.GetNeighbors())
                if not has_bonded_countercharge:
                    return False, "Zwitterion / ionic species (outside ESOL domain)"

        esol_val = logs if logs is not None else calculate_delaney_esol(mol)
        if esol_val is None or esol_val <= -10.0 or esol_val > 2.0:
            return False, "Outside ESOL domain"

        return True, ""
    except Exception:
        return False, "Outside ESOL domain"



def detect_functional_groups(mol: Optional[Chem.Mol], name: str = "") -> List[str]:
    """Identify key electrochemical functional groups present in a candidate molecule."""
    if not mol and not name:
        return []

    groups = []
    if mol:
        # Polyol / alcohol: check aliphatic hydroxyl groups
        oh_matches = len(mol.GetSubstructMatches(FG_PATTERNS["polyol"]))
        carb_matches = len(mol.GetSubstructMatches(FG_PATTERNS["carboxylate"]))
        aliphatic_oh_count = max(0, oh_matches - carb_matches)
        if aliphatic_oh_count >= 2:
            groups.append("polyol")
        elif aliphatic_oh_count == 1 and name:
            groups.append("alcohol")

        # Amide or Urea
        if mol.HasSubstructMatch(FG_PATTERNS["urea"]) or mol.HasSubstructMatch(FG_PATTERNS["amide"]):
            groups.append("amide_urea")

        # Thiocarbonyl or Thioamide
        if mol.HasSubstructMatch(FG_PATTERNS["thiocarbonyl"]) or mol.HasSubstructMatch(FG_PATTERNS["thioamide"]):
            groups.append("thiocarbonyl")

        # Quaternary ammonium
        if mol.HasSubstructMatch(FG_PATTERNS["quaternary_ammonium"]):
            groups.append("quaternary_ammonium")

        # Sulfoxide
        if mol.HasSubstructMatch(FG_PATTERNS["sulfoxide"]):
            groups.append("sulfoxide")

        # Ether / polyether
        if mol.HasSubstructMatch(FG_PATTERNS["ether"]) and "polyol" not in groups:
            groups.append("ether_polyether")

        # Carboxylate
        if carb_matches > 0:
            groups.append("carboxylate")

    if name:
        n_lower = name.lower()
        if "polyol" not in groups and any(k in n_lower for k in ("sorbitol", "mannitol", "glycerol", "glycol", "glucose", "xylitol")):
            groups.append("polyol")
        if "thiocarbonyl" not in groups and any(k in n_lower for k in ("thiourea", "thioacetamide", "thiocarbonyl")):
            groups.append("thiocarbonyl")
        if "quaternary_ammonium" not in groups and any(k in n_lower for k in ("ammonium", "choline", "tba", "tma")):
            groups.append("quaternary_ammonium")
        if "amide_urea" not in groups and any(k in n_lower for k in ("urea", "acrylamide", "formamide", "acetamide")):
            groups.append("amide_urea")

    if not groups:
        groups.append("general")

    return groups


def normalize_baseline_salt(baseline_electrolyte: Optional[str]) -> Optional[str]:
    """Parse raw baseline electrolyte string into one of the 6 canonical ZIB salt names."""
    if not baseline_electrolyte:
        return None
    sub_map = str.maketrans("₀₁₂₃₄₅₆₇₈₉⁺⁻", "0123456789+-")
    b_upper = baseline_electrolyte.strip().translate(sub_map).upper()
    
    if "ZNSO4" in b_upper or "SULFATE" in b_upper:
        return "ZnSO4"
    if "ZN(OTF)2" in b_upper or "TRIFLATE" in b_upper or "OTF" in b_upper:
        return "Zn(OTf)2"
    if "ZN(CLO4)2" in b_upper or "PERCHLORATE" in b_upper or "CLO4" in b_upper:
        return "Zn(ClO4)2"
    if "ZN(TFSI)2" in b_upper or "TFSI" in b_upper:
        return "Zn(TFSI)2"
    if "ZNCL2" in b_upper or "CHLORIDE" in b_upper:
        return "ZnCl2"
    if "ZN(CH3COO)2" in b_upper or "ACETATE" in b_upper or "OAC" in b_upper or "ZN(OAC)2" in b_upper:
        return "Zn(CH3COO)2"
    return None


def assess_electrolyte_compatibility(
    mol: Chem.Mol,
    baseline_electrolyte: Optional[str] = None
) -> Tuple[str, Dict[str, ElectrolyteCompatibilityDetail]]:
    """
    Evaluates compatibility and expected electrochemical effectiveness across all
    6 major aqueous ZIB electrolyte chemistries based on the candidate's functional groups.
    
    Returns:
        (electrolyte_compatibility_note, electrolyte_compatibility_breakdown)
    """
    groups = detect_functional_groups(mol)
    baseline_salt = normalize_baseline_salt(baseline_electrolyte)
    
    # Prioritize dominant functional group for mechanistic profile
    if "polyol" in groups:
        primary = "polyol"
    elif "amide_urea" in groups:
        primary = "amide_urea"
    elif "thiocarbonyl" in groups:
        primary = "thiocarbonyl"
    elif "quaternary_ammonium" in groups:
        primary = "quaternary_ammonium"
    elif "sulfoxide" in groups:
        primary = "sulfoxide"
    elif "ether_polyether" in groups:
        primary = "ether_polyether"
    elif "carboxylate" in groups:
        primary = "carboxylate"
    else:
        primary = "general"

    breakdown: Dict[str, ElectrolyteCompatibilityDetail] = {}

    if primary == "polyol":
        for salt in SUPPORTED_SALTS:
            if salt == baseline_salt:
                status = "Validated"
                eff = "High"
                notes = (
                    "Validated for sulfate chemistry; reconstructs Zn²⁺ solvation sheath by displacing active water, "
                    "suppresses byproduct basic zinc sulfate (BZS) passivation, and elevates HER overpotential for uniform (002) deposition."
                    if salt == "ZnSO4" else
                    f"Validated baseline in {salt} chemistry; modulates solvation structure and elevates HER overpotential."
                )
            elif salt == "ZnSO4":
                status = "Highly Compatible"
                eff = "High"
                notes = "Reconstructs Zn²⁺ solvation sheath by displacing active water, suppresses byproduct basic zinc sulfate (BZS) passivation, and elevates HER overpotential for uniform (002) deposition."
            elif salt == "Zn(OTf)2":
                status = "Highly Compatible"
                eff = "High"
                notes = "Theoretically compatible with Zn(OTf)2 due to neutral non-ionic solvation sheath modulation; works synergistically with bulky OTf- anions to promote in situ ZnF2-rich SEI while displacing coordinated water."
            elif salt == "Zn(ClO4)2":
                status = "Highly Compatible"
                eff = "High"
                notes = "Weakly coordinating ClO4- enables rapid Zn²⁺ desolvation kinetics; neutral polyols modulate water activity and suppress HER without impeding the ultrafast charge transfer kinetics of perchlorate."
            elif salt == "Zn(TFSI)2":
                status = "Highly Compatible"
                eff = "High"
                notes = "In water-in-salt (WiSE) or concentrated Zn(TFSI)2 systems, polyols bind residual free water molecules, expanding the electrochemical stability window (ESW > 2.2 V) and suppressing water splitting."
            elif salt == "ZnCl2":
                status = "Compatible"
                eff = "Moderate"
                notes = "Concentrated ZnCl2 exhibits rich chlorozinc complex anions ([ZnCl4]2-); polyols intercalate into the complex network to reduce solution viscosity, though high acidity at elevated ZnCl2 concentrations must be monitored."
            elif salt == "Zn(CH3COO)2":
                status = "Compatible"
                eff = "Moderate"
                notes = "Acetate buffers electrolyte pH (~5-6); polyols synergistically regulate Zn²⁺ nucleation morphology without perturbing the mild acetate buffer equilibrium."
            breakdown[salt] = ElectrolyteCompatibilityDetail(
                salt=salt,
                status=status,
                expected_effectiveness=eff,
                notes=notes
            )

    elif primary == "amide_urea":
        for salt in SUPPORTED_SALTS:
            if salt == baseline_salt:
                status = "Validated"
                eff = "High"
                notes = (
                    "Validated for sulfate chemistry; carbonyl O coordinates Zn²⁺ preferentially over H2O, "
                    "disrupting water hydrogen bonding, eliminating BZS accumulation, and mitigating hydrogen evolution."
                    if salt == "ZnSO4" else
                    f"Validated baseline in {salt} chemistry; strong carbonyl dipole reconstructs Zn²⁺ coordination sphere."
                )
            elif salt == "ZnSO4":
                status = "Highly Compatible"
                eff = "High"
                notes = "Carbonyl O coordinates Zn²⁺ preferentially over H2O, disrupting water hydrogen bonding, eliminating BZS accumulation, and mitigating hydrogen evolution."
            elif salt == "Zn(OTf)2":
                status = "Highly Compatible"
                eff = "High"
                notes = "Strongly compatible; amide coordination reconstructs the Zn²⁺ sheath alongside OTf- to promote a stable, compact fluoride/organic hybrid interphase."
            elif salt == "Zn(ClO4)2":
                status = "Highly Compatible"
                eff = "High"
                notes = "Amide carbonyls moderate the electric field and coordinate Zn²⁺, balancing the ultra-rapid desolvation of weakly coordinating ClO4- to prevent localized runaway dendrites."
            elif salt == "Zn(TFSI)2":
                status = "Highly Compatible"
                eff = "High"
                notes = "Amides readily participate in deep eutectic solvents (DES) or hybrid solvates with Zn(TFSI)2, significantly suppressing free water activity and widening the cathodic stability window."
            elif salt == "ZnCl2":
                status = "Highly Compatible"
                eff = "High"
                notes = "Readily forms low-temperature deep eutectic electrolytes (e.g., ZnCl2-urea DES), altering [ZnClx(H2O)y]2-x speciation, lowering freezing point, and inhibiting dendrites."
            elif salt == "Zn(CH3COO)2":
                status = "Compatible"
                eff = "Moderate"
                notes = "Co-exists stably with acetate anions, contributing solvation sheath modulation while acetate maintains mild pH buffering."
            breakdown[salt] = ElectrolyteCompatibilityDetail(
                salt=salt,
                status=status,
                expected_effectiveness=eff,
                notes=notes
            )

    elif primary == "thiocarbonyl":
        for salt in SUPPORTED_SALTS:
            if salt == baseline_salt:
                status = "Validated"
                eff = "High"
                notes = (
                    "Validated for sulfate chemistry; adsorbs strongly via soft sulfur on Zn (002) facets, "
                    "regulating 2D zinc ion diffusion and suppressing parasitic HER and BZS formation."
                    if salt == "ZnSO4" else
                    f"Validated baseline in {salt} chemistry; sulfur interfacial adsorption guides planar Zn deposition."
                )
            elif salt == "ZnSO4":
                status = "Highly Compatible"
                eff = "High"
                notes = "Adsorbs strongly via soft sulfur on Zn (002) facets, regulating 2D zinc ion diffusion and suppressing parasitic HER and BZS formation."
            elif salt == "Zn(OTf)2":
                status = "Highly Compatible"
                eff = "High"
                notes = "Synergistic dual-protection mechanism: sulfur interfacial adsorption directs smooth planar plating while OTf- reduction generates a fluorinated SEI."
            elif salt == "Zn(ClO4)2":
                status = "Compatible"
                eff = "Moderate"
                notes = "Interfacial thiocarbonyl adsorption acts as a kinetic regulator, moderating the rapid deposition flux induced by non-coordinating ClO4- anions."
            elif salt == "Zn(TFSI)2":
                status = "Compatible"
                eff = "Moderate"
                notes = "Acts effectively as an interfacial adsorption leveler; retains surface-directing capability in fluorinated WiSE formulations."
            elif salt == "ZnCl2":
                status = "Caution / Conditional"
                eff = "Conditional"
                notes = "High nucleophilic Cl- anion concentrations competitively adsorb on the Zn surface, potentially displacing thiocarbonyl ligands; concentration optimization is required to maintain effective surface coverage."
            elif salt == "Zn(CH3COO)2":
                status = "Compatible"
                eff = "Moderate"
                notes = "Operates effectively under acetate's buffered pH; sulfur adsorption guides planar zinc deposition without interfering with acetate coordination."
            breakdown[salt] = ElectrolyteCompatibilityDetail(
                salt=salt,
                status=status,
                expected_effectiveness=eff,
                notes=notes
            )

    elif primary == "quaternary_ammonium":
        for salt in SUPPORTED_SALTS:
            if salt == baseline_salt:
                status = "Validated"
                eff = "High"
                notes = (
                    "Validated for sulfate chemistry; classic dynamic electrostatic shielding layer (DESL); "
                    "selectively adsorbs at electric-field concentrations to eliminate dendrite initiation."
                    if salt == "ZnSO4" else
                    f"Validated baseline in {salt} chemistry; dynamic electrostatic shielding repels dendrite-forming Zn²⁺ accumulation."
                )
            elif salt == "ZnSO4":
                status = "Highly Compatible"
                eff = "High"
                notes = "Classic dynamic electrostatic shielding layer (DESL); selectively adsorbs at electric-field concentrations to eliminate dendrite initiation."
            elif salt == "Zn(OTf)2":
                status = "Highly Compatible"
                eff = "High"
                notes = "Cationic shielding operates synergistically with OTf- anion decomposition, creating a dual electrostatic and fluorinated interphase barrier against dendrites."
            elif salt == "Zn(ClO4)2":
                status = "Highly Compatible"
                eff = "High"
                notes = "Counteracts perchlorate's high ionic conductivity and fast kinetics by preventing localized charge clustering and rapid tip growth."
            elif salt == "Zn(TFSI)2":
                status = "Caution / Conditional"
                eff = "Conditional"
                notes = "Bulky organic quaternary ammonium cations may form low-solubility hydrophobic ion pairs [NR4][TFSI] with bulky TFSI- anions, risking phase separation or precipitation; verify solubility limits before robotic formulation."
            elif salt == "ZnCl2":
                status = "Compatible"
                eff = "Moderate"
                notes = "Electrostatic shielding is active, but high ionic strength in concentrated ZnCl2 compacts the electric double layer (EDL); additive dosage may need upward adjustment to sustain effective shielding."
            elif salt == "Zn(CH3COO)2":
                status = "Compatible"
                eff = "Moderate"
                notes = "Fully compatible with acetate buffer electrolytes; electrostatic shielding functions without altering bulk electrolyte equilibrium."
            breakdown[salt] = ElectrolyteCompatibilityDetail(
                salt=salt,
                status=status,
                expected_effectiveness=eff,
                notes=notes
            )

    elif primary == "sulfoxide":
        for salt in SUPPORTED_SALTS:
            if salt == baseline_salt:
                status = "Validated"
                eff = "High"
                notes = (
                    "Validated for sulfate chemistry; high Gutmann donor number S=O coordinates Zn²⁺ preferentially, "
                    "replaces inner-sphere water, lowers water activity, suppresses hydrogen evolution, and eliminates basic zinc sulfate precipitation."
                    if salt == "ZnSO4" else
                    f"Validated baseline in {salt} chemistry; strong donor coordination modulates primary solvation sheath."
                )
            elif salt == "ZnSO4":
                status = "Highly Compatible"
                eff = "High"
                notes = "High Gutmann donor number S=O coordinates Zn²⁺ preferentially, replaces inner-sphere water, lowers water activity, suppresses hydrogen evolution, and eliminates basic zinc sulfate precipitation."
            elif salt == "Zn(OTf)2":
                status = "Highly Compatible"
                eff = "High"
                notes = "Hybrid sulfoxide-triflate solvation suppresses water reactivity while assisting OTf- anion participation in the inner sheath to form a robust ZnF2-rich interphase."
            elif salt == "Zn(ClO4)2":
                status = "Highly Compatible"
                eff = "High"
                notes = "Strongly coordinates Zn²⁺ in non-coordinating ClO4- electrolytes, providing robust solvation protection and wide electrochemical stability windows without compromising rate capability."
            elif salt == "Zn(TFSI)2":
                status = "Highly Compatible"
                eff = "High"
                notes = "Highly effective in hybrid water-in-salt / molecular co-solvent electrolytes; depresses electrolyte freezing point, widens the voltage window beyond 2.5 V, and suppresses water reduction."
            elif salt == "ZnCl2":
                status = "Compatible"
                eff = "Moderate"
                notes = "Modulates [ZnClx(H2O)y]2-x coordination and lowers solution viscosity; however, strong Zn²⁺-sulfoxide coordination alters chlorozincate equilibrium and should be dosed cautiously."
            elif salt == "Zn(CH3COO)2":
                status = "Compatible"
                eff = "Moderate"
                notes = "Synergistically modulates solvation with acetate ligands, lowering water activity while maintaining stable pH buffering."
            breakdown[salt] = ElectrolyteCompatibilityDetail(
                salt=salt,
                status=status,
                expected_effectiveness=eff,
                notes=notes
            )

    else:
        for salt in SUPPORTED_SALTS:
            if salt == baseline_salt:
                status = "Validated"
                eff = "High"
                notes = f"Validated baseline in {salt} chemistry; confirms experimental efficacy in reported cell format."
            elif salt == "ZnSO4":
                status = "Compatible"
                eff = "Moderate"
                notes = "Standard aqueous baseline; observe whether additive prevents basic zinc sulfate (BZS) passivating flakes and stabilizes plating overpotentials."
            elif salt == "Zn(OTf)2":
                status = "Compatible"
                eff = "Moderate"
                notes = "Triflate electrolyte; assess interfacial synergy with in situ formed ZnF2-rich protective interphase."
            elif salt == "Zn(ClO4)2":
                status = "Compatible"
                eff = "Moderate"
                notes = "Perchlorate system with weakly coordinated ClO4-; check compatibility with high desolvation rate and fast mass transport."
            elif salt == "Zn(TFSI)2":
                status = "Compatible"
                eff = "Moderate"
                notes = "Water-in-salt / concentrated TFSI- regime; evaluate compatibility with expanded electrochemical stability window and solubility in high-concentration salt."
            elif salt == "ZnCl2":
                status = "Compatible"
                eff = "Moderate"
                notes = "Concentrated chloride complexation; evaluate stability against chlorozinc coordination and acidic solution environments."
            elif salt == "Zn(CH3COO)2":
                status = "Compatible"
                eff = "Moderate"
                notes = "Acetate buffer system; evaluate additive interaction with weakly acidic pH environment and acetate ligands."
            breakdown[salt] = ElectrolyteCompatibilityDetail(
                salt=salt,
                status=status,
                expected_effectiveness=eff,
                notes=notes
            )

    # Synthesize unified cross-chemistry guidance note
    if baseline_salt == "ZnSO4":
        note = "Validated for sulfate chemistry; verify compatibility before using in other electrolytes"
        if primary == "polyol":
            note += "; theoretically compatible with Zn(OTf)2 due to neutral non-ionic solvation sheath modulation."
        elif primary == "amide_urea":
            note += "; theoretically compatible with Zn(OTf)2 and Zn(ClO4)2 due to strong carbonyl coordination."
        elif primary == "thiocarbonyl":
            note += "; theoretically compatible with Zn(OTf)2; exercise caution in ZnCl2 due to competitive Cl- adsorption."
        elif primary == "quaternary_ammonium":
            note += "; theoretically compatible with Zn(OTf)2; caution with Zn(TFSI)2 due to possible ion-pair precipitation."
        elif primary == "sulfoxide":
            note += "; theoretically compatible with Zn(OTf)2, Zn(ClO4)2, and Zn(TFSI)2 via strong donor coordination."
        else:
            note += "; evaluate electrolyte-specific solvation and interphase behavior."
    elif baseline_salt == "Zn(OTf)2":
        note = "Validated for triflate chemistry; verify compatibility before using in other electrolytes; monitor basic zinc sulfate (BZS) suppression in sulfate chemistries."
    elif baseline_salt == "Zn(ClO4)2":
        note = "Validated for perchlorate chemistry; verify compatibility before using in other electrolytes."
    elif baseline_salt == "Zn(TFSI)2":
        note = "Validated for TFSI chemistry (water-in-salt); verify compatibility before using in other electrolytes."
    elif baseline_salt == "ZnCl2":
        note = "Validated for zinc chloride chemistry; verify compatibility before using in other electrolytes."
    elif baseline_salt == "Zn(CH3COO)2":
        note = "Validated for acetate chemistry; verify compatibility before using in other electrolytes."
    else:
        note = "Parent electrolyte not specified; verify baseline salt compatibility (e.g., 2 M ZnSO4) prior to SDL execution."

    return note, breakdown


class RDKitSimilarityEngine:
    def __init__(
        self,
        candidate_pool: Optional[List[Dict[str, str]]] = None,
        pubchem_client: Optional[Any] = None,
    ):
        self.candidate_pool = candidate_pool if candidate_pool is not None else DEFAULT_POOL
        self.pool_mols = []
        self._descriptors_cache: Dict[str, Dict[str, Any]] = {}
        self._fp_cache: Dict[str, Any] = {}
        if pubchem_client is not None:
            self.pubchem_client = pubchem_client
        elif PubChemClient is not None:
            self.pubchem_client = PubChemClient()
        else:
            self.pubchem_client = None
        
        for item in self.candidate_pool:
            mol = Chem.MolFromSmiles(item["smiles"])
            if mol:
                canon_smiles = Chem.MolToSmiles(mol)
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                functional_groups = detect_functional_groups(mol)
                descriptors = self._compute_mol_descriptors(mol)
                self.pool_mols.append({
                    "name": item["name"],
                    "smiles": item["smiles"],
                    "canon_smiles": canon_smiles,
                    "mol": mol,
                    "fp": fp,
                    "functional_groups": functional_groups,
                    "descriptors": descriptors,
                })
                self._descriptors_cache[item["smiles"]] = descriptors
                self._descriptors_cache[canon_smiles] = descriptors
                self._fp_cache[item["smiles"]] = fp
                self._fp_cache[canon_smiles] = fp

    @staticmethod
    def _compute_mol_descriptors(mol: Chem.Mol) -> Dict[str, Any]:
        """Compute physicochemical descriptors directly from an RDKit Mol object."""
        return {
            "molecular_weight": round(Descriptors.MolWt(mol), 2),
            "logp": round(Descriptors.MolLogP(mol), 2),
            "logs": calculate_delaney_esol(mol),
            "tpsa": round(rdMolDescriptors.CalcTPSA(mol), 2),
            "hbd": rdMolDescriptors.CalcNumLipinskiHBD(mol),
            "hba": rdMolDescriptors.CalcNumLipinskiHBA(mol),
            "rotatable_bonds": Descriptors.NumRotatableBonds(mol),
        }

    def calculate_descriptors(self, smiles: str) -> Optional[Dict[str, Any]]:
        """Calculate physicochemical descriptors for a given SMILES string."""
        if not smiles:
            logger.warning(f"Invalid SMILES string for descriptor calculation: {smiles}")
            return None
        cached = self._descriptors_cache.get(smiles)
        if cached is not None:
            return dict(cached)

        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            logger.warning(f"Invalid SMILES string for descriptor calculation: {smiles}")
            return None

        canon_smiles = Chem.MolToSmiles(mol)
        cached_canon = self._descriptors_cache.get(canon_smiles)
        if cached_canon is not None:
            self._descriptors_cache[smiles] = cached_canon
            return dict(cached_canon)

        desc = self._compute_mol_descriptors(mol)
        self._descriptors_cache[smiles] = desc
        self._descriptors_cache[canon_smiles] = desc
        return dict(desc)

    def compute_tanimoto(self, smiles1: str, smiles2: str) -> Optional[float]:
        """Compute Morgan Fingerprint (radius=2, 2048-bit) Tanimoto similarity between two SMILES."""
        fp1 = self._fp_cache.get(smiles1) if smiles1 else None
        if fp1 is None:
            mol1 = Chem.MolFromSmiles(smiles1) if smiles1 else None
            if not mol1:
                logger.warning(f"Invalid SMILES string for Tanimoto computation: {smiles1}, {smiles2}")
                return None
            fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
            self._fp_cache[smiles1] = fp1

        fp2 = self._fp_cache.get(smiles2) if smiles2 else None
        if fp2 is None:
            mol2 = Chem.MolFromSmiles(smiles2) if smiles2 else None
            if not mol2:
                logger.warning(f"Invalid SMILES string for Tanimoto computation: {smiles1}, {smiles2}")
                return None
            fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
            self._fp_cache[smiles2] = fp2

        return float(TanimotoSimilarity(fp1, fp2))

    def resolve_chemical_name_to_smiles(self, name: str) -> Optional[str]:
        """
        Attempt to resolve a chemical name to a valid SMILES string.
        First checks the curated pool and common aliases, then queries PubChem REST API (ChemCrow inspired).
        """
        if not name or not name.strip():
            return None

        cleaned_name = name.strip()
        cleaned_lower = cleaned_name.lower()

        # 1. Curated pool lookup
        for item in self.candidate_pool:
            if item["name"].lower() == cleaned_lower:
                return item["smiles"]

        # 2. Known common aliases lookup for fast offline resilience
        alias_map = {
            "edta-cuna2": "[Na+].[Na+].[Cu+2].[O-]C(=O)CN(CCN(CC(=O)[O-])CC(=O)[O-])CC(=O)[O-]",
            "copper disodium ethylenediaminetetraacetate": "[Na+].[Na+].[Cu+2].[O-]C(=O)CN(CCN(CC(=O)[O-])CC(=O)[O-])CC(=O)[O-]",
            "edta-2na": "C(CN(CC(=O)O)CC(=O)[O-])N(CC(=O)O)CC(=O)[O-].[Na+].[Na+]",
            "disodium edta": "C(CN(CC(=O)O)CC(=O)[O-])N(CC(=O)O)CC(=O)[O-].[Na+].[Na+]",
            "du": "CN1CCCN(C)C1=O",
            "n,n0-dimethylpropyleneurea": "CN1CCCN(C)C1=O",
            "n,n′-dimethylpropyleneurea": "CN1CCCN(C)C1=O",
            "dmi": "CN1CCN(C)C1=O",
            "1,3-dimethyl-2-imidazolidinone": "CN1CCN(C)C1=O",
            "cuso4": "[Cu+2].[O-]S(=O)(=O)[O-]",
            "copper sulfate": "[Cu+2].[O-]S(=O)(=O)[O-]",
            "mnso4": "[Mn+2].[O-]S(=O)(=O)[O-]",
            "manganese sulfate": "[Mn+2].[O-]S(=O)(=O)[O-]",
            "sc": "[Na+].[Na+].[Na+].C(C(=O)[O-])C(CC(=O)[O-])(C(=O)[O-])O",
            "sodium citrate": "[Na+].[Na+].[Na+].C(C(=O)[O-])C(CC(=O)[O-])(C(=O)[O-])O",
            "teabr": "CC[N+](CC)(CC)CC.[Br-]",
            "ctab": "CCCCCCCCCCCCCCCC[N+](C)(C)C.[Br-]",
            "sds": "CCCCCCCCCCCCOS(=O)(=O)[O-].[Na+]",
        }
        if cleaned_lower in alias_map:
            return alias_map[cleaned_lower]

        # 3. PubChem online query fallback (inspired by ChemCrow)
        if self.pubchem_client:
            try:
                smi = self.pubchem_client.name_to_smiles(cleaned_name)
                if smi:
                    mol = Chem.MolFromSmiles(smi)
                    if mol:
                        return smi
            except Exception as e:
                logger.debug(f"PubChem name resolution failed for '{name}': {e}")

        return None

    def find_similar_candidates(
        self, 
        query_smiles: str, 
        top_n: int = 5, 
        threshold: float = 0.15,
        baseline_electrolyte: Optional[str] = None,
        use_pubchem: bool = True,
        parent_name: Optional[str] = None,
    ) -> List[CandidateMolecule]:
        """
        Rank molecules based on Morgan fingerprint Tanimoto similarity.
        Optionally searches NCBI PubChem dynamically for structural analogues (ChemCrow approach)
        and integrates them with the curated reference library.
        Computes ESOL aqueous solubility and evaluates cross-electrolyte compatibility.
        """
        query_mol = Chem.MolFromSmiles(query_smiles)
        if not query_mol:
            logger.error(f"Invalid SMILES provided for similarity search: {query_smiles}")
            return []

        query_canon_smi = Chem.MolToSmiles(query_mol)
        query_fp = AllChem.GetMorganFingerprintAsBitVect(query_mol, 2, nBits=2048)

        candidates_to_score = []

        # 1. From curated reference pool
        for item in self.pool_mols:
            sim = float(TanimotoSimilarity(query_fp, item["fp"]))
            cand_canon = item["canon_smiles"]
            if cand_canon != query_canon_smi and (not parent_name or item["name"].strip().lower() != parent_name.strip().lower()):
                if sim >= 1.0:
                    sim = 0.99
                candidates_to_score.append({
                    "name": item["name"],
                    "smiles": item["smiles"],
                    "canon_smiles": cand_canon,
                    "mol": item["mol"],
                    "descriptors": item["descriptors"],
                    "functional_groups": item["functional_groups"],
                    "sim": sim,
                    "source": "Curated DB",
                    "metric": "Tanimoto (Morgan R=2, Curated DB)"
                })

        # 2. Dynamic PubChem fast 2D similarity search (ChemCrow inspired)
        if use_pubchem and self.pubchem_client:
            try:
                pubchem_cutoff = max(70, min(95, int(threshold * 100)))
                pubchem_hits = self.pubchem_client.search_similar_compounds(
                    query_smiles=query_smiles,
                    threshold=pubchem_cutoff,
                    max_records=15
                )
                for hit in pubchem_hits:
                    h_smi = hit.get("smiles")
                    if not h_smi:
                        continue
                    h_mol = Chem.MolFromSmiles(h_smi)
                    if not h_mol:
                        continue
                    h_canon = Chem.MolToSmiles(h_mol)
                    if h_canon == query_canon_smi:
                        continue

                    # Compute Morgan FP and precise Tanimoto similarity
                    h_fp = AllChem.GetMorganFingerprintAsBitVect(h_mol, 2, nBits=2048)
                    h_sim = float(TanimotoSimilarity(query_fp, h_fp))

                    candidates_to_score.append({
                        "name": hit.get("name") or f"PubChem CID {hit.get('cid')}",
                        "smiles": h_smi,
                        "canon_smiles": h_canon,
                        "mol": h_mol,
                        "sim": h_sim,
                        "source": "PubChem Live",
                        "metric": "Tanimoto (Morgan R=2, PubChem Live)"
                    })
            except Exception as e:
                logger.warning(f"Dynamic PubChem similarity search encountered an error: {e}")

        # Deduplicate candidates by canonical SMILES (keeping highest similarity score, prioritizing curated name if equal)
        unique_cands: Dict[str, Dict[str, Any]] = {}
        for c in candidates_to_score:
            c_canon = c.get("canon_smiles") or Chem.MolToSmiles(c["mol"])
            if c_canon not in unique_cands:
                unique_cands[c_canon] = c
            else:
                existing = unique_cands[c_canon]
                if c["sim"] > existing["sim"]:
                    unique_cands[c_canon] = c
                elif c["sim"] == existing["sim"] and existing["source"] != "Curated DB" and c["source"] == "Curated DB":
                    unique_cands[c_canon] = c

        scored_list = list(unique_cands.values())
        scored_list.sort(key=lambda x: x["sim"], reverse=True)

        filtered_matches = [c for c in scored_list if c["sim"] >= threshold]
        top_matches = filtered_matches[:top_n] if filtered_matches else scored_list[:top_n]

        results = []
        for item in top_matches:
            mol = item["mol"]
            sim = item["sim"]
            desc = item.get("descriptors") or self._compute_mol_descriptors(mol)

            mw = desc["molecular_weight"]
            logp = desc["logp"]
            logs = desc["logs"]
            tpsa = desc["tpsa"]
            hbd = desc["hbd"]
            hba = desc["hba"]
            rot_bonds = desc["rotatable_bonds"]

            solubility_warning = ""
            if logs < -4.0:
                solubility_warning = "WARNING: Low intrinsic solubility (LogS < -4). Requires heated stirring or co-solvent during SDL dispensing."
            else:
                solubility_warning = "Aqueous solubility limits optimal for automated liquid handling."

            source_tag = f" [{item['source']}]" if "source" in item else ""
            if sim >= threshold:
                rationale = (
                    f"Tanimoto similarity of {sim:.2f} to target{source_tag}. "
                    f"Shares structural motifs with {hbd} H-bond donors and {hba} acceptors."
                )
            else:
                rationale = (
                    f"Top baseline reference analogue (Tanimoto similarity: {sim:.2f}){source_tag}. "
                    f"Features compatible functional motifs ({hbd} H-bond donors, {hba} acceptors)."
                )

            # Electrochemical compatibility reasoning across 6 major ZIB electrolyte salts
            compat_note, compat_breakdown = assess_electrolyte_compatibility(
                mol=mol,
                baseline_electrolyte=baseline_electrolyte
            )

            parent_scaffold_name = parent_name or "Literature Reference Scaffold"
            chemo_rationale = generate_chemoinformatic_rationale(
                cand_name=item["name"],
                cand_mol=mol,
                parent_name=parent_scaffold_name,
                similarity_score=sim,
                logs=logs,
                lang="en"
            )

            cand = CandidateMolecule(
                name=item["name"],
                smiles=item["smiles"],
                similarity_score=sim,
                similarity_metric=item.get("metric", "Tanimoto (Morgan R=2)"),
                molecular_weight=mw,
                logp=logp,
                logs=logs,
                tpsa=tpsa,
                hbd=hbd,
                hba=hba,
                rotatable_bonds=rot_bonds,
                rationale=rationale,
                sdl_formulation_notes=solubility_warning,
                parent_baseline_electrolyte=baseline_electrolyte,
                electrolyte_compatibility_note=compat_note,
                electrolyte_compatibility_breakdown=compat_breakdown,
                parent_scaffold_name=parent_scaffold_name,
                parent_scaffold_smiles=query_smiles,
                parent_scaffold_similarity=sim,
                ai_rationale=chemo_rationale["full_rationale"],
                functional_motif_rationale=chemo_rationale["functional_motifs"],
                physical_hypothesis=chemo_rationale["physical_hypothesis"],
            )
            results.append(cand)

        return results


def generate_chemoinformatic_rationale(
    cand_name: str,
    cand_mol: Optional[Chem.Mol],
    parent_name: str,
    similarity_score: float,
    logs: Optional[float] = None,
    lang: str = "en",
) -> Dict[str, str]:
    """
    Generate structured chemoinformatic rationale for Tier 2 AI-suggested novel analogues.
    Explains:
    1. Parent literature scaffold anchor and Tanimoto similarity score.
    2. Why functional group motifs are chemically promising for Zn2+ solvation modulation or dendrite/HER suppression.
    3. The physical hypothesis (maintaining zincophilic adsorption while enhancing aqueous solubility).
    """
    groups = detect_functional_groups(cand_mol) if cand_mol else ["general"]
    theor_sat_mm = round((10 ** (logs if logs is not None else -2.0)) * 1000.0, 1) if (logs is not None and logs > -10) else 100.0
    hard_ceil_mm = round(0.80 * theor_sat_mm, 1)

    if lang == "zh":
        anchor_txt = f"锚定于文献已验证母体骨架 **{parent_name}**（Tanimoto 结构相似度：**{similarity_score:.2f}**）。"
        if "polyol" in groups:
            motif_txt = "富含多羟基 (-OH) 亲锌基序，能有效置换配位水分子重构 Zn²⁺ 初级溶剂化鞘层，打断活性氢键网络，抑制碱式硫酸锌 (BZS) 钝化并提升析氢反应 (HER) 过电位。"
        elif "amide_urea" in groups:
            motif_txt = "具有高偶极矩极性羰基/酰胺基团，优先于活性自由水与 Zn²⁺ 配位，改变内层溶剂化配位几何结构并抑制寄生析氢反应 (HER)。"
        elif "thiocarbonyl" in groups:
            motif_txt = "软硫供体原子在 Zn (002) 晶面表现出特异性亲锌吸附，引导 Zn²⁺ 二维表面均匀扩散并抑制枝晶尖端生长。"
        elif "quaternary_ammonium" in groups:
            motif_txt = "带正电的季铵基阳离子在微突起尖端建立动态自愈合静电屏蔽层，将后续 Zn²⁺ 通量重定向至平坦凹面，实现平面均匀沉积。"
        elif "sulfoxide" in groups:
            motif_txt = "高极性 S=O 亚砜基序配位进入 Zn²⁺ 内溶剂化鞘层，高效排斥自由水分子并拓宽电解液的电化学稳定窗口。"
        elif "ether_polyether" in groups:
            motif_txt = "柔性醚氧连接提供适度的路易斯碱性配位 Zn²⁺，平滑电解液-电极界面离子传输且不增加过高的去溶剂化能垒。"
        elif "carboxylate" in groups:
            motif_txt = "螯合羧酸基团引导均匀的 Zn²⁺ 界面定向迁移，并促进原位构建致密的有机-无机复合固态电解质界面膜 (SEI)。"
        else:
            motif_txt = "极性功能基序降低去溶剂化活化能垒，优化电极表面电场均匀分布，并平抑界面电荷转移极化。"

        if logs is not None and logs > -2.0:
            hypo_txt = f"在继承母体 {parent_name} 亲锌界面吸附与溶剂化调控特性的同时，保持优异的水相溶解度（理论饱和度 {theor_sat_mm:.1f} mM），消除自动化移液通道沉淀析出隐患并抑制晶核引发的枝晶短路。"
        else:
            hypo_txt = f"保留源自母体 {parent_name} 的关键 Zn²⁺ 界面配位特征，并通过严格执行 Delaney ESOL 80% 溶解度安全上限（{hard_ceil_mm:.1f} mM），在保证强界面吸附的同时确保自动化移液安全性。"

    elif lang == "fr":
        anchor_txt = f"Ancré sur l'échafaudage de la littérature **{parent_name}** (Similarité de Tanimoto : **{similarity_score:.2f}**)."
        if "polyol" in groups:
            motif_txt = "Les motifs hydroxyles (-OH) denses modulent la gaine de solvatation primaire de Zn²⁺ en déplaçant l'eau coordonnée, perturbant les réseaux hydrogène et supprimant la passivation par le sulfate basique de zinc (BZS)."
        elif "amide_urea" in groups:
            motif_txt = "Les motifs donneurs carbonyles et azotés polarisés se coordinent préférentiellement à Zn²⁺ par rapport à l'eau active, reconstruisant la sphère de coordination interne et augmentant la surtension de dégagement d'hydrogène (HER)."
        elif "thiocarbonyl" in groups:
            motif_txt = "Les motifs donneurs de soufre mous présentent une forte adsorption zincophile sur les facettes cristallines Zn (002), guidant une diffusion bidimensionnelle uniforme des ions zinc et supprimant la propagation dendritique."
        elif "quaternary_ammonium" in groups:
            motif_txt = "Les têtes cationiques d'ammonium quaternaire établissent un bouclier électrostatique dynamique sur les protubérances de zinc, repoussant le flux de Zn²⁺ vers les creux pour un dépôt planaire homogène."
        elif "sulfoxide" in groups:
            motif_txt = "Les motifs sulfoxydes S=O hautement polaires coordinent Zn²⁺ dans la sphère interne de solvatation, remplaçant l'eau libre et élargissant la fenêtre de stabilité électrochimique de l'électrolyte."
        elif "ether_polyether" in groups:
            motif_txt = "Les liaisons éther oxygène flexibles fournissent une basicité de Lewis modérée pour coordonner Zn²⁺, fluidifiant le transport ionique interfacial sans pénalité excessive de désolvatation."
        elif "carboxylate" in groups:
            motif_txt = "Les motifs carboxylates chélatants guident une migration interfaciale uniforme de Zn²⁺ et favorisent la formation in situ d'une interphase protectrice composite organique-inorganique (SEI)."
        else:
            motif_txt = "Les motifs fonctionnels polaires abaissent l'énergie d'activation de désolvatation, optimisent la distribution du champ électrique interfacial et modèrent la cinétique de transfert de charge."

        if logs is not None and logs > -2.0:
            hypo_txt = f"Préserve l'adsorption interfaciale zincophile et la régulation de solvatation de {parent_name} tout en garantissant une haute solubilité aqueuse (saturation théorique de {theor_sat_mm:.1f} mM) pour éliminer les précipitations en canaux de pipetage et inhiber les dendrites par germination cristalline."
        else:
            hypo_txt = f"Maintient la coordination interfaciale essentielle avec Zn²⁺ issue de {parent_name} tout en appliquant un plafond de sécurité Delaney ESOL de 80% ({hard_ceil_mm:.1f} mM) pour concilier forte adsorption de surface et sécurité de distribution liquide automatisée."

    else:
        # Default English
        anchor_txt = f"Anchored to literature scaffold **{parent_name}** (Tanimoto Similarity: **{similarity_score:.2f}**)."
        if "polyol" in groups:
            motif_txt = "Dense hydroxyl (-OH) motifs modulate the primary Zn²⁺ solvation sheath by displacing coordinated water molecules, disrupting hydrogen-bonding networks, and suppressing basic zinc sulfate (BZS) byproduct passivation."
        elif "amide_urea" in groups:
            motif_txt = "Polarized carbonyl and nitrogen donor motifs coordinate Zn²⁺ preferentially over active water, reconstructing the inner coordination sphere and elevating the hydrogen evolution (HER) overpotential."
        elif "thiocarbonyl" in groups:
            motif_txt = "Soft sulfur donor motifs exhibit strong zincophilic adsorption on Zn (002) crystal facets, directing uniform 2D zinc ion diffusion and suppressing dendrite propagation."
        elif "quaternary_ammonium" in groups:
            motif_txt = "Cationic quaternary ammonium heads establish a dynamic electrostatic shield on zinc protuberances, repelling incoming Zn²⁺ toward planar valleys for uniform deposition."
        elif "sulfoxide" in groups:
            motif_txt = "Highly polar S=O sulfoxide motifs coordinate Zn²⁺ into the inner solvation shell, replacing free water molecules and broadening the electrolyte electrochemical stability window."
        elif "ether_polyether" in groups:
            motif_txt = "Flexible ether oxygen linkages provide moderate Lewis basicity to coordinate Zn²⁺, smoothing interfacial ion transport without excessive desolvation activation penalty."
        elif "carboxylate" in groups:
            motif_txt = "Chelating carboxylate groups guide uniform Zn²⁺ interfacial migration and promote in situ organic-inorganic composite protective interphase (SEI) formation."
        else:
            motif_txt = "Polar functional motifs lower desolvation activation energy, optimize interfacial electric field distribution, and moderate charge transfer kinetics."

        if logs is not None and logs > -2.0:
            hypo_txt = f"Preserves the zincophilic interfacial adsorption and solvation structure of {parent_name} while maintaining favorable aqueous solubility ({theor_sat_mm:.1f} mM theoretical saturation) to eliminate pipetting channel precipitation and inhibit crystal-nucleated dendrites."
        else:
            hypo_txt = f"Maintains essential interfacial coordination with Zn²⁺ from {parent_name} while enforcing an 80% Delaney ESOL safety ceiling ({hard_ceil_mm:.1f} mM) to balance strong surface adsorption with automated liquid dispensing safety."

    full_txt = f"{anchor_txt} {motif_txt} {hypo_txt}"
    return {
        "parent_anchor": anchor_txt,
        "functional_motifs": motif_txt,
        "physical_hypothesis": hypo_txt,
        "full_rationale": full_txt,
    }
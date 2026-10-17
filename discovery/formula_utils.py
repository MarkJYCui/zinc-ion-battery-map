"""
Chemical Formula and Conventional Notation Utilities.
Module 1: Materials Acceleration Platform (MAP) for Zinc-Ion Batteries (ZIB).

Converts SMILES, RDKit Mols, and common chemical names into conventional IUPAC
chemical formulas using Unicode subscripts (e.g., C₁₂H₂₂O₁₁, LiCl, ZnSO₄).
"""

import re
from typing import Dict, Any, Optional
from rdkit import Chem
try:
    from discovery.search import clean_scientific_text
except ImportError:
    from search import clean_scientific_text

CURATED_FORMULAS: Dict[str, str] = {
    "D-Sorbitol": "C₆H₁₄O₆",
    "D-Mannitol": "C₆H₁₄O₆",
    "Glucose": "C₆H₁₂O₆",
    "Polyacrylamide": "(C₃H₅NO)ₙ",
    "Polyacrylamide (PAM)": "(C₃H₅NO)ₙ",
    "PAM": "(C₃H₅NO)ₙ",
    "Polyethylene Glycol": "(C₂H₄O)ₙ",
    "Polyethylene Glycol (PEG)": "(C₂H₄O)ₙ",
    "Polyethylene glycol": "(C₂H₄O)ₙ",
    "Polyethylene glycol (PEG)": "(C₂H₄O)ₙ",
    "PEG": "(C₂H₄O)ₙ",
    "Thiourea": "CH₄N₂S",
    "Urea": "CH₄N₂O",
    "Ethylene Glycol": "C₂H₆O₂",
    "Glycerol": "C₃H₈O₃",
    "Ascorbic Acid": "C₆H₈O₆",
    "Tetramethylammonium bromide": "(CH₃)₄NBr",
    "Tetrapropylammonium bromide": "(C₃H₇)₄NBr",
    "Tetrabutylammonium bromide": "(C₄H₉)₄NBr",
    "Choline chloride": "C₅H₁₄ClNO",
    "Dimethyl sulfoxide": "(CH₃)₂SO",
    "Trehalose": "C₁₂H₂₂O₁₁",
    "TEABr": "(C₂H₅)₄NBr",
    "Tetraethylammonium bromide": "(C₂H₅)₄NBr",
    "Succinic Acid": "C₄H₆O₄",
    "Citric Acid": "C₆H₈O₇",
    "Phosphonoglycolic acid": "C₂H₅O₆P",
    "phosphonoglycolic acid": "C₂H₅O₆P",
    "N,N'-methylenebis(acrylamide)": "C₇H₁₀N₂O₂",
    "N,N′-methylenebis(acrylamide)": "C₇H₁₀N₂O₂",
    "methylenebisacrylamide": "C₇H₁₀N₂O₂",
    "Hyaluronic acid": "(C₁₄H₂₁NO₁₁)ₙ",
    "hyaluronic acid": "(C₁₄H₂₁NO₁₁)ₙ",
    "PEG-200": "H(C₂H₄O)₄OH",
    "Copper disodium ethylenediaminetetraacetate (EDTA-CuNa2)": "C₁₀H₁₂CuN₂Na₂O₈",
    "Copper disodium ethylenediaminetetraacetate": "C₁₀H₁₂CuN₂Na₂O₈",
    "EDTA-CuNa2": "C₁₀H₁₂CuN₂Na₂O₈",
    "Disodium EDTA (EDTA-2Na)": "C₁₀H₁₄N₂Na₂O₈",
    "Disodium EDTA": "C₁₀H₁₄N₂Na₂O₈",
    "N,N′-dimethylpropyleneurea (DU)": "C₆H₁₂N₂O",
    "N,N′-dimethylpropyleneurea": "C₆H₁₂N₂O",
    "N,N0-dimethylpropyleneurea (DU)": "C₆H₁₂N₂O",
    "DU": "C₆H₁₂N₂O",
    "1,3-Dimethyl-2-imidazolidinone (DMI)": "C₅H₁₀N₂O",
    "1,3-Dimethyl-2-imidazolidinone": "C₅H₁₀N₂O",
    "Copper(II) sulfate (CuSO₄)": "CuSO₄",
    "Copper(II) sulfate": "CuSO₄",
    "CuSO4": "CuSO₄",
    "CuSO₄": "CuSO₄",
    "Manganese(II) sulfate (MnSO₄)": "MnSO₄",
    "Manganese(II) sulfate": "MnSO₄",
    "MnSO4": "MnSO₄",
    "MnSO₄": "MnSO₄",
    "Sodium citrate (SC)": "Na₃C₆H₅O₇",
    "Sodium citrate": "Na₃C₆H₅O₇",
    "Trisodium citrate": "Na₃C₆H₅O₇",
    "Cetyltrimethylammonium bromide (CTAB)": "C₁₉H₄₂BrN",
    "Cetyltrimethylammonium bromide": "C₁₉H₄₂BrN",
    "CTAB": "C₁₉H₄₂BrN",
    "Sodium dodecyl sulfate (SDS)": "NaC₁₂H₂₅SO₄",
    "Sodium dodecyl sulfate": "NaC₁₂H₂₅SO₄",
    "SDS": "NaC₁₂H₂₅SO₄",
    "Betaine": "C₅H₁₁NO₂",
    "Graphene oxide": "GO",
    "Graphene Oxide": "GO",
    "graphene oxide": "GO",
    "GO": "GO",
    "Reduced graphene oxide": "rGO",
    "Reduced graphene oxide (rGO)": "rGO",
    "rGO": "rGO",
    "Carbon nanotubes": "CNT",
    "CNT": "CNT",
    "MXene": "MXene",
    "Ti3C2Tx": "Ti₃C₂Tₓ",
    # Universal Conventional Inorganic Salts
    "Lithium chloride": "LiCl",
    "Lithium chloride (LiCl)": "LiCl",
    "LiCl": "LiCl",
    "Sodium chloride": "NaCl",
    "Sodium chloride (NaCl)": "NaCl",
    "NaCl": "NaCl",
    "Potassium chloride": "KCl",
    "Potassium chloride (KCl)": "KCl",
    "KCl": "KCl",
    "Zinc chloride": "ZnCl₂",
    "Zinc chloride (ZnCl2)": "ZnCl₂",
    "Zinc chloride (ZnCl₂)": "ZnCl₂",
    "ZnCl2": "ZnCl₂",
    "ZnCl₂": "ZnCl₂",
    "Magnesium sulfate": "MgSO₄",
    "Magnesium sulfate (MgSO4)": "MgSO₄",
    "Magnesium sulfate (MgSO₄)": "MgSO₄",
    "MgSO4": "MgSO₄",
    "MgSO₄": "MgSO₄",
    "Sodium fluoride": "NaF",
    "Sodium fluoride (NaF)": "NaF",
    "NaF": "NaF",
    "Potassium bromide": "KBr",
    "Potassium bromide (KBr)": "KBr",
    "KBr": "KBr",
    "Lithium iodide": "LiI",
    "Lithium iodide (LiI)": "LiI",
    "LiI": "LiI",
    "Lithium bromide": "LiBr",
    "Lithium bromide (LiBr)": "LiBr",
    "LiBr": "LiBr",
    "Ammonium chloride": "NH₄Cl",
    "Ammonium chloride (NH4Cl)": "NH₄Cl",
    "NH4Cl": "NH₄Cl",
    "Calcium chloride": "CaCl₂",
    "Calcium chloride (CaCl2)": "CaCl₂",
    "CaCl2": "CaCl₂",
    "Barium chloride": "BaCl₂",
    "BaCl2": "BaCl₂",
    "Nickel sulfate": "NiSO₄",
    "NiSO4": "NiSO₄",
    "Cobalt sulfate": "CoSO₄",
    "CoSO4": "CoSO₄",
    "Iron(II) sulfate": "FeSO₄",
    "FeSO4": "FeSO₄",
    "2-(N-morpholino)ethanesulfonic acid": "C₆H₁₃NO₄S",
    "2-(N-morpholino)ethanesulfonic acid (MES)": "C₆H₁₃NO₄S",
    "MES": "C₆H₁₃NO₄S",
}

PERIODIC_TABLE_ELEMENTS = {
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
    'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
    'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
    'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
    'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
    'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
    'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm'
}

KNOWN_FORMULA_ABBREVIATIONS = {'GO', 'rGO', 'CNT', 'MXene', 'OTf', 'TFSI', 'OAc', 'Ac', 'DU', 'DMI', 'SDS', 'CTAB', 'EDTA'}

INORGANIC_METALS = {
    'Li', 'Na', 'K', 'Rb', 'Cs', 'Fr',
    'Be', 'Mg', 'Ca', 'Sr', 'Ba', 'Ra',
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
    'Al', 'Ga', 'In', 'Tl', 'Sn', 'Pb', 'Bi',
    'La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Th', 'Pa', 'U'
}

INORGANIC_ELEMENT_PRIORITY = {
    'Li': 10, 'Na': 11, 'K': 12, 'Rb': 13, 'Cs': 14,
    'Be': 15, 'Mg': 16, 'Ca': 17, 'Sr': 18, 'Ba': 19,
    'Sc': 20, 'Ti': 21, 'V': 22, 'Cr': 23, 'Mn': 24,
    'Fe': 25, 'Co': 26, 'Ni': 27, 'Cu': 28, 'Zn': 29,
    'Al': 30, 'Ga': 31, 'In': 32, 'Sn': 33, 'Pb': 34, 'Bi': 35,
    'Ag': 36, 'Cd': 37, 'Zr': 38, 'Nb': 39, 'Mo': 40, 'Pd': 41, 'Pt': 42, 'Au': 43,
    'B': 52, 'Si': 54, 'P': 56, 'S': 58, 'Se': 60,
    'O': 75, 'H': 80,
    'F': 90, 'Cl': 92, 'Br': 94, 'I': 96
}

SUB_TO_DIGIT = str.maketrans('₀₁₂₃₄₅₆₇₈₉', '0123456789')
DIGIT_TO_SUB = str.maketrans('0123456789+-', '₀₁₂₃₄₅₆₇₈₉⁺⁻')


def is_chemical_formula(s: str) -> bool:
    """Check if a string represents an actual chemical formula rather than a chemical name or text."""
    if not s or not isinstance(s, str):
        return False
    s_clean = s.strip()
    if not s_clean:
        return False
    if s_clean in KNOWN_FORMULA_ABBREVIATIONS or s_clean in CURATED_FORMULAS.values():
        return True
    if ' ' in s_clean or ',' in s_clean or ';' in s_clean:
        return False
    temp = s_clean
    for ab in ['OTf', 'TFSI', 'OAc', 'Ac']:
        temp = temp.replace(ab, '')
    if re.search(r'[a-z]{2,}', temp):
        return False
    if re.search(r'\d+-[a-zA-Z]', s_clean):
        return False
    stripped = re.sub(r'[\d₀-₉()\[\]{}+^·.*²⁺⁻/\\=\-]', '', temp)
    if not stripped:
        return False
    tokens = re.findall(r'[A-Z][a-z]?', stripped)
    if ''.join(tokens) != stripped:
        return False
    return all(tok in PERIODIC_TABLE_ELEMENTS for tok in tokens)


def mol_to_conventional_formula(mol: Any) -> str:
    """
    Convert an RDKit Mol object to a conventional chemical formula.
    - Pure organic molecules: Hill system (C, then H, then alphabetical).
    - Inorganic compounds and metal salts: Cation/metal first (Li, Na, K, Zn, etc.),
      followed by central nonmetals (S, P), then oxygen (O), hydrogen (H), halogens (Cl, Br, I, F).
    """
    from collections import Counter
    atom_counts = Counter()
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        atom_counts[sym] += 1
        h_on_atom = atom.GetTotalNumHs()
        if h_on_atom > 0:
            atom_counts['H'] += h_on_atom

    elements = list(atom_counts.keys())
    has_carbon = 'C' in elements
    has_metal = any(el in INORGANIC_METALS for el in elements)

    def fmt_elem(el: str, count: int) -> str:
        c_str = str(count) if count > 1 else ''
        return f'{el}{c_str}'

    # Case 1: Pure organic molecule (has Carbon, no Metal)
    if has_carbon and not has_metal:
        ordered = []
        if 'C' in elements:
            ordered.append('C')
        if 'H' in elements:
            ordered.append('H')
        for el in sorted([e for e in elements if e not in ('C', 'H')]):
            ordered.append(el)
        raw = ''.join(fmt_elem(el, atom_counts[el]) for el in ordered)
        return raw.translate(DIGIT_TO_SUB)

    # Case 2: Metal-containing compound (inorganic salt or organometallic)
    if has_metal:
        metals = sorted([e for e in elements if e in INORGANIC_METALS], key=lambda e: INORGANIC_ELEMENT_PRIORITY.get(e, 25))
        non_metals = [e for e in elements if e not in INORGANIC_METALS]
        if has_carbon:
            ordered_nm = []
            if 'C' in non_metals:
                ordered_nm.append('C')
            if 'H' in non_metals:
                ordered_nm.append('H')
            for el in sorted([e for e in non_metals if e not in ('C', 'H')]):
                ordered_nm.append(el)
            ordered = metals + ordered_nm
        else:
            ordered = metals + sorted(non_metals, key=lambda e: INORGANIC_ELEMENT_PRIORITY.get(e, 85))

        raw = ''.join(fmt_elem(el, atom_counts[el]) for el in ordered)
        return raw.translate(DIGIT_TO_SUB)

    # Case 3: Inorganic non-metal compound (no Metal, no Carbon)
    if 'N' in elements and 'H' in elements and any(x in elements for x in ('F', 'Cl', 'Br', 'I', 'S', 'O')):
        ordered = ['N', 'H'] + sorted([e for e in elements if e not in ('N', 'H')], key=lambda e: INORGANIC_ELEMENT_PRIORITY.get(e, 90))
    elif 'H' in elements and len(elements) > 1:
        other_elems = [e for e in elements if e != 'H']
        ordered = ['H'] + sorted(other_elems, key=lambda e: INORGANIC_ELEMENT_PRIORITY.get(e, 80))
    else:
        ordered = sorted(elements, key=lambda e: INORGANIC_ELEMENT_PRIORITY.get(e, 80))

    raw = ''.join(fmt_elem(el, atom_counts[el]) for el in ordered)
    return raw.translate(DIGIT_TO_SUB)


def normalize_formula_to_conventional(raw_formula: str) -> str:
    """
    Universally convert any alphabetical Hill formula string (e.g., 'ClLi', 'ClNa', 'Cl2Zn', 'O4SZn', 'MgSO4')
    into conventional IUPAC chemical notation ('LiCl', 'NaCl', 'ZnCl₂', 'ZnSO₄', 'MgSO₄').
    If raw_formula is not a chemical formula (e.g. chemical name), returns it unchanged without mangling.
    """
    if not raw_formula:
        return ''
    clean_raw = raw_formula.strip()
    if clean_raw in CURATED_FORMULAS:
        return CURATED_FORMULAS[clean_raw]
    if not is_chemical_formula(clean_raw):
        return clean_raw
    tokens = re.findall(r'([A-Z][a-z]?)([₀₁₂₃₄₅₆₇₈₉\d]*)', clean_raw)
    if not tokens or ''.join(t[0] + t[1] for t in tokens) != clean_raw:
        return clean_raw

    counts: Dict[str, int] = {}
    for el, c_str in tokens:
        c_val = int(c_str.translate(SUB_TO_DIGIT)) if c_str else 1
        counts[el] = counts.get(el, 0) + c_val

    elements = list(counts.keys())
    has_carbon = 'C' in elements
    has_metal = any(el in INORGANIC_METALS for el in elements)

    def fmt_el(el: str, c: int) -> str:
        cnt_s = str(c) if c > 1 else ''
        return el + cnt_s

    if has_carbon and not has_metal:
        ordered = []
        if 'C' in elements:
            ordered.append('C')
        if 'H' in elements:
            ordered.append('H')
        for el in sorted([e for e in elements if e not in ('C', 'H')]):
            ordered.append(el)
        raw = ''.join(fmt_el(el, counts[el]) for el in ordered)
        return raw.translate(DIGIT_TO_SUB)

    if has_metal:
        metals = sorted([e for e in elements if e in INORGANIC_METALS], key=lambda e: INORGANIC_ELEMENT_PRIORITY.get(e, 25))
        non_metals = [e for e in elements if e not in INORGANIC_METALS]
        if has_carbon:
            ordered_nm = []
            if 'C' in non_metals:
                ordered_nm.append('C')
            if 'H' in non_metals:
                ordered_nm.append('H')
            for el in sorted([e for e in non_metals if e not in ('C', 'H')]):
                ordered_nm.append(el)
            ordered = metals + ordered_nm
        else:
            ordered = metals + sorted(non_metals, key=lambda e: INORGANIC_ELEMENT_PRIORITY.get(e, 85))
        raw = ''.join(fmt_el(el, counts[el]) for el in ordered)
        return raw.translate(DIGIT_TO_SUB)

    if 'N' in elements and 'H' in elements and any(x in elements for x in ('F', 'Cl', 'Br', 'I', 'S', 'O')):
        ordered = ['N', 'H'] + sorted([e for e in elements if e not in ('N', 'H')], key=lambda e: INORGANIC_ELEMENT_PRIORITY.get(e, 90))
    elif 'H' in elements and len(elements) > 1:
        other_elems = [e for e in elements if e != 'H']
        ordered = ['H'] + sorted(other_elems, key=lambda e: INORGANIC_ELEMENT_PRIORITY.get(e, 80))
    else:
        ordered = sorted(elements, key=lambda e: INORGANIC_ELEMENT_PRIORITY.get(e, 80))

    raw = ''.join(fmt_el(el, counts[el]) for el in ordered)
    return raw.translate(DIGIT_TO_SUB)


def get_chemical_formula(smiles: Optional[str] = None, name: Optional[str] = None) -> str:
    """Return conventional standard chemical / molecular formula with Unicode subscripts."""
    if name:
        if name in CURATED_FORMULAS:
            return CURATED_FORMULAS[name]
        name_lower = name.strip().lower()
        for k, v in CURATED_FORMULAS.items():
            if k.lower() == name_lower:
                return v
        clean_base = re.sub(r'\s*\([^)]*\)\s*$', '', name).strip()
        if clean_base and clean_base.lower() != name_lower:
            for k, v in CURATED_FORMULAS.items():
                if k.lower() == clean_base.lower():
                    return v
        if is_chemical_formula(name):
            norm = normalize_formula_to_conventional(name)
            if norm:
                return norm
    if smiles:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                return mol_to_conventional_formula(mol)
        except Exception:
            pass
    return ""


def format_chem_formula(formula: str) -> str:
    """Convert chemical formulas and unit strings to proper Unicode subscripts and superscripts."""
    if not formula:
        return formula
    res = clean_scientific_text(str(formula))
    if is_chemical_formula(res):
        norm = normalize_formula_to_conventional(res)
        if norm and norm != res:
            res = norm
    replacements = [
        # Complex coordination species
        ("[ZnClx(H2O)y]2-x", "[ZnClₓ(H₂O)ᵧ]²⁻ˣ"),
        ("[ZnCl4]2-", "[ZnCl₄]²⁻"),
        ("[ZnCl4]^2-", "[ZnCl₄]²⁻"),
        ("[ZnCl4]²⁻", "[ZnCl₄]²⁻"),
        ("[NR4][TFSI]", "[NR₄][TFSI⁻]"),
        ("[NR4]", "[NR₄]"),
        ("Zn4SO4(OH)6·xH2O", "Zn₄SO₄(OH)₆·xH₂O"),
        ("Zn4SO4(OH)6", "Zn₄SO₄(OH)₆"),
        ("Zn(OH)2", "Zn(OH)₂"),
        ("Zn5(OH)8Cl2·H2O", "Zn₅(OH)₈Cl₂·H₂O"),
        ("Zn5(OH)8Cl2", "Zn₅(OH)₈Cl₂"),
        ("ZnF2", "ZnF₂"),
        ("V2O5", "V₂O₅"),
        ("MnO2", "MnO₂"),
        ("NH4V4O10", "NH₄V₄O₁₀"),
        ("Na3V2(PO4)3", "Na₃V₂(PO₄)₃"),
        ("H2O", "H₂O"),
        ("H2O2", "H₂O₂"),
        ("H2", "H₂"),
        ("O2", "O₂"),
        ("OH-", "OH⁻"),
        ("H+", "H⁺"),
        ("Zn2+", "Zn²⁺"),
        ("Cu2+", "Cu²⁺"),
        ("Mn2+", "Mn²⁺"),
        ("Ni2+", "Ni²⁺"),
        ("Co2+", "Co²⁺"),
        ("Fe2+", "Fe²⁺"),
        ("Fe3+", "Fe³⁺"),
        ("Li+", "Li⁺"),
        ("Na+", "Na⁺"),
        ("K+", "K⁺"),
        ("Cl-", "Cl⁻"),
        ("Br-", "Br⁻"),
        ("I-", "I⁻"),
        ("F-", "F⁻"),
        ("SO4^2-", "SO₄²⁻"),
        ("SO4 2-", "SO₄²⁻"),
        ("SO42-", "SO₄²⁻"),
        ("SO4", "SO₄"),
        ("ClO4^-", "ClO₄⁻"),
        ("ClO4 -", "ClO₄⁻"),
        ("ClO4-", "ClO₄⁻"),
        ("ClO4", "ClO₄"),
        ("TFSI^-", "TFSI⁻"),
        ("TFSI -", "TFSI⁻"),
        ("TFSI-", "TFSI⁻"),
        ("OTf^-", "OTf⁻"),
        ("OTf -", "OTf⁻"),
        ("OTf-", "OTf⁻"),
        ("CH3COO^-", "CH₃COO⁻"),
        ("CH3COO -", "CH₃COO⁻"),
        ("CH3COO-", "CH₃COO⁻"),
        ("CH3COO", "CH₃COO"),
        ("OAc^-", "OAc⁻"),
        ("OAc -", "OAc⁻"),
        ("OAc-", "OAc⁻"),
        ("OAc", "OAc"),
        ("Zn4SO4", "Zn₄SO₄"),
        # Salts
        ("Zn(CH3COO)2", "Zn(CH₃COO)₂"),
        ("Zn(CH3COO)₂", "Zn(CH₃COO)₂"),
        ("Zn(CH₃COO)2", "Zn(CH₃COO)₂"),
        ("Zn(OAc)2", "Zn(OAc)₂"),
        ("Zn(OAc)₂", "Zn(OAc)₂"),
        ("Zn(ClO4)2", "Zn(ClO₄)₂"),
        ("Zn(ClO4)₂", "Zn(ClO₄)₂"),
        ("Zn(ClO₄)2", "Zn(ClO₄)₂"),
        ("Zn(TFSI)2", "Zn(TFSI)₂"),
        ("Zn(TFSI)₂", "Zn(TFSI)₂"),
        ("Zn(OTf)2", "Zn(OTf)₂"),
        ("Zn(OTf)₂", "Zn(OTf)₂"),
        ("ZnSO4", "ZnSO₄"),
        ("ZnCl2", "ZnCl₂"),
        ("ZnF2-rich", "ZnF₂-rich"),
        ("ZnF2", "ZnF₂"),
        # Standalone ClO4 / CH3COO / SO4
        ("ClO4-", "ClO₄⁻"),
        ("ClO4^-", "ClO₄⁻"),
        ("ClO4", "ClO₄"),
        ("CH3COO-", "CH₃COO⁻"),
        ("CH3COO^-", "CH₃COO⁻"),
        ("CH3COO", "CH₃COO"),
        ("SO4^2-", "SO₄²⁻"),
        ("SO42-", "SO₄²⁻"),
        ("Zn2+", "Zn²⁺"),
        ("Zn 2+", "Zn²⁺"),
        ("Zn^2+", "Zn²⁺"),
        ("CuSO4", "CuSO₄"),
        ("MnSO4", "MnSO₄"),
        ("NiSO4", "NiSO₄"),
        ("CoSO4", "CoSO₄"),
        ("FeSO4", "FeSO₄"),
        ("MgSO4", "MgSO₄"),
        ("LiCl", "LiCl"),
        ("NaCl", "NaCl"),
        ("KCl", "KCl"),
        ("NH4Cl", "NH₄Cl"),
        ("CaCl2", "CaCl₂"),
        ("BaCl2", "BaCl₂"),
        ("NaF", "NaF"),
        ("KBr", "KBr"),
        ("LiI", "LiI"),
        ("LiBr", "LiBr"),
    ]
    for old, new in replacements:
        res = res.replace(old, new)
    res = re.sub(r'Zn\s*\^?2\s*\+', 'Zn²⁺', res)
    res = re.sub(r'Zn\s*2\s*\^\s*\+', 'Zn²⁺', res)
    res = re.sub(r'(?<![A-Za-z0-9])SO4(?![A-Za-z0-9])', 'SO₄', res)
    res = re.sub(r'\bZn\s*\(\s*CH\s*[3₃]\s*COO\s*\)\s*[2₂]', 'Zn(CH₃COO)₂', res)
    res = re.sub(r'\bZn\s*\(\s*ClO\s*[4₄]\s*\)\s*[2₂]', 'Zn(ClO₄)₂', res)
    res = re.sub(r'(?<![A-Za-z0-9])CH3COO(?![A-Za-z0-9])', 'CH₃COO', res)
    res = re.sub(r'(?<![A-Za-z0-9])ClO4(?![A-Za-z0-9])', 'ClO₄', res)
    return res


def normalize_condition_key(name: str, conc: Optional[float] = None, salt: Optional[str] = None) -> str:
    """
    Generate a canonical condition-aware compound key:
    (Normalized Chemical Name, Rounded Concentration, Normalized Baseline Salt).

    Treats identical additive names with different concentrations (e.g. 10 mM vs 50 mM)
    or different baseline electrolytes (e.g. 2 M ZnSO4 vs 1 M Zn(OTf)2) as distinct
    formulation chemistries.
    """
    clean_name = re.sub(r'[\s_\-\(\)\[\]/\\.,;:\"\'#]', '', str(name or '').lower())

    # Normalize concentration
    if conc is None or str(conc).strip().lower() in ("", "none", "unspecified", "not reported"):
        conc_val = "unspecified"
    else:
        try:
            conc_val = round(float(conc), 1)
        except (ValueError, TypeError):
            conc_val = "unspecified"

    # Normalize salt
    raw_salt = str(salt or '').strip()
    if not raw_salt or 'unspecified' in raw_salt.lower() or 'not reported' in raw_salt.lower():
        clean_salt = 'unspecified'
    else:
        clean_salt = re.sub(r'[\s_\-\(\)\[\]/\\.,;:\"\'#]', '', raw_salt.lower())
        # Canonicalize common representations
        if 'znso4' in clean_salt or 'zincsulfate' in clean_salt:
            m_match = re.search(r'(\d+(?:\.\d+)?)m', clean_salt)
            m_val = f"{float(m_match.group(1)):g}m" if m_match else "2m"
            clean_salt = f"{m_val}znso4"
        elif 'znotf' in clean_salt or 'triflate' in clean_salt:
            m_match = re.search(r'(\d+(?:\.\d+)?)m', clean_salt)
            m_val = f"{float(m_match.group(1)):g}m" if m_match else "1m"
            clean_salt = f"{m_val}znotf2"
        elif 'zntfsi' in clean_salt or 'tfsi' in clean_salt:
            m_match = re.search(r'(\d+(?:\.\d+)?)m', clean_salt)
            m_val = f"{float(m_match.group(1)):g}m" if m_match else "4m"
            clean_salt = f"{m_val}zntfsi2"
        elif 'znclo4' in clean_salt or 'perchlorate' in clean_salt:
            m_match = re.search(r'(\d+(?:\.\d+)?)m', clean_salt)
            m_val = f"{float(m_match.group(1)):g}m" if m_match else "2m"
            clean_salt = f"{m_val}znclo42"
        elif 'zncl2' in clean_salt or 'chloride' in clean_salt:
            m_match = re.search(r'(\d+(?:\.\d+)?)m', clean_salt)
            m_val = f"{float(m_match.group(1)):g}m" if m_match else "1m"
            clean_salt = f"{m_val}zncl2"

    return f"{clean_name}::{conc_val}::{clean_salt}"

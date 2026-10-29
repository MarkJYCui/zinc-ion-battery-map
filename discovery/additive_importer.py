"""
Additive List Importer & Batch Processor for Autonomous ZIB Materials Acceleration Platform (MAP).
Module 1: Materials Acceleration Platform (MAP) for Zinc-Ion Batteries (ZIB).

Supports:
- File formats: .csv, .tsv, .xlsx, .xls, .txt, .json
- Up to 5,000 additives with sub-second local RDKit parsing
- Automatic deduplication and citation frequency tracking
- Smart column alias mapping without external AI calls
- Intelligent hybrid structure resolution:
    1. SMILES provided in file -> instant local RDKit validation (0 ms)
    2. Curated ZIB additives dictionary lookup
    3. Smart NIH PubChem REST resolution with persistent local caching
- Pure-Python template generation (.csv and .xlsx)
"""

import io
import re
import json
import logging
import zipfile
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Tuple, Optional, Callable
from pathlib import Path
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

from models import CandidateMolecule
from discovery.formula_utils import (
    get_chemical_formula,
    format_chem_formula,
    normalize_condition_key,
    convert_concentration_to_mM,
)
from discovery.similarity import CURATED_ZIB_ADDITIVES, calculate_delaney_esol
from discovery.pubchem_client import PubChemClient, canonicalize_smiles, is_valid_smiles
from export_engine import make_multi_sheet_xlsx

logger = logging.getLogger("AdditiveImporter")

# Column alias mappings (normalized lowercase without whitespace/punctuation)
COLUMN_ALIASES = {
    "name": [
        "name", "additivename", "additive", "chemicalname", "compound",
        "compoundname", "molecule", "moleculename", "substance",
        "nom", "nomdeladditif", "添加剂", "添加剂名称", "化学品", "化合物"
    ],
    "concentration": [
        "concentration", "reportedconcentration", "reportedconcentrationmm",
        "concentrationmm", "conc", "concmm", "molarity", "molaritymm", "value",
        "optimalconcentration", "concentrationrapportee", "浓度", "报告浓度", "浓度mm"
    ],
    "unit": [
        "unit", "concentrationunit", "unite", "单位"
    ],
    "salt": [
        "baselinesalt", "baselineelectrolyte", "literaturebaselineelectrolyte",
        "parentelectrolyte", "electrolyte", "salt", "baseline",
        "electrolytedebase", "seldebase", "基准电解液", "文献基准电解液", "盐", "基准盐"
    ],
    "smiles": [
        "smiles", "canonicalsmiles", "structure", "smilesstring", "smilesstructure",
        "structuresmiles", "smiles结构", "结构式"
    ],
    "formula": [
        "formula", "chemicalformula", "conventionalformula", "formule",
        "formulechimique", "化学式", "分子式"
    ],
    "mechanism": [
        "mechanism", "mechanisms", "reportedmechanisms", "function", "action",
        "mecanisme", "mecanismes", "作用机制", "机理"
    ]
}


def normalize_header(header: str) -> str:
    """Normalize a header string by removing non-alphanumeric chars and lowering case."""
    if not header:
        return ""
    return re.sub(r'[\s_\-\(\)\[\]/\\.,;:\"\'#]', '', str(header).lower())


def resolve_column_name(header: str) -> Optional[str]:
    """Map a raw column header to a canonical field key (name, concentration, salt, smiles, formula, mechanism)."""
    clean = normalize_header(header)
    if not clean:
        return None

    # Pass 1: Exact match against alias list
    for canonical, aliases in COLUMN_ALIASES.items():
        if clean in aliases:
            return canonical

    # Pass 2: Fuzzy substring match (prefer longest matching alias to avoid false positives)
    best_match = None
    best_len = 0
    for canonical, aliases in COLUMN_ALIASES.items():
        for a in aliases:
            if a in clean:
                if len(a) >= 4 and len(a) > best_len:
                    best_match = canonical
                    best_len = len(a)
    return best_match


def read_xlsx_bytes(file_bytes: bytes) -> List[List[str]]:
    """
    Parse a Microsoft Excel .xlsx workbook in pure Python with zero third-party dependencies.
    Extracts cells from the first worksheet, supporting inline strings, shared strings, and raw values.
    """
    rows_data: List[List[str]] = []
    try:
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as z:
            shared_strings = []
            if 'xl/sharedStrings.xml' in z.namelist():
                root = ET.fromstring(z.read('xl/sharedStrings.xml'))
                for si in root.iter('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}si'):
                    texts = [t.text or '' for t in si.iter('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t')]
                    shared_strings.append(''.join(texts))

            # Locate first sheet in sheet list
            sheet_files = [f for f in z.namelist() if f.startswith('xl/worksheets/sheet') and f.endswith('.xml')]
            if not sheet_files:
                return []
            sheet_files.sort()
            sheet_xml = z.read(sheet_files[0])
            root = ET.fromstring(sheet_xml)
            ns = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'

            for row in root.iter(f'{ns}row'):
                row_cells = []
                for c in row.iter(f'{ns}c'):
                    t = c.get('t')
                    val = ''
                    if t == 'inlineStr':
                        is_el = c.find(f'{ns}is')
                        if is_el is not None:
                            t_el = is_el.find(f'{ns}t')
                            val = t_el.text if (t_el is not None and t_el.text) else ''
                    elif t == 's':
                        v_el = c.find(f'{ns}v')
                        if v_el is not None and v_el.text and v_el.text.isdigit():
                            idx = int(v_el.text)
                            val = shared_strings[idx] if idx < len(shared_strings) else ''
                    else:
                        v_el = c.find(f'{ns}v')
                        val = v_el.text if (v_el is not None and v_el.text) else ''
                    row_cells.append(str(val).strip())
                if any(row_cells):
                    rows_data.append(row_cells)
    except Exception as e:
        logger.warning(f"pure-Python read_xlsx_bytes failed: {e}")
    return rows_data


def parse_additive_file(file_bytes: bytes, filename: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Parse an uploaded additive file (.csv, .tsv, .xlsx, .xls, .txt, .json).
    Returns (raw_records, notices).
    """
    notices: List[str] = []
    records: List[Dict[str, Any]] = []
    fname_lower = filename.lower().strip()

    # 1. JSON Format
    if fname_lower.endswith(".json"):
        try:
            content = json.loads(file_bytes.decode("utf-8", errors="replace"))
            items = []
            if isinstance(content, list):
                items = content
            elif isinstance(content, dict):
                items = content.get("additives") or content.get("data") or content.get("candidates") or [content]
            
            for it in items:
                if not isinstance(it, dict):
                    continue
                # Map keys
                mapped: Dict[str, Any] = {}
                for k, v in it.items():
                    can_key = resolve_column_name(k)
                    if can_key:
                        mapped[can_key] = v
                    else:
                        mapped[k] = v
                if "name" in mapped and mapped["name"]:
                    records.append(mapped)
            notices.append(f"Parsed {len(records)} JSON items from '{filename}'.")
            return records, notices
        except Exception as e:
            notices.append(f"JSON parsing error: {e}")
            return [], notices

    # 2. Excel Format (.xlsx, .xls)
    if fname_lower.endswith((".xlsx", ".xls")):
        rows = read_xlsx_bytes(file_bytes)
        if not rows and fname_lower.endswith(".xlsx"):
            try:
                # Fallback to pandas if openpyxl happened to be present
                df = pd.read_excel(io.BytesIO(file_bytes))
                rows = [df.columns.astype(str).tolist()] + df.astype(str).values.tolist()
            except Exception:
                pass

        if not rows or len(rows) < 2:
            notices.append("Excel file is empty or missing a header row with data.")
            return [], notices

        headers = [str(h).strip() for h in rows[0]]
        col_mapping = {idx: resolve_column_name(h) for idx, h in enumerate(headers)}
        name_col_idx = next((idx for idx, k in col_mapping.items() if k == "name"), 0)

        for row in rows[1:]:
            rec: Dict[str, Any] = {}
            for idx, val in enumerate(row):
                key = col_mapping.get(idx)
                if key:
                    rec[key] = val
            if "name" not in rec and len(row) > name_col_idx:
                rec["name"] = row[name_col_idx]
            if rec.get("name"):
                records.append(rec)
        notices.append(f"Parsed {len(records)} rows from Excel file '{filename}'.")
        return records, notices

    # 3. CSV / TSV / Delimited TXT
    try:
        text = file_bytes.decode("utf-8-sig", errors="replace")
    except Exception:
        text = file_bytes.decode("latin-1", errors="replace")

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        notices.append("File is empty.")
        return [], notices

    # Check if lines have delimiters (comma, tab, semicolon)
    first_line = lines[0]
    has_delimiter = any(d in first_line for d in (',', '\t', ';'))

    if has_delimiter:
        try:
            df = pd.read_csv(io.StringIO(text), sep=None, engine='python', on_bad_lines='skip')
            col_mapping = {col: resolve_column_name(col) for col in df.columns}
            name_col = next((col for col, k in col_mapping.items() if k == "name"), None)
            if not name_col:
                # If first column has text, assume it's the additive name
                name_col = df.columns[0]
                col_mapping[name_col] = "name"

            for _, row in df.iterrows():
                rec = {}
                for col in df.columns:
                    k = col_mapping.get(col)
                    val = row[col]
                    if pd.notna(val):
                        rec[k if k else col] = str(val).strip()
                if rec.get("name"):
                    records.append(rec)
            notices.append(f"Parsed {len(records)} tabular records from '{filename}'.")
            return records, notices
        except Exception as e:
            notices.append(f"Delimited table parsing fallback: {e}")

    # 4. Fallback for plain .txt (one additive per line)
    for line in lines:
        if line.startswith("#") or line.lower().startswith("additive name"):
            continue
        parts = [p.strip() for p in re.split(r'[,;\t]', line) if p.strip()]
        if not parts:
            continue
        name_val = parts[0]
        conc_val = None
        salt_val = "Unspecified / Not reported"
        smi_val = ""
        if len(parts) > 1:
            try:
                conv_v, _ = convert_concentration_to_mM(parts[1], raw_text=parts[1])
                if conv_v is not None:
                    conc_val = conv_v
                else:
                    c_matches = re.findall(r'[\d.]+', parts[1])
                    if c_matches:
                        conc_val = float(c_matches[0])
            except Exception:
                conc_val = None
        if len(parts) > 2 and parts[2].strip():
            salt_val = parts[2].strip()
        if len(parts) > 3:
            smi_val = parts[3].strip()
        records.append({
            "name": name_val,
            "concentration": conc_val,
            "salt": salt_val,
            "smiles": smi_val,
        })

    notices.append(f"Parsed {len(records)} additives line-by-line from '{filename}'.")
    return records, notices


def deduplicate_and_resolve_additives(
    raw_records: List[Dict[str, Any]],
    max_limit: int = 5000,
    resolve_pubchem: bool = True,
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> Tuple[List[CandidateMolecule], Dict[str, Any]]:
    """
    Deduplicate raw additive records, enforce the 5,000 capacity guardrail,
    and resolve chemistry using RDKit + local ZIB database + cached PubChem.
    Returns (candidate_molecules, summary_stats).
    """
    pubchem_client = PubChemClient() if resolve_pubchem else None
    curated_map = {re.sub(r'[\s_\-\(\)\[\]/\\.,;:\"\'#]', '', a["name"].lower()): a for a in CURATED_ZIB_ADDITIVES}

    unique_candidates: Dict[str, Dict[str, Any]] = {}
    citation_counts: Dict[str, int] = {}
    total_parsed = len(raw_records)

    for rec in raw_records:
        raw_name = str(rec.get("name", "")).strip()
        if not raw_name or len(raw_name) < 2:
            continue

        raw_conc = rec.get("concentration")
        raw_unit = str(rec.get("unit", "")).strip()
        conc_val = None
        unit_val = raw_unit or "mM"
        if raw_conc is not None:
            try:
                conv_v, conv_u = convert_concentration_to_mM(
                    raw_conc,
                    unit=raw_unit if raw_unit else None,
                    raw_text=str(raw_conc),
                )
                if conv_v is not None:
                    conc_val = conv_v
                    unit_val = conv_u
                else:
                    conc_m = re.findall(r'[\d.]+', str(raw_conc))
                    if conc_m:
                        conc_val = float(conc_m[0])
            except Exception:
                conc_val = None

        raw_salt_in = str(rec.get("salt", "")).strip()
        if not raw_salt_in or "unspecified" in raw_salt_in.lower() or "not reported" in raw_salt_in.lower():
            raw_salt = "Unspecified / Not reported"
        else:
            raw_salt = raw_salt_in

        raw_smiles = str(rec.get("smiles", "")).strip()
        raw_formula = str(rec.get("formula", "")).strip()
        raw_mech = str(rec.get("mechanism", "")).strip()

        cond_key = normalize_condition_key(raw_name, conc_val, raw_salt)
        citation_counts[cond_key] = citation_counts.get(cond_key, 0) + 1

        if cond_key in unique_candidates:
            continue

        unique_candidates[cond_key] = {
            "name": raw_name,
            "conc": conc_val,
            "unit": unit_val,
            "salt": raw_salt,
            "smiles": raw_smiles,
            "formula": raw_formula,
            "mechanism": raw_mech,
        }

    # Apply 5,000 capacity guardrail
    all_keys = list(unique_candidates.keys())
    truncated = False
    if len(all_keys) > max_limit:
        all_keys = all_keys[:max_limit]
        truncated = True

    candidate_molecules: List[CandidateMolecule] = []
    pubchem_lookups_count = 0

    for idx, key in enumerate(all_keys):
        item = unique_candidates[key]
        c_name = item["name"]
        if progress_callback:
            try:
                progress_callback(idx + 1, len(all_keys), c_name)
            except Exception:
                pass

        c_smiles = item["smiles"]
        c_formula = item["formula"]
        c_salt = item["salt"]
        c_conc = item["conc"]
        c_unit = item["unit"]
        c_mech = item["mechanism"]
        citations = citation_counts.get(key, 1)

        # 1. Structure Resolution
        mol = None
        if c_smiles and is_valid_smiles(c_smiles):
            mol = Chem.MolFromSmiles(c_smiles)
            c_smiles = Chem.MolToSmiles(mol, canonical=True) if mol else c_smiles

        # 2. Curated ZIB Additives lookup
        norm_name_lookup = re.sub(r'[\s_\-\(\)\[\]/\\.,;:\"\'#]', '', c_name.lower())
        if mol is None and norm_name_lookup in curated_map:
            c_smiles = curated_map[norm_name_lookup]["smiles"]
            mol = Chem.MolFromSmiles(c_smiles)

        # 2b. Check without parenthetical abbreviation (e.g. "Polyethylene Glycol (PEG)" -> "Polyethylene Glycol")
        if mol is None:
            clean_base_norm = re.sub(r'[\s_\-\(\)\[\]/\\.,;:\"\'#]', '', re.sub(r'\s*\([^)]*\)\s*$', '', c_name).lower())
            if clean_base_norm in curated_map:
                c_smiles = curated_map[clean_base_norm]["smiles"]
                mol = Chem.MolFromSmiles(c_smiles)

        # 3. PubChem cached resolution for unfamiliar names
        if mol is None and pubchem_client:
            resolved_smi = pubchem_client.name_to_smiles(c_name)
            if resolved_smi and is_valid_smiles(resolved_smi):
                c_smiles = resolved_smi
                mol = Chem.MolFromSmiles(resolved_smi)
                pubchem_lookups_count += 1

        # 4. Chemoinformatic property calculation via RDKit
        if mol is not None:
            mw = round(float(Descriptors.MolWt(mol)), 2)
            logp = round(float(Descriptors.MolLogP(mol)), 2)
            tpsa = round(float(Descriptors.TPSA(mol)), 2)
            hbd = int(Lipinski.NumHDonors(mol))
            hba = int(Lipinski.NumHAcceptors(mol))
            rot_bonds = int(Lipinski.NumRotatableBonds(mol))
            esol = calculate_delaney_esol(mol)
            if not c_formula:
                c_formula = get_chemical_formula(c_smiles, c_name)
            compat_note = "Aqueous Zn2+ electrolyte compatible"
        else:
            mw = 0.0
            logp = 0.0
            tpsa = 0.0
            hbd = 0
            hba = 0
            rot_bonds = 0
            esol = None
            if not c_formula:
                c_formula = get_chemical_formula("", c_name)
            compat_note = "Unassessed (chemical structure not provided)"

        notes = f"User uploaded additive. Reported in {citations} citation(s)."
        if citations > 1:
            notes += f" Highly cited chemistry ({citations}x mentions)."

        mechs = [m.strip() for m in re.split(r'[,;]', c_mech) if m.strip()] if c_mech else []
        mech_summary = ", ".join(mechs) if mechs else "Unspecified / Not reported"

        if c_conc is not None and mw > 0 and c_unit in ("g/L", "mg/mL", "mg/L", "µg/mL", "ug/mL"):
            c_conc, c_unit = convert_concentration_to_mM(c_conc, c_unit, mw=mw)

        cand = CandidateMolecule(
            name=c_name,
            smiles=c_smiles or "",
            similarity_score=1.0,
            similarity_metric="User Uploaded Additive",
            molecular_weight=mw,
            logp=logp,
            logs=esol,
            tpsa=tpsa,
            hbd=hbd,
            hba=hba,
            rotatable_bonds=rot_bonds,
            rationale=notes,
            sdl_formulation_notes=f"Uploaded formulation candidate. Baseline: {c_salt}",
            parent_baseline_electrolyte=c_salt,
            electrolyte_compatibility_note=compat_note,
            chemical_formula=c_formula,
            reported_concentration=c_conc,
            reported_concentration_unit=c_unit if c_conc is not None else None,
            reported_mechanisms=mechs,
            mechanism_summary=mech_summary,
            is_literature_additive=True,
            citation_count=citations,
        )
        candidate_molecules.append(cand)

    stats = {
        "total_records_parsed": total_parsed,
        "unique_additives_count": len(candidate_molecules),
        "truncated": truncated,
        "max_limit": max_limit,
        "pubchem_resolved": pubchem_lookups_count,
    }
    return candidate_molecules, stats


def generate_template_csv() -> bytes:
    """Generate standard CSV template for user additive uploads."""
    rows = [
        "Additive Name,Reported Concentration (mM),Baseline Electrolyte,SMILES,Chemical Formula,Mechanisms",
        "Trehalose,25.0,2 M ZnSO4,C(C1C(C(C(C(O1)OC2C(C(C(C(O2)CO)O)O)O)O)O)O)O,C12H22O11,Solvation sheath modulation; Dendrite suppression",
        "Glycine,30.0,2 M ZnSO4,C(C(=O)O)N,C2H5NO2,SEI film formation; Corrosion inhibition",
        "Thiourea,10.0,2 M ZnSO4,C(=S)(N)N,CH4N2S,Zincophilic adsorption; HER suppression",
        "D-Sorbitol,20.0,2 M ZnSO4,C(C(C(C(C(CO)O)O)O)O)O,C6H14O6,Hydrogen bond network reconstruction",
    ]
    return "\n".join(rows).encode("utf-8")


def generate_template_xlsx() -> bytes:
    """Generate standard Excel .xlsx template for user additive uploads."""
    sheet_data = [
        ["Additive Name", "Reported Concentration (mM)", "Baseline Electrolyte", "SMILES", "Chemical Formula", "Mechanisms"],
        ["Trehalose", 25.0, "2 M ZnSO4", "C(C1C(C(C(C(O1)OC2C(C(C(C(O2)CO)O)O)O)O)O)O)O", "C12H22O11", "Solvation sheath modulation; Dendrite suppression"],
        ["Glycine", 30.0, "2 M ZnSO4", "C(C(=O)O)N", "C2H5NO2", "SEI film formation; Corrosion inhibition"],
        ["Thiourea", 10.0, "2 M ZnSO4", "C(=S)(N)N", "CH4N2S", "Zincophilic adsorption; HER suppression"],
        ["D-Sorbitol", 20.0, "2 M ZnSO4", "C(C(C(C(C(CO)O)O)O)O)O", "C6H14O6", "Hydrogen bond network reconstruction"],
    ]
    return make_multi_sheet_xlsx({"ZIB_Additives_Template": sheet_data})

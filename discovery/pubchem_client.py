"""
PubChem REST PUG Client & Dynamic Chemical Exploration.
Module 1: Materials Acceleration Platform (MAP) for Zinc-Ion Batteries (ZIB).

Inspired by ChemCrow:
Portions of this code are adapted from ChemCrow:
MIT License
Copyright (c) 2023 White Laboratory
(See third_party_licenses/ChemCrow-LICENSE.txt)

Provides chemical name-to-SMILES resolution and dynamic 2D similarity searches
against the NCBI PubChem database to discover structural analogues of electrolyte additives.
"""

import json
import logging
import re
from pathlib import Path
from functools import lru_cache
from typing import List, Dict, Any, Optional
import requests
from rdkit import Chem

logger = logging.getLogger("PubChemClient")
CACHE_FILE = Path(__file__).resolve().parent / "pubchem_cache.json"
_PUBCHEM_LOCAL_CACHE: Dict[str, str] = {}


def _get_pubchem_cache() -> Dict[str, str]:
    global _PUBCHEM_LOCAL_CACHE
    if not _PUBCHEM_LOCAL_CACHE and CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                _PUBCHEM_LOCAL_CACHE = json.load(f)
        except Exception:
            _PUBCHEM_LOCAL_CACHE = {}
    return _PUBCHEM_LOCAL_CACHE


def _cache_pubchem_entry(name: str, smiles: str):
    global _PUBCHEM_LOCAL_CACHE
    cache = _get_pubchem_cache()
    cache[name.lower().strip()] = smiles
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
    except Exception:
        pass


def is_valid_smiles(text: str) -> bool:
    """Validate whether a string is a valid chemical SMILES representation."""
    if not text or not isinstance(text, str):
        return False
    try:
        mol = Chem.MolFromSmiles(text.strip(), sanitize=False)
        return mol is not None
    except Exception:
        return False


def canonicalize_smiles(smiles: str) -> Optional[str]:
    """Convert SMILES to canonical representation."""
    if not is_valid_smiles(smiles):
        return None
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol:
            return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        pass
    return smiles.strip()


class PubChemClient:
    """
    Client for querying the NCBI PubChem PUG REST API.
    Enables live chemical entity resolution and 2D similarity searches.
    Includes deterministic timeouts and offline fallback handling.
    """

    PUG_BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

    def __init__(self, timeout: float = 3.5, session: Optional[requests.Session] = None):
        self.timeout = timeout
        self.session = session or requests.Session()
        self.session.headers.update({
            "User-Agent": "Autonomous-ZIB-MAP/2.0 (Chemical-Discovery-Engine; contact: research@materials-map.org)",
            "Accept": "application/json"
        })

    def name_to_smiles(self, chemical_name: str) -> Optional[str]:
        """
        Query PubChem to resolve a chemical name into a canonical isomeric SMILES string.
        First checks persistent local disk cache to avoid redundant network queries.
        Returns None if not found, invalid, or offline.
        """
        if not chemical_name or not chemical_name.strip():
            return None

        clean_name = chemical_name.strip()
        # If it's already a valid SMILES, return its canonical form
        if is_valid_smiles(clean_name) and not re.search(r'\s', clean_name):
            return canonicalize_smiles(clean_name)

        # Check local disk cache
        cache = _get_pubchem_cache()
        if clean_name.lower() in cache:
            return cache[clean_name.lower()]

        url = f"{self.PUG_BASE_URL}/compound/name/{requests.utils.quote(clean_name)}/property/IsomericSMILES,CanonicalSMILES,SMILES,MolecularWeight,IUPACName/JSON"
        try:
            resp = self.session.get(url, timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                props = data.get("PropertyTable", {}).get("Properties", [])
                if props:
                    smi = props[0].get("IsomericSMILES") or props[0].get("CanonicalSMILES") or props[0].get("SMILES")
                    if smi:
                        can_smi = canonicalize_smiles(smi)
                        if can_smi:
                            _cache_pubchem_entry(clean_name, can_smi)
                            return can_smi
        except (requests.RequestException, ValueError, KeyError) as e:
            logger.debug(f"PubChem name_to_smiles query failed for '{clean_name}': {e}")

        return None

    def smiles_to_name(self, smiles: str) -> Optional[str]:
        """
        Query PubChem to resolve a SMILES string to its primary common chemical name or IUPAC synonym.
        """
        canon_smi = canonicalize_smiles(smiles)
        if not canon_smi:
            return None

        url = f"{self.PUG_BASE_URL}/compound/smiles/{requests.utils.quote(canon_smi)}/synonyms/JSON"
        try:
            resp = self.session.get(url, timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                info = data.get("InformationList", {}).get("Information", [])
                if info and "Synonym" in info[0] and info[0]["Synonym"]:
                    synonyms = info[0]["Synonym"]
                    # Pick first non-CAS, non-catalog ID synonym if possible
                    for syn in synonyms:
                        if not re.match(r'^\d{2,7}-\d{2}-\d$', syn) and not syn.startswith("NSC") and not syn.startswith("CHEMBL"):
                            return syn
                    return synonyms[0]
        except (requests.RequestException, ValueError, KeyError) as e:
            logger.debug(f"PubChem smiles_to_name query failed for '{smiles}': {e}")

        return None

    def search_similar_compounds(
        self,
        query_smiles: str,
        threshold: int = 70,
        max_records: int = 8
    ) -> List[Dict[str, Any]]:
        """
        Perform 2D fast similarity search on PubChem using Tanimoto threshold (percentage 0-100).
        Returns a list of candidate dictionaries with keys:
            name, smiles, cid, molecular_weight, iupac_name, source
        """
        canon_smi = canonicalize_smiles(query_smiles)
        if not canon_smi:
            return []

        url = (
            f"{self.PUG_BASE_URL}/compound/fastsimilarity_2d/smiles/{requests.utils.quote(canon_smi)}"
            f"/property/Title,IsomericSMILES,CanonicalSMILES,SMILES,MolecularWeight,IUPACName/JSON"
            f"?Threshold={threshold}&MaxRecords={max_records}"
        )

        candidates = []
        try:
            resp = self.session.get(url, timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                props = data.get("PropertyTable", {}).get("Properties", [])
                for item in props:
                    smi = item.get("IsomericSMILES") or item.get("CanonicalSMILES") or item.get("SMILES")
                    if not smi:
                        continue
                    c_smi = canonicalize_smiles(smi)
                    if not c_smi:
                        continue

                    cid_val = item.get("CID")
                    cand_name = item.get("Title") or item.get("IUPACName") or (f"PubChem CID {cid_val}" if cid_val else "PubChem Candidate")
                    mw = float(item.get("MolecularWeight", 0.0))

                    candidates.append({
                        "name": cand_name,
                        "smiles": c_smi,
                        "cid": cid_val,
                        "molecular_weight": mw,
                        "iupac_name": item.get("IUPACName"),
                        "source": "PubChem Dynamic Discovery (ChemCrow API)"
                    })
        except (requests.RequestException, ValueError, KeyError) as e:
            logger.debug(f"PubChem fast similarity search failed for '{query_smiles}': {e}")

        return candidates

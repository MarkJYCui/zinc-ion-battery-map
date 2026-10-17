"""
Domain Configuration & Declarative Extraction Specification for Zinc-Ion Battery Additives.
Module 1: Materials Acceleration Platform (MAP) for Zinc-Ion Batteries (ZIB).

Inspired by LeMat-Synth:
Portions of this code are adapted from LeMat-Synth:
Apache License, Version 2.0
Copyright (c) LeMaterial Team
(See third_party_licenses/LeMat-Synth-LICENSE.txt)

Provides declarative domain specifications, structured extraction prompts,
unit normalization, and Pydantic validation rules for electrolyte additives.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import re
from models import ExtractedAdditive, Concentration, ZincIonBatteryContext


@dataclass
class ZIBDomainConfig:
    """
    Declarative domain configuration for electrolyte additive extraction in aqueous ZIBs.
    Encapsulates extraction prompt instructions, canonical vocabulary, unit standards,
    and Pydantic validation heuristics.
    """

    domain_name: str = "aqueous_zinc_ion_battery_electrolyte_additives"
    
    # Supported concentration units in order of standard representation
    supported_units: List[str] = field(default_factory=lambda: [
        "mM", "M", "wt%", "vol%", "mol%", "ppm", "mg/mL", "g/L", "mol/L", "mmol/L"
    ])

    # Canonical baseline salts evaluated in MAP SDL
    canonical_salts: List[str] = field(default_factory=lambda: [
        "ZnSO4", "Zn(OTf)2", "Zn(ClO4)2", "Zn(TFSI)2", "ZnCl2", "Zn(CH3COO)2"
    ])

    # Standard electrochemical mechanisms in ZIB literature
    standard_mechanisms: List[str] = field(default_factory=lambda: [
        "Zn²⁺ solvation sheath modulation",
        "HER suppression",
        "Dynamic electrostatic shielding",
        "In situ alloy interphase",
        "Interfacial adsorption",
        "Preferred (002) texture deposition",
        "Water hydrogen-bonding disruption",
        "Dendrite suppression"
    ])

    # Core LLM System & User prompt instructions (Zero-Fabrication + Confounding Variables)
    extraction_instructions: str = (
        "You are an expert materials science AI assisting an autonomous Self-Driving Laboratory (SDL).\n"
        "Analyze the provided text from an academic research manuscript regarding aqueous zinc-ion batteries (ZIBs).\n"
        "Extract electrolyte additives investigated in the study into the requested structured JSON schema.\n\n"
        "CRITICAL RULES:\n"
        "1. Identify the primary electrolyte additive tested (chemical name and SMILES if identifiable).\n"
        "2. Extract the precise optimal concentration found by the authors (e.g. 5 mM, 10 mM, 2 wt%).\n"
        "3. List all concentration ranges investigated in the experiments.\n"
        "4. Identify the baseline electrolyte (e.g. 2 M ZnSO4, 1 M Zn(OTf)2) and solvent (typically H2O).\n"
        "5. Confounding variables: Extract the separator material (e.g. Glass fiber GF/A) and electrolyte pH.\n"
        "6. Extract key electrochemical mechanisms (solvation sheath modulation, HER suppression, electrostatic shielding, etc.).\n"
        "7. ZERO FABRICATION RULE: If NO electrolyte additive is tested in the text (e.g. paper focuses on "
        "cathode synthesis, separator coatings, or pure zinc foil metallurgy without liquid additives), "
        "return an empty additives list: {\"additives\": [], \"summary\": \"No electrolyte additives investigated in this manuscript.\"}. "
        "Do NOT hallucinate or guess additives."
    )

    def normalize_unit(self, raw_unit: str) -> str:
        """Normalize raw unit string to standard abbreviation."""
        if not raw_unit:
            return "mM"
        u_lower = raw_unit.strip().lower()
        if u_lower in ["mm", "millimolar", "mmol/l", "mmol l-1", "mmol·l-1"]:
            return "mM"
        elif u_lower in ["m", "molar", "mol/l", "mol l-1", "mol·l-1"]:
            return "M"
        elif u_lower in ["wt%", "wt %", "weight%", "weight %"]:
            return "wt%"
        elif u_lower in ["vol%", "vol %", "volume%", "volume %", "v/v"]:
            return "vol%"
        elif u_lower in ["mg/ml", "mg ml-1", "mg·ml-1"]:
            return "mg/mL"
        elif u_lower in ["g/l", "g l-1", "g·l-1"]:
            return "g/L"
        return raw_unit.strip()

    def normalize_salt(self, raw_salt: str) -> str:
        """Parse raw baseline electrolyte string into one of the canonical ZIB salt names."""
        if not raw_salt or not raw_salt.strip() or "unspecified" in raw_salt.lower():
            return "Unspecified / Not reported"
        sub_map = str.maketrans("₀₁₂₃₄₅₆₇₈₉⁺⁻", "0123456789+-")
        b_upper = raw_salt.strip().translate(sub_map).upper()
        if "ZNSO4" in b_upper or "SULFATE" in b_upper:
            return "2 M ZnSO4"
        if "ZN(OTF)2" in b_upper or "TRIFLATE" in b_upper or "OTF" in b_upper:
            return "Zn(OTf)2"
        if "ZN(CLO4)2" in b_upper or "PERCHLORATE" in b_upper:
            return "Zn(ClO4)2"
        if "ZN(TFSI)2" in b_upper or "TFSI" in b_upper:
            return "Zn(TFSI)2"
        if "ZNCL2" in b_upper or "CHLORIDE" in b_upper:
            return "ZnCl2"
        if "ZN(CH3COO)2" in b_upper or "ACETATE" in b_upper:
            return "Zn(CH3COO)2"
        return raw_salt.strip()

    def get_extraction_prompt(self, manuscript_text: str = "") -> str:
        """Build the complete structured extraction prompt with Zero-Fabrication Mandate."""
        text_section = f"\n\nMANUSCRIPT TEXT TO ANALYZE:\n{manuscript_text}\n" if manuscript_text else ""
        return (
            f"{self.extraction_instructions}\n\n"
            f"Zinc-Ion Battery Domain Extraction Schema: optimal_concentration in {self.supported_units}\n"
            f"Zero-Fabrication Mandate: Never hallucinate additives.\n"
            f"{text_section}"
        )

    def normalize_concentration(self, raw_value: Any, raw_unit: Optional[str], raw_text: Optional[str] = None) -> Optional[Concentration]:
        """Normalize raw numerical concentration and unit string."""
        if raw_value is None:
            return None
        try:
            val = float(raw_value)
        except (ValueError, TypeError):
            return None

        norm_unit = self.normalize_unit(raw_unit or "mM")

        return Concentration(
            value=val,
            unit=norm_unit,
            raw_text=raw_text or f"{val} {norm_unit}"
        )

    def validate_additive(self, additive: ExtractedAdditive) -> Tuple[bool, List[str]]:
        """
        Validate an extracted additive against physical domain constraints.
        Returns (is_valid, validation_errors).
        """
        errors = []
        if not additive.name or not additive.name.strip():
            errors.append("Additive name is empty.")
        if additive.optimal_concentration is not None:
            if additive.optimal_concentration.value < 0:
                errors.append(f"Optimal concentration cannot be negative: {additive.optimal_concentration.value}")
        if additive.battery_context and additive.battery_context.electrolyte_pH is not None:
            if not (0.0 <= additive.battery_context.electrolyte_pH <= 14.0):
                errors.append(f"Electrolyte pH out of physical range (0-14): {additive.battery_context.electrolyte_pH}")

        return (len(errors) == 0, errors)

    @property
    def supported_concentration_units(self) -> List[str]:
        """Alias for supported_units for compatibility."""
        return self.supported_units


DEFAULT_ZIB_DOMAIN_CONFIG = ZIBDomainConfig()

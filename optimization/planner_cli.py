"""
Command-Line Interface and JSON Orchestrator for Phase 2:
Autonomous Design of Experiments (DoE) & Electrolyte-Aware Robotic Dispensing.

Interfaces directly with Phase 1 ExtractionResult or runs standalone with user CLI flags.
Outputs machine-readable JSON schedules for robotic handlers (Opentrons OT-2, Hamilton).
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Optional, List

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from models import ExtractionResult
from optimization.concentration_anchor import ConcentrationAnchorEngine
from optimization.doe_engine import DoEEngine
from optimization.dispenser import RoboticDispenserEngine

logger = logging.getLogger(__name__)


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Autonomous DoE Formulation & Robotic Dispensing Orchestrator for Zinc-Ion Batteries."
    )

    input_group = parser.add_argument_group("Input Options (Phase 1 Integration or Manual)")
    input_group.add_argument(
        "--from-extraction",
        type=str,
        default=None,
        help="Path to Phase 1 ExtractionResult JSON file.",
    )
    input_group.add_argument(
        "--additive",
        type=str,
        default=None,
        help="Additive common or IUPAC name (e.g. 'D-Sorbitol').",
    )
    input_group.add_argument(
        "--published-conc",
        type=float,
        default=None,
        help="Published concentration in mM (e.g. 15.0).",
    )
    input_group.add_argument(
        "--parent-electrolyte",
        type=str,
        default="2 M ZnSO4",
        help="Baseline electrolyte in parent study (default: '2 M ZnSO4').",
    )
    input_group.add_argument(
        "--smiles",
        type=str,
        default=None,
        help="SMILES chemical structure string of the additive.",
    )
    input_group.add_argument(
        "--candidate-index",
        type=int,
        default=None,
        help="0-based index of CandidateMolecule in ExtractionResult to formulate.",
    )

    plan_group = parser.add_argument_group("DoE & Chemistry Planning Options")
    plan_group.add_argument(
        "--target-electrolyte",
        type=str,
        required=True,
        help="Target electrolyte system for screening (e.g. 'Zn(OTf)2', '1 M Zn(OTf)2', '5 M ZnCl2', 'Zn(TFSI)2').",
    )
    plan_group.add_argument(
        "--method",
        type=str,
        choices=["lhs", "ccd"],
        default="lhs",
        help="Design of experiments sampling methodology ('lhs' or 'ccd'). Default: 'lhs'.",
    )
    plan_group.add_argument(
        "--samples",
        type=int,
        default=9,
        help="Number of formulation samples for Latin Hypercube Sampling (default: 9).",
    )
    plan_group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for LHS sampling reproducibility (default: 42).",
    )
    plan_group.add_argument(
        "--temperature",
        type=float,
        nargs=2,
        metavar=("MIN_C", "MAX_C"),
        default=None,
        help="Include temperature as Factor 3 with bounds [MIN_C, MAX_C].",
    )

    robot_group = parser.add_argument_group("Robotic Dispenser Options")
    robot_group.add_argument(
        "--rack-format",
        type=str,
        choices=["24-vial rack", "96-well plate"],
        default="24-vial rack",
        help="Physical vessel format ('24-vial rack' or '96-well plate'). Default: '24-vial rack'.",
    )
    robot_group.add_argument(
        "--total-volume-ul",
        type=float,
        default=None,
        help="Total solution volume per vial/well in uL (default: 2000 for 24-vial rack, 200 for 96-well plate).",
    )
    robot_group.add_argument(
        "--additive-stock-conc",
        type=float,
        default=None,
        help="Concentration of additive stock solution in mM.",
    )
    robot_group.add_argument(
        "--salt-stock-conc",
        type=float,
        default=None,
        help="Concentration of zinc salt stock solution in M.",
    )

    output_group = parser.add_argument_group("Output Formatting Options")
    output_group.add_argument(
        "--json",
        action="store_true",
        help="Output the resulting DispensingSchedule strictly as JSON to stdout.",
    )
    output_group.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional file path to write the DispensingSchedule JSON.",
    )

    return parser.parse_args(args)


def run_planner(args: Optional[List[str]] = None) -> int:
    parsed = parse_args(args)

    additive_name = parsed.additive
    published_conc = parsed.published_conc
    parent_electrolyte = parsed.parent_electrolyte
    smiles = parsed.smiles

    esol_logs_override = None
    if parsed.from_extraction:
        ext_path = Path(parsed.from_extraction)
        if not ext_path.is_file():
            print(f"Error: Extraction file not found: {ext_path}", file=sys.stderr)
            return 1
        try:
            with open(ext_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            extraction = ExtractionResult.model_validate(data)
            if not extraction.additives:
                print("Error: ExtractionResult contains no additives.", file=sys.stderr)
                return 1

            parent_add = extraction.additives[0]
            published_conc = (
                parent_add.optimal_concentration.value
                if parent_add.optimal_concentration and parent_add.optimal_concentration.value is not None
                else 10.0
            )
            parent_electrolyte = (
                parent_add.battery_context.baseline_electrolyte
                if parent_add.battery_context and parent_add.battery_context.baseline_electrolyte
                else "2 M ZnSO4"
            )

            if parsed.candidate_index is not None:
                if parsed.candidate_index < 0 or parsed.candidate_index >= len(extraction.candidate_molecules):
                    print(
                        f"Error: candidate-index {parsed.candidate_index} out of range "
                        f"({len(extraction.candidate_molecules)} candidates available).",
                        file=sys.stderr,
                    )
                    return 1
                cand = extraction.candidate_molecules[parsed.candidate_index]
                additive_name = cand.name
                smiles = cand.smiles
                esol_logs_override = cand.logs
                if cand.parent_baseline_electrolyte:
                    parent_electrolyte = cand.parent_baseline_electrolyte
            elif parsed.additive:
                matched_cand = next(
                    (c for c in extraction.candidate_molecules if c.name.lower() == parsed.additive.lower()), None
                )
                matched_add = next(
                    (a for a in extraction.additives if a.name.lower() == parsed.additive.lower()), None
                )
                if matched_cand:
                    additive_name = matched_cand.name
                    smiles = matched_cand.smiles
                    esol_logs_override = matched_cand.logs
                    if matched_cand.parent_baseline_electrolyte:
                        parent_electrolyte = matched_cand.parent_baseline_electrolyte
                elif matched_add:
                    additive_name = matched_add.name
                    published_conc = (
                        matched_add.optimal_concentration.value
                        if matched_add.optimal_concentration and matched_add.optimal_concentration.value is not None
                        else published_conc
                    )
                    if matched_add.battery_context and matched_add.battery_context.baseline_electrolyte:
                        parent_electrolyte = matched_add.battery_context.baseline_electrolyte
                    smiles = matched_add.smiles
                else:
                    additive_name = parsed.additive
                    smiles = parsed.smiles
            else:
                additive_name = parent_add.name
                smiles = parent_add.smiles
        except Exception as e:
            print(f"Error parsing ExtractionResult JSON: {e}", file=sys.stderr)
            return 1
    elif not additive_name or published_conc is None:
        print(
            "Error: Either provide --from-extraction OR both --additive and --published-conc.",
            file=sys.stderr,
        )
        return 1

    anchor_engine = ConcentrationAnchorEngine()
    anchor = anchor_engine.calculate_anchor(
        additive_name=additive_name,
        published_conc_mM=published_conc,
        parent_electrolyte=parent_electrolyte,
        target_electrolyte=parsed.target_electrolyte,
        smiles=smiles,
        esol_logs_override=esol_logs_override,
    )

    doe_engine = DoEEngine()
    plan = doe_engine.create_plan(
        anchor=anchor,
        method=parsed.method,
        num_samples=parsed.samples,
        seed=parsed.seed,
        temperature_range=parsed.temperature,
    )

    dispenser = RoboticDispenserEngine()
    schedule = dispenser.generate_schedule(
        plan=plan,
        rack_format=parsed.rack_format,
        total_volume_uL=parsed.total_volume_ul,
        additive_stock_conc_mM=parsed.additive_stock_conc,
        salt_stock_conc_M=parsed.salt_stock_conc,
    )

    schedule_json = schedule.model_dump_json(indent=2)

    if parsed.output:
        try:
            out_path = Path(parsed.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(schedule_json, encoding="utf-8")
        except Exception as e:
            print(f"Error saving output to {parsed.output}: {e}", file=sys.stderr)
            return 1

    if parsed.json:
        print(schedule_json)
        return 0

    print("================================================================================")
    print("  MATERIALS ACCELERATION PLATFORM (MAP) - AUTONOMOUS DoE FORMULATION ENGINE")
    print("================================================================================")
    print(f"  Additive:             {anchor.additive_name}")
    if anchor.smiles:
        print(f"  SMILES:               {anchor.smiles}")
    print(f"  Published Baseline:   {anchor.published_concentration:.1f} mM in {anchor.parent_electrolyte}")
    print(f"  Target Electrolyte:   {anchor.target_electrolyte_salt} ({plan.target_salt_default_molarity:.1f} M standard)")
    print(f"  Chemistry Scaling:    {anchor.scaling_factor:.2f}x multiplier")
    print(f"  Shifted Anchor:       {anchor.shifted_anchor_conc:.2f} mM")
    print(f"  Exploration Bounds:   [{anchor.min_conc:.2f} mM, {anchor.max_conc:.2f} mM]")
    if anchor.solubility_ceiling_mM:
        print(f"  Solubility Ceiling:   {anchor.solubility_ceiling_mM:.2f} mM (80% Delaney ESOL Saturation)")
        if anchor.is_capped_by_solubility:
            print("                        * Upper bound TRUNCATED to prevent precipitation *")
    print(f"\n  Electrolyte Rationale:\n  {anchor.electrolyte_adjustment_rationale}")

    print("\n--------------------------------------------------------------------------------")
    print(f"  DoE METHOD: {plan.method.upper()} ({len(plan.formulations)} formulation points)")
    print("--------------------------------------------------------------------------------")
    print(
        f"  {'Vial':<6} {'Additive (mM)':<15} {'Salt (M)':<12} {'Add Stock (uL)':<16} {'Salt Stock (uL)':<16} {'Water (uL)':<12}"
    )
    print("  ----------------------------------------------------------------------------")
    for vial in schedule.vials:
        print(
            f"  {vial.vial_id:<6} {vial.additive_conc_mM:<15.2f} {vial.salt_conc_M:<12.2f} "
            f"{vial.additive_stock_uL:<16.1f} {vial.salt_stock_uL:<16.1f} {vial.water_uL:<12.1f}"
        )

    if schedule.warnings:
        print("\n  [SAFETY & LIQUID HANDLING WARNINGS]")
        for w in schedule.warnings:
            print(f"  ! {w}")
    print("================================================================================")
    return 0


if __name__ == "__main__":
    sys.exit(run_planner())

"""
Materials Acceleration Platform (MAP) for Aqueous Zinc-Ion Batteries (ZIB).
Top-level Command-Line Interface (CLI).

Commands:
    run-pipeline: End-to-end autonomous campaign (Discovery -> Formulation -> Analysis -> Reporting).
    run-discovery: Phase 1 literature search & extraction engine.
    run-formulation: Phase 2 salt-aware DoE formulation & robotic dispensing planner.
    run-analysis: Phase 3 electrochemical characterization, failure windowing & dashboard generator.
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Optional, List

from map_orchestrator import MAPOrchestrator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MAP_CLI")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="map",
        description="Autonomous Materials Acceleration Platform (MAP) for Aqueous Zinc-Ion Batteries.",
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # --------------------------------------------------------------------------
    # Subcommand: run-pipeline (End-to-End Campaign)
    # --------------------------------------------------------------------------
    pipe_parser = subparsers.add_parser(
        "run-pipeline",
        help="Execute full end-to-end autonomous MAP campaign across all three phases.",
    )
    # Input options
    pipe_in = pipe_parser.add_argument_group("Phase 1 / Additive Input Options")
    pipe_in.add_argument("--query", type=str, default=None, help="Literature search query (e.g. 'sorbitol').")
    pipe_in.add_argument("--from-extraction", type=str, default=None, help="Path to Phase 1 ExtractionResult JSON.")
    pipe_in.add_argument("--additive", type=str, default=None, help="Additive chemical name.")
    pipe_in.add_argument("--published-conc", type=float, default=None, help="Published baseline concentration (mM).")
    pipe_in.add_argument("--smiles", type=str, default=None, help="Additive SMILES chemical structure.")
    pipe_in.add_argument("--parent-electrolyte", type=str, default="2 M ZnSO4", help="Parent baseline electrolyte.")

    # Formulation options
    pipe_form = pipe_parser.add_argument_group("Phase 2 Formulation Options")
    pipe_form.add_argument(
        "--target-salt",
        type=str,
        default="Zn(OTf)2",
        help="Target electrolyte salt system (e.g. 'Zn(OTf)2', 'ZnSO4', '5 M ZnCl2', 'Zn(TFSI)2').",
    )
    pipe_form.add_argument("--doe-method", type=str, choices=["lhs", "ccd"], default="lhs", help="DoE method.")
    pipe_form.add_argument("--samples", type=int, default=9, help="Number of formulation candidates (LHS).")
    pipe_form.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of closed-loop iterations (default 1 for single pass, 2+ for adaptive active learning).",
    )
    pipe_form.add_argument(
        "--samples-per-iteration",
        type=int,
        default=3,
        help="Number of formulation samples per adaptive iteration (default 3).",
    )
    pipe_form.add_argument("--rack-format", type=str, choices=["24-vial rack", "96-well plate"], default="24-vial rack")
    pipe_form.add_argument("--total-volume-ul", type=float, default=None, help="Total volume per vial (uL).")
    pipe_form.add_argument("--seed", type=int, default=42, help="Random seed for DoE.")

    # Analysis & Ingestion options
    pipe_an = pipe_parser.add_argument_group("Phase 3 Analysis & Ingestion Options")
    pipe_an.add_argument("--cycler-data", type=str, default=None, help="Path to experimental cycler file or folder.")
    pipe_an.add_argument("--use-benchmark", action="store_true", help="Simulate benchmark cycling data across DoE points.")

    # Output options
    pipe_out = pipe_parser.add_argument_group("Output Options")
    pipe_out.add_argument("--output-dir", type=str, default="./campaign_results", help="Directory for campaign artifacts.")
    pipe_out.add_argument("--json", action="store_true", help="Print CampaignResult JSON directly to stdout.")

    # --------------------------------------------------------------------------
    # Subcommand: run-discovery (Phase 1 Only)
    # --------------------------------------------------------------------------
    disc_parser = subparsers.add_parser("run-discovery", help="Run Phase 1 literature discovery & candidate extraction.")
    disc_parser.add_argument("--query", type=str, default=None, help="Literature search query.")
    disc_parser.add_argument("--from-extraction", type=str, default=None, help="Path to pre-existing extraction JSON.")
    disc_parser.add_argument("--output", type=str, default=None, help="Output file path for ExtractionResult JSON.")
    disc_parser.add_argument("--json", action="store_true", help="Print ExtractionResult JSON directly to stdout.")

    # --------------------------------------------------------------------------
    # Subcommand: run-formulation (Phase 2 Only)
    # --------------------------------------------------------------------------
    form_parser = subparsers.add_parser("run-formulation", help="Run Phase 2 salt-aware DoE & dispensing schedule planner.")
    form_parser.add_argument("--from-extraction", type=str, default=None, help="Phase 1 ExtractionResult JSON.")
    form_parser.add_argument("--additive", type=str, default=None, help="Additive chemical name.")
    form_parser.add_argument("--published-conc", type=float, default=None, help="Published concentration in mM.")
    form_parser.add_argument("--smiles", type=str, default=None, help="SMILES chemical structure.")
    form_parser.add_argument("--parent-electrolyte", type=str, default="2 M ZnSO4", help="Parent baseline electrolyte.")
    form_parser.add_argument("--target-salt", type=str, required=True, help="Target electrolyte chemistry.")
    form_parser.add_argument("--method", type=str, choices=["lhs", "ccd"], default="lhs", help="DoE method.")
    form_parser.add_argument("--samples", type=int, default=9, help="Number of formulation points.")
    form_parser.add_argument("--rack-format", type=str, choices=["24-vial rack", "96-well plate"], default="24-vial rack")
    form_parser.add_argument("--total-volume-ul", type=float, default=None, help="Volume per vial in uL.")
    form_parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    form_parser.add_argument("--output", type=str, default=None, help="Path to save DispensingSchedule JSON.")
    form_parser.add_argument("--json", action="store_true", help="Print DispensingSchedule JSON to stdout.")

    # --------------------------------------------------------------------------
    # Subcommand: run-analysis (Phase 3 Only)
    # --------------------------------------------------------------------------
    ana_parser = subparsers.add_parser("run-analysis", help="Run Phase 3 electrochemical characterization & visualization.")
    ana_parser.add_argument("--file", type=str, required=True, help="Path to cycler data file (.csv, .mpt, .xlsx, .mpr).")
    ana_parser.add_argument("--technique", type=str, default="auto", help="Technique (auto, galvanostatic_symmetric, etc.).")
    ana_parser.add_argument("--area-cm2", type=float, default=1.13097, help="Electrode geometric area in cm2.")
    ana_parser.add_argument("--output-dir", type=str, default="./reports", help="Output directory for reports & dashboards.")
    ana_parser.add_argument("--no-html", action="store_true", help="Skip Plotly HTML dashboard generation.")
    ana_parser.add_argument("--json", action="store_true", help="Print report JSON to stdout.")

    return parser


def handle_pipeline(args: argparse.Namespace) -> int:
    orchestrator = MAPOrchestrator(output_dir=args.output_dir)
    try:
        if getattr(args, "iterations", 1) > 1:
            campaign = orchestrator.run_closed_loop_campaign(
                query=args.query,
                from_extraction=args.from_extraction,
                additive_name=args.additive,
                published_conc_mM=args.published_conc,
                smiles=args.smiles,
                parent_electrolyte=args.parent_electrolyte,
                target_salt=args.target_salt,
                doe_method=args.doe_method,
                initial_samples=args.samples,
                max_iterations=args.iterations,
                samples_per_iteration=args.samples_per_iteration,
                rack_format=args.rack_format,
                total_volume_uL=args.total_volume_ul,
                cycler_data=args.cycler_data,
                use_benchmark=args.use_benchmark,
                output_dir=args.output_dir,
                seed=args.seed,
            )
        else:
            campaign = orchestrator.run_full_campaign(
                query=args.query,
                from_extraction=args.from_extraction,
                additive_name=args.additive,
                published_conc_mM=args.published_conc,
                smiles=args.smiles,
                parent_electrolyte=args.parent_electrolyte,
                target_salt=args.target_salt,
                doe_method=args.doe_method,
                num_samples=args.samples,
                rack_format=args.rack_format,
                total_volume_uL=args.total_volume_ul,
                cycler_data=args.cycler_data,
                use_benchmark=args.use_benchmark,
                output_dir=args.output_dir,
                seed=args.seed,
            )

        if args.json:
            print(campaign.model_dump_json(indent=2))
            return 0

        print("================================================================================")
        print("  MATERIALS ACCELERATION PLATFORM (MAP) - END-TO-END CAMPAIGN RESULTS")
        print("================================================================================")
        print(f"  Campaign ID:        {campaign.campaign_id}")
        print(f"  Target Salt:        {campaign.target_salt}")
        print(f"  Output Directory:   {campaign.output_dir}")
        print(f"  Total Iterations:   {campaign.total_iterations}")
        print(f"  Formulations:       {len(campaign.cell_results)} cell candidates evaluated across {campaign.total_iterations} round(s)")
        if campaign.pareto_optimal_candidates:
            print(f"\n  [PARETO-OPTIMAL CANDIDATES ({len(campaign.pareto_optimal_candidates)} on Frontier)]")
            for pc in campaign.pareto_optimal_candidates:
                r_tag = f"Round {pc.iteration_round}"
                print(f"    - Sample {pc.sample_id} ({pc.vial_id}, {r_tag}): Additive {pc.additive_conc_mM:.2f} mM, Salt {pc.salt_conc_M:.2f} M | Score: {pc.performance_score:.2f}")

        if campaign.best_candidate:
            bc = campaign.best_candidate
            print("\n  [BEST PARETO-OPTIMAL FORMULATION CANDIDATE]")
            print(f"  Sample ID:          {bc.sample_id} (Rack Vial: {bc.vial_id}, Round {bc.iteration_round})")
            print(f"  Additive Conc:      {bc.additive_conc_mM:.2f} mM")
            print(f"  Salt Conc:          {bc.salt_conc_M:.2f} M")
            print(f"  Performance Score:  {bc.performance_score:.2f}")
            if bc.analysis_report and bc.analysis_report.symmetric_cell:
                symm = bc.analysis_report.symmetric_cell
                act = symm.active_cycling_metrics or {}
                print(f"  Active Lifespan:    {act.get('active_lifespan_hours', symm.total_lifespan_hours):.1f} h")
                print(f"  Mean Overpotential: {act.get('mean_overpotential_mv', symm.mean_overpotential_mv):.2f} mV")
                print(f"  Failure Mode:       {symm.failure_mode}")
        print("================================================================================")
        return 0
    except Exception as e:
        logger.error(f"Campaign execution failed: {e}", exc_info=True)
        return 1


def handle_discovery(args: argparse.Namespace) -> int:
    orchestrator = MAPOrchestrator()
    try:
        res = orchestrator.run_discovery(query=args.query, from_extraction=args.from_extraction)
        res_json = res.model_dump_json(indent=2)
        if args.output:
            out_p = Path(args.output)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_text(res_json, encoding="utf-8")
        if args.json or not args.output:
            print(res_json)
        return 0
    except Exception as e:
        logger.error(f"Discovery failed: {e}")
        return 1


def handle_formulation(args: argparse.Namespace) -> int:
    from optimization.planner_cli import run_planner
    cli_args = ["--target-electrolyte", args.target_salt, "--method", args.method, "--samples", str(args.samples)]
    if args.from_extraction:
        cli_args.extend(["--from-extraction", args.from_extraction])
    if args.additive:
        cli_args.extend(["--additive", args.additive])
    if args.published_conc is not None:
        cli_args.extend(["--published-conc", str(args.published_conc)])
    if args.smiles:
        cli_args.extend(["--smiles", args.smiles])
    if args.parent_electrolyte:
        cli_args.extend(["--parent-electrolyte", args.parent_electrolyte])
    if args.rack_format:
        cli_args.extend(["--rack-format", args.rack_format])
    if args.total_volume_ul is not None:
        cli_args.extend(["--total-volume-ul", str(args.total_volume_ul)])
    if args.seed is not None:
        cli_args.extend(["--seed", str(args.seed)])
    if args.output:
        cli_args.extend(["--output", args.output])
    if args.json:
        cli_args.append("--json")

    return run_planner(cli_args)


def handle_analysis(args: argparse.Namespace) -> int:
    orchestrator = MAPOrchestrator(output_dir=args.output_dir)
    try:
        report = orchestrator.run_analysis(
            data_source=args.file,
            technique=args.technique,
            area_cm2=args.area_cm2,
            output_dir=args.output_dir,
            no_html=args.no_html,
        )
        report_json = report.model_dump_json(indent=2)
        if args.json:
            print(report_json)
        else:
            print("================================================================================")
            print("  ELECTROCHEMICAL ANALYSIS REPORT")
            print("================================================================================")
            print(f"  Sample ID:         {report.sample_id}")
            print(f"  Technique:         {report.technique}")
            print(f"  File Source:       {report.file_source}")
            print(f"  Dashboard HTML:    {report.visualization_html_path}")
            if report.active_cycling_metrics:
                print(f"  Active Lifespan:   {report.active_cycling_metrics.get('active_lifespan_hours')} h")
                print(f"  Overpotential:     {report.active_cycling_metrics.get('mean_overpotential_mv')} mV")
            if report.total_test_summary:
                print(f"  Failure Mode:      {report.total_test_summary.get('failure_mode')}")
                print(f"  Short Circuit:     {report.total_test_summary.get('short_circuit_detected')}")
            print("================================================================================")
        return 0
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return 1


def main(args: Optional[List[str]] = None) -> int:
    parser = build_parser()
    parsed = parser.parse_args(args)

    if not parsed.subcommand:
        parser.print_help()
        return 1

    if parsed.subcommand == "run-pipeline":
        return handle_pipeline(parsed)
    elif parsed.subcommand == "run-discovery":
        return handle_discovery(parsed)
    elif parsed.subcommand == "run-formulation":
        return handle_formulation(parsed)
    elif parsed.subcommand == "run-analysis":
        return handle_analysis(parsed)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())

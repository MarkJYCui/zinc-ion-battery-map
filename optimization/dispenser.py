"""
Robotic Dispensing Schedule Generator for Zinc-Ion Batteries (ZIB).
Module 2: Autonomous Design of Experiments (DoE) & Robotic Dispensing.

Generates volumetric liquid handling pipetting schedules for automated liquid handlers
(e.g., Opentrons OT-2, Hamilton Microlab STAR, Tecan Fluent).
Computes exact stock solution requirements, maps coordinates to 24-vial racks or 96-well plates,
and checks liquid handling constraints (pipette limits, viscosity, precipitation).
"""

import math
import uuid
import warnings
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Tuple

from core.exceptions import OversaturatedFormulationWarning
from models import DoEPlan, RoboticVial, DispensingSchedule, FormulationPoint
from optimization.concentration_anchor import CANONICAL_ELECTROLYTE_PROFILES



class RoboticDispenserEngine:
    """Calculates robotic liquid handling dispensing volumes and coordinates."""

    def __init__(self):
        self.profiles = CANONICAL_ELECTROLYTE_PROFILES

    def generate_coordinate_grid(self, rack_format: str, count: int, offset: int = 0) -> List[Tuple[str, str, int]]:
        """
        Generate physical coordinate layout (vial_id, row, col).

        Supported formats:
            - '24-vial rack': 4 rows (A-D) x 6 columns (1-6)
            - '96-well plate': 8 rows (A-H) x 12 columns (1-12)

        Args:
            rack_format: '24-vial rack' or '96-well plate'.
            count: Number of coordinates to allocate.
            offset: Starting index offset in the rack (e.g. for adaptive multi-round dispensing).
        """
        if rack_format == "24-vial rack":
            max_capacity = 24
            cols_per_row = 6
        elif rack_format == "96-well plate":
            max_capacity = 96
            cols_per_row = 12
        else:
            raise ValueError(f"Unsupported container format: {rack_format}")

        if offset + count > max_capacity:
            raise ValueError(
                f"Deck capacity overflow: {count} new samples with an offset of {offset} "
                f"exceeds the maximum {max_capacity} wells for a {rack_format}."
            )

        coords = []
        for i in range(count):
            well_idx = offset + i
            row_letter = chr(65 + (well_idx // cols_per_row))
            col_number = (well_idx % cols_per_row) + 1
            coords.append((f"{row_letter}{col_number}", row_letter, col_number))

        return coords

    def _assign_coordinates(
        self,
        formulations: List[FormulationPoint],
        container_type: str,
        vial_offset: int = 0,
    ) -> List[Tuple[str, str, int]]:
        """
        Enforce canonical container formats and assign row-major coordinates to formulations.
        """
        if container_type == "24-vial rack":
            max_capacity = 24
            cols_per_row = 6
        elif container_type == "96-well plate":
            max_capacity = 96
            cols_per_row = 12
        else:
            raise ValueError(f"Unsupported container format: {container_type}")

        if vial_offset + len(formulations) > max_capacity:
            raise ValueError(
                f"Deck capacity overflow: {len(formulations)} new samples with an offset of {vial_offset} "
                f"exceeds the maximum {max_capacity} wells for a {container_type}."
            )

        coords = []
        for i, formulation in enumerate(formulations):
            well_idx = vial_offset + i
            row_letter = chr(65 + (well_idx // cols_per_row))
            col_number = (well_idx % cols_per_row) + 1
            formulation.coordinate = f"{row_letter}{col_number}"
            coords.append((formulation.coordinate, row_letter, col_number))

        return coords

    def generate_schedule(
        self,
        plan: DoEPlan,
        rack_format: str = "24-vial rack",
        total_volume_uL: Optional[float] = None,
        additive_stock_conc_mM: Optional[float] = None,
        salt_stock_conc_M: Optional[float] = None,
        deck_slot: int = 1,
        aspiration_rate_ul_s: Optional[float] = None,
        dispense_rate_ul_s: Optional[float] = None,
        initial_vial_offset: int = 0,
        existing_vials: Optional[List[RoboticVial]] = None,
        iteration_round: Optional[int] = None,
    ) -> DispensingSchedule:
        """
        Generate a complete DispensingSchedule from a DoEPlan.

        Args:
            plan: The DoEPlan containing formulation points.
            rack_format: '24-vial rack' (default) or '96-well plate'.
            total_volume_uL: Target volume per formulation container in uL
                             (default: 2000 uL for 24-vial rack, 200 uL for 96-well plate).
            additive_stock_conc_mM: Concentration of additive stock solution in mM.
            salt_stock_conc_M: Concentration of zinc salt stock solution in M.
            deck_slot: Assigned OT-2 deck slot (default 1).
            aspiration_rate_ul_s: Optional custom aspiration rate in uL/s.
            dispense_rate_ul_s: Optional custom dispense rate in uL/s.
            initial_vial_offset: Starting well index offset (default 0).
            existing_vials: Optional list of previously dispensed RoboticVial instances to offset against.
            iteration_round: Optional iteration round index (defaults to plan.iteration_round).

        Returns:
            DispensingSchedule instance.
        """
        if existing_vials is not None and initial_vial_offset == 0:
            initial_vial_offset = len(existing_vials)

        if iteration_round is None:
            iteration_round = getattr(plan, 'iteration_round', 0)

        target_salt = plan.target_electrolyte_salt
        profile = self.profiles.get(target_salt, self.profiles['ZnSO4'])

        if total_volume_uL is None:
            if rack_format == '96-well plate':
                total_volume_uL = 200.0
            else:
                total_volume_uL = 2000.0

        max_add_conc = max((f.additive_conc_mM for f in plan.formulations), default=0.0)
        max_salt_conc = max((f.salt_conc_M for f in plan.formulations), default=1.0)

        if salt_stock_conc_M is None:
            default_stock = profile.get('default_stock_molarity', 3.5)
            salt_stock_conc_M = max(default_stock, round(max_salt_conc * 1.15, 2))

        if additive_stock_conc_mM is None:
            if max_add_conc <= 0.0:
                additive_stock_conc_mM = 100.0
            else:
                min_feasible_stock = 0.0
                if salt_stock_conc_M and salt_stock_conc_M > 0:
                    for f in plan.formulations:
                        rem_frac = 1.0 - (f.salt_conc_M / salt_stock_conc_M)
                        if rem_frac > 0.01 and f.additive_conc_mM > 0:
                            min_feasible_stock = max(min_feasible_stock, (f.additive_conc_mM / rem_frac) * 1.05)
                suggested = max(max_add_conc * 2.0, min_feasible_stock)
                if suggested <= 50.0:
                    additive_stock_conc_mM = 50.0
                elif suggested <= 100.0:
                    additive_stock_conc_mM = 100.0
                elif suggested <= 250.0:
                    additive_stock_conc_mM = 250.0
                elif suggested <= 500.0:
                    additive_stock_conc_mM = 500.0
                elif suggested <= 1000.0:
                    additive_stock_conc_mM = 1000.0
                else:
                    additive_stock_conc_mM = round(suggested, 1)

        salt_stock_name = f"{target_salt} stock solution ({salt_stock_conc_M:.1f} M)"

        if rack_format in ("vial_rack_24", "24-vial rack"):
            container_type = "24-vial rack"
        elif rack_format in ("well_plate_96", "96-well plate"):
            container_type = "96-well plate"
        else:
            container_type = rack_format
        vial_offset = initial_vial_offset
        formulations = plan.formulations

        # Enforce canonical container formats and capacities
        if container_type == "24-vial rack":
            max_capacity = 24
            cols_per_row = 6
        elif container_type == "96-well plate":
            max_capacity = 96
            cols_per_row = 12
        else:
            raise ValueError(f"Unsupported container format: {container_type}")

        # Safeguard against deck collisions and capacity overflow
        if vial_offset + len(formulations) > max_capacity:
            raise ValueError(
                f"Deck capacity overflow: {len(formulations)} new samples with an offset of {vial_offset} "
                f"exceeds the maximum {max_capacity} wells for a {container_type}."
            )

        vials = []
        schedule_warnings = []

        if target_salt == 'ZnCl2' and (max_salt_conc > 4.0 or salt_stock_conc_M >= 5.0):
            schedule_warnings.append(
                f"High viscosity alert: ZnCl2 electrolyte at high concentration (up to {max_salt_conc:.1f} M) "
                f"exhibits elevated dynamic viscosity (> 10 mPa·s). "
                f"Recommend reverse pipetting or slow aspiration speed (50%) on automated liquid handlers."
            )

        if target_salt == 'Zn(TFSI)2' and (max_salt_conc >= 4.0 or salt_stock_conc_M >= 4.0):
            schedule_warnings.append(
                "High viscosity alert: Concentrated Zn(TFSI)2 WiSE electrolyte displays high kinematic viscosity "
                "and low surface tension. Recommend using positive displacement pipetting or extended equilibration delays."
            )

        if target_salt == 'Zn(TFSI)2' and 'quaternary_ammonium' in plan.anchor.functional_groups:
            schedule_warnings.append(
                "Precipitation risk alert: Quaternary ammonium cation paired with Zn(TFSI)2 electrolyte risks "
                "hydrophobic [NR4][TFSI] ion-pair precipitation. Inspect vials post-mixing."
            )

        has_low_volume_warning = False
        has_negative_water = False

        is_viscous = (target_salt in ('ZnCl2', 'Zn(TFSI)2') and (max_salt_conc > 4.0 or salt_stock_conc_M >= 4.0))
        calculated_asp_rate = 25.0 if is_viscous else 50.0
        calculated_disp_rate = 25.0 if is_viscous else 50.0

        effective_asp_rate = aspiration_rate_ul_s if aspiration_rate_ul_s is not None else calculated_asp_rate
        effective_disp_rate = dispense_rate_ul_s if dispense_rate_ul_s is not None else calculated_disp_rate

        for i, formulation in enumerate(formulations):
            well_idx = vial_offset + i

            # Map flat index to alphanumeric row-major coordinate (e.g., A1, A2...)
            row_letter = chr(65 + (well_idx // cols_per_row))
            col_number = (well_idx % cols_per_row) + 1
            formulation.coordinate = f"{row_letter}{col_number}"

            vial_id = formulation.coordinate
            row = row_letter
            col = col_number

            vial_warns = []

            v_add_stock = round((formulation.additive_conc_mM / additive_stock_conc_mM) * total_volume_uL, 2)
            v_salt_stock = round((formulation.salt_conc_M / salt_stock_conc_M) * total_volume_uL, 2)
            v_water = round(total_volume_uL - v_add_stock - v_salt_stock, 2)

            if v_water < 0:
                total_stock_vol = round(v_add_stock + v_salt_stock, 2)
                warnings.warn(
                    f"Target volume ({total_volume_uL} uL) exceeded by stock components ({total_stock_vol} uL). "
                    f"Clamping diluent volume to 0.0.",
                    OversaturatedFormulationWarning,
                    stacklevel=2,
                )
                v_water = 0.0
                vial_warns.append("Negative water balance: Stock concentrations too low to achieve target concentrations.")
                has_negative_water = True

            if 0 < v_add_stock < 5.0:
                vial_warns.append(
                    f"Pipette volume for additive stock ({v_add_stock:.1f} uL) is below robotic accuracy "
                    f"threshold (5.0 uL). Recommend preparing a more dilute intermediate stock."
                )
                has_low_volume_warning = True

            if 0 < v_salt_stock < 5.0:
                vial_warns.append(
                    f"Pipette volume for salt stock ({v_salt_stock:.1f} uL) is below robotic accuracy threshold (5.0 uL)."
                )
                has_low_volume_warning = True

            if 0 < v_water < 5.0:
                vial_warns.append(
                    f"Pipette volume for diluent water ({v_water:.1f} uL) is below robotic accuracy threshold (5.0 uL)."
                )
                has_low_volume_warning = True

            if plan.anchor.solubility_limit_mM:
                if formulation.additive_conc_mM > 0.75 * plan.anchor.solubility_limit_mM:
                    vial_warns.append(
                        f"Solubility risk alert: Additive concentration ({formulation.additive_conc_mM:.1f} mM) "
                        f"exceeds 75% of saturation limit ({plan.anchor.solubility_limit_mM:.1f} mM); "
                        f"risk of crystallization during storage."
                    )

            vials.append(
                RoboticVial(
                    vial_id=vial_id,
                    rack_format=rack_format,
                    row=row,
                    col=col,
                    additive_conc_mM=formulation.additive_conc_mM,
                    salt_conc_M=formulation.salt_conc_M,
                    target_volume_uL=total_volume_uL,
                    additive_stock_uL=v_add_stock,
                    salt_stock_uL=v_salt_stock,
                    water_uL=v_water,
                    warnings=vial_warns,
                    slot=deck_slot,
                    aspiration_rate_ul_s=effective_asp_rate,
                    dispense_rate_ul_s=effective_disp_rate,
                )
            )

        if has_low_volume_warning:
            schedule_warnings.append(
                "One or more pipetting steps are below the 5.0 uL robotic accuracy threshold; "
                "consider adjusting stock concentrations or scaling up total volume."
            )

        if has_negative_water:
            schedule_warnings.append(
                "Stock feasibility alert: One or more formulations require stock solutions more concentrated than provided."
            )
            schedule_warnings.append(
                "Negative water balance alert: Target concentrations cannot be reached without negative water volume."
            )

        schedule_id = f"SCHED_{target_salt}_{plan.additive_name.replace(' ', '_')}_{uuid.uuid4().hex[:6]}"

        return DispensingSchedule(
            schedule_id=schedule_id,
            target_electrolyte_salt=target_salt,
            electrolyte_adjustment_rationale=plan.electrolyte_adjustment_rationale,
            rack_format=rack_format,
            total_vials=len(vials),
            total_volume_per_vial_uL=total_volume_uL,
            additive_stock_conc_mM=additive_stock_conc_mM,
            salt_stock_conc_M=salt_stock_conc_M,
            salt_stock_name=salt_stock_name,
            vials=vials,
            warnings=schedule_warnings,
            ot2_deck_slot=deck_slot,
            default_aspiration_rate_ul_s=effective_asp_rate,
            default_dispense_rate_ul_s=effective_disp_rate,
            created_at=datetime.now(timezone.utc).isoformat(),
            iteration_round=iteration_round,
        )

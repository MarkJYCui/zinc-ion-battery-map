"""
Multi-Factor Design of Experiments (DoE) Sampling Engine for Zinc-Ion Batteries (ZIB).
Module 2: Autonomous Design of Experiments (DoE) & Robotic Dispensing.

Supports Latin Hypercube Sampling (LHS) and Central Composite Design (CCD).
Dynamically adapts zinc salt concentration bounds to the target electrolyte chemistry.
"""

import math
import uuid
from typing import Optional, List, Dict, Any, Tuple
import numpy as np

from models import ConcentrationAnchor, DoEParameter, FormulationPoint, DoEPlan
from optimization.concentration_anchor import CANONICAL_ELECTROLYTE_PROFILES


class DoEEngine:
    """Multi-factor Design of Experiments generator for autonomous electrolyte optimization."""

    def __init__(self):
        self.profiles = CANONICAL_ELECTROLYTE_PROFILES

    def create_plan(
        self,
        anchor: ConcentrationAnchor,
        method: str = "lhs",
        num_samples: int = 10,
        seed: int = 42,
        salt_range_override: Optional[List[float]] = None,
        temperature_range: Optional[List[float]] = None,
        ccd_center_points: int = 3,
        alpha_mode: str = "orthogonal",
    ) -> DoEPlan:
        """
        Generate a complete DoEPlan for autonomous battery formulation.

        Args:
            anchor: The ConcentrationAnchor instance for the additive.
            method: 'lhs' (Latin Hypercube Sampling) or 'ccd' (Central Composite Design).
            num_samples: Number of samples for LHS (ignored for CCD).
            seed: Random seed for reproducible sampling.
            salt_range_override: Optional custom salt concentration bounds [min, max] in M.
            temperature_range: Optional temperature bounds [min, max] in °C (adds Factor 3).
            ccd_center_points: Number of replicate center points for CCD (default 3).
            alpha_mode: 'face' (face-centered, alpha=1.0) or 'rotatable'.

        Returns:
            DoEPlan instance.
        """
        target_salt = anchor.target_electrolyte_salt
        profile = self.profiles.get(target_salt, self.profiles['ZnSO4'])

        if salt_range_override:
            salt_min = salt_range_override[0]
            salt_max = salt_range_override[1]
        else:
            salt_min = profile['molarity_range'][0]
            salt_max = profile['molarity_range'][1]

        param_additive = DoEParameter(
            name='additive_conc_mM',
            display_name=f"{anchor.additive_name} Concentration",
            unit='mM',
            min_val=round(anchor.min_conc, 4),
            max_val=round(anchor.max_conc, 4),
            baseline_val=round(anchor.shifted_anchor_conc, 4),
        )

        param_salt = DoEParameter(
            name='salt_conc_M',
            display_name=f"{target_salt} Salt Concentration",
            unit='M',
            min_val=float(salt_min),
            max_val=float(salt_max),
            baseline_val=float(profile['default_molarity']),
        )

        parameters = [param_additive, param_salt]

        if temperature_range:
            param_temp = DoEParameter(
                name='temperature_C',
                display_name='Operating / Formulation Temperature',
                unit='°C',
                min_val=float(temperature_range[0]),
                max_val=float(temperature_range[1]),
                baseline_val=25.0,
            )
            parameters.append(param_temp)

        method_lower = method.lower()
        if method_lower in ('lhs', 'latin_hypercube', 'latin hypercube'):
            formulations = self._generate_lhs(parameters=parameters, num_samples=num_samples, seed=seed)
        elif method_lower in ('ccd', 'central_composite', 'central composite'):
            formulations = self._generate_ccd(parameters=parameters, center_points=ccd_center_points, alpha_mode=alpha_mode)
        else:
            raise ValueError(f"Unsupported DoE method '{method}'. Must be 'lhs' or 'ccd'.")

        plan_id = f"DOE_{target_salt}_{anchor.additive_name.replace(' ', '_')}_{method_lower.upper()}_{uuid.uuid4().hex[:6]}"

        return DoEPlan(
            plan_id=plan_id,
            additive_name=anchor.additive_name,
            target_electrolyte_salt=target_salt,
            target_salt_default_molarity=profile['default_molarity'],
            target_salt_molarity_range=[float(salt_min), float(salt_max)],
            method=method_lower,
            anchor=anchor,
            parameters=parameters,
            formulations=formulations,
            electrolyte_adjustment_rationale=anchor.electrolyte_adjustment_rationale,
            metadata={
                'seed': seed,
                'num_formulations': len(formulations),
                'num_factors': len(parameters),
                'target_profile': profile['display_name'],
            },
        )

    def _generate_lhs(
        self,
        parameters: List[DoEParameter],
        num_samples: int,
        seed: int,
    ) -> List[FormulationPoint]:
        """
        Generate Latin Hypercube Sampling (LHS) matrix.
        Stratifies each dimension into num_samples equal intervals using vectorized NumPy operations.
        """
        rng = np.random.default_rng(seed)
        num_dimensions = len(parameters)

        # Vectorized LHS Stratification
        strata = np.linspace(0, 1, num_samples + 1)
        random_offsets = rng.uniform(0, 1, (num_samples, num_dimensions))

        lhs_samples = np.zeros((num_samples, num_dimensions))
        for d in range(num_dimensions):
            bins = strata[:-1] + random_offsets[:, d] * (strata[1:] - strata[:-1])
            rng.shuffle(bins)
            lhs_samples[:, d] = bins

        min_vals = np.array([p.min_val for p in parameters])
        max_vals = np.array([p.max_val for p in parameters])
        scaled_samples = min_vals + lhs_samples * (max_vals - min_vals)

        formulations = []
        for i in range(num_samples):
            sample_id = f"S{i+1:02d}"
            add_val = round(float(scaled_samples[i, 0]), 4)
            salt_val = round(float(scaled_samples[i, 1]), 4)
            temp_val = round(float(scaled_samples[i, 2]), 2) if num_dimensions >= 3 else None

            formulations.append(
                FormulationPoint(
                    sample_id=sample_id,
                    additive_conc_mM=add_val,
                    salt_conc_M=salt_val,
                    temperature_C=temp_val,
                    point_type='lhs',
                )
            )
        return formulations

    def _generate_ccd(
        self,
        parameters: List[DoEParameter],
        center_points: int,
        alpha_mode: str,
    ) -> List[FormulationPoint]:
        """
        Generate Central Composite Design (CCD) matrix.
        Consists of:
          1. 2^k Factorial points
          2. 2k Axial / star points
          3. Center points (replicated for pure error estimation)
        """
        k = len(parameters)
        if alpha_mode == 'face':
            alpha = 1.0
        else:
            alpha = math.sqrt(k)

        # Vectorized CCD Generation (factorial block)
        bounds = [(-1.0, 1.0) for _ in range(k)]
        bounds_arrays = [np.array([low, high]) for low, high in bounds]
        grid = np.meshgrid(*bounds_arrays)
        factorial_points = np.vstack(list(map(np.ravel, grid))).T

        # Vectorized star / axial points (2k points)
        axial_points = np.empty((2 * k, k))
        axial_points[0::2] = -alpha * np.eye(k)
        axial_points[1::2] = alpha * np.eye(k)

        # Vectorized center points
        center_points_matrix = np.zeros((center_points, k))

        # Concatenate blocks vectorially
        all_coded = np.vstack([factorial_points, axial_points, center_points_matrix])
        point_types = (
            ['factorial'] * len(factorial_points) +
            ['axial'] * len(axial_points) +
            ['center'] * center_points
        )

        min_vals = np.array([p.min_val for p in parameters])
        max_vals = np.array([p.max_val for p in parameters])
        mids = (min_vals + max_vals) / 2.0
        halfs = (max_vals - min_vals) / 2.0

        physical_points = np.clip(mids + all_coded * halfs, min_vals, max_vals)

        formulations = []
        for i, (coords, p_type) in enumerate(zip(physical_points, point_types)):
            sample_id = f"S{i+1:02d}"
            add_val = round(float(coords[0]), 4)
            salt_val = round(float(coords[1]), 4)
            temp_val = round(float(coords[2]), 2) if k >= 3 else None

            formulations.append(
                FormulationPoint(
                    sample_id=sample_id,
                    additive_conc_mM=add_val,
                    salt_conc_M=salt_val,
                    temperature_C=temp_val,
                    point_type=p_type,
                )
            )
        return formulations

    def generate_adaptive_plan(
        self,
        anchor: ConcentrationAnchor,
        prior_results: List[Any],
        num_samples: int = 3,
        iteration_round: int = 1,
        seed: int = 42,
        salt_range_override: Optional[List[float]] = None,
        temperature_range: Optional[List[float]] = None,
        start_sample_index: Optional[int] = None,
        al_strategy: str = "balanced",
    ) -> DoEPlan:
        """
        Generate an adaptive sequential DoE plan based on prior experimental observations.

        Detects response surface performance gradients:
          - Upward gradient (best candidate at max tested): exploit upward and interpolate gap.
          - Downward gradient (best candidate at min tested): probe lower and interpolate gap.
          - Interior peak (best candidate between bounds): interpolate both adjacent gaps.

        Strict physical guardrail:
          Enforces Delaney ESOL 80% saturation ceiling (anchor.solubility_ceiling_mM).
          Capping alerts are attached when extreme gradients suggest supersaturated regimes.

        Args:
            anchor: The ConcentrationAnchor instance.
            prior_results: List of completed FormulatedCellResult instances.
            num_samples: Number of adaptive candidates to propose.
            iteration_round: Iteration round index (Round 1, 2, ...).
            seed: Random seed.
            salt_range_override: Optional custom salt concentration range [min, max] in M.
            temperature_range: Optional custom temperature range [min, max] in °C.
            start_sample_index: Optional starting sample ID integer offset (e.g. 4 for S04).
            al_strategy: Active learning exploration vs. exploitation strategy
                ('fastest'/'exploration', 'refine'/'exploitation', or 'balanced').

        Returns:
            DoEPlan instance configured with method='adaptive'.
        """
        s_low = (al_strategy or "balanced").lower()
        if "fast" in s_low or "explor" in s_low:
            strategy_mode = "exploration"
        elif "refin" in s_low or "exploit" in s_low:
            strategy_mode = "exploitation"
        else:
            strategy_mode = "balanced"

        target_salt = anchor.target_electrolyte_salt
        profile = self.profiles.get(target_salt, self.profiles['ZnSO4'])

        if salt_range_override:
            salt_min, salt_max = salt_range_override[0], salt_range_override[1]
        else:
            salt_min, salt_max = profile['molarity_range'][0], profile['molarity_range'][1]

        if not prior_results:
            return self.create_plan(
                anchor=anchor,
                method="lhs",
                num_samples=num_samples,
                seed=seed,
                salt_range_override=salt_range_override,
                temperature_range=temperature_range,
            )

        # Extract prior cell scores
        evaluated_cells = []
        for cell in prior_results:
            conc = float(cell.additive_conc_mM)
            score = float(cell.performance_score or 0.0)
            if score == 0.0 and getattr(cell, 'analysis_report', None) and getattr(cell.analysis_report, 'symmetric_cell', None):
                symm = cell.analysis_report.symmetric_cell
                act_metrics = symm.active_cycling_metrics or {}
                act_h = act_metrics.get("active_lifespan_hours", symm.total_lifespan_hours)
                overpot_mv = act_metrics.get("mean_overpotential_mv", symm.mean_overpotential_mv)
                score = round(act_h / (1.0 + overpot_mv / 50.0), 3)
            evaluated_cells.append((conc, score, cell))

        best_conc, best_score, best_cell = max(evaluated_cells, key=lambda item: item[1])
        best_salt = float(getattr(best_cell, 'salt_conc_M', profile['default_molarity']))

        tested_concs = sorted(list(set(round(item[0], 4) for item in evaluated_cells)))

        raw_proposals = []

        # Analyze performance gradient
        if best_conc >= tested_concs[-1]:
            # Upward Gradient: Performance increases toward higher concentrations
            lower_neighbors = [c for c in tested_concs if c < best_conc]
            nearest_lower = max(lower_neighbors) if lower_neighbors else 0.0
            gap = best_conc - nearest_lower
            if gap <= 0:
                gap = max(1.0, best_conc * 0.25)

            if strategy_mode == "exploration":
                delta = round(gap / 2.0, 4)
                raw_proposals.append((round(best_conc + delta, 4), "adaptive_exploration"))
                raw_proposals.append((round(best_conc + 2.0 * delta, 4), "adaptive_exploration"))
                raw_proposals.append((round((best_conc + nearest_lower) / 2.0, 4), "adaptive_interpolation"))
                raw_proposals.append((round(best_conc + 3.0 * delta, 4), "adaptive_exploration"))
                raw_proposals.append((round(best_conc - delta, 4), "adaptive_interpolation"))
            elif strategy_mode == "exploitation":
                delta = round(gap / 5.0, 4)
                raw_proposals.append((round(best_conc + delta, 4), "adaptive_exploitation"))
                raw_proposals.append((round(best_conc - delta, 4), "adaptive_interpolation"))
                raw_proposals.append((round(best_conc + 0.5 * delta, 4), "adaptive_exploitation"))
                raw_proposals.append((round(best_conc - 0.5 * delta, 4), "adaptive_interpolation"))
                raw_proposals.append((round(best_conc + 2.0 * delta, 4), "adaptive_exploitation"))
            else:
                delta = round(gap / 3.0, 4)
                # Order: Interpolate (15 mM), Exploit (25 mM), Further Interpolate (10 mM), Further Exploit (30 mM), Midpoint (12.5 mM)
                raw_proposals.append((round(best_conc - delta, 4), "adaptive_interpolation"))
                raw_proposals.append((round(best_conc + delta, 4), "adaptive_exploitation"))
                raw_proposals.append((round(best_conc - 2.0 * delta, 4), "adaptive_interpolation"))
                raw_proposals.append((round(best_conc + 2.0 * delta, 4), "adaptive_exploitation"))
                raw_proposals.append((round((best_conc + nearest_lower) / 2.0, 4), "adaptive_interpolation"))
                raw_proposals.append((round(best_conc + 3.0 * delta, 4), "adaptive_exploitation"))

        elif best_conc <= tested_concs[0]:
            # Downward Gradient: Lower concentration yields higher performance
            higher_neighbors = [c for c in tested_concs if c > best_conc]
            nearest_higher = min(higher_neighbors) if higher_neighbors else best_conc + 5.0
            gap = nearest_higher - best_conc
            if gap <= 0:
                gap = max(1.0, best_conc * 0.25)

            if strategy_mode == "exploration":
                delta = round(gap / 2.0, 4)
                raw_proposals.append((round(max(0.0, best_conc - delta), 4), "adaptive_exploration"))
                raw_proposals.append((round((best_conc + nearest_higher) / 2.0, 4), "adaptive_interpolation"))
                raw_proposals.append((round(max(0.0, best_conc - 2.0 * delta), 4), "adaptive_exploration"))
                raw_proposals.append((round(best_conc + delta, 4), "adaptive_interpolation"))
            elif strategy_mode == "exploitation":
                delta = round(gap / 5.0, 4)
                raw_proposals.append((round(max(0.0, best_conc - delta), 4), "adaptive_exploitation"))
                raw_proposals.append((round(best_conc + delta, 4), "adaptive_interpolation"))
                raw_proposals.append((round(max(0.0, best_conc - 0.5 * delta), 4), "adaptive_exploitation"))
                raw_proposals.append((round(best_conc + 0.5 * delta, 4), "adaptive_interpolation"))
            else:
                delta = round(gap / 3.0, 4)
                raw_proposals.append((round(max(0.0, best_conc - delta), 4), "adaptive_exploitation"))
                raw_proposals.append((round(best_conc + delta, 4), "adaptive_interpolation"))
                raw_proposals.append((round(best_conc + 2.0 * delta, 4), "adaptive_interpolation"))
                raw_proposals.append((round((best_conc + nearest_higher) / 2.0, 4), "adaptive_interpolation"))

        else:
            # Interior Peak: Optimum resides between tested points
            lower_neighbors = [c for c in tested_concs if c < best_conc]
            higher_neighbors = [c for c in tested_concs if c > best_conc]
            nearest_lower = max(lower_neighbors)
            nearest_higher = min(higher_neighbors)

            if strategy_mode == "exploration":
                raw_proposals.append((round((best_conc + nearest_lower) / 2.0, 4), "adaptive_exploration"))
                raw_proposals.append((round((best_conc + nearest_higher) / 2.0, 4), "adaptive_exploration"))
                raw_proposals.append((round(nearest_lower + (best_conc - nearest_lower) / 4.0, 4), "adaptive_interpolation"))
                raw_proposals.append((round(nearest_higher - (nearest_higher - best_conc) / 4.0, 4), "adaptive_interpolation"))
            elif strategy_mode == "exploitation":
                raw_proposals.append((round(best_conc - (best_conc - nearest_lower) / 5.0, 4), "adaptive_exploitation"))
                raw_proposals.append((round(best_conc + (nearest_higher - best_conc) / 5.0, 4), "adaptive_exploitation"))
                raw_proposals.append((round(best_conc - (best_conc - nearest_lower) / 10.0, 4), "adaptive_exploitation"))
                raw_proposals.append((round(best_conc + (nearest_higher - best_conc) / 10.0, 4), "adaptive_exploitation"))
            else:
                raw_proposals.append((round((best_conc + nearest_lower) / 2.0, 4), "adaptive_interpolation"))
                raw_proposals.append((round((best_conc + nearest_higher) / 2.0, 4), "adaptive_interpolation"))
                raw_proposals.append((round(best_conc - (best_conc - nearest_lower) / 4.0, 4), "adaptive_interpolation"))
                raw_proposals.append((round(best_conc + (nearest_higher - best_conc) / 4.0, 4), "adaptive_interpolation"))

        # Physical Guardrail: Delaney ESOL 80% Saturation Ceiling Capping
        ceiling = anchor.solubility_ceiling_mM
        warnings = []
        is_capped_by_solubility = False

        selected_candidates: List[Tuple[float, str]] = []

        for c_raw, p_type in raw_proposals:
            if len(selected_candidates) >= num_samples:
                break
            c_val = max(0.0, float(c_raw))
            if ceiling is not None and c_val > ceiling:
                orig_val = c_val
                c_val = round(ceiling, 4)
                is_capped_by_solubility = True
                warn_msg = (
                    f"Physical guardrail alert: Proposed concentration ({orig_val:.2f} mM) "
                    f"exceeds Delaney ESOL 80% saturation ceiling ({ceiling:.2f} mM); "
                    f"capped at ceiling."
                )
                if warn_msg not in warnings:
                    warnings.append(warn_msg)

            # Avoid duplicates within tolerance
            if any(abs(c_val - s[0]) < 1e-3 for s in selected_candidates):
                continue
            selected_candidates.append((round(c_val, 4), p_type))

        # If capping caused deduplication leaving fewer than num_samples, backfill intermediate points
        if len(selected_candidates) < num_samples:
            max_allowed = ceiling if ceiling is not None else (tested_concs[-1] * 2.0)
            
            # Continually find the largest gaps and bisect them until we reach num_samples
            while len(selected_candidates) < num_samples:
                current_pts = sorted(list(set(tested_concs + [s[0] for s in selected_candidates])))
                
                # Introduce boundaries for gap calculation if not present
                if current_pts[0] > 0.0:
                    current_pts.insert(0, 0.0)
                if current_pts[-1] < max_allowed:
                    current_pts.append(max_allowed)
                
                # Identify the largest valid gap to interpolate
                gaps = [(current_pts[j+1] - current_pts[j], current_pts[j], current_pts[j+1]) 
                        for j in range(len(current_pts)-1)]
                gaps.sort(reverse=True, key=lambda x: x[0])
                
                best_gap = gaps[0]
                mid_pt = round((best_gap[1] + best_gap[2]) / 2.0, 4)
                
                # Ensure we don't duplicate or exceed precision limits
                if not any(abs(mid_pt - s[0]) < 1e-3 for s in selected_candidates):
                    selected_candidates.append((mid_pt, "adaptive_interpolation"))
                else:
                    break  # Break to prevent infinite loops if precision is exhausted


        # Formulation parameters
        cand_concs = [s[0] for s in selected_candidates]
        param_additive = DoEParameter(
            name='additive_conc_mM',
            display_name=f"{anchor.additive_name} Concentration (Adaptive)",
            unit='mM',
            min_val=round(min(cand_concs), 4),
            max_val=round(max(cand_concs), 4),
            baseline_val=round(best_conc, 4),
        )

        param_salt = DoEParameter(
            name='salt_conc_M',
            display_name=f"{target_salt} Salt Concentration",
            unit='M',
            min_val=round(float(salt_min), 4),
            max_val=round(float(salt_max), 4),
            baseline_val=round(float(best_salt), 4),
        )

        parameters = [param_additive, param_salt]

        temp_val = None
        if temperature_range:
            param_temp = DoEParameter(
                name='temperature_C',
                display_name='Operating / Formulation Temperature',
                unit='°C',
                min_val=float(temperature_range[0]),
                max_val=float(temperature_range[1]),
                baseline_val=25.0,
            )
            parameters.append(param_temp)
            temp_val = round((temperature_range[0] + temperature_range[1]) / 2.0, 2)

        if start_sample_index is None:
            start_sample_index = len(prior_results) + 1

        formulations = []
        for i, (add_val, p_type) in enumerate(selected_candidates):
            sample_id = f"S{start_sample_index + i:02d}"
            formulations.append(
                FormulationPoint(
                    sample_id=sample_id,
                    additive_conc_mM=add_val,
                    salt_conc_M=round(float(best_salt), 4),
                    temperature_C=temp_val,
                    point_type=p_type,
                )
            )

        plan_id = f"DOE_{target_salt}_{anchor.additive_name.replace(' ', '_')}_ADAPTIVE_R{iteration_round}_{uuid.uuid4().hex[:6]}"
        rationale = anchor.electrolyte_adjustment_rationale
        if is_capped_by_solubility and ceiling is not None:
            rationale += f" [Adaptive round {iteration_round}: Capped at Delaney ESOL ceiling {ceiling:.2f} mM]"

        # Also update anchor solubility cap flag if capped
        if is_capped_by_solubility:
            anchor.is_capped_by_solubility = True

        return DoEPlan(
            plan_id=plan_id,
            additive_name=anchor.additive_name,
            target_electrolyte_salt=target_salt,
            target_salt_default_molarity=profile['default_molarity'],
            target_salt_molarity_range=[float(salt_min), float(salt_max)],
            method="adaptive",
            anchor=anchor,
            parameters=parameters,
            formulations=formulations,
            electrolyte_adjustment_rationale=rationale,
            iteration_round=iteration_round,
            prior_observations=prior_results,
            metadata={
                'seed': seed,
                'num_formulations': len(formulations),
                'num_factors': len(parameters),
                'target_profile': profile['display_name'],
                'iteration_round': iteration_round,
                'best_prior_conc_mM': best_conc,
                'best_prior_score': best_score,
                'al_strategy': strategy_mode,
                'warnings': warnings,
                'is_capped_by_solubility': is_capped_by_solubility,
            },
        )

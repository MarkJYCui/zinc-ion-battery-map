"""
Interactive Plotly HTML Visualization Engine for Zinc-Ion Battery Characterization.
Generates publication-grade, self-contained interactive HTML dashboards.
Supports both native Plotly Python library and zero-dependency Plotly.js CDN fallback.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
import numpy as np
import plotly.graph_objects as go

from models import (
    ElectrochemicalTechnique,
    ElectrochemicalAnalysisReport,
    NormalizedElectrochemicalData,
)
from analysis.processing.math_engine import MathEngine


class ElectrochemicalPlotEngine:
    """Renders standalone interactive electrochemical analysis dashboards."""

    def __init__(self, output_dir: Union[str, Path] = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def render_report(
        self,
        report: ElectrochemicalAnalysisReport,
        output_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        tech_val = report.technique.value if hasattr(report.technique, "value") else str(report.technique)
        if output_path is None:
            safe_name = report.sample_id.replace(" ", "_").replace("/", "_")
            out_file = self.output_dir / f"{safe_name}_{tech_val}_dashboard.html"
        else:
            out_file = Path(output_path)
            out_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            tech = ElectrochemicalTechnique(tech_val) if isinstance(tech_val, str) else tech_val
        except ValueError:
            tech = report.technique
        if tech == ElectrochemicalTechnique.GALVANOSTATIC_SYMMETRIC:
            html_content = self._render_symmetric_dashboard(report)
        elif tech == ElectrochemicalTechnique.GALVANOSTATIC_ASYMMETRIC:
            html_content = self._render_asymmetric_dashboard(report)
        elif tech == ElectrochemicalTechnique.GALVANOSTATIC_FULL_CELL:
            html_content = self._render_full_cell_dashboard(report)
        elif tech == ElectrochemicalTechnique.LSV:
            html_content = self._render_lsv_dashboard(report)
        elif tech == ElectrochemicalTechnique.EIS:
            html_content = self._render_eis_dashboard(report)
        elif tech == ElectrochemicalTechnique.CV:
            html_content = self._render_cv_dashboard(report)
        else:
            html_content = self._render_generic_dashboard(report)

        out_file.write_text(html_content, encoding="utf-8")
        return out_file

    def _render_symmetric_dashboard(self, report: ElectrochemicalAnalysisReport) -> str:
        d = report.processed_data
        symm = report.symmetric_cell
        t_unit = d.time_unit

        # Visualization Memory Optimization: Min-Max Decimation for large arrays
        disp_t = d.display_time
        volts = d.voltage_v
        max_pts = 10000
        
        if disp_t and volts and len(disp_t) > max_pts:
            chunk_size = len(disp_t) // (max_pts // 2)
            dec_x, dec_y = [], []
            for i in range(0, len(disp_t), chunk_size):
                x_chunk, y_chunk = disp_t[i:i+chunk_size], volts[i:i+chunk_size]
                if not y_chunk: continue
                min_idx, max_idx = y_chunk.index(min(y_chunk)), y_chunk.index(max(y_chunk))
                if min_idx < max_idx:
                    dec_x.extend([x_chunk[min_idx], x_chunk[max_idx]])
                    dec_y.extend([y_chunk[min_idx], y_chunk[max_idx]])
                elif min_idx > max_idx:
                    dec_x.extend([x_chunk[max_idx], x_chunk[min_idx]])
                    dec_y.extend([y_chunk[max_idx], y_chunk[min_idx]])
                else:
                    dec_x.append(x_chunk[min_idx])
                    dec_y.append(y_chunk[min_idx])
            plot_x, plot_y = dec_x, dec_y
        else:
            plot_x, plot_y = disp_t, volts

        trace_v = {
            "x": plot_x,
            "y": plot_y,
            "type": "scatter",
            "mode": "lines",
            "name": "Voltage (V)",
            "line": {"color": "#1f77b4", "width": 1.5},
        }

        cyc_nums = [m.cycle_number for m in symm.cycle_metrics] if symm else []
        overpots = [m.overpotential_mv for m in symm.cycle_metrics] if symm else []
        hyst = [m.voltage_hysteresis_mv for m in symm.cycle_metrics] if symm else []

        trace_overpot = {
            "x": cyc_nums,
            "y": overpots,
            "type": "scatter",
            "mode": "lines+markers",
            "name": "Overpotential η (mV)",
            "line": {"color": "#d62728", "width": 2},
        }

        trace_hyst = {
            "x": cyc_nums,
            "y": hyst,
            "type": "scatter",
            "mode": "lines+markers",
            "name": "Hysteresis ΔV (mV)",
            "line": {"color": "#ff7f0e", "width": 1.5, "dash": "dash"},
        }

        if symm and symm.short_circuit_detected:
            status_badge = "<span style='color: #d9534f; font-weight: bold;'>DENDRITIC SHORT-CIRCUIT DETECTED</span>"
        else:
            status_badge = "<span style='color: #5cb85c; font-weight: bold;'>STABLE REVERSIBLE CYCLING</span>"

        # Adaptive Failure-Aware Cycling Windowing & Interactive Dual-Range Configuration (Approach C)
        xaxis_vt: Dict[str, Any] = {
            "title": f"Time ({t_unit})",
            "rangeslider": {"visible": True},
            "automargin": True,
        }

        shapes: List[Dict[str, Any]] = []
        annotations: List[Dict[str, Any]] = []

        has_failure = symm is not None and (
            symm.failure_time_hours is not None
            and (symm.short_circuit_detected or symm.soft_short_detected or symm.failure_mode != "none")
        )

        max_t = float(max(d.display_time)) if d.display_time else 0.0

        if has_failure and symm.failure_time_hours is not None:
            t_fail = symm.failure_time_hours
            buffer_h = max(5.0, 0.05 * t_fail)
            t_window_max = min(round(t_fail + buffer_h, 2), max_t if max_t > 0 else t_fail + buffer_h)
            xaxis_vt["range"] = [0, t_window_max]

            # Failure vertical dashed line
            shapes.append({
                "type": "line",
                "x0": t_fail,
                "x1": t_fail,
                "y0": 0,
                "y1": 1,
                "yref": "paper",
                "line": {"color": "#d62728", "width": 2, "dash": "dash"},
            })

            # Diagnostic annotation tag placed above the plot canvas so it never overlaps voltage traces
            diag_text = (
                symm.total_test_summary.get("diagnostic_message")
                if symm.total_test_summary and isinstance(symm.total_test_summary, dict)
                else None
            )
            if not diag_text:
                cyc = symm.failure_cycle if symm.failure_cycle else 1
                if symm.short_circuit_detected:
                    diag_text = f"Dendrite Short-Circuit Detected at {t_fail:.1f} h (Cycle {cyc}) - Potential < 5 mV"
                else:
                    diag_text = f"Failure Detected at {t_fail:.1f} h (Cycle {cyc}) - Mode: {symm.failure_mode}"

            annotations.append({
                "x": 0.5,
                "xref": "paper",
                "y": 1.03,
                "yref": "paper",
                "xanchor": "center",
                "yanchor": "bottom",
                "text": diag_text,
                "showarrow": False,
                "bgcolor": "rgba(255, 235, 235, 0.96)",
                "bordercolor": "#d62728",
                "borderwidth": 1,
                "borderpad": 4,
                "font": {"color": "#d62728", "size": 11},
            })
        else:
            t_window_max = max_t

        init_24h_max = min(24.0, max_t)
        toolbar_html = f"""
        <div class="plot-toolbar" role="group" aria-label="Time Range Selector">
            <button type="button" class="plot-range-btn active" onclick="setPlotVtRange(this, 'plot_vt', [0, {t_window_max}], false)">0 to Failure (Auto)</button>
            <button type="button" class="plot-range-btn" onclick="setPlotVtRange(this, 'plot_vt', [0, {max_t}], true)">Full Test</button>
            <button type="button" class="plot-range-btn" onclick="setPlotVtRange(this, 'plot_vt', [0, {init_24h_max}], false)">Initial 24h</button>
        </div>
        """

        curr_dens = d.current_density_ma_cm2[0] if d.current_density_ma_cm2 else 0.0
        layout_vt: Dict[str, Any] = {
            "title": {
                "text": f"Galvanostatic Voltage Profile ({curr_dens:.2f} mA/cm²)",
                "x": 0.5,
                "xanchor": "center",
                "y": 0.96 if annotations else 0.95,
                "yanchor": "top",
            },
            "xaxis": xaxis_vt,
            "yaxis": {"title": "Cell Potential vs Zn/Zn²⁺ (V)", "automargin": True},
            "template": "plotly_white",
            "margin": {"t": 92 if annotations else 60, "b": 65, "l": 65, "r": 30},
        }
        if shapes:
            layout_vt["shapes"] = shapes
        if annotations:
            layout_vt["annotations"] = annotations

        plots = [
            {
                "div_id": "plot_vt",
                "toolbar_html": toolbar_html,
                "data": [trace_v],
                "layout": layout_vt,
            },
            {
                "div_id": "plot_overpot",
                "data": [trace_overpot, trace_hyst],
                "layout": {
                    "title": {
                        "text": "Zinc Stripping/Plating Overpotential & Hysteresis Evolution",
                        "x": 0.5,
                        "xanchor": "center",
                    },
                    "xaxis": {"title": "Cycle Number", "automargin": True},
                    "yaxis": {"title": "Overpotential / Hysteresis (mV)", "automargin": True},
                    "legend": {
                        "orientation": "h",
                        "yanchor": "top",
                        "y": -0.22,
                        "xanchor": "center",
                        "x": 0.5,
                    },
                    "margin": {"t": 55, "b": 80, "l": 65, "r": 30},
                    "template": "plotly_white",
                },
            },
        ]

        summary_html = f"""
        <div class="metrics-card">
            <h3>Diagnostic Performance Summary</h3>
            <p><strong>Cell Status:</strong> {status_badge}</p>
            <p><strong>Total Lifespan:</strong> {symm.total_lifespan_hours if symm else 0} h ({symm.total_cycles if symm else 0} cycles)</p>
            <p><strong>Mean Overpotential (η):</strong> {symm.mean_overpotential_mv if symm else 0} mV</p>
            <p><strong>Mean Hysteresis (ΔV):</strong> {symm.mean_hysteresis_mv if symm else 0} mV</p>
            <p><strong>Active Area:</strong> {report.geometry.area_cm2:.3f} cm²</p>
        </div>
        """

        return self._generate_html_container(report, summary_html, plots)

    def _render_asymmetric_dashboard(self, report: ElectrochemicalAnalysisReport) -> str:
        d = report.processed_data
        asymm = report.asymmetric_cell

        cyc_nums = [m.cycle_number for m in asymm.cycle_metrics] if asymm else []
        ces = [m.coulombic_efficiency_percent for m in asymm.cycle_metrics] if asymm else []
        nucs = [m.nucleation_overpotential_mv for m in asymm.cycle_metrics] if asymm else []

        trace_ce = {
            "x": cyc_nums,
            "y": ces,
            "type": "scatter",
            "mode": "lines+markers",
            "name": "Coulombic Efficiency (%)",
            "line": {"color": "#2ca02c", "width": 2},
        }

        trace_nuc = {
            "x": cyc_nums,
            "y": nucs,
            "type": "scatter",
            "mode": "lines+markers",
            "name": "Nucleation Overpotential (mV)",
            "line": {"color": "#9467bd", "width": 1.5, "dash": "dash"},
        }

        trace_v = {
            "x": d.display_time,
            "y": d.voltage_v,
            "type": "scatter",
            "mode": "lines",
            "name": "Voltage (V)",
            "line": {"color": "#1f77b4"},
        }

        plots = [
            {
                "div_id": "plot_ce",
                "data": [trace_ce],
                "layout": {
                    "title": {
                        "text": "Coulombic Efficiency (CE) Retention across Plating/Stripping Cycles",
                        "x": 0.5,
                        "xanchor": "center",
                    },
                    "xaxis": {"title": "Cycle Number", "automargin": True},
                    "yaxis": {"title": "Coulombic Efficiency (%)", "range": [0, 105], "automargin": True},
                    "margin": {"t": 55, "b": 60, "l": 65, "r": 30},
                    "template": "plotly_white",
                },
            },
            {
                "div_id": "plot_nuc",
                "data": [trace_nuc],
                "layout": {
                    "title": {
                        "text": "Zinc Nucleation Overpotential Evolution",
                        "x": 0.5,
                        "xanchor": "center",
                    },
                    "xaxis": {"title": "Cycle Number", "automargin": True},
                    "yaxis": {"title": "Nucleation Overpotential (mV)", "automargin": True},
                    "margin": {"t": 55, "b": 60, "l": 65, "r": 30},
                    "template": "plotly_white",
                },
            },
            {
                "div_id": "plot_vt",
                "data": [trace_v],
                "layout": {
                    "title": {
                        "text": "Asymmetric Cell Plating/Stripping Profiles",
                        "x": 0.5,
                        "xanchor": "center",
                    },
                    "xaxis": {"title": f"Time ({d.time_unit})", "automargin": True},
                    "yaxis": {"title": "Potential vs Zn/Zn²⁺ (V)", "automargin": True},
                    "margin": {"t": 55, "b": 60, "l": 65, "r": 30},
                    "template": "plotly_white",
                },
            },
        ]

        summary_html = f"""
        <div class="metrics-card">
            <h3>Asymmetric Plating/Stripping Metrics</h3>
            <p><strong>Average CE:</strong> {asymm.average_ce_percent if asymm else 0}%</p>
            <p><strong>Final CE:</strong> {asymm.final_ce_percent if asymm else 0}%</p>
            <p><strong>Mean Nucleation Overpotential:</strong> {asymm.mean_nucleation_overpotential_mv if asymm else 0} mV</p>
            <p><strong>Cumulative Plating Capacity:</strong> {asymm.cumulative_capacity_mah_cm2 if asymm else 0} mAh/cm²</p>
        </div>
        """

        return self._generate_html_container(report, summary_html, plots)

    def _render_full_cell_dashboard(self, report: ElectrochemicalAnalysisReport) -> str:
        d = report.processed_data
        fc = report.full_cell

        trace_cap = {
            "x": fc.cycle_numbers if fc else [],
            "y": fc.discharge_capacities_mah_g if fc else [],
            "type": "scatter",
            "mode": "lines+markers",
            "name": "Discharge Capacity (mAh/g)",
            "line": {"color": "#1f77b4", "width": 2},
            "marker": {"symbol": "circle", "size": 6, "color": "#1f77b4"},
            "yaxis": "y1",
        }

        trace_ce = {
            "x": fc.cycle_numbers if fc else [],
            "y": fc.coulombic_efficiencies if fc else [],
            "type": "scatter",
            "mode": "lines+markers",
            "name": "Coulombic Efficiency (%)",
            "line": {"color": "#ff7f0e", "width": 1.5, "dash": "dot"},
            "marker": {"symbol": "diamond", "size": 6, "color": "#ff7f0e"},
            "yaxis": "y2",
        }

        min_ce = min(fc.coulombic_efficiencies) if (fc and fc.coulombic_efficiencies) else 80.0
        max_ce = max(fc.coulombic_efficiencies) if (fc and fc.coulombic_efficiencies) else 100.0
        y2_min = max(0.0, min(80.0, min_ce - 5.0))
        y2_max = min(150.0, max(105.0, max_ce + 5.0))

        plots = [
            {
                "div_id": "plot_cap",
                "data": [trace_cap, trace_ce],
                "layout": {
                    "title": {
                        "text": "Full Cell Specific Capacity & Coulombic Efficiency vs Cycle",
                        "x": 0.5,
                        "xanchor": "center",
                    },
                    "xaxis": {"title": "Cycle Number", "showgrid": True, "automargin": True},
                    "yaxis": {"title": "Specific Capacity (mAh/g)", "side": "left", "showgrid": True, "automargin": True},
                    "yaxis2": {
                        "title": "Coulombic Efficiency (%)",
                        "overlaying": "y",
                        "side": "right",
                        "showgrid": False,
                        "range": [y2_min, y2_max],
                        "automargin": True,
                    },
                    "hovermode": "x unified",
                    "legend": {"orientation": "h", "yanchor": "top", "y": -0.22, "xanchor": "center", "x": 0.5},
                    "margin": {"t": 60, "b": 80, "l": 65, "r": 65},
                    "template": "plotly_white",
                },
            },
        ]

        is_summary = d.metadata.get("is_cycle_summary", False)
        if not is_summary and len(d.voltage_v) > (len(fc.cycle_numbers) if fc else 1):
            trace_v = {
                "x": d.display_time,
                "y": d.voltage_v,
                "type": "scatter",
                "mode": "lines",
                "name": "Cell Voltage (V)",
                "line": {"color": "#1f77b4"},
            }
            plots.append({
                "div_id": "plot_vt",
                "data": [trace_v],
                "layout": {
                    "title": {
                        "text": "Full Cell Galvanostatic Voltage Curve",
                        "x": 0.5,
                        "xanchor": "center",
                    },
                    "xaxis": {"title": f"Time ({d.time_unit})", "automargin": True},
                    "yaxis": {"title": "Cell Voltage (V)", "automargin": True},
                    "margin": {"t": 55, "b": 60, "l": 65, "r": 30},
                    "template": "plotly_white",
                },
            })

        summary_html = f"""
        <div class="metrics-card">
            <h3>Full Cell Performance Summary</h3>
            <p><strong>Initial Capacity:</strong> {fc.initial_discharge_capacity_mah_g if fc else 0} mAh/g</p>
            <p><strong>Final Capacity:</strong> {fc.final_discharge_capacity_mah_g if fc else 0} mAh/g</p>
            <p><strong>Capacity Retention:</strong> {fc.capacity_retention_percent if fc else 0}%</p>
            <p><strong>Average CE:</strong> {fc.average_coulombic_efficiency_percent if fc else 0}%</p>
            <p><strong>Average EE:</strong> {fc.average_energy_efficiency_percent if fc else 0}%</p>
        </div>
        """

        return self._generate_html_container(report, summary_html, plots)

    def _render_lsv_dashboard(self, report: ElectrochemicalAnalysisReport) -> str:
        d = report.processed_data
        lsv = report.lsv

        trace_j = {
            "x": d.voltage_v,
            "y": d.current_density_ma_cm2,
            "type": "scatter",
            "mode": "lines",
            "name": "Linear Polarization j (mA/cm²)",
            "line": {"color": "#d62728", "width": 2},
        }

        import numpy as np

        log_j = [round(float(np.log10(max(1e-6, abs(val)))), 3) for val in d.current_density_ma_cm2]

        trace_tafel = {
            "x": d.voltage_v,
            "y": log_j,
            "type": "scatter",
            "mode": "lines",
            "name": "log₁₀ |j| (A/cm²)",
            "line": {"color": "#9467bd", "width": 2},
        }

        plots = [
            {
                "div_id": "plot_lsv",
                "data": [trace_j],
                "layout": {
                    "title": {
                        "text": "Linear Sweep Voltammetry (LSV) Polarization Curve",
                        "x": 0.5,
                        "xanchor": "center",
                    },
                    "xaxis": {"title": "Potential vs Zn/Zn²⁺ (V)", "automargin": True},
                    "yaxis": {"title": "Current Density j (mA/cm²)", "automargin": True},
                    "margin": {"t": 55, "b": 60, "l": 65, "r": 30},
                    "template": "plotly_white",
                },
            },
            {
                "div_id": "plot_tafel",
                "data": [trace_tafel],
                "layout": {
                    "title": {
                        "text": "Tafel Kinetics Plot (log₁₀ |j| vs Potential)",
                        "x": 0.5,
                        "xanchor": "center",
                    },
                    "xaxis": {"title": "Potential vs Zn/Zn²⁺ (V)", "automargin": True},
                    "yaxis": {"title": "log₁₀ |j| (mA/cm²)", "automargin": True},
                    "margin": {"t": 55, "b": 60, "l": 65, "r": 30},
                    "template": "plotly_white",
                },
            },
        ]

        summary_html = f"""
        <div class="metrics-card">
            <h3>HER Overpotential & Corrosion Kinetics</h3>
            <p><strong>HER Onset Potential:</strong> {lsv.her_onset_potential_v if lsv else 0} V (at {lsv.her_current_density_threshold_ma_cm2 if lsv else 0} mA/cm²)</p>
            <p><strong>Corrosion Potential (E_corr):</strong> {lsv.corrosion_potential_ecorr_v if lsv else 0} V</p>
            <p><strong>Corrosion Current (j_corr):</strong> {lsv.corrosion_current_jcorr_ma_cm2 if lsv else 0} mA/cm²</p>
            <p><strong>Cathodic Tafel Slope (β_c):</strong> {lsv.tafel_slope_cathodic_mv_per_dec if lsv else 0} mV/dec</p>
        </div>
        """

        return self._generate_html_container(report, summary_html, plots)

    def _render_eis_dashboard(self, report: ElectrochemicalAnalysisReport) -> str:
        d = report.processed_data
        eis = report.eis

        import numpy as np

        zr = d.metadata.get("z_real_ohm") or [] if hasattr(d, "metadata") and d.metadata else []
        zi = d.metadata.get("z_imag_ohm") or [] if hasattr(d, "metadata") and d.metadata else []

        trace_nyquist = {
            "x": zr,
            "y": zi,
            "type": "scatter",
            "mode": "lines+markers",
            "name": "Nyquist (-Z'' vs Z')",
            "line": {"color": "#1f77b4", "width": 2},
        }

        layout_nyquist = {
            "title": {
                "text": "Nyquist Impedance Spectrum",
                "x": 0.5,
                "xanchor": "center",
            },
            "xaxis": {"title": "Z' (Ω)", "automargin": True},
            "yaxis": {"title": "-Z'' (Ω)", "scaleanchor": "x", "scaleratio": 1, "automargin": True},
            "margin": {"t": 55, "b": 60, "l": 65, "r": 30},
            "template": "plotly_white",
        }
        if not zr or not zi:
            layout_nyquist["annotations"] = [{
                "text": "Warning: Raw impedance spectrum arrays (z_real_ohm, z_imag_ohm) are missing.",
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": 0.5,
                "showarrow": False,
                "font": {"color": "#d62728", "size": 13},
            }]

        plots = [
            {
                "div_id": "plot_nyquist",
                "data": [trace_nyquist] if (zr and zi) else [],
                "layout": layout_nyquist,
            }
        ]

        summary_html = f"""
        <div class="metrics-card">
            <h3>Impedance Spectroscopy Equivalent Circuit</h3>
            <p><strong>Equivalent Circuit:</strong> {eis.equivalent_circuit if eis else 'R_s + R_ct // CPE'}</p>
            <p><strong>Solution Resistance (R_s):</strong> {eis.solution_resistance_rs_ohm if eis else 0} Ω</p>
            <p><strong>Charge-Transfer Resistance (R_ct):</strong> {eis.charge_transfer_resistance_rct_ohm if eis else 0} Ω</p>
            <p><strong>Characteristic Frequency:</strong> {eis.characteristic_frequency_hz if eis else 'N/A'} Hz</p>
        </div>
        """

        return self._generate_html_container(report, summary_html, plots)

    def _render_cv_dashboard(self, report: ElectrochemicalAnalysisReport) -> str:
        d = report.processed_data
        cv = report.cv

        trace_cv = {
            "x": d.voltage_v,
            "y": d.current_ma,
            "type": "scatter",
            "mode": "lines",
            "name": "Cyclic Voltammogram",
            "line": {"color": "#ff7f0e", "width": 2},
        }

        traces = [trace_cv]
        if cv and cv.anodic_peak_potential_epa_v is not None and cv.anodic_peak_current_ipa_ma is not None:
            traces.append({
                "x": [cv.anodic_peak_potential_epa_v],
                "y": [cv.anodic_peak_current_ipa_ma],
                "type": "scatter",
                "mode": "markers+text",
                "name": f"E_pa ({cv.anodic_peak_potential_epa_v:.3f} V)",
                "text": ["E_pa"],
                "textposition": "top center",
                "marker": {"symbol": "triangle-up", "size": 12, "color": "#d62728"},
            })
        if cv and cv.cathodic_peak_potential_epc_v is not None and cv.cathodic_peak_current_ipc_ma is not None:
            traces.append({
                "x": [cv.cathodic_peak_potential_epc_v],
                "y": [cv.cathodic_peak_current_ipc_ma],
                "type": "scatter",
                "mode": "markers+text",
                "name": f"E_pc ({cv.cathodic_peak_potential_epc_v:.3f} V)",
                "text": ["E_pc"],
                "textposition": "bottom center",
                "marker": {"symbol": "triangle-down", "size": 12, "color": "#1f77b4"},
            })

        plots = [
            {
                "div_id": "plot_cv",
                "data": traces,
                "layout": {
                    "title": {
                        "text": "Zinc Cyclic Voltammetry (CV) Curve",
                        "x": 0.5,
                        "xanchor": "center",
                    },
                    "xaxis": {"title": "Potential vs Zn/Zn²⁺ (V)", "automargin": True},
                    "yaxis": {"title": "Current (mA)", "automargin": True},
                    "legend": {
                        "orientation": "h",
                        "yanchor": "top",
                        "y": -0.22,
                        "xanchor": "center",
                        "x": 0.5,
                    },
                    "margin": {"t": 55, "b": 80, "l": 65, "r": 30},
                    "template": "plotly_white",
                },
            }
        ]

        summary_html = f"""
        <div class="metrics-card">
            <h3>Cyclic Voltammetry Reversibility Metrics</h3>
            <p><strong>Anodic Peak (E_pa):</strong> {cv.anodic_peak_potential_epa_v if cv else 0} V ({cv.anodic_peak_current_ipa_ma if cv else 0} mA)</p>
            <p><strong>Cathodic Peak (E_pc):</strong> {cv.cathodic_peak_potential_epc_v if cv else 0} V ({cv.cathodic_peak_current_ipc_ma if cv else 0} mA)</p>
            <p><strong>Peak Separation (ΔE_p):</strong> {cv.peak_separation_delta_ep_mv if cv else 0} mV</p>
            <p><strong>Current Ratio (I_pa / |I_pc|):</strong> {cv.current_ratio_ipa_ipc if cv else 0}</p>
            <p><strong>Plating Onset Potential:</strong> {cv.plating_onset_potential_v if cv else 0} V</p>
        </div>
        """

        return self._generate_html_container(report, summary_html, plots)

    @staticmethod
    def create_cv_figure(report: ElectrochemicalAnalysisReport) -> go.Figure:
        """Create interactive Plotly Figure for Cyclic Voltammetry with anodic/cathodic peak markers."""
        d = report.processed_data
        cv = report.cv
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=d.voltage_v,
            y=d.current_ma,
            mode="lines",
            name="Cyclic Voltammogram",
            line=dict(color="#1f77b4", width=2),
        ))
        if cv and cv.anodic_peak_potential_epa_v is not None and cv.anodic_peak_current_ipa_ma is not None:
            fig.add_trace(go.Scatter(
                x=[cv.anodic_peak_potential_epa_v],
                y=[cv.anodic_peak_current_ipa_ma],
                mode="markers+text",
                name=f"E_pa ({cv.anodic_peak_potential_epa_v:.3f} V)",
                text=["E_pa"],
                textposition="top center",
                marker=dict(symbol="triangle-up", size=13, color="#d62728", line=dict(width=1, color="black")),
            ))
        if cv and cv.cathodic_peak_potential_epc_v is not None and cv.cathodic_peak_current_ipc_ma is not None:
            fig.add_trace(go.Scatter(
                x=[cv.cathodic_peak_potential_epc_v],
                y=[cv.cathodic_peak_current_ipc_ma],
                mode="markers+text",
                name=f"E_pc ({cv.cathodic_peak_potential_epc_v:.3f} V)",
                text=["E_pc"],
                textposition="bottom center",
                marker=dict(symbol="triangle-down", size=13, color="#2ca02c", line=dict(width=1, color="black")),
            ))
        if cv and cv.plating_onset_potential_v is not None:
            fig.add_vline(
                x=cv.plating_onset_potential_v,
                line_dash="dot",
                line_color="#7f7f7f",
                annotation_text=f"Plating Onset: {cv.plating_onset_potential_v:.3f} V",
                annotation_position="bottom left",
            )
        if cv and cv.stripping_onset_potential_v is not None:
            fig.add_vline(
                x=cv.stripping_onset_potential_v,
                line_dash="dot",
                line_color="#ff7f0e",
                annotation_text=f"Stripping Onset: {cv.stripping_onset_potential_v:.3f} V",
                annotation_position="top right",
            )
        fig.update_layout(
            title=dict(text=f"Cyclic Voltammogram ({report.file_source})", x=0.5, xanchor="center"),
            xaxis=dict(title="Cell Potential vs Zn/Zn²⁺ (V)", automargin=True),
            yaxis=dict(title="Current (mA)", automargin=True),
            legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="center", x=0.5),
            template="plotly_white",
            height=440,
            margin=dict(l=55, r=25, t=55, b=80),
        )
        return fig

    @staticmethod
    def create_lsv_figures(report: ElectrochemicalAnalysisReport) -> Tuple[go.Figure, go.Figure]:
        """Create dual interactive Plotly figures for LSV: polarization curve and Tafel slope kinetics."""
        d = report.processed_data
        lsv = report.lsv

        v_arr = np.array(d.voltage_v, dtype=float)
        j_arr = np.array(d.current_density_ma_cm2, dtype=float)

        if len(v_arr) > 10:
            if v_arr[-1] >= v_arr[0]:
                start_idx = int(np.argmin(v_arr[: min(50, len(v_arr))]))
            else:
                start_idx = int(np.argmax(v_arr[: min(50, len(v_arr))]))
            v_plot = v_arr[start_idx:].tolist()
            j_plot = j_arr[start_idx:].tolist()
        else:
            v_plot = d.voltage_v
            j_plot = d.current_density_ma_cm2

        # 1. Polarization curve (j vs V)
        fig_lsv = go.Figure()
        fig_lsv.add_trace(go.Scatter(
            x=v_plot,
            y=j_plot,
            mode="lines",
            name="j (mA/cm²)",
            line=dict(color="#d62728", width=2),
        ))
        if lsv and lsv.corrosion_potential_ecorr_v is not None:
            fig_lsv.add_vline(
                x=lsv.corrosion_potential_ecorr_v,
                line_dash="dash",
                line_color="#1f77b4",
                annotation_text=f"E_corr: {lsv.corrosion_potential_ecorr_v:.3f} V",
                annotation_position="top left",
            )
        if lsv and lsv.her_onset_potential_v is not None:
            fig_lsv.add_vline(
                x=lsv.her_onset_potential_v,
                line_dash="dot",
                line_color="#9467bd",
                annotation_text=f"HER: {lsv.her_onset_potential_v:.3f} V",
                annotation_position="bottom right",
            )
        fig_lsv.update_layout(
            title=dict(text="LSV Polarization Curve (j vs V)", x=0.5, xanchor="center"),
            xaxis=dict(title="Potential vs Zn/Zn²⁺ (V)", automargin=True),
            yaxis=dict(title="Current Density j (mA/cm²)", automargin=True),
            template="plotly_white",
            height=400,
            margin=dict(l=55, r=25, t=55, b=60),
        )

        # 2. Tafel slope plot (log10|j| vs V)
        j_safe_arr = np.abs(np.array(j_plot, dtype=float))
        safe_j = np.where(j_safe_arr < 1e-8, 1e-8, j_safe_arr)
        log_j = np.log10(safe_j)

        fig_tafel = go.Figure()
        fig_tafel.add_trace(go.Scatter(
            x=v_plot,
            y=log_j.tolist(),
            mode="lines",
            name="log₁₀|j|",
            line=dict(color="#9467bd", width=2),
        ))
        if lsv and lsv.corrosion_potential_ecorr_v is not None:
            fig_tafel.add_vline(
                x=lsv.corrosion_potential_ecorr_v,
                line_dash="dash",
                line_color="#1f77b4",
                annotation_text=f"E_corr: {lsv.corrosion_potential_ecorr_v:.3f} V",
                annotation_position="top left",
            )
        fig_tafel.update_layout(
            title=dict(text="Tafel Slope Kinetics (log₁₀|j| vs V)", x=0.5, xanchor="center"),
            xaxis=dict(title="Potential vs Zn/Zn²⁺ (V)", automargin=True),
            yaxis=dict(title="log₁₀ |j| (mA/cm²)", automargin=True),
            template="plotly_white",
            height=400,
            margin=dict(l=55, r=25, t=55, b=60),
        )

        return fig_lsv, fig_tafel

    @staticmethod
    def create_eis_figure(report: ElectrochemicalAnalysisReport) -> go.Figure:
        """Create interactive Nyquist plot with strict 1:1 aspect ratio and annotations."""
        d = report.processed_data
        eis = report.eis

        zr = d.metadata.get("z_real_ohm") or [] if hasattr(d, "metadata") and d.metadata else []
        zi = d.metadata.get("z_imag_ohm") or [] if hasattr(d, "metadata") and d.metadata else []
        neg_zi = [-z if len(zi) > 0 and np.mean(zi) < 0 else z for z in zi] if zi else []

        fig = go.Figure()
        if zr and neg_zi:
            fig.add_trace(go.Scatter(
                x=zr,
                y=neg_zi,
                mode="lines+markers",
                name="Nyquist (-Z'' vs Z')",
                line=dict(color="#1f77b4", width=2),
                marker=dict(size=7, color="#1f77b4"),
            ))

            if eis and eis.solution_resistance_rs_ohm is not None:
                fig.add_annotation(
                    x=eis.solution_resistance_rs_ohm,
                    y=0,
                    text=f"R_s ({eis.solution_resistance_rs_ohm:.1f} Ω)",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor="#2ca02c",
                    ax=0,
                    ay=-30,
                    font=dict(color="#2ca02c", size=11),
                )

            if eis and eis.charge_transfer_resistance_rct_ohm is not None and neg_zi:
                apex_idx = int(np.argmax(neg_zi))
                apex_x = zr[apex_idx] if apex_idx < len(zr) else 0.0
                apex_y = neg_zi[apex_idx]
                freq_text = f"f_apex: {eis.characteristic_frequency_hz:.1f} Hz" if eis.characteristic_frequency_hz else "Apex"
                fig.add_annotation(
                    x=apex_x,
                    y=apex_y,
                    text=freq_text,
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor="#d62728",
                    ax=30,
                    ay=-25,
                    font=dict(color="#d62728", size=11),
                )
        else:
            fig.add_annotation(
                text="Warning: Raw impedance spectrum arrays (z_real_ohm, z_imag_ohm) are missing.",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(color="#d62728", size=13),
            )

        fig.update_layout(
            title=dict(text=f"Nyquist Impedance Spectrum ({report.file_source})", x=0.5, xanchor="center"),
            xaxis=dict(title="Z' (Ω)", automargin=True),
            yaxis=dict(
                title="-Z'' (Ω)",
                scaleanchor="x",
                scaleratio=1,
                automargin=True,
            ),
            template="plotly_white",
            height=480,
            margin=dict(l=55, r=25, t=55, b=60),
        )
        return fig

    @staticmethod
    def create_dqdv_figure(
        report: ElectrochemicalAnalysisReport,
        window_length: int = 15,
        polyorder: int = 3,
    ) -> go.Figure:
        """Create differential capacity dQ/dV vs V plot with Savitzky-Golay filtering."""
        d = report.processed_data
        cap_data = d.areal_capacity_mah_cm2 if d.areal_capacity_mah_cm2 else d.capacity_mah
        dq_vals = MathEngine.compute_dq_dv_savgol(
            voltage_v=d.voltage_v,
            capacity_mah=cap_data,
            window_length=window_length,
            polyorder=polyorder,
        )

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=d.voltage_v,
            y=dq_vals,
            mode="lines",
            name="dQ/dV (Savitzky-Golay)",
            line=dict(color="#2ca02c", width=1.8),
        ))
        y_label = "dQ/dV (mAh/(V·cm²))" if d.areal_capacity_mah_cm2 else "dQ/dV (mAh/V)"
        fig.update_layout(
            title=dict(text=f"Differential Capacity Analysis dQ/dV ({report.file_source})", x=0.5, xanchor="center"),
            xaxis=dict(title="Cell Potential vs Zn/Zn²⁺ (V)", automargin=True),
            yaxis=dict(title=y_label, automargin=True),
            template="plotly_white",
            height=400,
            margin=dict(l=55, r=25, t=55, b=60),
        )
        return fig

    def _render_generic_dashboard(self, report: ElectrochemicalAnalysisReport) -> str:
        d = report.processed_data

        trace_v = {
            "x": d.display_time,
            "y": d.voltage_v,
            "type": "scatter",
            "mode": "lines",
            "name": "Voltage (V)",
        }

        plots = [
            {
                "div_id": "plot_gen",
                "data": [trace_v],
                "layout": {
                    "title": "Electrochemical Time Series",
                    "xaxis": {"title": f"Time ({d.time_unit})"},
                    "yaxis": {"title": "Voltage (V)"},
                    "template": "plotly_white",
                },
            }
        ]

        return self._generate_html_container(report, "", plots)

    def _generate_html_container(
        self,
        report: ElectrochemicalAnalysisReport,
        summary_card_html: str,
        plots: List[Dict[str, Any]],
    ) -> str:
        plot_divs = []
        plot_scripts = []

        for p in plots:
            div_id = p["div_id"]
            toolbar = p.get("toolbar_html", "")
            data_json = json.dumps(p["data"])
            layout_json = json.dumps(p["layout"])
            if toolbar:
                plot_divs.append(f'<div class="plot-box">{toolbar}<div id="{div_id}" style="min-height: 430px; width: 100%;"></div></div>')
            else:
                plot_divs.append(f'<div id="{div_id}" class="plot-box"></div>')
            plot_scripts.append(f'Plotly.newPlot("{div_id}", {data_json}, {layout_json}, {{responsive: true}});')

        divs_html = "\n".join(plot_divs)
        scripts_js = "\n".join(plot_scripts)

        tech_val = report.technique.value if hasattr(report.technique, "value") else str(report.technique)
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ZIB Characterization Report - {report.sample_id}</title>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            margin: 0;
            padding: 24px;
            background-color: #f8f9fa;
            color: #212529;
        }}
        .header {{
            background: linear-gradient(135deg, #1f4037, #99f2c8);
            color: white;
            padding: 20px 30px;
            border-radius: 8px;
            margin-bottom: 24px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .header h1 {{ margin: 0 0 8px 0; font-size: 24px; }}
        .header p {{ margin: 0; opacity: 0.9; font-size: 14px; }}
        .metrics-card {{
            background: white;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 24px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
        .metrics-card h3 {{ margin-top: 0; color: #1f4037; border-bottom: 2px solid #e9ecef; padding-bottom: 8px; }}
        .plot-container {{ display: flex; flex-direction: column; gap: 24px; }}
        .plot-box {{
            background: white;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            padding: 16px;
            min-height: 450px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
        .plot-toolbar {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            align-items: center;
            margin-bottom: 6px;
        }}
        .plot-range-btn {{
            background: #f1f5f9;
            color: #000000;
            border: 1px solid #cbd5e1;
            border-radius: 5px;
            padding: 5px 12px;
            font-size: 12px;
            font-weight: 600;
            cursor: pointer;
            white-space: nowrap;
            transition: background 0.15s ease, border-color 0.15s ease;
        }}
        .plot-range-btn:hover {{
            background: #e2e8f0;
            border-color: #94a3b8;
        }}
        .plot-range-btn.active {{
            background: #0284c7;
            color: #ffffff;
            border-color: #0284c7;
        }}
        .footer {{
            margin-top: 40px;
            text-align: center;
            font-size: 12px;
            color: #6c757d;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Autonomous ZIB Electrochemical Characterization Report</h1>
        <p><strong>Sample ID:</strong> {report.sample_id} | <strong>Technique:</strong> {tech_val} | <strong>Source:</strong> {report.file_source}</p>
    </div>

    {summary_card_html}

    <div class="plot-container">
        {divs_html}
    </div>

    <script>
        function setPlotVtRange(btn, divId, rangeArr, autoRange) {{
            if (btn && btn.parentElement) {{
                var btns = btn.parentElement.querySelectorAll('.plot-range-btn');
                for (var i = 0; i < btns.length; i++) {{
                    btns[i].classList.remove('active');
                }}
                btn.classList.add('active');
            }}
            if (typeof Plotly !== 'undefined') {{
                if (autoRange) {{
                    Plotly.relayout(divId, {{'xaxis.range': rangeArr, 'xaxis.autorange': true}});
                }} else {{
                    Plotly.relayout(divId, {{'xaxis.range': rangeArr, 'xaxis.autorange': false}});
                }}
            }}
        }}
        {scripts_js}
        if (typeof ResizeObserver !== 'undefined') {{
            var ro = new ResizeObserver(function(entries) {{
                for (var i = 0; i < entries.length; i++) {{
                    var el = entries[i].target;
                    if (el && el.offsetWidth > 0 && typeof Plotly !== 'undefined') {{
                        try {{ Plotly.Plots.resize(el); }} catch(e) {{}}
                    }}
                }}
            }});
            var plotNodes = document.querySelectorAll('[id^="plot_"]');
            for (var j = 0; j < plotNodes.length; j++) {{
                ro.observe(plotNodes[j]);
            }}
        }}
    </script>

    <div class="footer">
        Materials Acceleration Platform (MAP) for Aqueous Zinc-Ion Batteries • Generated autonomously on {report.created_at}
    </div>
</body>
</html>
"""

# Convenient top-level functional exports
create_cv_figure = ElectrochemicalPlotEngine.create_cv_figure
create_lsv_figures = ElectrochemicalPlotEngine.create_lsv_figures
create_eis_figure = ElectrochemicalPlotEngine.create_eis_figure
create_dqdv_figure = ElectrochemicalPlotEngine.create_dqdv_figure

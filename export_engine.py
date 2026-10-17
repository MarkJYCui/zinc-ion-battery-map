"""
Export Engine for Autonomous Materials Acceleration Platform (MAP).
Provides pure-Python generators for editable Office documents:
- Microsoft Excel Workbooks (.xlsx) via OpenXML SpreadsheetML
- Microsoft Word Documents (.docx) via OpenXML WordprocessingML
- Formatted Plain Text Reports (.txt)
All generators are self-contained in standard library (zipfile, xml.sax.saxutils, io).
"""

import csv
import io
import zipfile
import xml.sax.saxutils as saxutils
from typing import Dict, List, Any, Tuple, Optional

from models import CampaignResult, DispensingSchedule, DoEPlan, FormulatedCellResult, ElectrochemicalAnalysisReport


def make_multi_sheet_xlsx(sheets_dict: Dict[str, List[List[Any]]]) -> bytes:
    """
    Generate a valid Microsoft Excel .xlsx workbook containing multiple worksheets
    using pure Python standard library (zipfile + XML).
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as z:
        sheet_names = list(sheets_dict.keys())
        
        # [Content_Types].xml
        ct = [
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n',
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">\n',
            '  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>\n',
            '  <Default Extension="xml" ContentType="application/xml"/>\n',
            '  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>\n',
            '  <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>\n',
        ]
        for idx in range(1, len(sheet_names) + 1):
            ct.append(f'  <Override PartName="/xl/worksheets/sheet{idx}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>\n')
        ct.append('</Types>')
        z.writestr('[Content_Types].xml', ''.join(ct))

        # _rels/.rels
        rels = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n'
            '  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>\n'
            '</Relationships>'
        )
        z.writestr('_rels/.rels', rels)

        # xl/_rels/workbook.xml.rels
        wb_rels = [
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n',
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n',
        ]
        for idx in range(1, len(sheet_names) + 1):
            wb_rels.append(f'  <Relationship Id="rId{idx}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{idx}.xml"/>\n')
        styles_rid = len(sheet_names) + 1
        wb_rels.append(f'  <Relationship Id="rId{styles_rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>\n')
        wb_rels.append('</Relationships>')
        z.writestr('xl/_rels/workbook.xml.rels', ''.join(wb_rels))

        # xl/styles.xml
        styles = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">\n'
            '  <fonts count="2">\n'
            '    <font><sz val="11"/><name val="Calibri"/></font>\n'
            '    <font><b/><sz val="11"/><name val="Calibri"/></font>\n'
            '  </fonts>\n'
            '  <fills count="2">\n'
            '    <fill><patternFill patternType="none"/></fill>\n'
            '    <fill><patternFill patternType="gray125"/></fill>\n'
            '  </fills>\n'
            '  <borders count="1">\n'
            '    <border><left/><right/><top/><bottom/><diagonal/></border>\n'
            '  </borders>\n'
            '  <cellStyleXfs count="1">\n'
            '    <xf numFmtId="0" fontId="0" fillId="0" borderId="0"/>\n'
            '  </cellStyleXfs>\n'
            '  <cellXfs count="2">\n'
            '    <xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>\n'
            '    <xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0" applyFont="1"/>\n'
            '  </cellXfs>\n'
            '</styleSheet>'
        )
        z.writestr('xl/styles.xml', styles)

        # xl/workbook.xml
        wb = [
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n',
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">\n',
            '  <sheets>\n',
        ]
        for idx, sname in enumerate(sheet_names, 1):
            safe_sname = saxutils.escape(sname[:31])
            wb.append(f'    <sheet name="{safe_sname}" sheetId="{idx}" r:id="rId{idx}"/>\n')
        wb.append('  </sheets>\n</workbook>')
        z.writestr('xl/workbook.xml', ''.join(wb))

        # worksheets
        for idx, (sname, rows) in enumerate(sheets_dict.items(), 1):
            sheet_xml = [
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n',
                '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">\n',
                '<sheetData>\n',
            ]
            for r_idx, row in enumerate(rows, 1):
                sheet_xml.append(f'<row r="{r_idx}">\n')
                is_header = (r_idx == 1)
                style_attr = ' s="1"' if is_header else ''
                for c_idx, val in enumerate(row, 1):
                    temp = c_idx
                    col_letter = ''
                    while temp > 0:
                        temp, rem = divmod(temp - 1, 26)
                        col_letter = chr(65 + rem) + col_letter
                    cell_ref = f"{col_letter}{r_idx}"
                    if isinstance(val, (int, float)) and not isinstance(val, bool):
                        sheet_xml.append(f'<c r="{cell_ref}"{style_attr}><v>{val}</v></c>')
                    else:
                        escaped = saxutils.escape(str(val) if val is not None else "")
                        sheet_xml.append(f'<c r="{cell_ref}"{style_attr} t="inlineStr"><is><t>{escaped}</t></is></c>')
                sheet_xml.append('</row>\n')
            sheet_xml.append('</sheetData>\n</worksheet>')
            z.writestr(f'xl/worksheets/sheet{idx}.xml', ''.join(sheet_xml))

    return buffer.getvalue()


def make_docx(title: str, sections: List[Tuple[str, Any]]) -> bytes:
    """
    Generate a valid Microsoft Word .docx document using pure Python standard library.
    sections is a list of tuples:
      ('heading', text)
      ('paragraph', text)
      ('bullet', text)
      ('table', list_of_rows) where list_of_rows is List[List[Any]]
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as z:
        content_types = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">\n'
            '  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>\n'
            '  <Default Extension="xml" ContentType="application/xml"/>\n'
            '  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>\n'
            '</Types>'
        )
        z.writestr('[Content_Types].xml', content_types)

        rels = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n'
            '  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>\n'
            '</Relationships>'
        )
        z.writestr('_rels/.rels', rels)

        doc_xml = [
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n',
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">\n',
            '<w:body>\n',
        ]

        # Document Title
        esc_title = saxutils.escape(title)
        doc_xml.append(
            f'<w:p><w:pPr><w:jc w:val="center"/><w:spacing w:before="120" w:after="240"/></w:pPr>'
            f'<w:r><w:rPr><w:b/><w:sz w:val="36"/><w:color w:val="000000"/></w:rPr><w:t>{esc_title}</w:t></w:r></w:p>\n'
        )

        for sec_type, sec_val in sections:
            if sec_type == 'heading':
                esc_h = saxutils.escape(str(sec_val))
                doc_xml.append(
                    f'<w:p><w:pPr><w:spacing w:before="240" w:after="100"/></w:pPr>'
                    f'<w:r><w:rPr><w:b/><w:sz w:val="26"/><w:color w:val="000000"/></w:rPr>'
                    f'<w:t>{esc_h}</w:t></w:r></w:p>\n'
                )
            elif sec_type in ('subheading', 'heading2'):
                esc_sh = saxutils.escape(str(sec_val))
                doc_xml.append(
                    f'<w:p><w:pPr><w:spacing w:before="160" w:after="60"/></w:pPr>'
                    f'<w:r><w:rPr><w:b/><w:sz w:val="22"/><w:color w:val="1E3A8A"/></w:rPr>'
                    f'<w:t>{esc_sh}</w:t></w:r></w:p>\n'
                )
            elif sec_type == 'paragraph':
                esc_p = saxutils.escape(str(sec_val))
                doc_xml.append(
                    f'<w:p><w:pPr><w:spacing w:before="40" w:after="100"/></w:pPr>'
                    f'<w:r><w:rPr><w:sz w:val="22"/><w:color w:val="000000"/></w:rPr>'
                    f'<w:t>{esc_p}</w:t></w:r></w:p>\n'
                )
            elif sec_type == 'bullet':
                esc_b = saxutils.escape(str(sec_val))
                doc_xml.append(
                    f'<w:p><w:pPr><w:spacing w:before="20" w:after="60"/><w:ind w:left="360"/></w:pPr>'
                    f'<w:r><w:rPr><w:b/><w:sz w:val="22"/><w:color w:val="000000"/></w:rPr><w:t>• </w:t></w:r>'
                    f'<w:r><w:rPr><w:sz w:val="22"/><w:color w:val="000000"/></w:rPr><w:t>{esc_b}</w:t></w:r></w:p>\n'
                )
            elif sec_type == 'table':
                doc_xml.append(
                    '<w:tbl><w:tblPr>'
                    '<w:tblW w:w="0" w:type="auto"/>'
                    '<w:tblBorders>'
                    '<w:top w:val="single" w:sz="6" w:space="0" w:color="000000"/>'
                    '<w:bottom w:val="single" w:sz="6" w:space="0" w:color="000000"/>'
                    '<w:insideH w:val="single" w:sz="4" w:space="0" w:color="D3D3D3"/>'
                    '<w:insideV w:val="single" w:sz="4" w:space="0" w:color="D3D3D3"/>'
                    '</w:tblBorders>'
                    '</w:tblPr>\n'
                )
                for r_i, row in enumerate(sec_val):
                    doc_xml.append('<w:tr>\n')
                    is_h = (r_i == 0)
                    for cell in row:
                        esc_c = saxutils.escape(str(cell) if cell is not None else "")
                        bold_xml = '<w:b/>' if is_h else ''
                        bg_xml = '<w:shd w:val="clear" w:color="auto" w:fill="F4F4F5"/>' if is_h else ''
                        doc_xml.append(
                            f'<w:tc><w:tcPr>{bg_xml}<w:tcMar><w:top w:w="80"/><w:bottom w:w="80"/><w:left w:w="120"/><w:right w:w="120"/></w:tcMar></w:tcPr>'
                            f'<w:p><w:r><w:rPr>{bold_xml}<w:sz w:val="20"/><w:color w:val="000000"/></w:rPr><w:t>{esc_c}</w:t></w:r></w:p>'
                            f'</w:tc>\n'
                        )
                    doc_xml.append('</w:tr>\n')
                doc_xml.append('</w:tbl>\n')

        doc_xml.append('</w:body></w:document>')
        z.writestr('word/document.xml', ''.join(doc_xml))

    return buffer.getvalue()


# -----------------------------------------------------------------------------
# Domain-Specific File Generators
# -----------------------------------------------------------------------------

def generate_dispensing_schedule_xlsx(schedule: DispensingSchedule) -> bytes:
    """Generate Excel .xlsx workbook for Opentrons OT-2 dispensing schedule."""
    headers = [
        "Vial ID",
        "Row",
        "Column",
        "Well",
        "Additive Stock (µL)",
        "Salt Stock (µL)",
        "Water Diluent (µL)",
        "Target Total Vol (µL)",
        "Additive Conc (mM)",
        "Salt Conc (M)",
        "Aspiration Rate (µL/s)",
        "Dispense Rate (µL/s)",
        "Warnings",
    ]
    rows = [headers]
    for v in schedule.vials:
        rows.append([
            v.vial_id,
            v.row,
            v.col,
            f"{v.row}{v.col}",
            round(v.additive_stock_uL, 2),
            round(v.salt_stock_uL, 2),
            round(v.water_uL, 2),
            round(v.target_volume_uL, 2),
            round(v.additive_conc_mM, 2),
            round(v.salt_conc_M, 2),
            round(v.aspiration_rate_ul_s, 1),
            round(v.dispense_rate_ul_s, 1),
            "; ".join(v.warnings) if v.warnings else "None",
        ])
    return make_multi_sheet_xlsx({"Dispensing Schedule": rows})


def generate_dispensing_schedule_docx(schedule: DispensingSchedule) -> bytes:
    """Generate Word .docx protocol document for Opentrons OT-2 dispensing schedule."""
    created = getattr(schedule, "created_at", None) or "N/A"
    tot_vol = getattr(schedule, "total_volume_per_vial_uL", 2000.0)
    
    table_rows = [
        ["Vial", "Well", "Additive (µL)", "Salt (µL)", "Water (µL)", "Total (µL)", "Additive (mM)", "Salt (M)"]
    ]
    for v in schedule.vials:
        table_rows.append([
            v.vial_id,
            f"{v.row}{v.col}",
            f"{v.additive_stock_uL:.1f}",
            f"{v.salt_stock_uL:.1f}",
            f"{v.water_uL:.1f}",
            f"{v.target_volume_uL:.1f}",
            f"{v.additive_conc_mM:.2f}",
            f"{v.salt_conc_M:.2f}",
        ])

    sections = [
        ("heading", "1. Opentrons OT-2 Robot Deck Configuration"),
        ("bullet", f"Schedule Identifier: {schedule.schedule_id}"),
        ("bullet", f"Target Electrolyte Salt: {schedule.target_electrolyte_salt}"),
        ("bullet", f"Rack Format & Location: {schedule.rack_format} in Deck Slot {schedule.ot2_deck_slot}"),
        ("bullet", f"Target Dispense Volume: {tot_vol:.1f} µL per vial"),
        ("bullet", f"Additive Stock Concentration: {schedule.additive_stock_conc_mM:.2f} mM"),
        ("bullet", f"Salt Stock Concentration: {schedule.salt_stock_conc_M:.2f} M ({schedule.salt_stock_name})"),
        ("bullet", f"Fluid Flow Rates: Aspiration = {schedule.default_aspiration_rate_ul_s:.1f} µL/s, Dispense = {schedule.default_dispense_rate_ul_s:.1f} µL/s"),
        ("bullet", f"Created Timestamp: {created}"),
        ("heading", "2. Vial-by-Vial Liquid Pipetting Schedule"),
        ("paragraph", "The following pipetting steps are calibrated for accurate volumetric delivery with automatic diluent water adjustment:"),
        ("table", table_rows),
        ("heading", "3. Quality Assurance & Viscosity Guardrails"),
        ("bullet", "Minimum Liquid Transfer: All pipetting transfers maintain > 5.0 µL threshold for high volumetric accuracy."),
        ("bullet", "Viscosity Adaptation: Aspiration rates are throttled to 25 µL/s for high-salinity salts (e.g. ZnCl₂ or Zn(TFSI)₂) to prevent cavitation."),
        ("bullet", "Diluent Continuity: Water volume is dynamically balanced to maintain constant total electrolyte volume across all vials."),
    ]
    return make_docx("Opentrons OT-2 Robotic Liquid Handling Protocol", sections)


def generate_dispensing_schedule_txt(schedule: DispensingSchedule) -> str:
    """Generate human-readable, editable text report of Opentrons OT-2 dispensing schedule."""
    created = getattr(schedule, "created_at", None) or "N/A"
    tot_vol = getattr(schedule, "total_volume_per_vial_uL", 2000.0)
    lines = [
        "=" * 80,
        "OPENTRONS OT-2 ROBOTIC DISPENSING SCHEDULE (FORMAL LABORATORY PROTOCOL)",
        "=" * 80,
        f"Schedule ID           : {schedule.schedule_id}",
        f"Target Salt           : {schedule.target_electrolyte_salt}",
        f"Created At            : {created}",
        f"Labware Rack Format   : {schedule.rack_format} (Deck Slot {schedule.ot2_deck_slot})",
        f"Total Target Volume   : {tot_vol:.1f} µL per vial",
        f"Additive Stock Conc   : {schedule.additive_stock_conc_mM:.2f} mM",
        f"Salt Stock Conc       : {schedule.salt_stock_conc_M:.2f} M ({schedule.salt_stock_name})",
        f"Default Flow Rates    : Aspiration = {schedule.default_aspiration_rate_ul_s:.1f} µL/s, Dispense = {schedule.default_dispense_rate_ul_s:.1f} µL/s",
        "-" * 80,
        "VIAL-BY-VIAL PIPETTING INSTRUCTIONS (Editable Lab Protocol):",
        "-" * 80,
        f"{'Vial':<6} | {'Well':<5} | {'Additive (µL)':<13} | {'Salt (µL)':<10} | {'Water (µL)':<10} | {'Total (µL)':<10} | {'Additive (mM)':<13} | {'Salt (M)':<8}",
        "-" * 80,
    ]
    for v in schedule.vials:
        lines.append(
            f"{v.vial_id:<6} | {f'{v.row}{v.col}':<5} | {v.additive_stock_uL:<13.1f} | {v.salt_stock_uL:<10.1f} | {v.water_uL:<10.1f} | {v.target_volume_uL:<10.1f} | {v.additive_conc_mM:<13.2f} | {v.salt_conc_M:<8.2f}"
        )
    lines.extend([
        "-" * 80,
        "SAFETY & PROTOCOL NOTES:",
        "1. All liquid transfers maintain > 5.0 µL minimum volume threshold for volumetric accuracy.",
        "2. For viscous electrolytes, ensure 25 µL/s aspiration throttling is maintained.",
        "3. Water diluent is calculated automatically to maintain total volume continuity.",
        "=" * 80,
    ])
    return "\n".join(lines)


def generate_doe_plan_xlsx(plan: DoEPlan) -> bytes:
    """Generate Excel .xlsx workbook for Electrolyte Formulation DoE Plan."""
    matrix_headers = ["Sample ID", "Additive Conc (mM)", "Salt Conc (M)", "Sampling Point Type"]
    matrix_rows = [matrix_headers]
    for f in plan.formulations:
        matrix_rows.append([f.sample_id, round(f.additive_conc_mM, 2), round(f.salt_conc_M, 2), f.point_type])

    anchor = plan.anchor
    ceil_str = f"{anchor.solubility_ceiling_mM:.1f} mM" if anchor.solubility_ceiling_mM is not None else "N/A"
    param_headers = ["Design Parameter", "Configured Value"]
    param_rows = [
        param_headers,
        ["Plan Identifier", plan.plan_id],
        ["Sampling Method", plan.method.upper()],
        ["Created At", str(getattr(plan, "created_at", "N/A"))],
        ["Additive Name", anchor.additive_name],
        ["Target Salt System", anchor.target_electrolyte_salt],
        ["Literature Baseline", f"{anchor.published_concentration:.1f} mM in {anchor.parent_salt}"],
        ["Concentration Scaling Factor", f"{anchor.scaling_factor:.2f}x"],
        ["Shifted Reference Center", f"{anchor.shifted_anchor_conc:.2f} mM"],
        ["Exploration Range [Min, Max]", f"[{anchor.min_conc:.1f}, {anchor.max_conc:.1f}] mM"],
        ["Delaney ESOL 80% Safety Cap", f"{ceil_str} (Enforced: {anchor.is_capped_by_solubility})"],
        ["Scaling Rationale", anchor.electrolyte_adjustment_rationale],
    ]
    return make_multi_sheet_xlsx({
        "Formulation Matrix": matrix_rows,
        "Design Parameters": param_rows,
    })


def generate_doe_plan_docx(plan: DoEPlan) -> bytes:
    """Generate Word .docx document for Electrolyte Formulation DoE Plan."""
    anchor = plan.anchor
    ceil_str = f"{anchor.solubility_ceiling_mM:.1f} mM" if anchor.solubility_ceiling_mM is not None else "N/A"
    created = getattr(plan, "created_at", None) or "N/A"

    table_rows = [["Sample ID", "Additive Conc (mM)", "Salt Conc (M)", "Sampling Point Type"]]
    for f in plan.formulations:
        table_rows.append([f.sample_id, f"{f.additive_conc_mM:.2f}", f"{f.salt_conc_M:.2f}", f.point_type])

    sections = [
        ("heading", "1. Design of Experiments (DoE) Overview"),
        ("bullet", f"Plan Identifier: {plan.plan_id}"),
        ("bullet", f"Sampling Strategy: {plan.method.upper()}"),
        ("bullet", f"Additive Candidate: {anchor.additive_name}"),
        ("bullet", f"Target Salt System: {anchor.target_electrolyte_salt}"),
        ("bullet", f"Created Timestamp: {created}"),
        ("heading", "2. Concentration Scaling & Delaney ESOL Physical Limits"),
        ("bullet", f"Literature Baseline: {anchor.published_concentration:.1f} mM in {anchor.parent_salt}"),
        ("bullet", f"Concentration Scaling Factor: {anchor.scaling_factor:.2f}x"),
        ("bullet", f"Shifted Reference Concentration: {anchor.shifted_anchor_conc:.2f} mM"),
        ("bullet", f"Parameter Exploration Range: [{anchor.min_conc:.1f}, {anchor.max_conc:.1f}] mM"),
        ("bullet", f"Delaney ESOL 80% Safety Ceiling: {ceil_str} (Active Capping: {anchor.is_capped_by_solubility})"),
        ("bullet", f"Mechanistic Scaling Rationale: {anchor.electrolyte_adjustment_rationale}"),
        ("heading", "3. Planned Formulation Coordinates"),
        ("paragraph", "The table below lists all planned electrolyte test formulations respecting solubility guardrails:"),
        ("table", table_rows),
    ]
    return make_docx("Electrolyte Formulation Design of Experiments (DoE) Plan", sections)


def generate_doe_plan_txt(plan: DoEPlan) -> str:
    """Generate human-readable, editable text report of Formulation DoE Plan."""
    anchor = plan.anchor
    ceil_str = f"{anchor.solubility_ceiling_mM:.1f} mM" if anchor.solubility_ceiling_mM is not None else "N/A"
    created = getattr(plan, "created_at", None) or "N/A"
    lines = [
        "=" * 80,
        "ELECTROLYTE FORMULATION DESIGN OF EXPERIMENTS (DoE) PLAN",
        "=" * 80,
        f"Plan ID               : {plan.plan_id}",
        f"Sampling Method       : {plan.method.upper()}",
        f"Created At            : {created}",
        f"Additive Name         : {anchor.additive_name}",
        f"Target Salt           : {anchor.target_electrolyte_salt}",
        "-" * 80,
        "CONCENTRATION SCALING & SAFETY GUARDRAILS:",
        "-" * 80,
        f"Reported Baseline     : {anchor.published_concentration:.1f} mM in {anchor.parent_salt}",
        f"Scaling Factor        : {anchor.scaling_factor:.2f}x",
        f"Shifted Reference     : {anchor.shifted_anchor_conc:.2f} mM",
        f"Exploration Range     : [{anchor.min_conc:.1f}, {anchor.max_conc:.1f}] mM",
        f"ESOL 80% Safety Cap   : {ceil_str} (Enforced: {anchor.is_capped_by_solubility})",
        f"Scaling Rationale     : {anchor.electrolyte_adjustment_rationale}",
        "-" * 80,
        "PLANNED FORMULATION POINTS (Modify parameters below as needed):",
        "-" * 80,
        f"{'Sample ID':<10} | {'Additive Conc (mM)':<18} | {'Salt Conc (M)':<13} | {'Sampling Point Type':<15}",
        "-" * 80,
    ]
    for f in plan.formulations:
        lines.append(
            f"{f.sample_id:<10} | {f.additive_conc_mM:<18.2f} | {f.salt_conc_M:<13.2f} | {f.point_type:<15}"
        )
    lines.extend([
        "-" * 80,
        "NOTES:",
        "1. All proposed formulation concentrations strictly respect the Delaney ESOL 80% saturation cap.",
        "2. Modify concentration targets above as required for manual benchtop testing or custom robot workflows.",
        "=" * 80,
    ])
    return "\n".join(lines)


def _format_failure_mode(mode: Optional[str]) -> str:
    """Format raw machine failure mode token into publication-grade human phrasing without underscores."""
    if not mode or str(mode).lower() in ("none", "", "null", "n/a"):
        return "None"
    clean = str(mode).replace("_", " ").strip().title()
    if "Dendritic" in clean and "Circuit" not in clean:
        clean = "Dendritic Short Circuit"
    return clean


def generate_campaign_summary_xlsx(campaign: CampaignResult) -> bytes:
    """
    Generate publication-grade 5-sheet Microsoft Excel master dossier summarizing
    the end-to-end closed-loop materials acceleration campaign.
    Worksheets:
      1. 1_Executive_Overview
      2. 2_Literature_Baseline
      3. 3_Robotic_Dispensing
      4. 4_Evaluated_Cells
      5. 5_Diagnostic_Summary
    """
    best = campaign.best_candidate
    best_id = best.sample_id if best else "N/A"
    best_score = round(best.performance_score, 3) if (best and best.performance_score is not None) else 0.0
    best_conc = f"{best.additive_conc_mM:.1f} mM" if best else "N/A"
    best_salt = f"{best.salt_conc_M:.2f} M" if best else "N/A"

    # Sheet 1: 1_Executive_Overview
    overview_headers = ["Executive Audit Metric", "Recorded Value", "Unit / Scope", "Description"]
    overview_rows = [
        overview_headers,
        ["Campaign Identifier", campaign.campaign_id, "String", "Unique autonomous campaign tracking code"],
        ["Target Electrolyte Salt", campaign.target_salt, "Salt Chemistry", "Host deployment electrolyte salt"],
        ["Tested Additive Candidate", campaign.formulation_plan.additive_name, "Molecule", "Target functional additive evaluated"],
        ["Champion Sample ID", best_id, "Sample", "Top-performing formulation discovered"],
        ["Champion Formulation", f"{best_conc} in {best_salt} {campaign.target_salt}", "Composition", "Optimal concentration balance"],
        ["Champion Performance Score", best_score, "Score", "Multi-objective composite merit score"],
        ["Total Optimization Iterations", campaign.total_iterations, "Rounds", "Number of closed-loop active learning rounds"],
        ["Total Formulations Evaluated", len(campaign.cell_results), "Formulations", "Total individual cell formulations synthesized & tested"],
        ["Pareto-Optimal Candidates", len(campaign.pareto_optimal_candidates), "Candidates", "Non-dominated candidates on the trade-off frontier"],
        ["Optimization Strategy", getattr(campaign.formulation_plan, "method", "Active Learning / BO").upper(), "Algorithm", "Sampling and optimization approach"],
        ["Robotic Labware Format", campaign.dispensing_schedule.rack_format, "Labware", "Opentrons OT-2 vial rack configuration"],
        ["OT-2 Deck Slot", campaign.dispensing_schedule.ot2_deck_slot, "Slot Index", "Physical deck position on Opentrons OT-2"],
        ["Campaign Created Timestamp", campaign.created_at or "N/A", "ISO Timestamp", "Execution and archiving timestamp"],
    ]

    # Sheet 2: 2_Literature_Baseline
    anchor = campaign.formulation_plan.anchor
    ceil_str = round(anchor.solubility_ceiling_mM, 2) if anchor.solubility_ceiling_mM is not None else "N/A"
    lim_str = round(anchor.solubility_limit_mM, 2) if anchor.solubility_limit_mM is not None else "N/A"
    cap_status = "Active (Enforced 80% Saturation Ceiling)" if anchor.is_capped_by_solubility else "Safe Margin (Below Saturation)"

    lit_headers = ["Literature / Physicochemical Metric", "Value", "Unit", "Physical Scaling Rationale & Mechanism"]
    lit_rows = [
        lit_headers,
        ["Mined Candidate Molecule Name", campaign.formulation_plan.additive_name, "", "Functional additive extracted from peer-reviewed literature/patent corpus"],
        ["Literature Baseline Parent Salt", anchor.parent_salt, "", "Reference electrolyte salt system reported in source publication"],
        ["Published Literature Concentration", round(anchor.published_concentration, 2), anchor.published_unit, "Original reported additive dosing concentration"],
        ["Target Electrolyte Salt Chemistry", campaign.target_salt, "", "Deployment host electrolyte salt for autonomous testing"],
        ["Concentration Scaling Factor", round(anchor.scaling_factor, 3), "x", "Thermodynamic scaling factor aligning solvation coordination numbers"],
        ["Shifted Reference Concentration", round(anchor.shifted_anchor_conc, 2), anchor.anchor_unit, "Concentration shifted for target salt prior to DoE bounding"],
        ["DoE Exploration Min Concentration", round(anchor.min_conc, 2), "mM", "Lower boundary of parameter exploration space"],
        ["DoE Exploration Max Concentration", round(anchor.max_conc, 2), "mM", "Upper boundary of parameter exploration space"],
        ["Delaney ESOL Estimated Solubility", lim_str, "mM", "Thermodynamic saturation solubility limit estimated via Delaney ESOL model"],
        ["Delaney ESOL 80% Saturation Ceiling", ceil_str, "mM", "Enforced safety boundary preventing salt precipitation in robotic channels"],
        ["Delaney ESOL Capping Status", cap_status, "Status", "Guardrail enforcement state for autonomous robotic safety"],
        ["Physical Scaling Rationale", anchor.electrolyte_adjustment_rationale, "", "Mechanistic rationale for concentration shifting and solvation adjustment"],
        ["Identified Functional Groups", ", ".join(anchor.functional_groups) if anchor.functional_groups else "None detected", "", "Molecular motifs responsible for interphase regulation and desolvation"],
    ]

    # Sheet 3: 3_Robotic_Dispensing
    disp_headers = [
        "Vial ID",
        "Deck Slot",
        "Rack Format",
        "Row",
        "Column",
        "Target Total Vol (µL)",
        "Additive Stock (µL)",
        "Salt Stock (µL)",
        "Water Diluent (µL)",
        "Additive Conc (mM)",
        "Salt Conc (M)",
        "Aspiration Rate (µL/s)",
        "Dispense Rate (µL/s)",
        "Warnings / Status",
    ]
    disp_rows = [disp_headers]
    for v in campaign.dispensing_schedule.vials:
        disp_rows.append([
            v.vial_id,
            getattr(v, "slot", campaign.dispensing_schedule.ot2_deck_slot),
            campaign.dispensing_schedule.rack_format,
            v.row,
            v.col,
            round(v.target_volume_uL, 1),
            round(v.additive_stock_uL, 1),
            round(v.salt_stock_uL, 1),
            round(v.water_uL, 1),
            round(v.additive_conc_mM, 2),
            round(v.salt_conc_M, 2),
            round(getattr(v, "aspiration_rate_ul_s", campaign.dispensing_schedule.default_aspiration_rate_ul_s), 1),
            round(getattr(v, "dispense_rate_ul_s", campaign.dispensing_schedule.default_dispense_rate_ul_s), 1),
            "; ".join(v.warnings) if v.warnings else "Ready",
        ])

    # Sheet 4: 4_Evaluated_Cells
    eval_headers = [
        "Sample ID",
        "Vial ID",
        "Round",
        "Additive Conc (mM)",
        "Salt Conc (M)",
        "Active Lifespan (h)",
        "Mean Overpotential (mV)",
        "Failure Mode",
        "Score",
        "Pareto Optimal",
    ]
    eval_rows = [eval_headers]
    for c in campaign.cell_results:
        act_h = "N/A"
        overpot = "N/A"
        fail_mode = "None"
        if c.analysis_report:
            if c.analysis_report.symmetric_cell:
                symm = c.analysis_report.symmetric_cell
                if symm.active_cycling_metrics:
                    act_h = round(symm.active_cycling_metrics.get("active_lifespan_hours", 0.0), 1)
                    overpot = round(symm.active_cycling_metrics.get("mean_overpotential_mv", 0.0), 1)
                elif symm.total_lifespan_hours:
                    act_h = round(symm.total_lifespan_hours, 1)
                    overpot = round(symm.mean_overpotential_mv, 1)
                if symm.total_test_summary:
                    fail_mode = _format_failure_mode(symm.total_test_summary.get("failure_mode"))
                elif symm.failure_mode:
                    fail_mode = _format_failure_mode(symm.failure_mode)
            elif c.analysis_report.asymmetric_cell:
                asym = c.analysis_report.asymmetric_cell
                act_h = f"{asym.total_cycles} cycles"
                overpot = round(asym.mean_nucleation_overpotential_mv, 1)
                fail_mode = "None"
            elif c.analysis_report.full_cell:
                fc = c.analysis_report.full_cell
                act_h = f"{fc.total_cycles} cycles"
                overpot = f"{fc.capacity_retention_percent:.1f}% ret"
                fail_mode = "None"
            elif c.analysis_report.summary_metrics:
                act_h = c.analysis_report.summary_metrics.get("active_lifespan_hours", "N/A")
                overpot = c.analysis_report.summary_metrics.get("mean_overpotential_mv", "N/A")
                fail_mode = _format_failure_mode(c.analysis_report.summary_metrics.get("failure_mode"))

        eval_rows.append([
            c.sample_id,
            c.vial_id,
            getattr(c, "iteration_round", 0),
            round(c.additive_conc_mM, 2),
            round(c.salt_conc_M, 2),
            act_h,
            overpot,
            fail_mode,
            round(c.performance_score, 3) if c.performance_score is not None else 0.0,
            "Yes" if c.is_pareto_optimal else "No",
        ])

    # Sheet 5: 5_Diagnostic_Summary
    diag_headers = ["Diagnostic Metric", "Measured Value", "Unit", "Physical Meaning & Interpretation"]
    diag_rows = [diag_headers]

    if best:
        diag_rows.extend([
            ["Champion Sample Identifier", best.sample_id, "", "Designated top-performing electrolyte formulation"],
            ["Robotic Vial Allocation", best.vial_id, "", "Assigned Opentrons OT-2 rack position"],
            ["Additive Dosing Concentration", round(best.additive_conc_mM, 2), "mM", f"Optimized {campaign.formulation_plan.additive_name} concentration"],
            ["Target Salt Molarity", round(best.salt_conc_M, 2), "M", f"Optimized concentration of {campaign.target_salt}"],
            ["Optimization Composite Score", round(best.performance_score, 3) if best.performance_score is not None else 0.0, "Score", "Pareto-weighted multi-objective merit score"],
            ["Pareto Frontier Status", "Optimal (Non-dominated)" if best.is_pareto_optimal else "Sub-optimal", "Status", "Non-dominated trade-off balance on Pareto frontier"],
        ])

        if best.analysis_report:
            rep = best.analysis_report
            diag_rows.extend([
                ["Electrochemical Technique", rep.technique, "", "Diagnostic testing protocol executed on champion candidate"],
                ["Electrode Active Area", round(rep.geometry.area_cm2, 4), "cm²", "Working electrode geometric active area"],
                ["Anode Material", rep.geometry.anode_material or "Zn foil", "", "Negative working electrode substrate"],
                ["Cathode Material", rep.geometry.cathode_material or "Zn foil / Counter", "", "Positive or counter electrode substrate"],
                ["Data Source File", rep.file_source or "Autonomous Ingestion", "", "Raw data origin and ingestion source"],
            ])

            if rep.symmetric_cell:
                symm = rep.symmetric_cell
                act = symm.active_cycling_metrics or {}
                tts = symm.total_test_summary or {}
                act_hours = act.get("active_lifespan_hours", symm.total_lifespan_hours)
                mean_op = act.get("mean_overpotential_mv", symm.mean_overpotential_mv)
                mean_hyst = act.get("mean_hysteresis_mv", symm.mean_hysteresis_mv)
                f_mode = _format_failure_mode(tts.get("failure_mode", symm.failure_mode))
                f_time = tts.get("failure_time_hours", symm.failure_time_hours)
                short_sc = tts.get("short_circuit_detected", symm.short_circuit_detected)
                soft_sc = symm.soft_short_detected

                diag_rows.extend([
                    ["Active Cycling Lifespan", round(act_hours, 2), "h", "Stable plating/stripping duration before catastrophic failure"],
                    ["Mean Cycling Overpotential", round(mean_op, 2), "mV", "Average kinetic polarization during active cycling"],
                    ["Mean Voltage Hysteresis", round(mean_hyst, 2), "mV", "Average voltage difference between plating and stripping plateau"],
                    ["Total Cycles Evaluated", symm.total_cycles or act.get("active_cycles", 0), "cycles", "Total completed plating/stripping cycles"],
                    ["Diagnosed Failure Mode", f_mode, "", "Dominant electrochemical degradation mechanism"],
                    ["Hard Short-Circuit Detected", "Yes" if short_sc else "No", "", "Dendritic micro-short penetrating the separator"],
                    ["Soft Short-Circuit Detected", "Yes" if soft_sc else "No", "", "Localized transient dendritic contact or micro-shorts"],
                    ["Time to Failure", round(f_time, 2) if f_time is not None else "N/A", "h", "Time elapsed when cell polarization or impedance runaway occurred"],
                ])

            if rep.asymmetric_cell:
                asym = rep.asymmetric_cell
                diag_rows.extend([
                    ["Average Coulombic Efficiency", round(asym.average_ce_percent, 2), "%", "Average Zn plating/stripping reversibility across all cycles"],
                    ["Final Cycle Coulombic Efficiency", round(asym.final_ce_percent, 2), "%", "End-of-test reversibility indicating interphase stability"],
                    ["Cumulative Capacity Stripped", round(asym.cumulative_capacity_mah_cm2, 2), "mAh/cm²", "Total charge passed before cell cutoff"],
                    ["Mean Nucleation Overpotential", round(asym.mean_nucleation_overpotential_mv, 2), "mV", "Thermodynamic energy barrier to initial Zn seed formation"],
                    ["Total Cycles Completed", asym.total_cycles, "cycles", "Reversibility cycling count"],
                ])

            if rep.full_cell:
                fc = rep.full_cell
                diag_rows.extend([
                    ["Initial Discharge Capacity", round(fc.initial_discharge_capacity_mah_g, 2), "mAh/g", "Cathode-specific capacity in cycle 1"],
                    ["Final Discharge Capacity", round(fc.final_discharge_capacity_mah_g, 2), "mAh/g", "Cathode-specific capacity at final cycle"],
                    ["Capacity Retention", round(fc.capacity_retention_percent, 2), "%", "Ratio of final to initial specific discharge capacity"],
                    ["Average Coulombic Efficiency", round(fc.average_coulombic_efficiency_percent, 2), "%", "Mean charge-discharge coulombic reversibility"],
                    ["Average Energy Efficiency", round(fc.average_energy_efficiency_percent or 0.0, 2), "%", "Round-trip energetic efficiency"],
                    ["Total Full Cell Cycles", fc.total_cycles, "cycles", "Full coin cell cycling lifespan"],
                ])

            if rep.cv:
                cv = rep.cv
                diag_rows.extend([
                    ["Anodic Peak Potential (Epa)", round(cv.anodic_peak_potential_epa_v, 4) if cv.anodic_peak_potential_epa_v is not None else "N/A", "V", "Stripping peak potential"],
                    ["Cathodic Peak Potential (Epc)", round(cv.cathodic_peak_potential_epc_v, 4) if cv.cathodic_peak_potential_epc_v is not None else "N/A", "V", "Plating peak potential"],
                    ["Peak Potential Separation (ΔEp)", round(cv.peak_separation_delta_ep_mv, 2) if cv.peak_separation_delta_ep_mv is not None else "N/A", "mV", "Electrode kinetic reversibility index"],
                    ["Anodic Peak Current (Ipa)", round(cv.anodic_peak_current_ipa_ma, 3) if cv.anodic_peak_current_ipa_ma is not None else "N/A", "mA", "Peak stripping current"],
                    ["Cathodic Peak Current (Ipc)", round(cv.cathodic_peak_current_ipc_ma, 3) if cv.cathodic_peak_current_ipc_ma is not None else "N/A", "mA", "Peak plating current"],
                    ["Current Ratio (Ipa/Ipc)", round(cv.current_ratio_ipa_ipc, 3) if cv.current_ratio_ipa_ipc is not None else "N/A", "", "Stripping-to-plating current ratio"],
                    ["Plating Onset Potential", round(cv.plating_onset_potential_v, 4) if cv.plating_onset_potential_v is not None else "N/A", "V", "Onset of cathodic Zn reduction"],
                    ["Nucleation Overpotential", round(cv.nucleation_overpotential_mv, 2) if cv.nucleation_overpotential_mv is not None else "N/A", "mV", "Zn electrocrystallization nucleation barrier"],
                ])

            if rep.lsv:
                lsv = rep.lsv
                diag_rows.extend([
                    ["HER Onset Potential", round(lsv.her_onset_potential_v, 4) if lsv.her_onset_potential_v is not None else "N/A", "V", "Potential at which parasitic hydrogen evolution initiates"],
                    ["Corrosion Potential (Ecorr)", round(lsv.corrosion_potential_ecorr_v, 4) if lsv.corrosion_potential_ecorr_v is not None else "N/A", "V", "Open-circuit resting thermodynamic corrosion potential"],
                    ["Corrosion Current Density (Jcorr)", round(lsv.corrosion_current_jcorr_ma_cm2, 5) if lsv.corrosion_current_jcorr_ma_cm2 is not None else "N/A", "mA/cm²", "Tafel-extrapolated spontaneous corrosion rate"],
                    ["Cathodic Tafel Slope", round(lsv.tafel_slope_cathodic_mv_per_dec, 2) if lsv.tafel_slope_cathodic_mv_per_dec is not None else "N/A", "mV/dec", "Hydrogen evolution reaction kinetic barrier"],
                    ["Anodic Tafel Slope", round(lsv.tafel_slope_anodic_mv_per_dec, 2) if lsv.tafel_slope_anodic_mv_per_dec is not None else "N/A", "mV/dec", "Zinc dissolution reaction kinetic barrier"],
                    ["Electrochemical Stability Window", round(lsv.electrochemical_stability_window_v, 3) if lsv.electrochemical_stability_window_v is not None else "N/A", "V", "Operating voltage range before electrolyte breakdown"],
                ])

            if rep.eis:
                eis = rep.eis
                diag_rows.extend([
                    ["Solution Resistance (Rs)", round(eis.solution_resistance_rs_ohm, 3) if eis.solution_resistance_rs_ohm is not None else "N/A", "Ω", "Ohmic electrolyte and hardware bulk resistance"],
                    ["Charge Transfer Resistance (Rct)", round(eis.charge_transfer_resistance_rct_ohm, 3) if eis.charge_transfer_resistance_rct_ohm is not None else "N/A", "Ω", "Interfacial electrochemical charge transfer resistance"],
                    ["Characteristic Frequency (f0)", round(eis.characteristic_frequency_hz, 2) if eis.characteristic_frequency_hz is not None else "N/A", "Hz", "Apex relaxation frequency of high-frequency semicircle"],
                    ["Equivalent Circuit Model", eis.equivalent_circuit or "R(RQ)", "", "Fitted Randles equivalent electrical circuit"],
                ])
    else:
        diag_rows.append(["Status", "No champion cell analyzed in this campaign run", "", "Diagnostic data pending"])

    return make_multi_sheet_xlsx({
        "1_Executive_Overview": overview_rows,
        "2_Literature_Baseline": lit_rows,
        "3_Robotic_Dispensing": disp_rows,
        "4_Evaluated_Cells": eval_rows,
        "5_Diagnostic_Summary": diag_rows,
    })


def generate_campaign_summary_docx(campaign: CampaignResult) -> bytes:
    """
    Generate formal scientific executive report (.docx) summarizing the autonomous campaign
    with numbered sections and structured OpenXML tables across all 3 modules.
    """
    best = campaign.best_candidate
    best_id = best.sample_id if best else "N/A"
    best_score = f"{best.performance_score:.3f}" if (best and best.performance_score is not None) else "N/A"
    best_conc = f"{best.additive_conc_mM:.1f} mM" if best else "N/A"
    best_salt = f"{best.salt_conc_M:.2f} M" if best else "N/A"
    anchor = campaign.formulation_plan.anchor
    ceil_str = f"{anchor.solubility_ceiling_mM:.1f} mM" if anchor.solubility_ceiling_mM is not None else "N/A"
    cap_status = "Enforced 80% Saturation Ceiling" if anchor.is_capped_by_solubility else "Safe Margin (Sub-saturation)"

    # Table 1: Campaign Milestones
    milestone_table = [
        ["Milestone Parameter", "Experimental Value / Configuration"],
        ["Campaign Identifier", str(campaign.campaign_id)],
        ["Target Electrolyte Salt Chemistry", str(campaign.target_salt)],
        ["Tested Functional Additive", str(campaign.formulation_plan.additive_name)],
        ["Optimization Strategy", str(getattr(campaign.formulation_plan, "method", "Active Learning / BO")).upper()],
        ["Total Formulations Screened", str(len(campaign.cell_results))],
        ["Optimization Rounds Executed", str(campaign.total_iterations)],
        ["Pareto-Optimal Formulations", str(len(campaign.pareto_optimal_candidates))],
        ["Champion Sample Identifier", str(best_id)],
        ["Champion Formulation Composition", f"{best_conc} additive in {best_salt} {campaign.target_salt}"],
        ["Champion Performance Merit Score", str(best_score)],
        ["Dossier Generation Date", str(campaign.created_at or "N/A")],
    ]

    # Table 2: Literature & Physical Scaling Guardrails
    lit_table = [
        ["Physicochemical Parameter", "Literature Anchor", "MAP Scaled Formulation", "Guardrail & Status"],
        ["Host Electrolyte Salt", str(anchor.parent_salt), str(campaign.target_salt), "Host System Transfer"],
        ["Additive Baseline Concentration", f"{anchor.published_concentration:.1f} {anchor.published_unit}", f"{anchor.shifted_anchor_conc:.2f} {anchor.anchor_unit}", f"{anchor.scaling_factor:.2f}x Scaling Ratio"],
        ["Parameter Exploration Range", "Single Data Point", f"[{anchor.min_conc:.1f}, {anchor.max_conc:.1f}] {anchor.anchor_unit}", "DoE Optimization Bounds"],
        ["Delaney ESOL Solubility Ceiling", "Unreported", ceil_str, cap_status],
        ["Mechanistic Scaling Rationale", "N/A", str(anchor.electrolyte_adjustment_rationale), "Solvation Sheath Coordination"],
    ]

    # Table 3: Opentrons OT-2 Dispensing Table
    dispensing_table = [
        ["Vial ID", "Row/Col", "Deck Slot", "Additive Stock (µL)", "Salt Stock (µL)", "Water (µL)", "Additive (mM)", "Salt (M)"]
    ]
    for v in campaign.dispensing_schedule.vials:
        dispensing_table.append([
            v.vial_id,
            f"{v.row}{v.col}",
            str(getattr(v, "slot", campaign.dispensing_schedule.ot2_deck_slot)),
            f"{v.additive_stock_uL:.1f}",
            f"{v.salt_stock_uL:.1f}",
            f"{v.water_uL:.1f}",
            f"{v.additive_conc_mM:.1f}",
            f"{v.salt_conc_M:.2f}",
        ])
    if len(dispensing_table) == 1:
        dispensing_table.append(["None", "-", "-", "0.0", "0.0", "0.0", "0.0", "0.0"])

    # Table 4: Evaluated Formulation Matrix
    form_table = [
        ["Sample ID", "Vial", "Round", "Additive (mM)", "Salt (M)", "Active Life (h)", "Overpot (mV)", "Failure Mode", "Score", "Pareto?"]
    ]
    for c in campaign.cell_results:
        act_h = "N/A"
        overpot = "N/A"
        fail_mode = "None"
        if c.analysis_report:
            if c.analysis_report.symmetric_cell:
                symm = c.analysis_report.symmetric_cell
                if symm.active_cycling_metrics:
                    act_h = f"{symm.active_cycling_metrics.get('active_lifespan_hours', 0.0):.1f}"
                    overpot = f"{symm.active_cycling_metrics.get('mean_overpotential_mv', 0.0):.1f}"
                elif symm.total_lifespan_hours:
                    act_h = f"{symm.total_lifespan_hours:.1f}"
                    overpot = f"{symm.mean_overpotential_mv:.1f}"
                if symm.total_test_summary:
                    fail_mode = _format_failure_mode(symm.total_test_summary.get("failure_mode"))
                elif symm.failure_mode:
                    fail_mode = _format_failure_mode(symm.failure_mode)
            elif c.analysis_report.asymmetric_cell:
                asym = c.analysis_report.asymmetric_cell
                act_h = f"{asym.total_cycles} cyc"
                overpot = f"{asym.mean_nucleation_overpotential_mv:.1f}"
            elif c.analysis_report.full_cell:
                fc = c.analysis_report.full_cell
                act_h = f"{fc.total_cycles} cyc"
                overpot = f"{fc.capacity_retention_percent:.1f}%"
            elif c.analysis_report.summary_metrics:
                act_h = str(c.analysis_report.summary_metrics.get("active_lifespan_hours", "N/A"))
                overpot = str(c.analysis_report.summary_metrics.get("mean_overpotential_mv", "N/A"))
                fail_mode = _format_failure_mode(c.analysis_report.summary_metrics.get("failure_mode"))

        score_val = f"{c.performance_score:.2f}" if c.performance_score is not None else "0.00"
        form_table.append([
            c.sample_id,
            c.vial_id,
            str(getattr(c, "iteration_round", 0)),
            f"{c.additive_conc_mM:.1f}",
            f"{c.salt_conc_M:.2f}",
            act_h,
            overpot,
            fail_mode,
            score_val,
            "Yes" if c.is_pareto_optimal else "No",
        ])
    if len(form_table) == 1:
        form_table.append(["None", "-", "0", "0.0", "0.00", "N/A", "N/A", "None", "0.00", "No"])

    # Table 5: Active Learning Iteration Progression
    hist_table = [["Round", "New Samples", "Total Cumulative Cells", "Champion Sample", "Champion Score"]]
    if campaign.history:
        for h in campaign.history:
            hist_table.append([
                str(h.get("iteration_round", 0)),
                str(h.get("new_samples", 0)),
                str(h.get("total_cells", 0)),
                str(h.get("best_sample_id", "N/A")),
                f"{h.get('best_score', 0.0):.2f}",
            ])

    # Table 6: Champion Electrochemical Diagnostics
    diag_table = [["Diagnostic Metric", "Measured Value", "Unit", "Physical Meaning & Interpretation"]]
    if best and best.analysis_report:
        rep = best.analysis_report
        diag_table.append(["Electrochemical Technique", rep.technique, "", "Diagnostic testing protocol"])
        diag_table.append(["Electrode Active Area", f"{rep.geometry.area_cm2:.4f}", "cm²", "Working electrode geometric area"])
        diag_table.append(["Anode Substrate", rep.geometry.anode_material or "Zn foil", "", "Negative working electrode substrate"])

        if rep.symmetric_cell:
            symm = rep.symmetric_cell
            act = symm.active_cycling_metrics or {}
            tts = symm.total_test_summary or {}
            act_hours = act.get("active_lifespan_hours", symm.total_lifespan_hours)
            mean_op = act.get("mean_overpotential_mv", symm.mean_overpotential_mv)
            mean_hyst = act.get("mean_hysteresis_mv", symm.mean_hysteresis_mv)
            f_mode = _format_failure_mode(tts.get("failure_mode", symm.failure_mode))
            diag_table.append(["Active Lifespan", f"{act_hours:.1f}", "h", "Stable plating/stripping duration before failure"])
            diag_table.append(["Mean Overpotential", f"{mean_op:.1f}", "mV", "Average kinetic polarization during active cycling"])
            diag_table.append(["Mean Voltage Hysteresis", f"{mean_hyst:.1f}", "mV", "Difference between plating and stripping plateau"])
            diag_table.append(["Diagnosed Failure Mode", f_mode, "", "Dominant electrochemical degradation mechanism"])

        if rep.asymmetric_cell:
            asym = rep.asymmetric_cell
            diag_table.append(["Average Coulombic Efficiency", f"{asym.average_ce_percent:.1f}", "%", "Average Zn plating/stripping reversibility"])
            diag_table.append(["Final Cycle Coulombic Efficiency", f"{asym.final_ce_percent:.1f}", "%", "End-of-test reversibility indicating interphase stability"])
            diag_table.append(["Mean Nucleation Overpotential", f"{asym.mean_nucleation_overpotential_mv:.1f}", "mV", "Energy barrier for Zn seed formation"])

        if rep.full_cell:
            fc = rep.full_cell
            diag_table.append(["Initial Discharge Capacity", f"{fc.initial_discharge_capacity_mah_g:.1f}", "mAh/g", "Cathode-specific capacity in cycle 1"])
            diag_table.append(["Final Discharge Capacity", f"{fc.final_discharge_capacity_mah_g:.1f}", "mAh/g", "Cathode-specific capacity at final cycle"])
            diag_table.append(["Capacity Retention", f"{fc.capacity_retention_percent:.1f}", "%", "Final to initial specific discharge capacity ratio"])
            diag_table.append(["Average Coulombic Efficiency", f"{fc.average_coulombic_efficiency_percent:.1f}", "%", "Mean charge-discharge reversibility"])

        if rep.cv:
            cv = rep.cv
            if cv.peak_separation_delta_ep_mv is not None:
                diag_table.append(["Peak Potential Separation (ΔEp)", f"{cv.peak_separation_delta_ep_mv:.1f}", "mV", "Electrode kinetic reversibility index"])
            if cv.anodic_peak_current_ipa_ma is not None:
                diag_table.append(["Anodic Peak Current (Ipa)", f"{cv.anodic_peak_current_ipa_ma:.2f}", "mA", "Peak stripping current"])

        if rep.lsv:
            lsv = rep.lsv
            if lsv.corrosion_potential_ecorr_v is not None:
                diag_table.append(["Corrosion Potential (Ecorr)", f"{lsv.corrosion_potential_ecorr_v:.3f}", "V", "Open-circuit thermodynamic corrosion potential"])
            if lsv.tafel_slope_cathodic_mv_per_dec is not None:
                diag_table.append(["Cathodic Tafel Slope", f"{lsv.tafel_slope_cathodic_mv_per_dec:.1f}", "mV/dec", "HER kinetic activation barrier"])

        if rep.eis:
            eis = rep.eis
            if eis.solution_resistance_rs_ohm is not None:
                diag_table.append(["Solution Resistance (Rs)", f"{eis.solution_resistance_rs_ohm:.2f}", "Ω", "Bulk electrolyte and contact resistance"])
            if eis.charge_transfer_resistance_rct_ohm is not None:
                diag_table.append(["Charge Transfer Resistance (Rct)", f"{eis.charge_transfer_resistance_rct_ohm:.2f}", "Ω", "Interfacial desolvation & charge transfer resistance"])
    else:
        diag_table.append(["Diagnostic Status", "Pending", "", "Electrochemical analysis data pending or not evaluated"])

    sections = [
        ("heading", "1. Executive Summary & Autonomous MAP Campaign Overview"),
        ("paragraph", f"This executive dossier compiles the complete closed-loop Materials Acceleration Platform (MAP) experimental investigation ({campaign.campaign_id}) for next-generation aqueous zinc battery electrolytes. The autonomous workflow seamlessly bridges Module 1 literature knowledge mining and physical solubility scaling, Module 2 Design of Experiments (DoE) formulation planning and Opentrons OT-2 robotic execution, and Module 3 multi-technique electrochemical cycling and failure mode diagnostics."),
        ("paragraph", f"Across {campaign.total_iterations} sequential optimization iteration(s), a total of {len(campaign.cell_results)} candidate electrolyte formulations were systematically synthesized, assembled, and electrochemically profiled. The platform identified {len(campaign.pareto_optimal_candidates)} Pareto-optimal formulation(s) balancing kinetic polarization and operational lifespan, converging on champion candidate {best_id} ({best_conc} additive in {best_salt} {campaign.target_salt}) with an optimal performance score of {best_score}."),
        ("table", milestone_table),

        ("heading", "2. Module 1 Literature Mining, Delaney ESOL Solubility Capping & Scaling"),
        ("paragraph", f"Module 1 extracts reported additive chemistries from peer-reviewed literature and patent corpora, establishing a baseline formulation anchor. To transfer this knowledge safely to the target {campaign.target_salt} host electrolyte, physical concentration scaling and thermodynamic solubility capping were applied."),
        ("paragraph", f"The reference literature reported {campaign.formulation_plan.additive_name} in {anchor.parent_salt} at {anchor.published_concentration:.1f} {anchor.published_unit}. Applying a mechanistic scaling factor of {anchor.scaling_factor:.2f}x shifted the anchor concentration to {anchor.shifted_anchor_conc:.2f} {anchor.anchor_unit} for {campaign.target_salt} ({anchor.electrolyte_adjustment_rationale})."),
        ("paragraph", f"To prevent solute precipitation, salt salting-out, or fluidic channel clogging during automated liquid dispensing, aqueous solubility was estimated using the Delaney ESOL thermodynamic model. The platform enforced an 80% saturation safety ceiling ({ceil_str}), resulting in a guardrail status of: {cap_status}."),
        ("table", lit_table),

        ("heading", "3. Module 2 DoE Formulation Matrix & Opentrons OT-2 Pipetting Protocol"),
        ("paragraph", f"Module 2 generated a high-dimensional formulation matrix using {campaign.formulation_plan.method.upper()} Design of Experiments (DoE), exploring additive concentrations [{anchor.min_conc:.1f}, {anchor.max_conc:.1f}] mM against target salt molarity ranges. The design points were automatically translated into an Opentrons OT-2 robotic dispensing schedule with deck labware mapping and microfluidic flow rate parameters."),
        ("bullet", f"Opentrons OT-2 Labware Format: {campaign.dispensing_schedule.rack_format}"),
        ("bullet", f"Deck Allocation: Slot {campaign.dispensing_schedule.ot2_deck_slot} with {len(campaign.dispensing_schedule.vials)} total vials configured"),
        ("bullet", f"Pipetting Flow Dynamics: {campaign.dispensing_schedule.default_aspiration_rate_ul_s} µL/s aspiration rate, {campaign.dispensing_schedule.default_dispense_rate_ul_s} µL/s dispense rate"),
        ("paragraph", "Robotic liquid handling formulation coordinates and stock dispensing instructions:"),
        ("table", dispensing_table),

        ("heading", "4. Module 3 Battery Cycling Evaluation, Polarization & Failure Diagnostics"),
        ("paragraph", "Module 3 executed rigorous electrochemical testing on all formulated cells. Voltage timeseries data were analyzed via a two-stage signal processing engine to compute active cycling lifespan, mean overpotential, voltage hysteresis, and identify early diagnostic failure mechanisms (including dendritic short circuits, soft shorts, and interfacial passivation)."),
        ("paragraph", "The complete formulation evaluation matrix across all optimization rounds is summarized below:"),
        ("table", form_table),
    ]

    if campaign.history:
        sections.extend([
            ("paragraph", "Sequential active learning progression across optimization rounds:"),
            ("table", hist_table),
        ])

    sections.extend([
        ("heading", "5. Top Recommended Formulation & Future Screening Recommendations"),
        ("paragraph", f"Based on multi-objective Pareto trade-off optimization between cycling lifespan and polarization overpotential, Sample {best_id} is designated as the champion electrolyte formulation for {campaign.target_salt} chemistry."),
        ("bullet", f"Recommended Formulation: {best_conc} {campaign.formulation_plan.additive_name} in {best_salt} {campaign.target_salt}"),
        ("bullet", f"Performance Merit Score: {best_score} (Optimal Pareto non-dominated candidate)"),
        ("paragraph", "Champion candidate electrochemical diagnostic profile:"),
        ("table", diag_table),
        ("paragraph", "Strategic Next Steps & Translation Recommendations:"),
        ("bullet", "Pouch Cell & Prototype Scaling: Transition champion formulation from coin cells to multi-layer pouch cells to evaluate performance under commercial areal capacities (>2 mAh/cm²)."),
        ("bullet", "Operando Interfacial Spectroscopy: Perform in situ Raman, FTIR, and XRD to confirm the molecular coordination structure in the Zn²⁺ inner solvation sheath and passivating SEI film."),
        ("bullet", "Narrow-Range DoE Refinement: Launch an active learning round localized within ±10% of champion coordinates to pinpoint the global optimum."),
        ("bullet", "Accelerated Calendar & Temperature Aging: Conduct high-temperature (45°C) and resting float tests to quantify hydrogen evolution suppression and corrosion inhibition over 1000+ hours."),
    ])

    return make_docx("Autonomous Zinc Battery Optimization — Laboratory Experiment Dossier", sections)


def generate_campaign_summary_txt(campaign: CampaignResult) -> str:
    """Generate human-readable, editable text report of the complete laboratory experiment dossier."""
    best = campaign.best_candidate
    best_id = best.sample_id if best else "N/A"
    best_score = f"{best.performance_score:.3f}" if (best and best.performance_score is not None) else "N/A"
    best_conc = f"{best.additive_conc_mM:.1f} mM" if best else "N/A"
    best_salt = f"{best.salt_conc_M:.2f} M" if best else "N/A"
    anchor = campaign.formulation_plan.anchor
    ceil_str = f"{anchor.solubility_ceiling_mM:.1f} mM" if anchor.solubility_ceiling_mM is not None else "N/A"

    lines = [
        "=" * 96,
        "AUTONOMOUS MATERIALS ACCELERATION PLATFORM (MAP) - LABORATORY EXPERIMENT DOSSIER",
        "=" * 96,
        "",
        "SECTION 1: EXECUTIVE SUMMARY & AUTONOMOUS MAP CAMPAIGN OVERVIEW",
        "-" * 96,
        f"Campaign Identifier       : {campaign.campaign_id}",
        f"Target Electrolyte Salt   : {campaign.target_salt}",
        f"Tested Functional Additive: {campaign.formulation_plan.additive_name}",
        f"Optimization Iterations   : {campaign.total_iterations} Round(s)",
        f"Total Formulations Tested : {len(campaign.cell_results)} Candidates",
        f"Pareto Optimal Candidates : {len(campaign.pareto_optimal_candidates)}",
        f"Champion Sample Discovered: {best_id} ({best_conc} additive in {best_salt} {campaign.target_salt})",
        f"Champion Merit Score      : {best_score}",
        f"Created Timestamp         : {campaign.created_at or 'N/A'}",
        "",
        "SECTION 2: MODULE 1 LITERATURE MINING, DELANEY ESOL SOLUBILITY CAPPING & SCALING",
        "-" * 96,
        f"Literature Baseline Salt  : {anchor.parent_salt}",
        f"Reported Baseline Conc    : {anchor.published_concentration:.1f} {anchor.published_unit}",
        f"Concentration Scaling     : {anchor.scaling_factor:.2f}x",
        f"Shifted Reference Conc    : {anchor.shifted_anchor_conc:.2f} {anchor.anchor_unit}",
        f"Exploration Range         : [{anchor.min_conc:.1f}, {anchor.max_conc:.1f}] {anchor.anchor_unit}",
        f"Delaney ESOL Saturation Cap: {ceil_str} (Active Capping: {anchor.is_capped_by_solubility})",
        f"Scaling Rationale         : {anchor.electrolyte_adjustment_rationale}",
        "",
        "SECTION 3: MODULE 2 DoE FORMULATION MATRIX & OPENTRONS OT-2 PIPETTING PROTOCOL",
        "-" * 96,
        f"Robotic Labware Format    : {campaign.dispensing_schedule.rack_format}",
        f"Opentrons OT-2 Deck Slot  : Slot {campaign.dispensing_schedule.ot2_deck_slot}",
        f"Aspiration / Dispense Rate: {campaign.dispensing_schedule.default_aspiration_rate_ul_s} µL/s / {campaign.dispensing_schedule.default_dispense_rate_ul_s} µL/s",
        f"Total Dispensing Vials    : {len(campaign.dispensing_schedule.vials)}",
        f"{'Vial ID':<8} | {'Pos':<5} | {'Slot':<5} | {'Additive (µL)':<14} | {'Salt (µL)':<10} | {'Water (µL)':<11} | {'Additive (mM)':<14} | {'Salt (M)':<9}",
        "-" * 96,
    ]
    for v in campaign.dispensing_schedule.vials:
        lines.append(
            f"{v.vial_id:<8} | {f'{v.row}{v.col}':<5} | {getattr(v, 'slot', campaign.dispensing_schedule.ot2_deck_slot):<5} | {v.additive_stock_uL:<14.1f} | {v.salt_stock_uL:<10.1f} | {v.water_uL:<11.1f} | {v.additive_conc_mM:<14.1f} | {v.salt_conc_M:<9.2f}"
        )

    lines.extend([
        "",
        "SECTION 4: MODULE 3 BATTERY CYCLING EVALUATION, POLARIZATION & FAILURE DIAGNOSTICS",
        "-" * 96,
        f"{'Sample ID':<10} | {'Vial':<5} | {'Round':<6} | {'Additive (mM)':<14} | {'Salt (M)':<9} | {'Active (h)':<11} | {'Overpot (mV)':<13} | {'Failure Mode':<20} | {'Score':<6} | {'Pareto':<6}",
        "-" * 96,
    ])
    for c in campaign.cell_results:
        act_h = "N/A"
        overpot = "N/A"
        fail_mode = "None"
        if c.analysis_report:
            if c.analysis_report.symmetric_cell:
                symm = c.analysis_report.symmetric_cell
                if symm.active_cycling_metrics:
                    act_h = f"{symm.active_cycling_metrics.get('active_lifespan_hours', 0.0):.1f}"
                    overpot = f"{symm.active_cycling_metrics.get('mean_overpotential_mv', 0.0):.1f}"
                elif symm.total_lifespan_hours:
                    act_h = f"{symm.total_lifespan_hours:.1f}"
                    overpot = f"{symm.mean_overpotential_mv:.1f}"
                if symm.total_test_summary:
                    fail_mode = _format_failure_mode(symm.total_test_summary.get("failure_mode"))
                elif symm.failure_mode:
                    fail_mode = _format_failure_mode(symm.failure_mode)
            elif c.analysis_report.asymmetric_cell:
                asym = c.analysis_report.asymmetric_cell
                act_h = f"{asym.total_cycles} cyc"
                overpot = f"{asym.mean_nucleation_overpotential_mv:.1f}"
            elif c.analysis_report.full_cell:
                fc = c.analysis_report.full_cell
                act_h = f"{fc.total_cycles} cyc"
                overpot = f"{fc.capacity_retention_percent:.1f}%"
            elif c.analysis_report.summary_metrics:
                act_h = str(c.analysis_report.summary_metrics.get("active_lifespan_hours", "N/A"))
                overpot = str(c.analysis_report.summary_metrics.get("mean_overpotential_mv", "N/A"))
                fail_mode = _format_failure_mode(c.analysis_report.summary_metrics.get("failure_mode"))

        score_val = f"{c.performance_score:.2f}" if c.performance_score is not None else "0.00"
        lines.append(
            f"{c.sample_id:<10} | {c.vial_id:<5} | {getattr(c, 'iteration_round', 0):<6} | {c.additive_conc_mM:<14.1f} | {c.salt_conc_M:<9.2f} | {act_h:<11} | {overpot:<13} | {fail_mode:<20} | {score_val:<6} | {'Yes' if c.is_pareto_optimal else 'No':<6}"
        )

    if campaign.history:
        lines.extend([
            "",
            "SEQUENTIAL ACTIVE LEARNING ITERATION PROGRESSION:",
            "-" * 96,
            f"{'Round':<6} | {'New Samples':<12} | {'Cumulative Cells':<17} | {'Champion Sample':<16} | {'Champion Score':<14}",
            "-" * 96,
        ])
        for h in campaign.history:
            lines.append(
                f"{h.get('iteration_round', 0):<6} | {h.get('new_samples', 0):<12} | {h.get('total_cells', 0):<17} | {h.get('best_sample_id', 'N/A'):<16} | {h.get('best_score', 0.0):<14.2f}"
            )

    lines.extend([
        "",
        "SECTION 5: TOP RECOMMENDED FORMULATION & FUTURE SCREENING RECOMMENDATIONS",
        "-" * 96,
        f"Champion Formulation      : Sample {best_id} ({best_conc} in {best_salt} {campaign.target_salt})",
        f"Champion Score            : {best_score}",
        "Translation Roadmap:",
        "1. Prototype Scaling       : Assemble multilayer pouch cells (>2 mAh/cm²) to assess commercial viability.",
        "2. Operando Characterization: In situ Raman / FTIR / XRD to verify Zn desolvation kinetics and SEI formation.",
        "3. Active Learning Bounds  : Launch localized DoE round around champion coordinates for fine-tuning.",
        "4. Long-Term Calendar Aging: 1000+ hour resting tests at elevated temperatures (45°C) for corrosion rate audit.",
        "=" * 96,
        "END OF LABORATORY EXPERIMENT DOSSIER",
        "=" * 96,
    ])
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Clean Electrochemical Dataset Exporters (OriginLab / GraphPad Prism)
# -----------------------------------------------------------------------------

def generate_electrochemical_dataset_csv(report: ElectrochemicalAnalysisReport) -> str:
    """
    Generate clean, OriginLab / GraphPad Prism compatible CSV dataset.
    Exports timeseries columns:
      Time (h), Potential (V), Current (mA), Current Density (mA/cm²), Capacity (mAh).
    """
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    headers = ["Time (h)", "Potential (V)", "Current (mA)", "Current Density (mA/cm²)", "Capacity (mAh)"]
    writer.writerow(headers)

    pdata = report.processed_data
    n_pts = len(pdata.voltage_v) if pdata.voltage_v else len(pdata.time_s)
    area = report.geometry.area_cm2 if (report.geometry and report.geometry.area_cm2 > 0) else 1.0

    for i in range(n_pts):
        if pdata.time_s and i < len(pdata.time_s):
            t_h = round(pdata.time_s[i] / 3600.0, 5)
        elif pdata.display_time and i < len(pdata.display_time):
            if pdata.time_unit == "h":
                t_h = round(pdata.display_time[i], 5)
            elif pdata.time_unit == "min":
                t_h = round(pdata.display_time[i] / 60.0, 5)
            elif pdata.time_unit == "s":
                t_h = round(pdata.display_time[i] / 3600.0, 5)
            elif pdata.time_unit == "d":
                t_h = round(pdata.display_time[i] * 24.0, 5)
            else:
                t_h = round(pdata.display_time[i], 5)
        else:
            t_h = 0.0

        v = round(pdata.voltage_v[i], 5) if pdata.voltage_v and i < len(pdata.voltage_v) else 0.0
        curr = round(pdata.current_ma[i], 5) if pdata.current_ma and i < len(pdata.current_ma) else 0.0

        if pdata.current_density_ma_cm2 and i < len(pdata.current_density_ma_cm2):
            j = round(pdata.current_density_ma_cm2[i], 5)
        else:
            j = round(curr / area, 5)

        cap = round(pdata.capacity_mah[i], 5) if pdata.capacity_mah and i < len(pdata.capacity_mah) else 0.0

        writer.writerow([t_h, v, curr, j, cap])

    return output.getvalue()


def generate_electrochemical_dataset_xlsx(report: ElectrochemicalAnalysisReport) -> bytes:
    """
    Generate multi-sheet Microsoft Excel workbook (.xlsx) with clean, publication-ready
    electrochemical datasets structured for OriginLab / GraphPad Prism:
    - Sheet 1 (Timeseries_Data): Time (h), Potential (V), Current (mA), Current Density (mA/cm²), Capacity (mAh).
    - Sheet 2 (Cycle_Summary): Cycle Number, Plating Plateau (V), Stripping Plateau (V), Overpotential (mV), Voltage Hysteresis (mV), Coulombic Efficiency (%).
    - Sheet 3 (Technique_Diagnostics): Technique-adaptive extracted metrics.
    """
    # -------------------------------------------------------------------------
    # Sheet 1: Timeseries_Data
    # -------------------------------------------------------------------------
    timeseries_headers = ["Time (h)", "Potential (V)", "Current (mA)", "Current Density (mA/cm²)", "Capacity (mAh)"]
    timeseries_rows = [timeseries_headers]

    pdata = report.processed_data
    n_pts = len(pdata.voltage_v) if pdata.voltage_v else len(pdata.time_s)
    area = report.geometry.area_cm2 if (report.geometry and report.geometry.area_cm2 > 0) else 1.0

    for i in range(n_pts):
        if pdata.time_s and i < len(pdata.time_s):
            t_h = round(pdata.time_s[i] / 3600.0, 5)
        elif pdata.display_time and i < len(pdata.display_time):
            if pdata.time_unit == "h":
                t_h = round(pdata.display_time[i], 5)
            elif pdata.time_unit == "min":
                t_h = round(pdata.display_time[i] / 60.0, 5)
            elif pdata.time_unit == "s":
                t_h = round(pdata.display_time[i] / 3600.0, 5)
            elif pdata.time_unit == "d":
                t_h = round(pdata.display_time[i] * 24.0, 5)
            else:
                t_h = round(pdata.display_time[i], 5)
        else:
            t_h = 0.0

        v = round(pdata.voltage_v[i], 5) if pdata.voltage_v and i < len(pdata.voltage_v) else 0.0
        curr = round(pdata.current_ma[i], 5) if pdata.current_ma and i < len(pdata.current_ma) else 0.0

        if pdata.current_density_ma_cm2 and i < len(pdata.current_density_ma_cm2):
            j = round(pdata.current_density_ma_cm2[i], 5)
        else:
            j = round(curr / area, 5)

        cap = round(pdata.capacity_mah[i], 5) if pdata.capacity_mah and i < len(pdata.capacity_mah) else 0.0

        timeseries_rows.append([t_h, v, curr, j, cap])

    # -------------------------------------------------------------------------
    # Sheet 2: Cycle_Summary
    # -------------------------------------------------------------------------
    cycle_headers = [
        "Cycle Number",
        "Plating Plateau (V)",
        "Stripping Plateau (V)",
        "Overpotential (mV)",
        "Voltage Hysteresis (mV)",
        "Coulombic Efficiency (%)",
    ]
    cycle_rows = [cycle_headers]

    if report.symmetric_cell and report.symmetric_cell.cycle_metrics:
        for m in report.symmetric_cell.cycle_metrics:
            cycle_rows.append([
                m.cycle_number,
                round(m.plating_plateau_v, 4),
                round(m.stripping_plateau_v, 4),
                round(m.overpotential_mv, 2),
                round(m.voltage_hysteresis_mv, 2),
                "N/A",
            ])
    elif report.asymmetric_cell and report.asymmetric_cell.cycle_metrics:
        for am in report.asymmetric_cell.cycle_metrics:
            cycle_rows.append([
                am.cycle_number,
                "N/A",
                "N/A",
                round(am.plateau_overpotential_mv, 2),
                round(am.nucleation_overpotential_mv, 2),
                round(am.coulombic_efficiency_percent, 2),
            ])
    elif report.full_cell and report.full_cell.cycle_numbers:
        fc = report.full_cell
        for idx, c_num in enumerate(fc.cycle_numbers):
            ce = round(fc.coulombic_efficiencies[idx], 2) if idx < len(fc.coulombic_efficiencies) else "N/A"
            cycle_rows.append([
                c_num,
                "N/A",
                "N/A",
                "N/A",
                "N/A",
                ce,
            ])
    elif pdata.cycle_number and pdata.coulombic_efficiency:
        seen_cycles = set()
        for c_num, ce_val in zip(pdata.cycle_number, pdata.coulombic_efficiency):
            if c_num not in seen_cycles:
                seen_cycles.add(c_num)
                cycle_rows.append([
                    c_num,
                    "N/A",
                    "N/A",
                    "N/A",
                    "N/A",
                    round(ce_val, 2) if ce_val is not None else "N/A",
                ])

    # -------------------------------------------------------------------------
    # Sheet 3: Technique_Diagnostics
    # -------------------------------------------------------------------------
    diag_headers = ["Diagnostic Metric", "Value", "Unit"]
    diag_rows = [diag_headers]

    # General metadata
    diag_rows.append(["Sample Identifier", report.sample_id, "-"])
    diag_rows.append(["File Source", report.file_source, "-"])
    tech_val = report.technique.value if hasattr(report.technique, "value") else str(report.technique)
    diag_rows.append(["Evaluated Technique", tech_val, "-"])
    if report.geometry:
        diag_rows.append(["Electrode Area", round(report.geometry.area_cm2, 4), "cm²"])
        if report.geometry.active_mass_mg is not None:
            diag_rows.append(["Active Mass", round(report.geometry.active_mass_mg, 4), "mg"])

    # Technique-adaptive extracted metrics
    if report.symmetric_cell:
        symm = report.symmetric_cell
        act_m = symm.active_cycling_metrics or {}
        diag_rows.append(["Total Cycles", symm.total_cycles, "cycles"])
        diag_rows.append(["Total Lifespan", round(symm.total_lifespan_hours, 2), "h"])
        if "active_lifespan_hours" in act_m:
            diag_rows.append(["Active Lifespan", round(act_m["active_lifespan_hours"], 2), "h"])
        if "active_cycles" in act_m:
            diag_rows.append(["Active Cycles", act_m["active_cycles"], "cycles"])
        mean_overpot = act_m.get("mean_overpotential_mv", symm.mean_overpotential_mv)
        diag_rows.append(["Mean Overpotential", round(mean_overpot, 2), "mV"])
        mean_hyst = act_m.get("mean_hysteresis_mv", symm.mean_hysteresis_mv)
        diag_rows.append(["Mean Voltage Hysteresis", round(mean_hyst, 2), "mV"])
        diag_rows.append(["Failure Mode", symm.failure_mode, "-"])
        diag_rows.append(["Failure Cycle", symm.failure_cycle if symm.failure_cycle is not None else "None", "-"])
        if symm.failure_time_hours is not None:
            diag_rows.append(["Failure Time", round(symm.failure_time_hours, 2), "h"])
        diag_rows.append(["Short Circuit Detected", "Yes" if symm.short_circuit_detected else "No", "-"])
        diag_rows.append(["Soft Short Detected", "Yes" if symm.soft_short_detected else "No", "-"])

    if report.asymmetric_cell:
        asymm = report.asymmetric_cell
        diag_rows.append(["Total Cycles", asymm.total_cycles, "cycles"])
        if asymm.average_ce_percent is not None:
            diag_rows.append(["Average Coulombic Efficiency", round(asymm.average_ce_percent, 2), "%"])
        if asymm.final_ce_percent is not None:
            diag_rows.append(["Final Coulombic Efficiency", round(asymm.final_ce_percent, 2), "%"])
        if asymm.cumulative_capacity_mah_cm2 is not None:
            diag_rows.append(["Cumulative Plating Capacity", round(asymm.cumulative_capacity_mah_cm2, 3), "mAh/cm²"])
        if asymm.mean_nucleation_overpotential_mv is not None:
            diag_rows.append(["Mean Nucleation Overpotential", round(asymm.mean_nucleation_overpotential_mv, 2), "mV"])

    if report.full_cell:
        fc = report.full_cell
        diag_rows.append(["Total Cycles", fc.total_cycles, "cycles"])
        if fc.initial_discharge_capacity_mah_g is not None:
            diag_rows.append(["Initial Discharge Capacity", round(fc.initial_discharge_capacity_mah_g, 2), "mAh/g"])
        if fc.final_discharge_capacity_mah_g is not None:
            diag_rows.append(["Final Discharge Capacity", round(fc.final_discharge_capacity_mah_g, 2), "mAh/g"])
        if fc.capacity_retention_percent is not None:
            diag_rows.append(["Capacity Retention", round(fc.capacity_retention_percent, 2), "%"])
        if fc.average_coulombic_efficiency_percent is not None:
            diag_rows.append(["Average Coulombic Efficiency", round(fc.average_coulombic_efficiency_percent, 2), "%"])
        ee = fc.average_energy_efficiency_percent or fc.energy_efficiency_percent
        if ee is not None:
            diag_rows.append(["Average Energy Efficiency", round(ee, 2), "%"])

    if report.cv:
        cv_res = report.cv
        if cv_res.anodic_peak_potential_epa_v is not None:
            diag_rows.append(["Anodic Peak Potential (Epa)", round(cv_res.anodic_peak_potential_epa_v, 4), "V"])
        if cv_res.cathodic_peak_potential_epc_v is not None:
            diag_rows.append(["Cathodic Peak Potential (Epc)", round(cv_res.cathodic_peak_potential_epc_v, 4), "V"])
        if cv_res.peak_separation_delta_ep_mv is not None:
            diag_rows.append(["Peak Separation (ΔEp)", round(cv_res.peak_separation_delta_ep_mv, 2), "mV"])
        if cv_res.anodic_peak_current_ipa_ma is not None:
            diag_rows.append(["Anodic Peak Current (Ipa)", round(cv_res.anodic_peak_current_ipa_ma, 4), "mA"])
        if cv_res.cathodic_peak_current_ipc_ma is not None:
            diag_rows.append(["Cathodic Peak Current (Ipc)", round(cv_res.cathodic_peak_current_ipc_ma, 4), "mA"])
        if cv_res.current_ratio_ipa_ipc is not None:
            diag_rows.append(["Peak Current Ratio (|Ipa/Ipc|)", round(cv_res.current_ratio_ipa_ipc, 3), "-"])
        if cv_res.plating_onset_potential_v is not None:
            diag_rows.append(["Plating Onset Potential", round(cv_res.plating_onset_potential_v, 4), "V"])
        if cv_res.stripping_onset_potential_v is not None:
            diag_rows.append(["Stripping Onset Potential", round(cv_res.stripping_onset_potential_v, 4), "V"])
        if cv_res.nucleation_overpotential_mv is not None:
            diag_rows.append(["Nucleation Overpotential", round(cv_res.nucleation_overpotential_mv, 2), "mV"])

    if report.lsv:
        lsv_res = report.lsv
        if lsv_res.corrosion_potential_ecorr_v is not None:
            diag_rows.append(["Corrosion Potential (E_corr)", round(lsv_res.corrosion_potential_ecorr_v, 4), "V"])
        if lsv_res.corrosion_current_jcorr_ma_cm2 is not None:
            diag_rows.append(["Corrosion Current Density (j_corr)", round(lsv_res.corrosion_current_jcorr_ma_cm2, 5), "mA/cm²"])
        if lsv_res.tafel_slope_cathodic_mv_per_dec is not None:
            diag_rows.append(["Cathodic Tafel Slope (βc)", round(lsv_res.tafel_slope_cathodic_mv_per_dec, 2), "mV/dec"])
        if lsv_res.tafel_slope_anodic_mv_per_dec is not None:
            diag_rows.append(["Anodic Tafel Slope (βa)", round(lsv_res.tafel_slope_anodic_mv_per_dec, 2), "mV/dec"])
        if lsv_res.her_onset_potential_v is not None:
            diag_rows.append(["HER Onset Potential", round(lsv_res.her_onset_potential_v, 4), "V"])
        if lsv_res.her_current_density_threshold_ma_cm2 is not None:
            diag_rows.append(["HER Threshold Current Density", round(lsv_res.her_current_density_threshold_ma_cm2, 3), "mA/cm²"])
        if lsv_res.electrochemical_stability_window_v is not None:
            diag_rows.append(["Electrochemical Stability Window", round(lsv_res.electrochemical_stability_window_v, 3), "V"])

    if report.eis:
        eis_res = report.eis
        if eis_res.solution_resistance_rs_ohm is not None:
            diag_rows.append(["Solution Resistance (Rs)", round(eis_res.solution_resistance_rs_ohm, 3), "Ω"])
        if eis_res.charge_transfer_resistance_rct_ohm is not None:
            diag_rows.append(["Charge Transfer Resistance (Rct)", round(eis_res.charge_transfer_resistance_rct_ohm, 3), "Ω"])
        if eis_res.characteristic_frequency_hz is not None:
            diag_rows.append(["Characteristic Frequency (f_apex)", round(eis_res.characteristic_frequency_hz, 2), "Hz"])
        if eis_res.equivalent_circuit:
            diag_rows.append(["Equivalent Circuit Model", eis_res.equivalent_circuit, "-"])

    # Additional summary metrics if present
    if report.summary_metrics:
        existing_keys = {r[0] for r in diag_rows}
        for k, v in report.summary_metrics.items():
            readable_k = k.replace("_", " ").title()
            if readable_k not in existing_keys and not isinstance(v, (dict, list)):
                val = round(v, 4) if isinstance(v, float) else v
                diag_rows.append([readable_k, val, "-"])

    return make_multi_sheet_xlsx({
        "Timeseries_Data": timeseries_rows,
        "Cycle_Summary": cycle_rows,
        "Technique_Diagnostics": diag_rows,
    })


def _extract_paper_fields(paper: Any) -> Dict[str, Any]:
    if not paper:
        return {"title": "", "authors": "", "year": "", "venue": "", "doi": "", "oa": "", "cites": "", "abstract": ""}
    if isinstance(paper, dict):
        authors_raw = paper.get("authors", [])
        return {
            "title": paper.get("title", "") or "",
            "authors": ", ".join(authors_raw) if isinstance(authors_raw, list) else str(authors_raw or ""),
            "year": paper.get("year", "") or "",
            "venue": paper.get("venue", "") or "",
            "doi": paper.get("doi", "") or "",
            "oa": paper.get("open_access_pdf_url", "") or "",
            "cites": paper.get("citation_count", "") or "",
            "abstract": paper.get("abstract", "") or ""
        }
    authors_raw = getattr(paper, "authors", []) or []
    return {
        "title": getattr(paper, "title", "") or "",
        "authors": ", ".join(authors_raw) if isinstance(authors_raw, list) else str(authors_raw or ""),
        "year": getattr(paper, "year", "") or "",
        "venue": getattr(paper, "venue", "") or "",
        "doi": getattr(paper, "doi", "") or "",
        "oa": getattr(paper, "open_access_pdf_url", "") or "",
        "cites": getattr(paper, "citation_count", "") or "",
        "abstract": getattr(paper, "abstract", "") or ""
    }


def _extract_additive_fields(add: Any) -> Dict[str, Any]:
    if not add:
        return {
            "name": "Unspecified / No additive",
            "smiles": "",
            "formula": "",
            "c_val": "",
            "c_unit": "",
            "c_raw": "",
            "base_salt": "Unspecified / Not reported",
            "mech_str": "",
            "mech_sum": "",
            "conf": "",
            "evid": ""
        }
    if isinstance(add, dict):
        add_name = add.get("name", "") or ""
        add_smi = add.get("smiles", "") or ""
        add_formula = add.get("chemical_formula", "") or ""
        if not add_formula and (add_smi or add_name):
            from discovery.formula_utils import get_chemical_formula
            add_formula = get_chemical_formula(add_smi, add_name)
        opt_c = add.get("optimal_concentration") or {}
        c_val = opt_c.get("value", "") if isinstance(opt_c, dict) else getattr(opt_c, "value", "")
        c_unit = opt_c.get("unit", "") if isinstance(opt_c, dict) else getattr(opt_c, "unit", "")
        c_raw = opt_c.get("raw_text", "") if isinstance(opt_c, dict) else getattr(opt_c, "raw_text", "")
        bc = add.get("battery_context") or {}
        base_salt = bc.get("baseline_electrolyte", "") if isinstance(bc, dict) else getattr(bc, "baseline_electrolyte", "")
        mechs = add.get("primary_mechanisms") or []
        mech_str = ", ".join(mechs) if isinstance(mechs, list) else str(mechs or "")
        return {
            "name": add_name,
            "smiles": add_smi,
            "formula": add_formula,
            "c_val": c_val,
            "c_unit": c_unit,
            "c_raw": c_raw,
            "base_salt": base_salt or "Unspecified / Not reported",
            "mech_str": mech_str,
            "mech_sum": add.get("mechanism_summary", "") or "",
            "conf": add.get("confidence_score", "") or "",
            "evid": add.get("extraction_evidence", "") or ""
        }
    add_name = getattr(add, "name", "") or ""
    add_smi = getattr(add, "smiles", "") or ""
    add_formula = getattr(add, "chemical_formula", "") or ""
    if not add_formula and (add_smi or add_name):
        from discovery.formula_utils import get_chemical_formula
        add_formula = get_chemical_formula(add_smi, add_name)
    opt_c = getattr(add, "optimal_concentration", None)
    c_val = getattr(opt_c, "value", "") if opt_c else getattr(add, "reported_concentration", "")
    c_unit = getattr(opt_c, "unit", "") if opt_c else getattr(add, "reported_concentration_unit", "")
    c_raw = getattr(opt_c, "raw_text", "") if opt_c else ""
    bc = getattr(add, "battery_context", None)
    base_salt = getattr(bc, "baseline_electrolyte", "") if bc else getattr(add, "parent_baseline_electrolyte", "")
    mechs = getattr(add, "primary_mechanisms", []) or getattr(add, "reported_mechanisms", []) or []
    mech_str = ", ".join(mechs) if isinstance(mechs, list) else str(mechs or "")
    return {
        "name": add_name,
        "smiles": add_smi,
        "formula": add_formula,
        "c_val": c_val,
        "c_unit": c_unit,
        "c_raw": c_raw,
        "base_salt": base_salt or "Unspecified / Not reported",
        "mech_str": mech_str,
        "mech_sum": getattr(add, "mechanism_summary", "") or "",
        "conf": getattr(add, "confidence_score", "") or "",
        "evid": getattr(add, "extraction_evidence", "") or ""
    }


def generate_mined_literature_csv(records: List[Dict[str, Any]]) -> str:
    """
    Generate an editable CSV table of all mined academic papers and their extracted
    electrolyte additive formulations, baseline salts, and electrochemical mechanisms.
    """
    output = io.StringIO()
    writer = csv.writer(output, lineterminator='\n')
    writer.writerow([
        "Index",
        "Paper Title",
        "Authors",
        "Year",
        "Venue",
        "DOI",
        "DOI URL",
        "Open Access PDF URL",
        "Citation Count",
        "Extracted Additive",
        "Chemical Formula",
        "SMILES",
        "Optimal Concentration",
        "Optimal Concentration Unit",
        "Optimal Concentration Raw Text",
        "Baseline Electrolyte Salt",
        "Primary Mechanisms",
        "Mechanism Summary",
        "Confidence Score",
        "Extraction Evidence",
        "Abstract",
    ])
    for idx, rec in enumerate(records or [], start=1):
        paper = rec.get("paper") if isinstance(rec, dict) else None
        if not paper and isinstance(rec, dict) and rec.get("filename"):
            paper = {"title": rec.get("filename"), "abstract": rec.get("summary", "")}
        add = rec.get("additive") if isinstance(rec, dict) else None

        p_info = _extract_paper_fields(paper)
        a_info = _extract_additive_fields(add)
        p_doi_url = f"https://doi.org/{p_info['doi']}" if p_info['doi'] else ""

        writer.writerow([
            idx,
            p_info["title"],
            p_info["authors"],
            p_info["year"],
            p_info["venue"],
            p_info["doi"],
            p_doi_url,
            p_info["oa"],
            p_info["cites"],
            a_info["name"],
            a_info["formula"],
            a_info["smiles"],
            a_info["c_val"],
            a_info["c_unit"],
            a_info["c_raw"],
            a_info["base_salt"],
            a_info["mech_str"],
            a_info["mech_sum"],
            a_info["conf"],
            a_info["evid"],
            p_info["abstract"],
        ])
    return output.getvalue()


def generate_mined_literature_xlsx(records: List[Dict[str, Any]]) -> bytes:
    """
    Generate an editable multi-worksheet Excel workbook (.xlsx) containing mined literature
    papers, extracted additive formulations, and unique candidate summaries.
    """
    headers = [
        "Index", "Paper Title", "Authors", "Year", "Venue", "DOI", "DOI URL",
        "Open Access PDF URL", "Citations", "Extracted Additive", "Chemical Formula",
        "SMILES", "Optimal Conc", "Conc Unit", "Conc Raw Text", "Baseline Salt",
        "Primary Mechanisms", "Mechanism Summary", "Confidence Score", "Extraction Evidence"
    ]
    paper_rows = [headers]
    unique_adds_map: Dict[str, List[Any]] = {}

    for idx, rec in enumerate(records or [], start=1):
        paper = rec.get("paper") if isinstance(rec, dict) else None
        if not paper and isinstance(rec, dict) and rec.get("filename"):
            paper = {"title": rec.get("filename"), "abstract": rec.get("summary", "")}
        add = rec.get("additive") if isinstance(rec, dict) else None

        p_info = _extract_paper_fields(paper)
        a_info = _extract_additive_fields(add)
        p_doi_url = f"https://doi.org/{p_info['doi']}" if p_info['doi'] else ""

        if a_info["name"] and a_info["name"].lower() not in unique_adds_map and a_info["name"].lower() != "unspecified / no additive":
            conc_disp = a_info["c_raw"] or (f"{a_info['c_val']} {a_info['c_unit']}".strip())
            unique_adds_map[a_info["name"].lower()] = [
                a_info["name"], a_info["formula"], a_info["smiles"], conc_disp, a_info["base_salt"], a_info["mech_str"], p_info["title"]
            ]

        paper_rows.append([
            idx, p_info["title"], p_info["authors"], p_info["year"], p_info["venue"], p_info["doi"], p_doi_url,
            p_info["oa"], p_info["cites"], a_info["name"], a_info["formula"], a_info["smiles"], a_info["c_val"], a_info["c_unit"], a_info["c_raw"],
            a_info["base_salt"], a_info["mech_str"], a_info["mech_sum"], a_info["conf"], a_info["evid"]
        ])

    add_headers = ["Additive Name", "Chemical Formula", "SMILES", "Reported Optimal Conc", "Baseline Salt", "Mechanisms", "Source Paper"]
    add_rows = [add_headers]
    for row in unique_adds_map.values():
        add_rows.append(row)

    return make_multi_sheet_xlsx({
        "Mined_Literature": paper_rows,
        "Unique_Additives": add_rows
    })




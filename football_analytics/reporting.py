import os
import re
import math
import unicodedata

import pandas as pd
from jinja2 import Environment, select_autoescape


def format_defensive_prose(
    multiplier: float,
    team_name: str,
    opponent_name: str,
) -> str:
    """Converts one independent defensive coefficient into directional prose."""
    try:
        numeric_value = float(multiplier)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid defensive multiplier: {multiplier!r}") from error
    if not math.isfinite(numeric_value):
        raise ValueError(f"Invalid defensive multiplier: {multiplier!r}")

    percentage_change = round((numeric_value - 1.0) * 100)
    absolute_percentage = abs(percentage_change)

    if percentage_change > 0:
        return (
            f"{opponent_name}'s defensive shape increases {team_name}'s "
            f"shooting opportunities by ~{absolute_percentage}%."
        )
    if percentage_change < 0:
        return (
            f"{opponent_name}'s defensive shape decreases {team_name}'s "
            f"shooting opportunities by ~{absolute_percentage}%."
        )
    return (
        "Tactical defensive shapes mirror field expectations perfectly, "
        "resulting in a neutral 0% net impact on shooting opportunities."
    )


def _pdf_safe_text(value: str) -> str:
    """Transliterates dynamic labels for reliable xhtml2pdf font rendering."""
    text = unicodedata.normalize("NFKD", str(value or ""))
    return "".join(char for char in text if not unicodedata.combining(char))


HTML_REPORT_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>World Cup 2026 Simulation Intelligence Report</title>
    <style>
        @page {
            size: A4 landscape;
            margin: 11mm 11mm 15mm 11mm;
            @bottom-left {
                content: "CONFIDENTIAL // SIMULATION INTELLIGENCE";
                font-family: Arial, sans-serif;
                font-size: 7pt;
                color: #94a3b8;
                font-weight: bold;
                letter-spacing: 0.4px;
            }
            @bottom-right {
                content: "Page " counter(page) " of " counter(pages);
                font-family: Arial, sans-serif;
                font-size: 7pt;
                color: #94a3b8;
            }
        }

        body {
            font-family: Arial, sans-serif;
            color: #0f172a;
            background: #ffffff;
            line-height: 1.32;
            margin: 0;
            padding: 0;
            font-size: 8.5pt;
        }

        .report-header {
            border-bottom: 2px solid #1e293b;
            padding-bottom: 7px;
            margin-bottom: 8px;
        }

        .header-accent {
            color: #3b82f6;
            font-size: 8.5pt;
            font-weight: bold;
            text-transform: uppercase;
            letter-spacing: 1.2px;
            margin: 0 0 3px 0;
        }

        .report-title {
            font-size: 18pt;
            color: #1e293b;
            font-weight: bold;
            margin: 0 0 6px 0;
        }

        .metadata-table,
        .context-table,
        .simulation-table,
        .footer-table {
            width: 100%;
            border-collapse: collapse;
        }

        .metadata-table td {
            padding: 2px 4px 2px 0;
            font-size: 8.3pt;
        }

        .metadata-label {
            font-weight: bold;
            color: #475569;
            width: 8%;
        }

        .metadata-value {
            color: #0f172a;
            width: 42%;
        }

        .toolbar {
            text-align: right;
            margin: 0 0 8px 0;
        }

        .print-button {
            background: #1e293b;
            border: 1px solid #1e293b;
            color: #ffffff;
            cursor: pointer;
            font-size: 8pt;
            font-weight: bold;
            padding: 5px 9px;
            text-transform: uppercase;
        }

        .section-heading {
            font-size: 10.5pt;
            color: #1e293b;
            font-weight: bold;
            border-bottom: 1px solid #dbe4ef;
            padding-bottom: 3px;
            margin: 8px 0 6px 0;
            text-transform: uppercase;
            letter-spacing: 0.35px;
        }

        .context-table {
            table-layout: fixed;
            margin-bottom: 9px;
        }

        .context-table td {
            width: 33.33%;
            padding: 0 4px;
            vertical-align: top;
        }

        .context-table td:first-child {
            padding-left: 0;
        }

        .context-table td:last-child {
            padding-right: 0;
        }

        .context-card {
            background: #f8fafc;
            border-left: 4px solid #3b82f6;
            padding: 7px 8px;
            min-height: 72px;
            page-break-inside: avoid;
        }

        .context-title {
            color: #1e293b;
            font-size: 8.5pt;
            font-weight: bold;
            margin: 0 0 4px 0;
            text-transform: uppercase;
            letter-spacing: 0.2px;
        }

        .context-copy {
            color: #334155;
            font-size: 7.5pt;
            margin: 0;
        }

        .metric {
            color: #1d4ed8;
            font-weight: bold;
        }

        .simulation-table {
            table-layout: fixed;
            margin-bottom: 0;
        }

        .simulation-table th {
            background: #1e293b;
            color: #ffffff;
            text-align: left;
            padding: 4px 4px;
            font-size: 6.4pt;
            font-weight: bold;
            text-transform: uppercase;
            border: 1px solid #1e293b;
            vertical-align: middle;
            white-space: nowrap;
            word-break: normal;
        }

        .simulation-table td {
            padding: 2px 4px;
            font-size: 6.7pt;
            line-height: 1.08;
            border: 1px solid #dbe4ef;
            vertical-align: middle;
            white-space: nowrap;
            word-break: normal;
            overflow-wrap: normal;
            hyphens: none;
        }

        .simulation-table tbody tr:nth-child(even) {
            background: #f8fafc;
        }

        .simulation-table tr {
            page-break-inside: avoid;
        }

        .simulation-table thead {
            display: table-header-group;
        }

        .text-center {
            text-align: center;
        }

        .text-right {
            text-align: right;
        }

        .bold {
            font-weight: bold;
        }

        .probability-cell {
            color: #1e3a8a;
            background: #eff6ff;
            font-weight: bold;
        }

        .xhtml-footer {
            display: none;
        }

        .footer-table td {
            color: #94a3b8;
            font-size: 7pt;
            font-weight: bold;
        }

        @media print {
            .toolbar {
                display: none;
            }
        }
    </style>
</head>
<body>
    <div id="xhtml-footer" class="xhtml-footer">
        <table class="footer-table">
            <tr>
                <td>CONFIDENTIAL // SIMULATION INTELLIGENCE</td>
                <td class="text-right">Page <pdf:pagenumber> of <pdf:pagecount></td>
            </tr>
        </table>
    </div>

    <div class="report-header">
        <p class="header-accent">Syndicate Quantitative Research Group</p>
        <h1 class="report-title">Starting XI Shooting Simulation Intelligence</h1>
        <table class="metadata-table">
            <tr>
                <td class="metadata-label">TO:</td>
                <td class="metadata-value">Investment Committee / Trading Desk</td>
                <td class="metadata-label">DATE:</td>
                <td class="metadata-value">{{ date }}</td>
            </tr>
            <tr>
                <td class="metadata-label">FROM:</td>
                <td class="metadata-value">Senior Portfolio Quant Engine</td>
                <td class="metadata-label">FIXTURE:</td>
                <td class="metadata-value bold">{{ fixture_name }}</td>
            </tr>
        </table>
    </div>

    <h2 class="section-heading">Match Context &amp; Simulation Baselines</h2>
    <table class="context-table">
        <tr>
            <td>
                <div class="context-card">
                    <h3 class="context-title">Team-Level Defensive Containment</h3>
                    <p class="context-copy">
                        Each defensive coefficient is calculated independently from
                        leave-one-defense-out common-opponent history. {{ away_team }}'s
                        shape modifies {{ home_team }} attack, while {{ home_team }}'s
                        shape separately modifies {{ away_team }} attack across
                        {{ simulation_count }} simulation draws.
                    </p>
                </div>
            </td>
            <td>
                <div class="context-card">
                    <h3 class="context-title">{{ away_context_title }}</h3>
                    <p class="context-copy">
                        {{ home_defensive_prose }}
                    </p>
                </div>
            </td>
            <td>
                <div class="context-card">
                    <h3 class="context-title">{{ home_context_title }}</h3>
                    <p class="context-copy">
                        {{ away_defensive_prose }}
                    </p>
                </div>
            </td>
        </tr>
    </table>

    <div class="toolbar">
        <button class="print-button" type="button" onclick="window.print()">Print PDF</button>
    </div>

    <h2 class="section-heading">Complete Starting XI Simulation Matrix</h2>
    <table class="simulation-table">
        <colgroup>
            <col style="width: 9%;">
            <col style="width: 18%;">
            <col style="width: 7%;">
            <col style="width: 12%;">
            <col style="width: 15%;">
            <col style="width: 14%;">
            <col style="width: 12%;">
            <col style="width: 13%;">
        </colgroup>
        <thead>
            <tr>
                <th>Team</th>
                <th>Player Name</th>
                <th class="text-center">Position</th>
                <th class="text-center">Projected Shots (Mean)</th>
                <th class="text-center">Projected Shots on Target (Mean)</th>
                <th class="text-center">Projected Shots Missed (Mean)</th>
                <th class="text-center">Projected Goals (Mean)</th>
                <th class="text-center">Simulated Any Shot Probability (%)</th>
            </tr>
        </thead>
        <tbody>
            {% for row in projection_rows %}
            <tr>
                <td>{{ row.Team }}</td>
                <td class="bold">{{ row.Player_Name }}</td>
                <td class="text-center">{{ row.Position }}</td>
                <td class="text-center">{{ "%.2f"|format(row.Projected_Shots) }}</td>
                <td class="text-center">{{ "%.2f"|format(row.Projected_SOT) }}</td>
                <td class="text-center">{{ "%.2f"|format(row.Projected_Missed) }}</td>
                <td class="text-center">{{ "%.2f"|format(row.Projected_Goals) }}</td>
                <td class="text-center probability-cell">{{ "%.2f"|format(row.Any_Shot_Probability) }}%</td>
            </tr>
            {% endfor %}
        </tbody>
    </table>

</body>
</html>
"""


def generate_html_report(
    date: str,
    fixture_name: str,
    projections_df: pd.DataFrame,
    *,
    home_team: str,
    away_team: str,
    home_attack_modifier: float,
    away_attack_modifier: float,
    simulation_count: int,
) -> str:
    """Renders the simulation-only stakeholder report."""
    required_columns = [
        "Team",
        "Player Name",
        "Position",
        "Projected Shots (Mean)",
        "Projected Shots on Target (Mean)",
        "Projected Shots Missed (Mean)",
        "Projected Goals (Mean)",
        "Simulated Any Shot Probability (%)",
    ]
    missing = [column for column in required_columns if column not in projections_df.columns]
    if missing:
        raise ValueError(f"Projection data is missing required columns: {missing}")

    visible = projections_df[
        projections_df["Position"] != "G"
    ].copy()
    visible = visible.sort_values(
        by="Projected Shots (Mean)",
        ascending=False,
        kind="mergesort",
    )

    projection_rows = []
    for _, row in visible.iterrows():
        projection_rows.append({
            "Team": _pdf_safe_text(row["Team"]),
            "Player_Name": _pdf_safe_text(row["Player Name"]),
            "Position": row["Position"],
            "Projected_Shots": float(row["Projected Shots (Mean)"]),
            "Projected_SOT": float(row["Projected Shots on Target (Mean)"]),
            "Projected_Missed": float(row["Projected Shots Missed (Mean)"]),
            "Projected_Goals": float(row["Projected Goals (Mean)"]),
            "Any_Shot_Probability": float(
                row["Simulated Any Shot Probability (%)"]
            ),
        })

    environment = Environment(
        autoescape=select_autoescape(
            enabled_extensions=("html", "xml"),
            default_for_string=True,
        )
    )
    template = environment.from_string(HTML_REPORT_TEMPLATE)
    return template.render(
        date=date,
        fixture_name=fixture_name,
        projection_rows=projection_rows,
        home_team=home_team,
        away_team=away_team,
        away_context_title=f"{away_team} Defensive Shape Impact",
        home_context_title=f"{home_team} Defensive Shape Impact",
        home_defensive_prose=format_defensive_prose(
            home_attack_modifier,
            home_team,
            away_team,
        ),
        away_defensive_prose=format_defensive_prose(
            away_attack_modifier,
            away_team,
            home_team,
        ),
        simulation_count=int(simulation_count),
    )


def compile_pdf_report(html_content: str, output_pdf_path: str):
    """Compiles semantic HTML to PDF, using WeasyPrint then xhtml2pdf."""
    print("Compiling PDF report...")
    output_dir = os.path.dirname(os.path.abspath(output_pdf_path))
    os.makedirs(output_dir, exist_ok=True)

    try:
        import weasyprint

        print("Using primary WeasyPrint engine...")
        weasyprint.HTML(string=html_content).write_pdf(output_pdf_path)
        print(f"Success: PDF generated via WeasyPrint at: {output_pdf_path}")
        return
    except Exception as error:
        print(f"Primary WeasyPrint compilation bypassed/failed: {error}")
        print("Initiating fallback pure-Python xhtml2pdf engine...")

    try:
        clean_html = re.sub(r"@bottom-left\s*\{[^{}]*\}", "", html_content)
        clean_html = re.sub(r"@bottom-right\s*\{[^{}]*\}", "", clean_html)
        xhtml_page_css = """
        @page {
            size: A4 landscape;
            margin: 11mm 11mm 15mm 11mm;
            @frame footer_frame {
                -pdf-frame-content: xhtml-footer;
                left: 11mm;
                right: 11mm;
                bottom: 4mm;
                height: 7mm;
            }
        }
        """
        clean_html = re.sub(
            r"@page\s*\{[^{}]*\}",
            xhtml_page_css,
            clean_html,
            count=1,
        )
        clean_html = clean_html.replace(
            ".xhtml-footer {\n            display: none;\n        }",
            ".xhtml-footer { display: block; }",
        )

        from xhtml2pdf import pisa

        with open(output_pdf_path, "wb") as pdf_file:
            status = pisa.CreatePDF(clean_html, dest=pdf_file)
        if status.err:
            raise RuntimeError(f"xhtml2pdf compiler error code: {status.err}")
        print(f"Success: PDF generated via fallback xhtml2pdf at: {output_pdf_path}")
    except Exception as error:
        raise RuntimeError(
            "Dual PDF compiler failed. Install either WeasyPrint or xhtml2pdf."
        ) from error

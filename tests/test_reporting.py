import os

import pandas as pd

from football_analytics.reporting import (
    generate_html_report,
    compile_pdf_report,
    format_defensive_prose,
)


def _projection_frame():
    return pd.DataFrame([
        {
            "Team": "Ecuador",
            "Starting Player": "John Yeboah",
            "Position": "F",
            "Projected Shots (Mean)": 1.42,
            "Projected Shots on Target (Mean)": 0.68,
            "Projected Shots Missed (Mean)": 0.74,
            "Projected Goals (Mean)": 0.11,
            "Simulated Any Shot Probability (%)": 75.40,
        },
        {
            "Team": "Germany",
            "Starting Player": "Deniz Undav",
            "Position": "F",
            "Projected Shots (Mean)": 2.10,
            "Projected Shots on Target (Mean)": 0.90,
            "Projected Shots Missed (Mean)": 1.20,
            "Projected Goals (Mean)": 0.20,
            "Simulated Any Shot Probability (%)": 87.30,
        },
        {
            "Team": "Germany",
            "Starting Player": "Oliver Baumann",
            "Position": "G",
            "Projected Shots (Mean)": 0.0,
            "Projected Shots on Target (Mean)": 0.0,
            "Projected Shots Missed (Mean)": 0.0,
            "Projected Goals (Mean)": 0.0,
            "Simulated Any Shot Probability (%)": 0.0,
        },
    ])


def _render_html():
    return generate_html_report(
        "June 25, 2026",
        "Germany vs. Ecuador",
        _projection_frame(),
        home_team="Germany",
        away_team="Ecuador",
        home_attack_modifier=1.2167,
        away_attack_modifier=0.7833,
        simulation_count=10000,
    )


def test_generate_html_report_is_simulation_only():
    html = _render_html()

    assert "Germany vs. Ecuador" in html
    assert "Match Context &amp; Simulation Baselines" in html
    assert "Team-Level Defensive Containment" in html
    assert "Ecuador Defensive Shape Impact" in html
    assert "Germany Defensive Shape Impact" in html
    assert "Complete Starting XI Simulation Matrix" in html

    for column in [
        "Team",
        "Starting Player",
        "Position",
        "Projected Shots (Mean)",
        "Projected Shots on Target (Mean)",
        "Projected Shots Missed (Mean)",
        "Projected Goals (Mean)",
        "Simulated Any Shot Probability (%)",
    ]:
        assert column in html

    assert "Deniz Undav" in html
    assert "John Yeboah" in html
    assert html.index("Deniz Undav") < html.index("John Yeboah")
    assert "Oliver Baumann" not in html
    assert "87.30%" in html
    assert "increases their baseline shot creation by ~22%" in html
    assert "decreases Ecuador&#39;s baseline shot creation by ~22%" in html
    assert "1.217x" not in html
    assert "0.783x" not in html
    assert "~-22%" not in html

    for forbidden in [
        "Retail Odds",
        "Stake",
        "Tactical Action",
        "Strategic Risk Mandates",
        "Pre-Match &amp; In-Play Execution Directives",
        "Hedging Playbook",
    ]:
        assert forbidden not in html

    assert "\\$" not in html
    assert "CONFIDENTIAL // SIMULATION INTELLIGENCE" in html


def test_defensive_prose_formats_direction_with_positive_percentages():
    positive = format_defensive_prose(1.217, "Japan", "Sweden")
    negative = format_defensive_prose(0.783, "Sweden", "Japan")
    neutral = format_defensive_prose(1.0, "Japan", "Sweden")

    assert "increases their baseline shot creation by ~22%" in positive
    assert "against Sweden's defensive shape" in positive
    assert "decreases Sweden's baseline shot creation by ~22%" in negative
    assert "~-22%" not in negative
    assert "neutral 0% net impact" in neutral


def test_generic_matchup_context_uses_selected_teams():
    html = generate_html_report(
        "June 25, 2026",
        "Curaçao vs. Ivory Coast",
        _projection_frame(),
        home_team="Curaçao",
        away_team="Ivory Coast",
        home_attack_modifier=0.6647,
        away_attack_modifier=1.3353,
        simulation_count=10000,
    )

    assert "Ivory Coast Defensive Shape Impact" in html
    assert "Curaçao Defensive Shape Impact" in html
    assert "increases their baseline shot creation by ~34%" in html
    assert "decreases Curaçao&#39;s baseline shot creation by ~34%" in html


def test_player_names_are_pdf_safe_without_changing_data_contract():
    projections = _projection_frame()
    projections.loc[0, "Starting Player"] = "R. Dōan"
    html = generate_html_report(
        "June 25, 2026",
        "Japan vs. Sweden",
        projections,
        home_team="Japan",
        away_team="Sweden",
        home_attack_modifier=1.0,
        away_attack_modifier=1.0,
        simulation_count=10000,
    )

    assert "R. Doan" in html
    assert projections.loc[0, "Starting Player"] == "R. Dōan"


def test_compile_pdf_report(tmp_path):
    pdf_path = os.path.join(tmp_path, "test_report.pdf")
    compile_pdf_report(_render_html(), pdf_path)

    assert os.path.exists(pdf_path)
    assert os.path.getsize(pdf_path) > 0

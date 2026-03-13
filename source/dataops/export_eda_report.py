"""
US-9: Builds an HTML EDA report from artifacts under `data/processed/eda/`.

Run this script AFTER `run_eda.py`.

Usage:
    python source/dataops/export_eda_report.py
"""

from __future__ import annotations

import html
import os
from datetime import datetime

import pandas as pd


PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
EDA_DIR = os.path.join(PROCESSED_DIR, "eda")
REPORT_PATH = os.path.join(EDA_DIR, "eda_report.html")

CSV_FILES = [
    "churn_rate_by_city.csv",
    "churn_rate_by_avg_plan_days.csv",
    "churn_rate_by_payment_method.csv",
    "churn_distribution.csv",
]

IMAGE_FILES = [
    ("01_payment_method_distribution_before.png", "Payment Method Distribution (Before Grouping)"),
    ("02_payment_method_distribution_after.png", "Payment Method Distribution (After Grouping)"),
    ("03_payment_plan_days_histogram.png", "Distribution of Payment Plan Days"),
    ("04_plan_list_price_boxplot.png", "Plan List Price Distribution"),
    ("05_auto_renew_countplot.png", "Auto-Renewal Distribution"),
    ("06_member_city_countplot.png", "Member Distribution by City"),
    ("07_member_gender_countplot.png", "Member Distribution by Gender"),
    ("08_member_bd_boxplot_before.png", "Member Age (bd) Distribution - Before Cleaning"),
    ("09_member_bd_boxplot_after.png", "Member Age (bd) Distribution - After Cleaning"),
    ("10_registered_via_countplot.png", "Member Distribution by Registration Channel"),
    ("11_top10_numeric_distributions.png", "Distribution of Top 10 Numeric Features (by variance)"),
    ("12_correlation_heatmap.png", "Correlation Heatmap of Numeric Features"),
    ("13_churn_countplot.png", "Distribution of Churn (is_churn)"),
]


def csv_table_html(csv_name: str) -> str:
    path = os.path.join(EDA_DIR, csv_name)
    if not os.path.exists(path):
        return f"<p><em>Missing file: {html.escape(csv_name)}</em></p>"

    df = pd.read_csv(path)
    table = df.to_html(index=False, border=0, classes="table", max_rows=20)
    return f"<h3>{html.escape(csv_name)}</h3>\n{table}"


def image_html(image_name: str, title: str) -> str:
    path = os.path.join(EDA_DIR, image_name)
    if not os.path.exists(path):
        return f"<p><em>Missing image: {html.escape(image_name)}</em></p>"
    return (
        f"<h3>{html.escape(title)}</h3>"
        f"<p class='muted'>{html.escape(image_name)}</p>"
        f"<img src='{html.escape(image_name)}' alt='{html.escape(title)}' style='max-width:100%;height:auto;border:1px solid #ddd;padding:4px;'/>"
    )


def build_report() -> str:
    csv_section = "\n".join(csv_table_html(name) for name in CSV_FILES)
    image_section = "\n".join(image_html(name, title) for name, title in IMAGE_FILES)
    generated_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>EDA Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #222; }}
    h1, h2 {{ margin-bottom: 8px; }}
    .muted {{ color: #666; margin-top: 0; }}
    .table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
    .table th, .table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
    .table th {{ background: #f5f5f5; }}
    .section {{ margin-top: 28px; }}
  </style>
</head>
<body>
  <h1>KKBox EDA Report</h1>
  <p class="muted">Generated at: {generated_at}</p>

  <div class="section">
    <h2>Summary Tables</h2>
    {csv_section}
  </div>

  <div class="section">
    <h2>Charts (13)</h2>
    {image_section}
  </div>
</body>
</html>
"""


def main() -> None:
    os.makedirs(EDA_DIR, exist_ok=True)
    report_html = build_report()
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_html)
    print(f"HTML report written: {REPORT_PATH}")


if __name__ == "__main__":
    main()

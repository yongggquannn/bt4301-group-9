"""
US-5: Runs EDA on `data/processed/df_train_final.csv` and exports artifacts to `data/processed/eda/`.

Run this script AFTER `generate_lineage.py`.

Usage:
    python source/dataops/run_eda.py
"""

from __future__ import annotations

import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


matplotlib.use("Agg")

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
EDA_DIR = os.path.join(PROCESSED_DIR, "eda")


def main() -> None:
    print("=" * 60)
    print("Running EDA and exporting artifacts")
    print("=" * 60)

    print("\n[1/4] Loading df_train_final.csv...")
    input_path = os.path.join(PROCESSED_DIR, "df_train_final.csv")
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Missing input file: {input_path}. Run cleanse_data.py first."
        )

    os.makedirs(EDA_DIR, exist_ok=True)
    df_train_final = pd.read_csv(input_path)
    print(f"  Loaded {len(df_train_final):,} rows.")

    print("\n[2/4] Computing summary churn metrics...")
    # Overall churn rate.
    overall_churn_rate = df_train_final["is_churn"].mean()
    print(f"Overall Churn Rate: {overall_churn_rate:.4f}")

    # Churn rate by city.
    churn_rate_by_city = (
        df_train_final.groupby("city")["is_churn"].mean().sort_values(ascending=False)
    )
    print("Churn Rate by City:\n", churn_rate_by_city)
    churn_rate_by_city.to_csv(os.path.join(EDA_DIR, "churn_rate_by_city.csv"), header=True)

    # Churn rate by average plan days.
    churn_rate_by_payment_plan = (
        df_train_final.groupby("avg_plan_days")["is_churn"].mean().sort_values(ascending=False)
    )
    print("Churn Rate by Avg Payment Plan Days:\n", churn_rate_by_payment_plan)
    churn_rate_by_payment_plan.to_csv(
        os.path.join(EDA_DIR, "churn_rate_by_avg_plan_days.csv"), header=True
    )

    # Churn rate by payment method.
    churn_rate_by_payment_method = (
        df_train_final.groupby("payment_method_id")["is_churn"].mean().sort_values(ascending=False)
    )
    print("Churn Rate by Payment Method:\n", churn_rate_by_payment_method)
    churn_rate_by_payment_method.to_csv(
        os.path.join(EDA_DIR, "churn_rate_by_payment_method.csv"), header=True
    )

    print("\n[3/4] Generating EDA visualizations...")
    # Top-variance numeric feature distributions.
    numeric_features = df_train_final.select_dtypes(include=["number"]).drop(columns=["is_churn"])
    top_10 = numeric_features.var().sort_values(ascending=False).head(10).index.tolist()
    fig, axes = plt.subplots(2, 5, figsize=(22, 8))
    for i, col in enumerate(top_10):
        ax = axes[i // 5, i % 5]
        df_train_final[col].hist(bins=30, ax=ax, edgecolor="black")
        ax.set_title(col, fontsize=10)
        ax.set_ylabel("Frequency")
    plt.suptitle("Distribution of Top 10 Numeric Features (by variance)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, "top10_numeric_distributions.png"), dpi=150)
    plt.close(fig)

    # Numeric feature correlation heatmap.
    numeric_df = df_train_final.select_dtypes(include=["number"]).drop(columns=["is_churn"])
    correlation_matrix = numeric_df.corr()
    plt.figure(figsize=(20, 18))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap of Numeric Features in df_train_final", fontsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, "correlation_heatmap.png"), dpi=150)
    plt.close()

    # Distribution of churn values.
    churn_distribution = df_train_final["is_churn"].value_counts()
    print("Distribution of Churn (is_churn):\n", churn_distribution)
    churn_distribution.to_csv(os.path.join(EDA_DIR, "churn_distribution.csv"), header=True)

    # Churn count plot.
    plt.figure(figsize=(6, 4))
    sns.countplot(x="is_churn", data=df_train_final)
    plt.title("Distribution of Churn (is_churn)")
    plt.xlabel("Is Churn")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, "churn_countplot.png"), dpi=150)
    plt.close()

    print("\n[4/4] Writing outputs...")
    print(f"  EDA artifacts saved to {EDA_DIR}")
    
    print("\n" + "=" * 60)
    print("Done. EDA step complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()

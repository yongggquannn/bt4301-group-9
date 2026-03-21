"""
Generates 13 EDA charts and builds the HTML EDA report.

Run this script AFTER `run_eda.py`.

Usage:
    python source/dataops/generate_eda_images_report.py
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, os.path.dirname(__file__))
import export_eda_report

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
EDA_DIR = os.path.join(PROCESSED_DIR, "eda")


def _require(path: str, msg: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required file: {path}. {msg}")


def _save(filename: str) -> None:
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, filename), dpi=150)
    plt.close()


def main() -> None:
    os.makedirs(EDA_DIR, exist_ok=True)

    train_final = os.path.join(PROCESSED_DIR, "df_train_final.csv")
    transactions = os.path.join(DATA_DIR, "transactions_v2.csv")
    members = os.path.join(DATA_DIR, "members_v3.csv")

    _require(train_final, "Run cleanse_data.py first.")
    _require(transactions, "Ensure raw dataset is available in data/raw.")
    _require(members, "Ensure raw dataset is available in data/raw.")

    df_train_final = pd.read_csv(train_final)
    df_transaction = pd.read_csv(transactions)
    df_member = pd.read_csv(members)

    plt.figure(figsize=(16, 8), dpi=60)
    sns.countplot(x="payment_method_id", data=df_transaction)
    plt.title("Payment Method Distribution (Before Grouping)")
    plt.xlabel("Payment Method ID")
    plt.ylabel("Count")
    plt.xticks(rotation=90)
    _save("01_payment_method_distribution_before.png")

    low_freq_methods = [3, 6, 8, 10, 11, 12, 13, 14, 16, 18, 20, 22, 26]
    df_transaction["payment_method_id"] = df_transaction["payment_method_id"].replace(low_freq_methods, 1)
    plt.figure(figsize=(12, 6))
    sns.countplot(x="payment_method_id", data=df_transaction)
    plt.title("Payment Method Distribution (After Grouping)")
    plt.xlabel("Payment Method ID")
    plt.ylabel("Count")
    _save("02_payment_method_distribution_after.png")

    plt.figure(figsize=(10, 6))
    df_transaction["payment_plan_days"].hist(bins=30)
    plt.title("Distribution of Payment Plan Days")
    plt.xlabel("Payment Plan Days")
    plt.ylabel("Frequency")
    plt.grid(True)
    _save("03_payment_plan_days_histogram.png")

    plt.figure(figsize=(10, 4))
    sns.boxplot(x="plan_list_price", data=df_transaction)
    plt.title("Plan List Price Distribution")
    plt.xlabel("Plan List Price")
    plt.ylabel("Value")
    _save("04_plan_list_price_boxplot.png")

    plt.figure(figsize=(8, 4))
    sns.countplot(x="is_auto_renew", data=df_transaction)
    plt.title("Auto-Renewal Distribution")
    plt.xlabel("Is Auto Renew")
    plt.ylabel("Count")
    _save("05_auto_renew_countplot.png")

    plt.figure(figsize=(12, 5))
    sns.countplot(x="city", data=df_member)
    plt.title("Member Distribution by City")
    plt.xlabel("City")
    plt.ylabel("Count")
    _save("06_member_city_countplot.png")

    plt.figure(figsize=(8, 4))
    sns.countplot(x="gender", data=df_member)
    plt.title("Member Distribution by Gender")
    plt.xlabel("Gender")
    plt.ylabel("Count")
    _save("07_member_gender_countplot.png")

    plt.figure(figsize=(10, 4))
    sns.boxplot(x="bd", data=df_member)
    plt.title("Member Age (bd) Distribution - Before Cleaning")
    plt.xlabel("bd")
    plt.ylabel("Value")
    _save("08_member_bd_boxplot_before.png")

    df_member_clean = df_member.copy()
    df_member_clean.loc[df_member_clean["bd"] < 18, "bd"] = np.nan
    df_member_clean.loc[df_member_clean["bd"] > 90, "bd"] = np.nan
    df_member_clean["bd"] = df_member_clean["bd"].fillna(df_member_clean["bd"].median())
    plt.figure(figsize=(10, 4))
    sns.boxplot(x="bd", data=df_member_clean)
    plt.title("Member Age (bd) Distribution - After Cleaning")
    plt.xlabel("bd")
    plt.ylabel("Value")
    _save("09_member_bd_boxplot_after.png")

    plt.figure(figsize=(10, 4))
    sns.countplot(x="registered_via", data=df_member)
    plt.title("Member Distribution by Registration Channel")
    plt.xlabel("Registered Via")
    plt.ylabel("Count")
    _save("10_registered_via_countplot.png")

    numeric_features = df_train_final.select_dtypes(include=["number"]).drop(columns=["is_churn"])
    top_10 = numeric_features.var().sort_values(ascending=False).head(10).index.tolist()
    fig, axes = plt.subplots(2, 5, figsize=(22, 8))
    for i, col in enumerate(top_10):
        ax = axes[i // 5, i % 5]
        df_train_final[col].hist(bins=30, ax=ax, edgecolor="black")
        ax.set_title(col, fontsize=10)
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
    plt.suptitle("Distribution of Top 10 Numeric Features (by variance)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, "11_top10_numeric_distributions.png"), dpi=150)
    plt.close(fig)

    numeric_df = df_train_final.select_dtypes(include=["number"]).drop(columns=["is_churn"])
    correlation_matrix = numeric_df.corr()
    plt.figure(figsize=(20, 18))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap of Numeric Features in df_train_final", fontsize=18)
    plt.xlabel("Features")
    plt.ylabel("Features")
    _save("12_correlation_heatmap.png")

    plt.figure(figsize=(6, 4))
    sns.countplot(x="is_churn", data=df_train_final)
    plt.title("Distribution of Churn (is_churn)")
    plt.xlabel("Is Churn")
    plt.ylabel("Count")
    _save("13_churn_countplot.png")

    export_eda_report.main()
    print(f"Generated 13 images and HTML report in: {EDA_DIR}")


if __name__ == "__main__":
    main()

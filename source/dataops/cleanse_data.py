"""
Notebook-parity cleansing script based on `data_cleaning_eda.ipynb`.

This mirrors the notebook cleaning/merging feature-prep steps and writes:
    data/processed/df_train_final.csv
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")


def main() -> None:
    print("=" * 60)
    print("Cleansing and preparing df_train_final")
    print("=" * 60)

    # Load raw CSV tables (notebook cell 1).
    print("\n[1/6] Loading raw CSV files...")
    df_train = pd.read_csv(os.path.join(DATA_DIR, "train_v2.csv"))
    df_member = pd.read_csv(os.path.join(DATA_DIR, "members_v3.csv"))
    df_transaction = pd.read_csv(os.path.join(DATA_DIR, "transactions_v2.csv"))
    df_log = pd.read_csv(os.path.join(DATA_DIR, "user_logs_v2.csv"))
    print("  Raw files loaded.")

    # Replace low-frequency payment methods with a collective code (notebook cell 12).
    print("\n[2/6] Normalizing low-frequency payment methods...")
    df_transaction.payment_method_id = df_transaction.payment_method_id.replace(
        [3, 6, 8, 10, 11, 12, 13, 14, 16, 18, 20, 22, 26], 1
    )
    df_transaction.payment_method_id = df_transaction.payment_method_id.astype("category")

    # Process duplicate msno in transactions (notebook cell 16).
    print("\n[3/6] Aggregating transaction duplicates...")
    df_duplicate_msno = df_transaction[df_transaction.duplicated(subset=["msno"])]
    duplicated_msno = df_duplicate_msno.msno.unique()
    transaction_group = df_transaction.groupby("msno")
    df_transaction_uniq = pd.DataFrame(
        data=None,
        columns=[
            "msno",
            "payment_method_id",
            "total_payment_plan_days",
            "total_plan_list_price",
            "total_actual_amount_paid",
            "last_plan_days",
            "last_plan_price",
            "is_auto_renewal",
            "first_transaction_date",
            "last_transaction_date",
            "membership_expire_date",
            "is_cancel",
            "no_of_transactions",
        ],
    )

    for msno in duplicated_msno:
        msno_group = transaction_group.get_group(msno)
        payment_method_id = msno_group.payment_method_id.iloc[0]
        total_payment_plan_days = msno_group.payment_plan_days.sum()
        total_plan_list_price = msno_group.plan_list_price.sum()
        total_actual_amount_paid = msno_group.actual_amount_paid.sum()
        is_auto_renewal = msno_group.is_auto_renew.iloc[0]
        first_transaction_date = msno_group.transaction_date.min()
        last_transaction_date = msno_group.transaction_date.max()
        membership_expire_date = msno_group.membership_expire_date.max()
        last_date = msno_group[msno_group.transaction_date == last_transaction_date]
        last_plan_days = 0
        last_plan_price = 0
        if len(last_date) > 0:
            last_plan_days = last_date["payment_plan_days"].values.max()
            last_plan_price = last_date["plan_list_price"].values.max()
        canceled = msno_group.is_cancel.values
        is_cancel = 0 if 1 in canceled else 1
        no_of_transactions = len(msno_group)
        pointer = len(df_transaction_uniq)
        df_transaction_uniq.loc[pointer] = [
            msno,
            payment_method_id,
            total_payment_plan_days,
            total_plan_list_price,
            total_actual_amount_paid,
            last_plan_days,
            last_plan_price,
            is_auto_renewal,
            first_transaction_date,
            last_transaction_date,
            membership_expire_date,
            is_cancel,
            no_of_transactions,
        ]

    # Add single-transaction users back after duplicate aggregation (notebook cells 17, 19, 20).
    single_msno = df_transaction[~df_transaction.duplicated(subset=["msno"], keep=False)]
    single_msno_processed = pd.DataFrame(
        {
            "msno": single_msno["msno"],
            "payment_method_id": single_msno["payment_method_id"],
            "total_payment_plan_days": single_msno["payment_plan_days"],
            "total_plan_list_price": single_msno["plan_list_price"],
            "total_actual_amount_paid": single_msno["actual_amount_paid"],
            "last_plan_days": single_msno["payment_plan_days"],
            "last_plan_price": single_msno["plan_list_price"],
            "is_auto_renewal": single_msno["is_auto_renew"],
            "first_transaction_date": single_msno["transaction_date"],
            "last_transaction_date": single_msno["transaction_date"],
            "membership_expire_date": single_msno["membership_expire_date"],
            "is_cancel": single_msno["is_cancel"],
            "no_of_transactions": 1,
        }
    )

    df_transaction_final = pd.concat([df_transaction_uniq, single_msno_processed], ignore_index=True)
    print(f"  Transaction rows after processing: {len(df_transaction_final):,}")

    # Clean member age outliers and median-impute bd (notebook cell 30).
    print("\n[4/6] Cleaning member age field (bd)...")
    df_member.loc[df_member.bd < 18, "bd"] = np.nan
    df_member.loc[df_member.bd > 90, "bd"] = np.nan
    df_member["bd"].fillna(df_member["bd"].median(), inplace=True)

    # Merge members with transaction aggregates (notebook cell 33).
    print("\n[5/6] Building merged training dataset...")
    df_member_transaction = pd.merge(df_member, df_transaction_final, how="inner", on="msno")
    # Cast date-like integer fields to datetime (notebook cell 35).
    df_member_transaction["registration_init_time"] = pd.to_datetime(
        df_member_transaction["registration_init_time"], format="%Y%m%d"
    )
    df_member_transaction["first_transaction_date"] = pd.to_datetime(
        df_member_transaction["first_transaction_date"], format="%Y%m%d"
    )
    df_member_transaction["last_transaction_date"] = pd.to_datetime(
        df_member_transaction["last_transaction_date"], format="%Y%m%d"
    )
    df_member_transaction["membership_expire_date"] = pd.to_datetime(
        df_member_transaction["membership_expire_date"], format="%Y%m%d"
    )
    # Drop gender column after merge (notebook cell 37).
    df_member_transaction = df_member_transaction.drop(columns=["gender"])
    # Merge churn labels with member+transaction data (notebook cell 39).
    df_train_final = pd.merge(df_train, df_member_transaction, how="inner", on="msno")

    # Expand date components (notebook cell 40).
    date_columns = [
        "registration_init_time",
        "first_transaction_date",
        "last_transaction_date",
        "membership_expire_date",
    ]
    for col in date_columns:
        df_train_final[f"{col}_year"] = df_train_final[col].dt.year
        df_train_final[f"{col}_month"] = df_train_final[col].dt.month
        df_train_final[f"{col}_day"] = df_train_final[col].dt.day

    # Build duration features (notebook cell 41).
    df_train_final["membership_duration_days"] = (
        df_train_final["membership_expire_date"] - df_train_final["registration_init_time"]
    ).dt.days
    df_train_final["transaction_span_days"] = (
        df_train_final["last_transaction_date"] - df_train_final["first_transaction_date"]
    ).dt.days
    # Build price/plan features and remove intermediate columns (notebook cell 42).
    df_train_final["avg_plan_price"] = (
        df_train_final["total_plan_list_price"] / df_train_final["no_of_transactions"]
    )
    df_train_final["avg_plan_days"] = (
        df_train_final["total_payment_plan_days"] / df_train_final["no_of_transactions"]
    )
    df_train_final["amount_due"] = (
        df_train_final["total_plan_list_price"] - df_train_final["total_actual_amount_paid"]
    )
    df_train_final = df_train_final.drop(
        columns=[
            "total_plan_list_price",
            "total_payment_plan_days",
            "total_actual_amount_paid",
            "first_transaction_date",
            "last_transaction_date",
            "membership_expire_date",
            "registration_init_time",
        ]
    )
    df_train_final = df_train_final.drop(
        columns=[
            "registration_init_time_year",
            "registration_init_time_month",
            "registration_init_time_day",
            "first_transaction_date_year",
            "first_transaction_date_month",
            "first_transaction_date_day",
            "last_transaction_date_year",
            "last_transaction_date_month",
            "last_transaction_date_day",
            "membership_expire_date_year",
            "membership_expire_date_month",
            "membership_expire_date_day",
        ]
    )

    # Keep notebook parity by touching df_log in pipeline context (loaded in notebook cell 1).
    print(f"user_logs rows loaded (not merged in df_train_final path): {len(df_log):,}")

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    # Save final cleaned/engineered table (notebook cell 60).
    print("\n[6/6] Saving df_train_final.csv...")
    output_path = os.path.join(PROCESSED_DIR, "df_train_final.csv")
    df_train_final.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")
    print(f"  Rows : {len(df_train_final):,}")
    
    print("\n" + "=" * 60)
    print("Done. Run build_customer_features.py next.")
    print("=" * 60)


if __name__ == "__main__":
    main()

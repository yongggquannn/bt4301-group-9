from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from feature_registry import FEATURE_SPECS, feature_names


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


@dataclass(frozen=True)
class SourceConfig:
    path: Path
    chunksize: int


TRAIN = SourceConfig(RAW_DIR / "train_v2.csv", 200_000)
MEMBERS = SourceConfig(RAW_DIR / "members_v3.csv", 200_000)
TRANSACTIONS = SourceConfig(RAW_DIR / "transactions_v2.csv", 250_000)
USER_LOGS = SourceConfig(RAW_DIR / "user_logs_v2.csv", 250_000)


def parse_yyyymmdd(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series.astype(str), format="%Y%m%d", errors="coerce")


def clean_age(series: pd.Series) -> pd.Series:
    age = pd.to_numeric(series, errors="coerce")
    return age.where(age.between(10, 80))


def load_train() -> pd.DataFrame:
    train = pd.read_csv(TRAIN.path)
    train["is_churn"] = pd.to_numeric(train["is_churn"], errors="coerce")
    return train[["msno", "is_churn"]]


def load_members() -> pd.DataFrame:
    members = pd.read_csv(MEMBERS.path)
    members["age"] = clean_age(members["bd"])
    members["registration_init_date"] = parse_yyyymmdd(members["registration_init_time"])

    result = members[
        [
            "msno",
            "city",
            "gender",
            "registered_via",
            "age",
            "registration_init_date",
        ]
    ].copy()

    result["gender"] = result["gender"].fillna("unknown")
    result["registration_year"] = result["registration_init_date"].dt.year
    result["registration_month"] = result["registration_init_date"].dt.month
    return result.drop(columns=["registration_init_date"])


def aggregate_transactions() -> pd.DataFrame:
    aggregations: dict[str, dict[str, object]] = {}

    for chunk in pd.read_csv(TRANSACTIONS.path, chunksize=TRANSACTIONS.chunksize):
        chunk["transaction_date"] = parse_yyyymmdd(chunk["transaction_date"])
        chunk["membership_expire_date"] = parse_yyyymmdd(chunk["membership_expire_date"])
        chunk["days_until_expiry"] = (
            chunk["membership_expire_date"] - chunk["transaction_date"]
        ).dt.days

        grouped = chunk.groupby("msno", sort=False).agg(
            txn_count=("msno", "size"),
            txn_plan_days_sum=("payment_plan_days", "sum"),
            txn_list_price_sum=("plan_list_price", "sum"),
            txn_amount_paid_sum=("actual_amount_paid", "sum"),
            txn_auto_renew_sum=("is_auto_renew", "sum"),
            txn_cancel_sum=("is_cancel", "sum"),
            txn_days_until_expiry_sum=("days_until_expiry", "sum"),
            txn_last_transaction_date=("transaction_date", "max"),
            txn_last_membership_expire_date=("membership_expire_date", "max"),
        )
        payment_method_sets = chunk.groupby("msno", sort=False)["payment_method_id"].agg(
            lambda values: set(values.dropna().tolist())
        )

        for msno, row in grouped.iterrows():
            current = aggregations.setdefault(
                msno,
                {
                    "txn_count": 0.0,
                    "txn_plan_days_sum": 0.0,
                    "txn_list_price_sum": 0.0,
                    "txn_amount_paid_sum": 0.0,
                    "txn_auto_renew_sum": 0.0,
                    "txn_cancel_sum": 0.0,
                    "txn_days_until_expiry_sum": 0.0,
                    "txn_payment_method_ids": set(),
                    "txn_last_transaction_date": pd.NaT,
                    "txn_last_membership_expire_date": pd.NaT,
                },
            )

            current["txn_count"] += float(row["txn_count"])
            current["txn_plan_days_sum"] += float(row["txn_plan_days_sum"])
            current["txn_list_price_sum"] += float(row["txn_list_price_sum"])
            current["txn_amount_paid_sum"] += float(row["txn_amount_paid_sum"])
            current["txn_auto_renew_sum"] += float(row["txn_auto_renew_sum"])
            current["txn_cancel_sum"] += float(row["txn_cancel_sum"])
            current["txn_days_until_expiry_sum"] += float(
                row["txn_days_until_expiry_sum"] if pd.notna(row["txn_days_until_expiry_sum"]) else 0.0
            )

            current["txn_payment_method_ids"].update(payment_method_sets.get(msno, set()))

            if pd.notna(row["txn_last_transaction_date"]):
                if pd.isna(current["txn_last_transaction_date"]) or row["txn_last_transaction_date"] > current["txn_last_transaction_date"]:
                    current["txn_last_transaction_date"] = row["txn_last_transaction_date"]

            if pd.notna(row["txn_last_membership_expire_date"]):
                if pd.isna(current["txn_last_membership_expire_date"]) or row["txn_last_membership_expire_date"] > current["txn_last_membership_expire_date"]:
                    current["txn_last_membership_expire_date"] = row["txn_last_membership_expire_date"]

    transactions = pd.DataFrame.from_dict(aggregations, orient="index").reset_index(names="msno")
    transactions["txn_payment_method_nunique"] = transactions["txn_payment_method_ids"].apply(len)

    transactions["txn_avg_plan_days"] = transactions["txn_plan_days_sum"] / transactions["txn_count"]
    transactions["txn_avg_list_price"] = transactions["txn_list_price_sum"] / transactions["txn_count"]
    transactions["txn_avg_amount_paid"] = transactions["txn_amount_paid_sum"] / transactions["txn_count"]
    transactions["txn_auto_renew_rate"] = transactions["txn_auto_renew_sum"] / transactions["txn_count"]
    transactions["txn_cancel_rate"] = transactions["txn_cancel_sum"] / transactions["txn_count"]
    transactions["txn_avg_days_until_expiry"] = (
        transactions["txn_days_until_expiry_sum"] / transactions["txn_count"]
    )
    transactions["txn_last_membership_days"] = (
        transactions["txn_last_membership_expire_date"] - transactions["txn_last_transaction_date"]
    ).dt.days

    return transactions.drop(
        columns=[
            "txn_plan_days_sum",
            "txn_list_price_sum",
            "txn_amount_paid_sum",
            "txn_auto_renew_sum",
            "txn_cancel_sum",
            "txn_days_until_expiry_sum",
            "txn_payment_method_ids",
            "txn_last_transaction_date",
            "txn_last_membership_expire_date",
        ]
    )


def aggregate_user_logs() -> pd.DataFrame:
    aggregations: dict[str, dict[str, float]] = {}

    for chunk in pd.read_csv(USER_LOGS.path, chunksize=USER_LOGS.chunksize):
        chunk["date"] = parse_yyyymmdd(chunk["date"])
        chunk["plays_total"] = chunk[
            ["num_25", "num_50", "num_75", "num_985", "num_100"]
        ].sum(axis=1)

        grouped = chunk.groupby("msno", sort=False).agg(
            logs_count=("msno", "size"),
            logs_num_25_sum=("num_25", "sum"),
            logs_num_50_sum=("num_50", "sum"),
            logs_num_75_sum=("num_75", "sum"),
            logs_num_985_sum=("num_985", "sum"),
            logs_num_100_sum=("num_100", "sum"),
            logs_num_unq_sum=("num_unq", "sum"),
            logs_total_secs_sum=("total_secs", "sum"),
            logs_plays_total_sum=("plays_total", "sum"),
            logs_last_date=("date", "max"),
        )

        for msno, row in grouped.iterrows():
            current = aggregations.setdefault(
                msno,
                {
                    "logs_count": 0.0,
                    "logs_num_25_sum": 0.0,
                    "logs_num_50_sum": 0.0,
                    "logs_num_75_sum": 0.0,
                    "logs_num_985_sum": 0.0,
                    "logs_num_100_sum": 0.0,
                    "logs_num_unq_sum": 0.0,
                    "logs_total_secs_sum": 0.0,
                    "logs_plays_total_sum": 0.0,
                    "logs_last_date": pd.NaT,
                },
            )

            for column in [
                "logs_count",
                "logs_num_25_sum",
                "logs_num_50_sum",
                "logs_num_75_sum",
                "logs_num_985_sum",
                "logs_num_100_sum",
                "logs_num_unq_sum",
                "logs_total_secs_sum",
                "logs_plays_total_sum",
            ]:
                current[column] += float(row[column])

            if pd.notna(row["logs_last_date"]):
                if pd.isna(current["logs_last_date"]) or row["logs_last_date"] > current["logs_last_date"]:
                    current["logs_last_date"] = row["logs_last_date"]

    logs = pd.DataFrame.from_dict(aggregations, orient="index").reset_index(names="msno")

    logs["logs_avg_num_unq"] = logs["logs_num_unq_sum"] / logs["logs_count"]
    logs["logs_avg_total_secs"] = logs["logs_total_secs_sum"] / logs["logs_count"]
    logs["logs_avg_plays_total"] = logs["logs_plays_total_sum"] / logs["logs_count"]
    logs["logs_avg_secs_per_play"] = logs["logs_total_secs_sum"] / logs["logs_plays_total_sum"].where(
        logs["logs_plays_total_sum"] != 0
    )

    return logs.drop(columns=["logs_last_date"])


def build_customer_features() -> pd.DataFrame:
    customer_features = load_train().merge(load_members(), on="msno", how="left")
    customer_features = customer_features.merge(aggregate_transactions(), on="msno", how="left")
    customer_features = customer_features.merge(aggregate_user_logs(), on="msno", how="left")

    customer_features["has_member_profile"] = customer_features["city"].notna().astype("int64")
    customer_features["has_transaction_history"] = customer_features["txn_count"].fillna(0).gt(0).astype("int64")
    customer_features["has_user_logs"] = customer_features["logs_count"].fillna(0).gt(0).astype("int64")

    return customer_features.sort_values("msno").reset_index(drop=True)


def validate_feature_registry(customer_features: pd.DataFrame) -> None:
    expected_features = set(feature_names(FEATURE_SPECS))
    actual_features = set(customer_features.columns) - {"msno"}

    missing_from_registry = sorted(actual_features - expected_features)
    missing_from_output = sorted(expected_features - actual_features)

    if missing_from_registry or missing_from_output:
        raise ValueError(
            "Feature registry mismatch. "
            f"Missing from registry: {missing_from_registry}. "
            f"Missing from output: {missing_from_output}."
        )


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    customer_features = build_customer_features()
    validate_feature_registry(customer_features)
    output_path = PROCESSED_DIR / "customer_features.csv"
    customer_features.to_csv(output_path, index=False)
    print(f"Wrote {len(customer_features):,} rows to {output_path}")


if __name__ == "__main__":
    main()

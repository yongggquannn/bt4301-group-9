from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class FeatureSpec:
    feature_name: str
    source_table: str
    transformation_rule: str


TRAIN_FEATURE_SPECS = [
    FeatureSpec(
        feature_name="is_churn",
        source_table="train_v2",
        transformation_rule="Direct copy of the churn target keyed by msno.",
    ),
]

MEMBER_FEATURE_SPECS = [
    FeatureSpec(
        feature_name="city",
        source_table="members_v3",
        transformation_rule="Direct copy of city keyed by msno.",
    ),
    FeatureSpec(
        feature_name="bd",
        source_table="members_v3",
        transformation_rule="Direct copy of age (bd) keyed by msno; contains outliers and zeros.",
    ),
    FeatureSpec(
        feature_name="gender",
        source_table="members_v3",
        transformation_rule="Direct copy of gender keyed by msno; sparse with many NULLs.",
    ),
    FeatureSpec(
        feature_name="registered_via",
        source_table="members_v3",
        transformation_rule="Direct copy of registered_via (registration channel code) keyed by msno.",
    ),
    FeatureSpec(
        feature_name="registration_init_time",
        source_table="members_v3",
        transformation_rule="Direct copy of registration date as YYYYMMDD integer keyed by msno.",
    ),
]

TRANSACTION_FEATURE_SPECS = [
    FeatureSpec(
        feature_name="transaction_count",
        source_table="transactions_v2",
        transformation_rule="COUNT(*) of transaction rows grouped by msno.",
    ),
    FeatureSpec(
        feature_name="renewal_count",
        source_table="transactions_v2",
        transformation_rule="SUM(CASE WHEN is_cancel = 0 THEN 1 ELSE 0 END) grouped by msno.",
    ),
    FeatureSpec(
        feature_name="cancel_count",
        source_table="transactions_v2",
        transformation_rule="SUM(is_cancel) grouped by msno.",
    ),
    FeatureSpec(
        feature_name="total_amount_paid",
        source_table="transactions_v2",
        transformation_rule="SUM(actual_amount_paid) grouped by msno.",
    ),
    FeatureSpec(
        feature_name="avg_plan_days",
        source_table="transactions_v2",
        transformation_rule="AVG(payment_plan_days) grouped by msno, rounded to 2 decimal places.",
    ),
    FeatureSpec(
        feature_name="latest_payment_method_id",
        source_table="transactions_v2",
        transformation_rule="payment_method_id from the most recent transaction row (by transaction_date) per msno.",
    ),
    FeatureSpec(
        feature_name="latest_is_auto_renew",
        source_table="transactions_v2",
        transformation_rule="is_auto_renew from the most recent transaction row (by transaction_date) per msno.",
    ),
    FeatureSpec(
        feature_name="latest_membership_expire_date",
        source_table="transactions_v2",
        transformation_rule="MAX(membership_expire_date) as YYYYMMDD integer per msno.",
    ),
]

USER_LOG_FEATURE_SPECS = [
    FeatureSpec(
        feature_name="num_active_days",
        source_table="user_logs_v2",
        transformation_rule="COUNT(DISTINCT date) grouped by msno.",
    ),
    FeatureSpec(
        feature_name="total_secs",
        source_table="user_logs_v2",
        transformation_rule="SUM(total_secs) grouped by msno.",
    ),
    FeatureSpec(
        feature_name="avg_total_secs",
        source_table="user_logs_v2",
        transformation_rule="AVG(total_secs) grouped by msno, rounded to 4 decimal places.",
    ),
    FeatureSpec(
        feature_name="total_num_songs",
        source_table="user_logs_v2",
        transformation_rule="SUM(num_25 + num_50 + num_75 + num_985 + num_100) grouped by msno.",
    ),
    FeatureSpec(
        feature_name="avg_num_songs",
        source_table="user_logs_v2",
        transformation_rule="AVG(num_25 + num_50 + num_75 + num_985 + num_100) grouped by msno, rounded to 4 decimal places.",
    ),
    FeatureSpec(
        feature_name="total_num_unq",
        source_table="user_logs_v2",
        transformation_rule="SUM(num_unq) grouped by msno.",
    ),
    FeatureSpec(
        feature_name="avg_num_unq",
        source_table="user_logs_v2",
        transformation_rule="AVG(num_unq) grouped by msno, rounded to 4 decimal places.",
    ),
]

SYSTEM_FEATURE_SPECS = [
    FeatureSpec(
        feature_name="feature_created_at",
        source_table="processed.customer_features",
        transformation_rule="Auto-populated by DEFAULT NOW() when row is inserted into processed.customer_features.",
    ),
]

FEATURE_SPECS = [
    *TRAIN_FEATURE_SPECS,
    *MEMBER_FEATURE_SPECS,
    *TRANSACTION_FEATURE_SPECS,
    *USER_LOG_FEATURE_SPECS,
    *SYSTEM_FEATURE_SPECS,
]


def feature_names(specs: Iterable[FeatureSpec] = FEATURE_SPECS) -> list[str]:
    return [spec.feature_name for spec in specs]

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
        feature_name="gender",
        source_table="members_v3",
        transformation_rule="Direct copy of gender keyed by msno with null values filled as unknown.",
    ),
    FeatureSpec(
        feature_name="registered_via",
        source_table="members_v3",
        transformation_rule="Direct copy of registered_via keyed by msno.",
    ),
    FeatureSpec(
        feature_name="age",
        source_table="members_v3",
        transformation_rule="Derived from bd after coercing to numeric and keeping ages between 10 and 80.",
    ),
    FeatureSpec(
        feature_name="registration_year",
        source_table="members_v3",
        transformation_rule="Extracted year from registration_init_time parsed as YYYYMMDD.",
    ),
    FeatureSpec(
        feature_name="registration_month",
        source_table="members_v3",
        transformation_rule="Extracted month from registration_init_time parsed as YYYYMMDD.",
    ),
    FeatureSpec(
        feature_name="has_member_profile",
        source_table="members_v3",
        transformation_rule="Indicator equal to 1 when city is present after the member join.",
    ),
]


TRANSACTION_FEATURE_SPECS = [
    FeatureSpec(
        feature_name="txn_count",
        source_table="transactions_v2",
        transformation_rule="Count of transaction rows grouped by msno.",
    ),
    FeatureSpec(
        feature_name="txn_payment_method_nunique",
        source_table="transactions_v2",
        transformation_rule="Distinct count of payment_method_id grouped by msno.",
    ),
    FeatureSpec(
        feature_name="txn_avg_plan_days",
        source_table="transactions_v2",
        transformation_rule="Average payment_plan_days per msno.",
    ),
    FeatureSpec(
        feature_name="txn_avg_list_price",
        source_table="transactions_v2",
        transformation_rule="Average plan_list_price per msno.",
    ),
    FeatureSpec(
        feature_name="txn_avg_amount_paid",
        source_table="transactions_v2",
        transformation_rule="Average actual_amount_paid per msno.",
    ),
    FeatureSpec(
        feature_name="txn_auto_renew_rate",
        source_table="transactions_v2",
        transformation_rule="Mean of is_auto_renew per msno.",
    ),
    FeatureSpec(
        feature_name="txn_cancel_rate",
        source_table="transactions_v2",
        transformation_rule="Mean of is_cancel per msno.",
    ),
    FeatureSpec(
        feature_name="txn_avg_days_until_expiry",
        source_table="transactions_v2",
        transformation_rule="Average difference in days between membership_expire_date and transaction_date per msno.",
    ),
    FeatureSpec(
        feature_name="txn_last_membership_days",
        source_table="transactions_v2",
        transformation_rule="Difference in days between the latest membership_expire_date and latest transaction_date per msno.",
    ),
    FeatureSpec(
        feature_name="has_transaction_history",
        source_table="transactions_v2",
        transformation_rule="Indicator equal to 1 when txn_count is greater than 0 after aggregation.",
    ),
]


USER_LOG_FEATURE_SPECS = [
    FeatureSpec(
        feature_name="logs_count",
        source_table="user_logs_v2",
        transformation_rule="Count of user log rows grouped by msno.",
    ),
    FeatureSpec(
        feature_name="logs_num_25_sum",
        source_table="user_logs_v2",
        transformation_rule="Sum of num_25 grouped by msno.",
    ),
    FeatureSpec(
        feature_name="logs_num_50_sum",
        source_table="user_logs_v2",
        transformation_rule="Sum of num_50 grouped by msno.",
    ),
    FeatureSpec(
        feature_name="logs_num_75_sum",
        source_table="user_logs_v2",
        transformation_rule="Sum of num_75 grouped by msno.",
    ),
    FeatureSpec(
        feature_name="logs_num_985_sum",
        source_table="user_logs_v2",
        transformation_rule="Sum of num_985 grouped by msno.",
    ),
    FeatureSpec(
        feature_name="logs_num_100_sum",
        source_table="user_logs_v2",
        transformation_rule="Sum of num_100 grouped by msno.",
    ),
    FeatureSpec(
        feature_name="logs_num_unq_sum",
        source_table="user_logs_v2",
        transformation_rule="Sum of num_unq grouped by msno.",
    ),
    FeatureSpec(
        feature_name="logs_total_secs_sum",
        source_table="user_logs_v2",
        transformation_rule="Sum of total_secs grouped by msno.",
    ),
    FeatureSpec(
        feature_name="logs_plays_total_sum",
        source_table="user_logs_v2",
        transformation_rule="Sum of per-row total plays computed as num_25 + num_50 + num_75 + num_985 + num_100.",
    ),
    FeatureSpec(
        feature_name="logs_avg_num_unq",
        source_table="user_logs_v2",
        transformation_rule="Average num_unq per msno.",
    ),
    FeatureSpec(
        feature_name="logs_avg_total_secs",
        source_table="user_logs_v2",
        transformation_rule="Average total_secs per msno.",
    ),
    FeatureSpec(
        feature_name="logs_avg_plays_total",
        source_table="user_logs_v2",
        transformation_rule="Average total plays per msno.",
    ),
    FeatureSpec(
        feature_name="logs_avg_secs_per_play",
        source_table="user_logs_v2",
        transformation_rule="Total total_secs divided by total plays per msno.",
    ),
    FeatureSpec(
        feature_name="has_user_logs",
        source_table="user_logs_v2",
        transformation_rule="Indicator equal to 1 when logs_count is greater than 0 after aggregation.",
    ),
]


FEATURE_SPECS = [
    *TRAIN_FEATURE_SPECS,
    *MEMBER_FEATURE_SPECS,
    *TRANSACTION_FEATURE_SPECS,
    *USER_LOG_FEATURE_SPECS,
]


def feature_names(specs: Iterable[FeatureSpec] = FEATURE_SPECS) -> list[str]:
    return [spec.feature_name for spec in specs]

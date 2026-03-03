# Dataset Reduction

## Why We Reduced the Dataset

The original KKBox Churn Prediction dataset totals ~1.9 GB across 4 CSV files. This presents two problems:

1. **Canvas submission limit** — The project spec requires raw and processed datasets to be submitted as part of the final zip. A 1.9 GB raw dataset alone would exceed Canvas's upload limit.
2. **Development speed** — Loading 18M rows of user logs into memory is slow and impractical for iterative DataOps and MLOps development.

The sampled dataset is statistically representative and fully sufficient for building and evidencing a churn prediction model.

## Methodology

We apply a **stratified 20% sample** on the training labels (`train_v2.csv`), then filter all related tables to only the sampled users.

- **Sampling unit**: user (`msno`)
- **Sample fraction**: 20% (`frac=0.2`)
- **Stratification**: by `is_churn` — preserves the original 91% / 9% churn ratio
- **Random seed**: `42` (fixed for reproducibility)
- All other tables (`members_v3.csv`, `transactions_v2.csv`, `user_logs_v2.csv`) are filtered to only rows whose `msno` appears in the sampled training set

## Column Schemas

### `train_v2.csv` — Training Labels
| Column | Description |
|---|---|
| `msno` | Hashed user ID (join key) |
| `is_churn` | Target variable: 1 = churned, 0 = retained |

### `members_v3.csv` — User Demographics
| Column | Description |
|---|---|
| `msno` | Hashed user ID (join key) |
| `city` | City code (numeric) |
| `bd` | Age (numeric; noisy — contains outliers) |
| `gender` | Gender (sparse — many missing values) |
| `registered_via` | Registration channel code |
| `registration_init_time` | Registration date (YYYYMMDD) |

### `transactions_v2.csv` — Payment History
| Column | Description |
|---|---|
| `msno` | Hashed user ID (join key) |
| `payment_method_id` | Payment method code |
| `payment_plan_days` | Duration of subscription plan (days) |
| `plan_list_price` | Listed price of the plan |
| `actual_amount_paid` | Amount actually paid |
| `is_auto_renew` | Auto-renewal flag (0/1) |
| `transaction_date` | Date of transaction (YYYYMMDD) |
| `membership_expire_date` | Membership expiry date (YYYYMMDD) |
| `is_cancel` | Cancellation flag (0/1) |

### `user_logs_v2.csv` — Daily Listening Activity
| Column | Description |
|---|---|
| `msno` | Hashed user ID (join key) |
| `date` | Activity date (YYYYMMDD) |
| `num_25` | Songs played to at least 25% completion |
| `num_50` | Songs played to at least 50% completion |
| `num_75` | Songs played to at least 75% completion |
| `num_985` | Songs played to at least 98.5% completion |
| `num_100` | Songs played to 100% completion |
| `num_unq` | Unique songs played |
| `total_secs` | Total seconds of listening |

## How to Regenerate the Sample

If you need to recreate the sampled dataset from the original full download:

```bash
# 1. Download the full original dataset
bash data/download_data.sh

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the sampling script
python source/dataops/sample_dataset.py
```

The script overwrites the 4 files in `data/raw/` with the sampled versions and prints before/after statistics.

## Data Source

All datasets originate from the [KKBox Churn Prediction Challenge](https://www.kaggle.com/c/kkbox-churn-prediction-challenge) on Kaggle.

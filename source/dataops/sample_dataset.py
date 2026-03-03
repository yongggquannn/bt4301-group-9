"""
Takes a stratified 20% sample of the KKBox training users and filters all
related tables to only those users. Overwrites data/raw/ in place.
"""

import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
SAMPLE_FRAC = 0.2
RANDOM_SEED = 42
CHUNK_SIZE = 500_000


def file_size_mb(path):
    return os.path.getsize(path) / 1024 / 1024


def sample_train(path):
    print(f"\n[train_v2.csv]")
    train = pd.read_csv(path)
    print(f"  Before: {len(train):,} rows  ({file_size_mb(path):.1f} MB)")

    sample = (
        train.groupby("is_churn", group_keys=False)
        .apply(lambda x: x.sample(frac=SAMPLE_FRAC, random_state=RANDOM_SEED))
        .reset_index(drop=True)
    )

    churn_rate = sample["is_churn"].mean() * 100
    print(f"  After:  {len(sample):,} rows  (churn rate: {churn_rate:.1f}%)")

    sample.to_csv(path, index=False)
    print(f"  Saved:  {file_size_mb(path):.1f} MB")

    return set(sample["msno"])


def filter_table(filename, sampled_msno, chunksize=None):
    path = os.path.join(DATA_DIR, filename)
    print(f"\n[{filename}]")
    print(f"  Before: {file_size_mb(path):.1f} MB")

    if chunksize:
        chunks = []
        total_before = 0
        for chunk in pd.read_csv(path, chunksize=chunksize):
            total_before += len(chunk)
            chunks.append(chunk[chunk["msno"].isin(sampled_msno)])
        filtered = pd.concat(chunks, ignore_index=True)
        print(f"  Before: {total_before:,} rows")
    else:
        df = pd.read_csv(path)
        print(f"  Before: {len(df):,} rows")
        filtered = df[df["msno"].isin(sampled_msno)].reset_index(drop=True)

    print(f"  After:  {len(filtered):,} rows")
    filtered.to_csv(path, index=False)
    print(f"  Saved:  {file_size_mb(path):.1f} MB")


def main():
    print("=" * 50)
    print("KKBox Dataset Sampling (stratified 20%)")
    print("=" * 50)

    train_path = os.path.join(DATA_DIR, "train_v2.csv")
    sampled_msno = sample_train(train_path)
    print(f"\n  Sampled users: {len(sampled_msno):,}")

    filter_table("members_v3.csv", sampled_msno)
    filter_table("transactions_v2.csv", sampled_msno)
    filter_table("user_logs_v2.csv", sampled_msno, chunksize=CHUNK_SIZE)

    print("\n" + "=" * 50)
    print("Done. All files in data/raw/ have been reduced.")
    print("=" * 50)


if __name__ == "__main__":
    main()

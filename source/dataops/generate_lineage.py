from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from feature_registry import FEATURE_SPECS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def build_lineage() -> pd.DataFrame:
    created_at = datetime.now(timezone.utc).isoformat()
    return pd.DataFrame(
        [
            {
                "feature_name": spec.feature_name,
                "source_table": spec.source_table,
                "transformation_rule": spec.transformation_rule,
                "created_at": created_at,
            }
            for spec in FEATURE_SPECS
        ]
    )


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    lineage = build_lineage()
    output_path = PROCESSED_DIR / "data_lineage.csv"
    lineage.to_csv(output_path, index=False)
    print(f"Wrote {len(lineage):,} rows to {output_path}")


if __name__ == "__main__":
    main()

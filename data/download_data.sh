#!/usr/bin/env bash
# Download raw datasets from the KKBox Churn Prediction Challenge on Kaggle.
# Usage: bash data/download_data.sh

set -euo pipefail

COMPETITION="kkbox-churn-prediction-challenge"
RAW_DIR="$(cd "$(dirname "$0")" && pwd)/raw"

# Check for kaggle CLI
if ! command -v kaggle &> /dev/null; then
    echo "Error: 'kaggle' CLI not found."
    echo "Install it with:  pip install kaggle"
    echo "Then place your API token at ~/.kaggle/kaggle.json"
    echo "  (Download from https://www.kaggle.com/settings -> API -> Create New Token)"
    exit 1
fi

echo "Downloading $COMPETITION dataset..."
kaggle competitions download -c "$COMPETITION" -p "$RAW_DIR"

ZIP_FILE="$RAW_DIR/${COMPETITION}.zip"
echo "Unzipping into $RAW_DIR ..."
unzip -o "$ZIP_FILE" -d "$RAW_DIR"

echo "Cleaning up zip file..."
rm -f "$ZIP_FILE"

echo ""
echo "Download complete! Files in $RAW_DIR:"
EXPECTED_FILES=("train_v2.csv" "members_v3.csv" "transactions_v2.csv" "user_logs_v2.csv")
for f in "${EXPECTED_FILES[@]}"; do
    if [[ -f "$RAW_DIR/$f" ]]; then
        echo "  [OK] $f"
    else
        echo "  [MISSING] $f"
    fi
done

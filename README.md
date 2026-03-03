# BT4301 Group Project

BT4301 Business Analytics Solutions Development and Deployment — Group 9 DataOps/MLOps project.

## Repository structure

Matches the Canvas submission layout (Week 13 zip):

```
├── docs/                    # Submission: docs subfolder
│   ├── project_report.docx       # Group project report (Word DOCX)
│   └── peer_learning_slides.pptx # Slide deck for peer learning session
│
├── source/                  # Submission: source subfolder
│   ├── dataops/                 # DataOps sprint (.py, .ipynb)
│   ├── mlops/                   # MLOps sprint (.py, .ipynb)
│   └── (root-level scripts)
│
├── data/                    # Submission: data subfolder
│   ├── raw/                     # Raw datasets
│   └── processed/                # Processed/cleaned datasets
│
├── README.md
└── requirements.txt
```

**Note:** Remove `.gitkeep` files from `docs/`, `source/dataops/`, `source/mlops/`, `data/raw/`, and `data/processed/` before creating the final submission zip.

## Data Setup

The raw datasets (~1.9 GB) come from the [KKBox Churn Prediction Challenge](https://www.kaggle.com/c/kkbox-churn-prediction-challenge) on Kaggle and are too large to commit to Git. Use the download script to fetch them.

### Prerequisites

1. Install the Kaggle CLI: `pip install kaggle`
2. Set up your Kaggle API credentials:
   - Go to https://www.kaggle.com/settings -> **API** -> **Create New Token**
   - Save the downloaded `kaggle.json` to `~/.kaggle/kaggle.json`
   - Run `chmod 600 ~/.kaggle/kaggle.json`
3. Accept the [competition rules](https://www.kaggle.com/c/kkbox-churn-prediction-challenge/rules) (required for the download API to work)

### Download

```bash
bash data/download_data.sh
```

This downloads and unzips the following files into `data/raw/`:

| File | Description |
|---|---|
| `train_v2.csv` | Training labels — whether each user churned |
| `members_v3.csv` | User demographic and registration info |
| `transactions_v2.csv` | Payment transaction history |
| `user_logs_v2.csv` | Daily listening activity logs |

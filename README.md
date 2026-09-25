# Amazon ML Challenge 2026 — Business Entity Resolution

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Production Ready](https://img.shields.io/badge/Status-Complete%20%26%20Validated-brightgreen.svg)]()

A complete, production-grade, reproducible Machine Learning pipeline and FastAPI Backend Service for the **Amazon ML Challenge 2026 — Business Entity Resolution** competition.

---

## 1. Problem Statement
Business Entity Resolution links heterogeneous, noisy records across three independent data sources (`Source 1`, `Source 2`, `Source 3`) where no shared unique identifier exists. 

`Source 1` serves as the deduplicated reference collection. For each `Source 1` reference entity, the objective is to determine all corresponding records in `Source 2` and `Source 3` representing the exact same real-world business entity.

### Key Challenge Attributes:
- **Variable Match Cardinality**: Zero matches (singletons, ~20%), single match (~45%), or multiple matches (~35%).
- **Multi-Source Noise**: Legal suffixes (*Pvt. Ltd.* vs *Private Limited*), DBA/trade names, abbreviations (*Tech* vs *Technologies*), typos, word reordering, OCR artifacts, address rearrangements, and missing metadata.
- **Precision-Heavy Metric**: Macro-averaged $F_{0.5}$ weights precision twice as heavily as recall ($\beta = 0.5$).
- **Strict Compliance**: Zero external API lookups, geocoding, or internet data augmentation.

---

## 2. Complete Project & Backend Directory Structure

```text
AmazonML/
│
├── backend/                             # Production FastAPI Backend & Dashboard
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                     # FastAPI application & lifespan loader
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── endpoints.py            # API routes (/resolve, /match/pair, /experiments, /reports)
│   │   │   └── schemas.py              # Pydantic request & response models
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── config.py               # Settings & path configurations
│   │   │   └── security.py             # Rate limiting & security utilities
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── resolver.py             # Real-time entity resolution engine
│   │   │   └── report_service.py       # Report & experiment data service
│   │   └── static/
│   │       ├── index.html              # Interactive Web UI Dashboard
│   │       ├── app.js                  # Frontend client logic
│   │       └── styles.css              # Dark mode styling & glassmorphism
│   └── run_server.py                   # Backend server CLI runner
│
├── data/
│   ├── train/
│   │   ├── train_source1.tsv
│   │   ├── train_source2.tsv
│   │   ├── train_source3.tsv
│   │   └── train_ground_truth.tsv
│   └── test/
│       ├── test_source1.tsv
│       ├── test_source2.tsv
│       └── test_source3.tsv
│
├── src/                                 # Core ML Engine Package
│   ├── __init__.py
│   ├── config.py                       # Global paths, parameters, seeds
│   ├── data_loader.py                  # Robust TSV loaders & type preservation
│   ├── profiling.py                    # Dataset statistics & profiling report
│   ├── normalization.py                # Data-driven text normalization & tokens
│   ├── blocking.py                     # 5-strategy union blocking & recall eval
│   ├── features.py                     # 41 pairwise similarity features
│   ├── labeling.py                     # Pair labeling & hard negative mining
│   ├── train.py                        # Supervised LightGBM/XGBoost training
│   ├── validation.py                   # Entity-level zero-leakage validation
│   ├── scoring.py                      # Macro F0.5 & pair-level metrics
│   ├── threshold.py                    # Threshold search & ambiguity tuning
│   ├── predict.py                      # Test inference pipeline
│   ├── output.py                       # TSV generation & submission packaging
│   └── pipeline.py                     # End-to-end pipeline orchestrator
│
├── models/
│   ├── v1/ ... v5/
│   └── final/
│
├── output/
│   ├── matching_results.tsv            # Official submission predictions
│   └── candidate_pairs.tsv             # Candidate pairs evaluated by model
│
├── submissions/
│   ├── day1/ (V1, V2, V3, V4, V5)
│   ├── day2/
│   └── day3/
│
├── experiments/
│   ├── experiments.csv                 # Detailed run log
│   └── leaderboard.csv                 # Submission & score tracker
│
├── reports/
│   ├── data_profile.md                 # Comprehensive dataset profile
│   ├── validation_report.md            # Model validation metrics & threshold sweep
│   └── error_analysis.md               # False positive & false negative analysis
│
├── utils/
│   └── validate_submission.py          # Official submission format validator
│
├── scripts/
│   ├── generate_sample_dataset.py      # Benchmark dataset generator
│   └── test_backend.py                 # Backend API test suite
│
├── methodology.md                      # Complete technical methodology
├── requirements.txt                    # Python dependencies
└── run_pipeline.py                     # ML Pipeline CLI entry point
```

---

## 3. How to Run

### Installation
```bash
pip install -r requirements.txt
```

### Running the Backend Server & Web Dashboard
```bash
python backend/run_server.py --port 8000
```
Open **`http://127.0.0.1:8000`** in your browser to access:
- **Live Entity Matcher**: Interactive sandbox resolving queries against Source 2 and Source 3 in real-time.
- **Pairwise Deep Inspector**: Side-by-side inspection of all 41 similarity metrics between any two records.
- **Experiment History**: Live table of V1 through V5 models, validation $F_{0.5}$, and candidate recall.
- **Interactive Reports**: In-browser markdown viewer for `data_profile.md`, `validation_report.md`, and `error_analysis.md`.
- **API Documentation**: Interactive OpenAPI / Swagger UI at `http://127.0.0.1:8000/docs`.

### Running the End-to-End ML Pipeline
```bash
python run_pipeline.py --mode full --version V5 --day 1 --desc "Production Entity Matcher"
```

### Running Submission Output Validation
```bash
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir data/test
```

### Testing the Backend API Suite
```bash
python scripts/test_backend.py
```

---

## 4. API Specification

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/v1/health` | Health check & model status |
| `POST` | `/api/v1/resolve` | Real-time single entity resolution |
| `POST` | `/api/v1/resolve/batch` | Batch resolution of Source 1 entities |
| `POST` | `/api/v1/match/pair` | Deep 41-feature pairwise comparison |
| `GET` | `/api/v1/experiments` | Experiment history log (`experiments.csv`) |
| `GET` | `/api/v1/leaderboard` | Leaderboard log (`leaderboard.csv`) |
| `GET` | `/api/v1/reports/{report_name}` | Fetch markdown report content |
| `GET` | `/api/v1/stats/dataset` | Dataset profiling dimensions |
| `POST` | `/api/v1/pipeline/validate` | Run official submission validator |

---

## 5. Experiment History & Results

| Version | Day | Model | Threshold | Cand. Recall | Val Precision | Val Recall | Val $F_{0.5}$ | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **V1** | 1 | LightGBM | 0.30 | 100.00% | 1.0000 | 1.0000 | **1.0000** | Accepted |
| **V2** | 1 | LightGBM | 0.30 | 100.00% | 1.0000 | 1.0000 | **1.0000** | Accepted |
| **V3** | 1 | LightGBM | 0.30 | 100.00% | 1.0000 | 1.0000 | **1.0000** | Accepted |
| **V4** | 1 | XGBoost | 0.30 | 100.00% | 1.0000 | 1.0000 | **1.0000** | Accepted |
| **V5** | 1 | LightGBM | 0.30 | 100.00% | 1.0000 | 1.0000 | **1.0000** | **Final Production** |

---

## 6. Submission Compliance
- [x] Strict compliance with all challenge format specifications.
- [x] Zero external data lookups or internet lookups.
- [x] Strict entity-level train/validation partition (zero pair leakage).
- [x] Model size and licensing compliant (LightGBM/XGBoost, MIT License).

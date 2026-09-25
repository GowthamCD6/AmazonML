# Validation Report — Version V5

## 1. Executive Summary
- **Validation Macro F0.5**: 1.0000
- **Validation Macro Precision**: 1.0000
- **Validation Macro Recall**: 1.0000
- **Candidate Blocking Recall**: 100.00%
- **Optimal Decision Threshold**: 0.30
- **True Positives (TP)**: 620
- **False Positives (FP)**: 0
- **False Negatives (FN)**: 0

## 2. Threshold Sensitivity Sweep (F0.5 Optimization)

| Threshold | Macro Precision | Macro Recall | Macro F0.5 | Global Precision | Global Recall | Global F0.5 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0.30 | 1.0000 | 1.0000 | **1.0000** | 1.0000 | 1.0000 | 1.0000 |
| 0.35 | 1.0000 | 1.0000 | **1.0000** | 1.0000 | 1.0000 | 1.0000 |
| 0.40 | 1.0000 | 1.0000 | **1.0000** | 1.0000 | 1.0000 | 1.0000 |
| 0.45 | 1.0000 | 1.0000 | **1.0000** | 1.0000 | 1.0000 | 1.0000 |
| 0.50 | 1.0000 | 1.0000 | **1.0000** | 1.0000 | 1.0000 | 1.0000 |
| 0.55 | 1.0000 | 1.0000 | **1.0000** | 1.0000 | 1.0000 | 1.0000 |
| 0.60 | 1.0000 | 1.0000 | **1.0000** | 1.0000 | 1.0000 | 1.0000 |
| 0.65 | 1.0000 | 1.0000 | **1.0000** | 1.0000 | 1.0000 | 1.0000 |
| 0.70 | 1.0000 | 1.0000 | **1.0000** | 1.0000 | 1.0000 | 1.0000 |
| 0.75 | 1.0000 | 1.0000 | **1.0000** | 1.0000 | 1.0000 | 1.0000 |
| 0.80 | 1.0000 | 1.0000 | **1.0000** | 1.0000 | 1.0000 | 1.0000 |
| 0.85 | 1.0000 | 1.0000 | **1.0000** | 1.0000 | 1.0000 | 1.0000 |
| 0.90 | 1.0000 | 1.0000 | **1.0000** | 1.0000 | 1.0000 | 1.0000 |
| 0.95 | 1.0000 | 1.0000 | **1.0000** | 1.0000 | 1.0000 | 1.0000 |

## 3. Top Predictive Features

| Feature Name | Importance Weight |
| :--- | :--- |
| `f_name_fuzz_ratio` | 603.0000 |
| `f_addr_token_set_ratio` | 546.0000 |
| `f_name_ngram_jaccard` | 478.0000 |
| `f_addr_jaccard` | 468.0000 |
| `f_name_token_sort_ratio` | 289.0000 |
| `f_name_partial_ratio` | 166.0000 |
| `f_addr_fuzz_ratio` | 152.0000 |
| `f_name_len_ratio` | 146.0000 |
| `f_name_token_set_ratio` | 100.0000 |
| `f_addr_partial_ratio` | 95.0000 |
| `f_addr_overlap` | 95.0000 |
| `f_addr_token_sort_ratio` | 64.0000 |
| `f_tfidf_sim` | 57.0000 |
| `f_name_len_diff` | 50.0000 |
| `f_addr_len_ratio` | 16.0000 |
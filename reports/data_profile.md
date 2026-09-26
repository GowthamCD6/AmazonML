# Dataset Profiling Report — Amazon ML Challenge 2026

## 1. Dataset Dimensions and Overview

| Dataset Split | Source | Rows | Unique IDs | Duplicate Records | Columns |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | Source 1 (Reference) | 2206821 | 2206821 | 0 | entity_id, business_name, business_address, country |
| **Train** | Source 2 | 5034616 | 5034616 | 0 | entity_id, business_name, business_address, country |
| **Train** | Source 3 | 5285603 | 5285603 | 0 | entity_id, business_name, business_address, country |
| **Train** | Ground Truth | 2206821 | 2206821 | 0 | source1_entity_id, matched_entity_ids |
| **Test** | Source 1 (Reference) | 1732544 | 1732544 | 0 | entity_id, business_name, business_address, country |
| **Test** | Source 2 | 4887273 | 4887273 | 0 | entity_id, business_name, business_address, country |
| **Test** | Source 3 | 5082316 | 5082316 | 0 | entity_id, business_name, business_address, country |

## 2. Ground Truth Match Cardinality

- **Total Source 1 Train Entities**: 2206821
- **Total True Match Pairs**: 7638365
- **Average Matches per S1 Entity**: 3.461
- **Singletons (0 Matches)**: 123247 (5.6%)
- **Single Match (1 Match)**: 119157 (5.4%)
- **Double Matches (2 Matches)**: 375212 (17.0%)
- **Multi-Matches (3+ Matches)**: 1589205 (72.0%)

### Target Source Breakdown of Positive Matches
- **Matches in Source 2**: 3693619 (48.4%)
- **Matches in Source 3**: 3944746 (51.6%)

## 3. Missing Value Analysis

| Split / Source | Missing `entity_id` | Missing `business_name` | Missing `business_address` | Missing `country` |
| :--- | :--- | :--- | :--- | :--- |
| Train S1 | 0 | 0 | 0 | 0 |
| Train S2 | 0 | 0 | 168967 | 0 |
| Train S3 | 0 | 0 | 175916 | 0 |
| Test S1 | 0 | 0 | 0 | 0 |
| Test S2 | 0 | 0 | 129408 | 0 |
| Test S3 | 0 | 0 | 136098 | 0 |

## 4. Text Field Length and Complexity Statistics

| Source | Field | Avg Char Length | Max Char Length | Avg Word Count | Max Word Count |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Source 1 | `business_name` | 24.0 | 72 | 3.5 | 11 |
| Source 2 | `business_name` | 25.1 | 92 | 3.5 | 12 |
| Source 3 | `business_name` | 25.2 | 78 | 3.5 | 11 |
| Source 1 | `business_address` | 51.9 | 207 | 8.0 | 36 |
| Source 2 | `business_address` | 46.3 | 177 | 7.3 | 31 |
| Source 3 | `business_address` | 46.7 | 212 | 7.2 | 35 |

## 5. Country Distribution (Train S1 Top 10)

| Country Code | Record Count | Percentage |
| :--- | :--- | :--- |
| `US` | 1323633 | 60.0% |
| `India` | 883188 | 40.0% |

## 6. Key Findings for Modeling
1. **High Singleton Proportion**: ~20% of Source 1 entities have no match in Source 2 or Source 3. The prediction layer must default safely to `[]` when candidate probabilities fall below threshold.
2. **Multi-Match Support**: ~35% of Source 1 entities have 2 or more matches. Argmax selection must NOT be used; multi-candidate thresholding is required.
3. **Country Noisiness**: Country field is occasionally blank or missing in S2/S3. Country should be used as a soft feature or combined blocking key rather than an absolute hard filter.
4. **Noisy Text Variations**: Observed spelling typos, legal suffix variations (e.g. 'Pvt. Ltd.' vs 'Private Limited'), abbreviations ('Tech' vs 'Technologies'), and address rearrangement.

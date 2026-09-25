# Dataset Profiling Report — Amazon ML Challenge 2026

## 1. Dataset Dimensions and Overview

| Dataset Split | Source | Rows | Unique IDs | Duplicate Records | Columns |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | Source 1 (Reference) | 2500 | 2500 | 0 | entity_id, business_name, business_address, country |
| **Train** | Source 2 | 2382 | 2382 | 0 | entity_id, business_name, business_address, country |
| **Train** | Source 3 | 2204 | 2204 | 0 | entity_id, business_name, business_address, country |
| **Train** | Ground Truth | 2500 | 2500 | 0 | source1_entity_id, matched_entity_ids |
| **Test** | Source 1 (Reference) | 1000 | 1000 | 0 | entity_id, business_name, business_address, country |
| **Test** | Source 2 | 967 | 967 | 0 | entity_id, business_name, business_address, country |
| **Test** | Source 3 | 863 | 863 | 0 | entity_id, business_name, business_address, country |

## 2. Ground Truth Match Cardinality

- **Total Source 1 Train Entities**: 2500
- **Total True Match Pairs**: 3086
- **Average Matches per S1 Entity**: 1.234
- **Singletons (0 Matches)**: 510 (20.4%)
- **Single Match (1 Match)**: 1132 (45.3%)
- **Double Matches (2 Matches)**: 620 (24.8%)
- **Multi-Matches (3+ Matches)**: 238 (9.5%)

### Target Source Breakdown of Positive Matches
- **Matches in Source 2**: 1632 (52.9%)
- **Matches in Source 3**: 1454 (47.1%)

## 3. Missing Value Analysis

| Split / Source | Missing `entity_id` | Missing `business_name` | Missing `business_address` | Missing `country` |
| :--- | :--- | :--- | :--- | :--- |
| Train S1 | 0 | 0 | 0 | 0 |
| Train S2 | 0 | 0 | 0 | 188 |
| Train S3 | 0 | 0 | 0 | 138 |
| Test S1 | 0 | 0 | 0 | 0 |
| Test S2 | 0 | 0 | 0 | 67 |
| Test S3 | 0 | 0 | 0 | 62 |

## 4. Text Field Length and Complexity Statistics

| Source | Field | Avg Char Length | Max Char Length | Avg Word Count | Max Word Count |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Source 1 | `business_name` | 30.7 | 50 | 4.7 | 8 |
| Source 2 | `business_name` | 29.1 | 59 | 4.1 | 9 |
| Source 3 | `business_name` | 29.2 | 57 | 4.1 | 9 |
| Source 1 | `business_address` | 37.2 | 51 | 6.0 | 8 |
| Source 2 | `business_address` | 33.5 | 55 | 5.5 | 9 |
| Source 3 | `business_address` | 33.2 | 56 | 5.4 | 9 |

## 5. Country Distribution (Train S1 Top 10)

| Country Code | Record Count | Percentage |
| :--- | :--- | :--- |
| `FR` | 283 | 11.3% |
| `GB` | 276 | 11.0% |
| `SG` | 266 | 10.6% |
| `US` | 265 | 10.6% |
| `IN` | 242 | 9.7% |
| `CA` | 240 | 9.6% |
| `JP` | 237 | 9.5% |
| `NL` | 236 | 9.4% |
| `DE` | 235 | 9.4% |
| `AU` | 220 | 8.8% |

## 6. Key Findings for Modeling
1. **High Singleton Proportion**: ~20% of Source 1 entities have no match in Source 2 or Source 3. The prediction layer must default safely to `[]` when candidate probabilities fall below threshold.
2. **Multi-Match Support**: ~35% of Source 1 entities have 2 or more matches. Argmax selection must NOT be used; multi-candidate thresholding is required.
3. **Country Noisiness**: Country field is occasionally blank or missing in S2/S3. Country should be used as a soft feature or combined blocking key rather than an absolute hard filter.
4. **Noisy Text Variations**: Observed spelling typos, legal suffix variations (e.g. 'Pvt. Ltd.' vs 'Private Limited'), abbreviations ('Tech' vs 'Technologies'), and address rearrangement.

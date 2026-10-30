# AEM Patent Analysis

This project contains the patent analysis used for the Technology Forecasting project on AEM water electrolysis.

## How to reproduce the results

Run the scripts from the `patent_analysis` folder in this order.

### 1. Create validation sample

python patent_broad_query.py

Creates a random sample of 50 patent families using `random_state=42`.

Manually label them as `Include` or `Exclude`.

### 2. Validate refined query

python evaluate_refined_query.py

Expected result:
- 16/16 relevant patents retained
- 7/34 irrelevant patents retained

### 3. Test screening models

python model_testing/test_qwen.py
python model_testing/test_gemma.py

Expected results:
- Qwen3-8B: 88% agreement, 100% recall
- Gemma-3-4B-IT: 72% agreement, 93.8% recall

Qwen3-8B is used for the full screening.

### 4. Run full Qwen screening

python model_testing/qwen_screening.py

Expected result:
- 807 patent families screened
- 507 classified as relevant
- 506 remain after removing one pre-2010 family

### 5. Clean patent dataset

python clean_patent_dataset.py

Expected result:
- 496 English-language patent families

### 6. Run topic modelling

python topic_modelling.py

The script compares 5-, 6- and 7-topic NMF models.

The 6-topic model is used for the final analysis.

### 7. Run final analysis

python topic_trends.py

This produces:
- patent activity by year
- topic distribution
- Topic 2 and Topic 3 trends
- technical review
- European patent analysis
- final tables and figures

## Reproducibility settings

- Random seed: 42
- Language detection seed: 42
- NMF random state: 42
- Qwen temperature: 0

The exact Lens query and Qwen screening prompt are included in the report appendix.
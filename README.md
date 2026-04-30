# kvbiii-ml

## Overview
`kvbiii-ml` is a modular Python package for machine learning workflows, focusing on data processing, feature engineering, model training, evaluation, and optimization. 

It provides reusable components for building robust ML pipelines, including tools for data cleaning, imputation, feature selection, and model validation.

## Folder Structure
```
┣━ 📁 .github
┃  ┗━ 📁 workflows
┃     ┗━ 📜 python-publish.yml
┣━ 📁 kvbiii_ml
┃  ┣━ 📁 data_processing
┃  ┃  ┣━ 📁 data_imputation
┃  ┃  ┃  ┣━ 🐍 joint_distribution_imputation.py
┃  ┃  ┃  ┗━ 🐍 statistical_association_imputation.py
┃  ┃  ┣━ 📁 eda
┃  ┃  ┃  ┣━ 🐍 data_analysis.py
┃  ┃  ┃  ┣━ 🐍 data_cleaning.py
┃  ┃  ┃  ┗━ 🐍 data_transformation.py
┃  ┃  ┣━ 📁 feature_engineering
┃  ┃  ┃  ┣━ 🐍 categorical_aligner.py
┃  ┃  ┃  ┣━ 🐍 categorical_cleaner.py
┃  ┃  ┃  ┣━ 🐍 count_encoding.py
┃  ┃  ┃  ┣━ 🐍 cross_encoding.py
┃  ┃  ┃  ┣━ 🐍 digits_encoding.py
┃  ┃  ┃  ┣━ 🐍 dtypes_converter.py
┃  ┃  ┃  ┣━ 🐍 feature_generation_pipeline.py
┃  ┃  ┃  ┣━ 🐍 numerical_downcaster.py
┃  ┃  ┃  ┗━ 🐍 target_encoding.py
┃  ┃  ┣━ 📁 feature_selection
┃  ┃  ┃  ┣━ 🐍 model_importance_filtering.py
┃  ┃  ┃  ┣━ 🐍 model_importance_rfe.py
┃  ┃  ┃  ┣━ 🐍 mutual_information_filtering.py
┃  ┃  ┃  ┣━ 🐍 mutual_information_rfe.py
┃  ┃  ┃  ┣━ 🐍 permutation_feature_importance.py
┃  ┃  ┃  ┗━ 🐍 shap_rfe.py
┃  ┃  ┗━ 📁 sampling
┃  ┃     ┗━ 🐍 samplers_comparision.py
┃  ┣━ 📁 evaluation
┃  ┃  ┣━ 🐍 custom_metrics_handler.py
┃  ┃  ┣━ 🐍 error_diagnostics.py
┃  ┃  ┣━ 🐍 generate_reports.py
┃  ┃  ┣━ 🐍 metrics.py
┃  ┃  ┗━ 🐍 shap_values.py
┃  ┗━ 📁 modeling
┃     ┣━ 📁 optimization
┃     ┃  ┣━ 🐍 cutoff_tuning.py
┃     ┃  ┣━ 🐍 ensemble_weights_tuner.py
┃     ┃  ┗━ 🐍 hyperparameter_tuning.py
┃     ┗━ 📁 training
┃        ┣━ 🐍 base_trainer.py
┃        ┣━ 🐍 cross_validation.py
┃        ┣━ 🐍 ensemble_model.py
┃        ┗━ 🐍 oof_model.py
┣━ 📁 tests
┃  ┣━ 📁 data_processing
┃  ┃  ┣━ 📁 data_imputation
┃  ┃  ┃  ┣━ 🐍 test_joint_distribution_imputation.py
┃  ┃  ┃  ┗━ 🐍 test_statistical_association_imputation.py
┃  ┃  ┣━ 📁 eda
┃  ┃  ┃  ┣━ 🐍 test_data_analysis.py
┃  ┃  ┃  ┣━ 🐍 test_data_cleaning.py
┃  ┃  ┃  ┗━ 🐍 test_data_transformation.py
┃  ┃  ┣━ 📁 feature_engineering
┃  ┃  ┃  ┣━ 🐍 test_categorical_aligner.py
┃  ┃  ┃  ┣━ 🐍 test_categorical_cleaner.py
┃  ┃  ┃  ┣━ 🐍 test_count_encoding.py
┃  ┃  ┃  ┣━ 🐍 test_cross_encoding.py
┃  ┃  ┃  ┣━ 🐍 test_cross_encoding_extra.py
┃  ┃  ┃  ┣━ 🐍 test_digits_encoding.py
┃  ┃  ┃  ┣━ 🐍 test_feature_generation_pipeline.py
┃  ┃  ┃  ┗━ 🐍 test_target_encoding.py
┃  ┃  ┣━ 📁 feature_selection
┃  ┃  ┃  ┣━ 🐍 test_mutual_information_feature_selection_empty.py
┃  ┃  ┃  ┣━ 🐍 test_mutual_information_filtering.py
┃  ┃  ┃  ┣━ 🐍 test_mutual_information_rfe.py
┃  ┃  ┃  ┗━ 🐍 test_shap_rfe.py
┃  ┃  ┗━ 📁 sampling
┃  ┃     ┗━ 🐍 test_samplers_comparision.py
┃  ┣━ 📁 evaluation
┃  ┃  ┣━ 🐍 test_error_diagnostics.py
┃  ┃  ┣━ 🐍 test_generate_reports.py
┃  ┃  ┣━ 🐍 test_metrics.py
┃  ┃  ┣━ 🐍 test_shap_values.py
┃  ┃  ┗━ 🐍 test_shap_values_additional.py
┃  ┣━ 📁 modeling
┃  ┃  ┣━ 📁 optimization
┃  ┃  ┃  ┣━ 🐍 test_cutoff_tuning.py
┃  ┃  ┃  ┣━ 🐍 test_ensemble_weights_tuner.py
┃  ┃  ┃  ┣━ 🐍 test_ensemble_weights_tuner_cv.py
┃  ┃  ┃  ┗━ 🐍 test_hyperparameter_tuning.py
┃  ┃  ┗━ 📁 training
┃  ┃     ┣━ 🐍 test_base_trainer.py
┃  ┃     ┣━ 🐍 test_cross_validation.py
┃  ┃     ┣━ 🐍 test_ensemble_model.py
┃  ┃     ┗━ 🐍 test_oof_model.py
┃  ┗━ 🐍 conftest.py
┣━ 👻 .gitignore
┣━ ⚙️ pyproject.toml
┣━ 📖 README.md
┣━ 📃 requirements.txt
┗━ 🐚 setup_venv.sh
```

## Files Description
* 📁 `.github`: GitHub configuration files.
    - 📁 `workflows`: CI/CD workflow definitions.
        - 📜 `python-publish.yml`: Publishes the package to PyPI.
* 📁 `kvbiii_ml`: Main package source code.
    - 📁 `data_processing`: Folder with data processing tools.
        - 📁 `data_imputation`: Folder with scripts for data imputation.
            - 🐍 `joint_distribution_imputation.py`: Impute using joint distributions.
            - 🐍 `statistical_association_imputation.py`: Impute based on statistical associations.
        - 📁 `eda`: Folder with scripts for exploratory data analysis.
            - 🐍 `data_analysis.py`: Data profiling and summary.
            - 🐍 `data_cleaning.py`: Data cleaning utilities.
            - 🐍 `data_transformation.py`: Data transformation functions.
        - 📁 `feature_engineering`: Folder with scripts for feature engineering.
            - 🐍 `categorical_cleaner.py`: Clean categorical features.
            - 🐍 `categories_assigner.py`: Assign categories to features.
            - 🐍 `count_encoding.py`: Count encoding for categorical variables.
            - 🐍 `cross_encoding.py`: Cross-feature encoding.
			- 🐍 `digits_encoding.py`: Digits encoding for numerical variables.
			- 🐍 `dtypes_converter.py`: Convert data types.
			- 🐍 `feature_generation_pipeline.py`: Pipeline for generating new features.
			- 🐍 `numerical_downcaster.py`: Downcast numerical variables.
			- 🐍 `target_encoding.py`: Target encoding for categorical variables.
		- 📁 `feature_selection`: Folder with scripts for feature selection.
			- 🐍 `model_importance_filtering.py`: Filter features using model-based importances.
			- 🐍 `model_importance_rfe.py`: Recursive feature elimination using model importances.
			- 🐍 `mutual_information_filtering.py`: Filter features by mutual information.
			- 🐍 `mutual_information_rfe.py`: Recursive feature elimination with mutual information scores.
			- 🐍 `permutation_feature_importance.py`: Permutation-importance based feature filtering.
			- 🐍 `shap_rfe.py`: SHAP-driven recursive feature elimination.
		- 📁 `sampling`: Sampling strategy utilities.
			- 🐍 `samplers_comparision.py`: Compare sampling strategies.
	- 📁 `evaluation`: Model evaluation utilities.
		- 🐍 `custom_metrics_handler.py`: Register and compute custom metrics.
		- 🐍 `error_diagnostics.py`: Error analysis and diagnostics.
		- 🐍 `generate_reports.py`: Generate evaluation reports.
		- 🐍 `metrics.py`: Metric calculation utilities.
		- 🐍 `shap_values.py`: SHAP value computation helpers.
	- 📁 `modeling`: Model training and optimization tools.
		- 📁 `optimization`: Optimization and tuning utilities.
			- 🐍 `cutoff_tuning.py`: Optimize classification cutoff thresholds.
			- 🐍 `ensemble_weights_tuner.py`: Tune ensemble weights.
			- 🐍 `hyperparameter_tuning.py`: Hyperparameter search utilities.
		- 📁 `training`: Model training workflows.
			- 🐍 `base_trainer.py`: Base trainer implementation.
			- 🐍 `cross_validation.py`: Cross-validation utilities.
			- 🐍 `ensemble_model.py`: Ensemble model training logic.
			- 🐍 `oof_model.py`: Out-of-fold training and predictions.
* 📁 `tests`: Test suite for the package.
	- 📁 `data_processing`: Tests for data processing modules.
		- 📁 `data_imputation`: Tests for data imputation utilities.
			- 🐍 `test_joint_distribution_imputation.py`: Tests joint distribution imputation.
			- 🐍 `test_statistical_association_imputation.py`: Tests statistical association imputation.
		- 📁 `eda`: Tests for exploratory data analysis utilities.
			- 🐍 `test_data_analysis.py`: Tests data analysis helpers.
			- 🐍 `test_data_cleaning.py`: Tests data cleaning utilities.
			- 🐍 `test_data_transformation.py`: Tests data transformation functions.
		- 📁 `feature_engineering`: Tests for feature engineering components.
			- 🐍 `test_categorical_aligner.py`: Tests categorical alignment.
			- 🐍 `test_categorical_cleaner.py`: Tests categorical cleaning.
			- 🐍 `test_count_encoding.py`: Tests count encoding.
			- 🐍 `test_cross_encoding.py`: Tests cross-feature encoding.
			- 🐍 `test_cross_encoding_extra.py`: Additional cross-encoding tests.
			- 🐍 `test_digits_encoding.py`: Tests digits encoding.
			- 🐍 `test_feature_generation_pipeline.py`: Tests feature generation pipeline.
			- 🐍 `test_target_encoding.py`: Tests target encoding.
		- 📁 `feature_selection`: Tests for feature selection methods.
			- 🐍 `test_mutual_information_feature_selection_empty.py`: Tests MI selection on empty inputs.
			- 🐍 `test_mutual_information_filtering.py`: Tests mutual information filtering.
			- 🐍 `test_mutual_information_rfe.py`: Tests mutual information RFE.
			- 🐍 `test_shap_rfe.py`: Tests SHAP-based RFE.
		- 📁 `sampling`: Tests for sampling utilities.
			- 🐍 `test_samplers_comparision.py`: Tests sampling comparisons.
	- 📁 `evaluation`: Tests for evaluation utilities.
		- 🐍 `test_error_diagnostics.py`: Tests error diagnostics.
		- 🐍 `test_generate_reports.py`: Tests report generation.
		- 🐍 `test_metrics.py`: Tests metrics calculations.
		- 🐍 `test_shap_values.py`: Tests SHAP value helpers.
		- 🐍 `test_shap_values_additional.py`: Additional SHAP value tests.
	- 📁 `modeling`: Tests for modeling modules.
		- 📁 `optimization`: Tests for optimization utilities.
			- 🐍 `test_cutoff_tuning.py`: Tests cutoff tuning.
			- 🐍 `test_ensemble_weights_tuner.py`: Tests ensemble weights tuning.
			- 🐍 `test_ensemble_weights_tuner_cv.py`: Tests ensemble weights tuning with CV.
			- 🐍 `test_hyperparameter_tuning.py`: Tests hyperparameter tuning.
		- 📁 `training`: Tests for training workflows.
			- 🐍 `test_base_trainer.py`: Tests base trainer behavior.
			- 🐍 `test_cross_validation.py`: Tests cross-validation utilities.
			- 🐍 `test_ensemble_model.py`: Tests ensemble model training.
			- 🐍 `test_oof_model.py`: Tests out-of-fold training.
	- 🐍 `conftest.py`: Pytest fixtures and shared test setup.
* 📄 `.env`: Environment variable definitions.
* 👻 `.gitignore`: Files and folders to ignore in git.
* 🐚 `install_package.sh`: Shell script for installation.
* ⚙️ `pyproject.toml`: Project metadata and build configuration.
* 📖 `README.md`: Project documentation.
* 📃 `requirements.txt`: Python dependencies list.

## Installation
To install the repository, follow these steps:
1. Clone the repository:
```bash
git clone https://github.com/kvbiii/kvbiii-ml.git
```

2. Navigate to the repository directory:
```bash
cd kvbiii-ml
```

3. Create a virtual environment (optional but recommended):
```bash
bash setup_venv.sh
```

4. Activate the virtual environment:
```bash
source kvbiii-ml_venv/bin/activate
```

## Usage
## Usage
You can use `kvbiii-ml` to build and evaluate machine learning pipelines. Here are some example usage ideas:
### Example: Data Analysis
```python
from kvbiii_ml.data_processing.eda.data_analysis import DataAnalyzer
data_analyzer = DataAnalyzer()
data_analyzer.base_information(train_data)
```

> **Output:**
>
> | Feature                   | dtypes  | Missing Values | % Missing | Unique Values | Count   |
> |---------------------------|---------|---------------|-----------|---------------|---------|
> | RhythmScore               | float64 | 0             | 0.00%     | 322,528       | 524,164 |
> | AudioLoudness             | float64 | 0             | 0.00%     | 310,411       | 524,164 |
> | VocalContent              | float64 | 0             | 0.00%     | 229,305       | 524,164 |
> | AcousticQuality           | float64 | 0             | 0.00%     | 270,478       | 524,164 |
> | InstrumentalScore         | float64 | 0             | 0.00%     | 218,979       | 524,164 |
> | LivePerformanceLikelihood | float64 | 0             | 0.00%     | 279,591       | 524,164 |
> | MoodScore                 | float64 | 0             | 0.00%     | 306,504       | 524,164 |
> | TrackDurationMs           | float64 | 0             | 0.00%     | 377,442       | 524,164 |
> | Energy                    | float64 | 0             | 0.00%     | 11,606        | 524,164 |
> | BeatsPerMinute            | float64 | 0             | 0.00%     | 14,622        | 524,164 |

---

### Example: Data Memory Reduction

```python
from kvbiii_ml.data_processing.eda.data_analysis import DataAnalyzer
data_analyzer = DataAnalyzer()
from kvbiii_ml.data_processing.eda.data_transformation import DataTransformer
data_transformer = DataTransformer()

categorical_features = data_analyzer.get_categorical_features(train_data, unique_threshold=100)
independent_features = list(set(train_data.columns) - {target_feature})
print("Train data memory optimization:")
train_data = data_transformer.optimize_memory(train_data, categorical_features=categorical_features, verbose=True)
print("\nTest data memory optimization:")
test_data = data_transformer.optimize_memory(test_data, categorical_features=categorical_features, verbose=True)
if target_feature in categorical_features:
    categorical_features.remove(target_feature)
    train_data[target_feature] = train_data[target_feature].cat.codes
```

> **Output:**
>
> ```
> Train data memory optimization:
> Numerical dtypes reduced: 2877.32 MB → 1883.55 MB (34.5% reduction)
> Categorical dtypes converted: 1883.55 MB → 1883.55 MB (0.0% reduction)
> Total memory usage reduced: 2877.32 MB → 1883.55 MB (34.5% reduction)
>
> Test data memory optimization:
> Numerical dtypes reduced: 1035.76 MB → 705.17 MB (31.9% reduction)
> Categorical dtypes converted: 705.17 MB → 705.17 MB (0.0% reduction)
> Total memory usage reduced: 1035.76 MB → 705.17 MB (31.9% reduction)
> ```

---

### Example: Cross-Validation Training
```python
from lightgbm import LGBMRegressor
from kvbiii_ml.modeling.training.cross_validation import CrossValidationTrainer

SEED = 17
PROBLEM_TYPE = "regression"
METRIC_NAME = "RMSE"
CV = KFold(n_splits=5, shuffle=True, random_state=SEED)

cross_validation_trainer = CrossValidationTrainer(metric_name = METRIC_NAME, problem_type=PROBLEM_TYPE, cv = CV, verbose=False)

model = LGBMRegressor(boosting_type='gbdt', random_state=SEED, n_jobs=-1, objective='regression', metric='rmse', verbose=-1, early_stopping_rounds=100)
train_scores, valid_scores, _ = cross_validation_trainer.fit(estimator=model, X=train_data[independent_features], y=train_data[target_feature])
print(f"Train {METRIC_NAME}: {np.mean(train_scores):.4f} +- {np.std(train_scores):.4f}")
print(f"Validation {METRIC_NAME}: {np.mean(valid_scores):.4f} +- {np.std(valid_scores):.4f}")
```

> **Output:**
>
> ```
> Train RMSE: 26.4147 +- 0.0186
> Validation RMSE: 26.4594 +- 0.0454
> ```

-------------------------------------------
**Last updated on 2026-04-29 18:40:26**

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
┃  ┃  ┃  ┣━ 🐍 categorical_cleaner.py
┃  ┃  ┃  ┣━ 🐍 categories_assigner.py
┃  ┃  ┃  ┣━ 🐍 count_encoding.py
┃  ┃  ┃  ┣━ 🐍 cross_encoding.py
┃  ┃  ┃  ┣━ 🐍 example_feature_generation_pipeline.py
┃  ┃  ┃  ┣━ 🐍 feature_generation_pipeline.py
┃  ┃  ┃  ┣━ 🐍 target_encoding old.py
┃  ┃  ┃  ┗━ 🐍 target_encoding.py
┃  ┃  ┣━ 📁 feature_selection
┃  ┃  ┃  ┣━ 🐍 mutual_information_feature_selection.py
┃  ┃  ┃  ┣━ 🐍 mutual_information_filtering.py
┃  ┃  ┃  ┣━ 🐍 mutual_information_rfe.py
┃  ┃  ┃  ┗━ 🐍 shap_rfe.py
┃  ┃  ┗━ 📁 sampling
┃  ┃     ┗━ 🐍 samplers_comparision.py
┃  ┣━ 📁 evaluation
┃  ┃  ┣━ 🐍 error_diagnostics.py
┃  ┃  ┣━ 🐍 generate_reports.py
┃  ┃  ┣━ 🐍 metrics.py
┃  ┃  ┗━ 🐍 shap_values.py
┃  ┗━ 📁 modeling
┃     ┣━ 📁 optimization
┃     ┃  ┣━ 🐍 cutoff_tuning.py
┃     ┃  ┣━ 🐍 ensemble_weights_tuner.py
┃     ┃  ┣━ 🐍 ensemble_weights_tuner_old.py
┃     ┃  ┗━ 🐍 hyperparameter_tuning.py
┃     ┗━ 📁 training
┃        ┣━ 🐍 base_trainer.py
┃        ┣━ 🐍 cross_validation.py
┃        ┣━ 🐍 ensemble_model.py
┃        ┗━ 🐍 oof_model.py
┣━ 📁 tests
┃  ┣━ 📁 data_processing
┃  ┃  ┣━ 📁 data_imputation
┃  ┃  ┃  ┗━ 🐍 test_joint_distribution_imputation.py
┃  ┃  ┣━ 📁 eda
┃  ┃  ┃  ┣━ 🐍 test_data_analysis.py
┃  ┃  ┃  ┣━ 🐍 test_data_cleaning.py
┃  ┃  ┃  ┗━ 🐍 test_data_transformation.py
┃  ┃  ┣━ 📁 feature_engineering
┃  ┃  ┃  ┣━ 🐍 test_categories_assigner.py
┃  ┃  ┃  ┗━ 🐍 test_target_encoding.py
┃  ┃  ┣━ 📁 feature_selection
┃  ┃  ┃  ┗━ 🐍 test_mutual_information_filtering.py
┃  ┃  ┗━ 📁 sampling
┃  ┃     ┗━ ...
┃  ┣━ 📁 evaluation
┃  ┃  ┣━ 🐍 test_error_diagnostics.py
┃  ┃  ┗━ 🐍 test_metrics.py
┃  ┣━ 📁 modeling
┃  ┃  ┣━ 📁 optimization
┃  ┃  ┃  ┗━ ...
┃  ┃  ┗━ 📁 training
┃  ┃     ┣━ 🐍 test_base_trainer.py
┃  ┃     ┣━ 🐍 test_cross_validation.py
┃  ┃     ┣━ 🐍 test_ensemble_model.py
┃  ┃     ┗━ 🐍 test_oof_model.py
┃  ┗━ 🐍 conftest.py
┣━ 📄 .env
┣━ 👻 .gitignore
┣━ 🐚 install_package.sh
┣━ ⚙️ pyproject.toml
┣━ 📖 README.md
┗━ 📃 requirements.txt
```

## Files Description
* 📁 `.github`: GitHub configuration files.
    - 📁 `workflows`: CI/CD workflow definitions.
        - 📜 `python-publish.yml`: Publishes the package to PyPI.
* 📁 `kvbiii_ml`: Main package source code.
    - 📁 `data_processing`: Data preparation modules.
        - 📁 `data_imputation`: Missing value handling.
            - 🐍 `joint_distribution_imputation.py`: Impute using joint distributions.
            - 🐍 `statistical_association_imputation.py`: Impute based on statistical associations.
        - 📁 `eda`: Exploratory data analysis tools.
            - 🐍 `data_analysis.py`: Data profiling and summary.
            - 🐍 `data_cleaning.py`: Data cleaning utilities.
            - 🐍 `data_transformation.py`: Data transformation functions.
        - 📁 `feature_engineering`: Feature creation and encoding.
            - 🐍 `categorical_cleaner.py`: Clean categorical features.
            - 🐍 `categories_assigner.py`: Assign categories to features.
            - 🐍 `count_encoding.py`: Count encoding for categorical variables.
            - 🐍 `cross_encoding.py`: Cross-feature encoding.
            - 🐍 `example_feature_generation_pipeline.py`: Example feature pipeline.
            - 🐍 `feature_generation_pipeline.py`: Feature generation pipeline.
            - 🐍 `target_encoding old.py`: Legacy target encoding.
            - 🐍 `target_encoding.py`: Target encoding implementation.
        - 📁 `feature_selection`: Feature selection algorithms.
            - 🐍 `mutual_information_feature_selection.py`: Select features by mutual information.
            - 🐍 `mutual_information_filtering.py`: Filter features by mutual information.
            - 🐍 `mutual_information_rfe.py`: Recursive feature elimination using mutual information.
            - 🐍 `shap_rfe.py`: Feature selection using SHAP values.
        - 📁 `sampling`: Data sampling strategies.
            - 🐍 `samplers_comparision.py`: Compare different samplers.
    - 📁 `evaluation`: Model evaluation and diagnostics.
        - 🐍 `error_diagnostics.py`: Analyze model errors.
        - 🐍 `generate_reports.py`: Generate evaluation reports.
        - 🐍 `metrics.py`: Model metrics calculation.
        - 🐍 `shap_values.py`: SHAP value computation.
    - 📁 `modeling`: Model training and optimization.
        - 📁 `optimization`: Model optimization utilities.
            - 🐍 `cutoff_tuning.py`: Tune decision cutoffs.
            - 🐍 `ensemble_weights_tuner.py`: Tune ensemble weights.
            - 🐍 `ensemble_weights_tuner_old.py`: Legacy ensemble tuner.
            - 🐍 `hyperparameter_tuning.py`: Hyperparameter optimization.
        - 📁 `training`: Model training modules.
            - 🐍 `base_trainer.py`: Base trainer class.
            - 🐍 `cross_validation.py`: Cross-validation routines.
            - 🐍 `ensemble_model.py`: Ensemble model implementation.
            - 🐍 `oof_model.py`: Out-of-fold model training.
* 📁 `tests`: Unit and integration tests.
    - 📁 `data_processing`: Tests for data processing.
        - 📁 `data_imputation`: Imputation tests.
            - 🐍 `test_joint_distribution_imputation.py`: Test joint distribution imputation.
        - 📁 `eda`: EDA tests.
            - 🐍 `test_data_analysis.py`: Test data analysis.
            - 🐍 `test_data_cleaning.py`: Test data cleaning.
            - 🐍 `test_data_transformation.py`: Test data transformation.
        - 📁 `feature_engineering`: Feature engineering tests.
            - 🐍 `test_categories_assigner.py`: Test categories assigner.
            - 🐍 `test_target_encoding.py`: Test target encoding.
        - 📁 `feature_selection`: Feature selection tests.
            - 🐍 `test_mutual_information_filtering.py`: Test mutual information filtering.
        - 📁 `sampling`: Sampling tests.
    - 📁 `evaluation`: Evaluation tests.
        - 🐍 `test_error_diagnostics.py`: Test error diagnostics.
        - 🐍 `test_metrics.py`: Test metrics.
    - 📁 `modeling`: Modeling tests.
        - 📁 `optimization`: Optimization tests.
        - 📁 `training`: Training tests.
            - 🐍 `test_base_trainer.py`: Test base trainer.
            - 🐍 `test_cross_validation.py`: Test cross-validation.
            - 🐍 `test_ensemble_model.py`: Test ensemble model.
            - 🐍 `test_oof_model.py`: Test out-of-fold model.
    - 🐍 `conftest.py`: Pytest configuration.
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
git clone <repo_url>
```

2. Navigate to the repository directory:
```bash
cd kvbiii-ml
```

3. Create a virtual environment (optional but recommended):
```bash
python -m venv <venv_name>
```

4. Activate the virtual environment:
```bash
source <venv_name>/bin/activate
```

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
**Last updated on 2025-09-03 21:33:44**

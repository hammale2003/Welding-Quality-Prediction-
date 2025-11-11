# Semi-Supervised Learning for All Group 2 Targets

## Overview

This folder contains a comprehensive notebook that trains semi-supervised learning models for **all Group 2 target properties** (except Hardness, which is processed separately) in a single unified workflow.

## Target Properties

| Property | Column Name | Unit | Availability | Description |
|----------|-------------|------|--------------|-------------|
| FATT 50% | `FATT_50%` | °C | ~31 (1.9%) | Fracture Appearance Transition Temperature |
| Primary Ferrite | `Primary_Ferrite_%` | % | ~138 (8.4%) | Main microstructure phase |
| Ferrite 2nd Phase | `Ferrite_2nd_Phase_%` | % | ~100 (6.1%) | Secondary ferrite formation |
| Acicular Ferrite | `Acicular_Ferrite_%` | % | ~120 (7.3%) | Needle-like structure for optimal toughness |
| Martensite | `Martensite_%` | % | ~110 (6.7%) | Hard transformation phase |
| Ferrite Carbide | `Ferrite_Carbide_%` | % | ~105 (6.4%) | Carbide precipitation phase |

## Methodology

### Self-Training Approach
Same methodology as `Hardness_2nd_Approach`:

1. **Data Preprocessing**:
   - MinMaxScaler normalization (0-1 range)
   - KNN Imputation (k=5 neighbors, distance weighting)
   - No PCA - using original 52 features

2. **Custom Self-Training Regressor**:
   - Iterative pseudo-labeling (15% per iteration, max 10 iterations)
   - Confidence estimation via prediction variance
   - Compatible with GridSearchCV

3. **Custom Cross-Validation**:
   - Training folds: labeled (fold) + all unlabeled data
   - Validation folds: labeled (fold) only
   - Ensures proper semi-supervised evaluation

4. **Model Comparison**:
   - Random Forest (Supervised vs Semi-Supervised)
   - XGBoost (Supervised vs Semi-Supervised)
   - Best model selected based on R² score

## Notebook Structure

### Part 1: Setup & Data Exploration
- Load welddb_new.csv dataset
- Visualize data availability for all targets
- Define Group 2 target properties

### Part 2: Custom Classes
- `SelfTrainingRegressor`: Sklearn-compatible self-training wrapper
- `CustomLabeledUnlabeledKFold`: CV splitter for semi-supervised learning

### Part 3: Sequential Training
- Train 4 models (2 supervised + 2 semi-supervised) for each target
- Hyperparameter optimization with GridSearchCV
- Performance evaluation (R², Adj. R², RMSE, MAE)

### Part 4: Results & Analysis
- Overall performance summary across all targets
- Best model selection per target
- Distribution comparison (original vs predicted)
- Complete dataset generation

## Generated Files

### Models (6 files)
- `models/best_fatt_50_model.pkl`
- `models/best_primary_ferrite__model.pkl`
- `models/best_ferrite_2nd_phase__model.pkl`
- `models/best_acicular_ferrite__model.pkl`
- `models/best_martensite__model.pkl`
- `models/best_ferrite_carbide__model.pkl`

### Data Files
- `data/group2_all_targets_complete.csv` - Complete dataset with all predictions
- `data/best_models_summary.csv` - Best model performance for each target
- `data/all_models_comparison.csv` - All 24 models (4 per target × 6 targets)
- `data/final_statistics_summary.csv` - Distribution statistics comparison
- `data/*_comparison.csv` - Individual target comparison (6 files)

### Visualizations
- `figures/group2_data_availability.png` - Data availability by target
- `figures/overall_performance_summary.png` - R², RMSE, model type distribution
- `figures/distribution_comparison_all_targets.png` - Original vs predicted distributions

## Key Features

✅ **Unified Workflow**: Train all 6 targets in one notebook execution  
✅ **Consistent Methodology**: Same approach as Hardness_2nd_Approach  
✅ **Comprehensive Evaluation**: 4 models × 6 targets = 24 model comparisons  
✅ **Automatic Selection**: Best model chosen based on R² score  
✅ **Complete Dataset**: All missing values predicted and saved  
✅ **Rich Visualizations**: Distribution analysis and performance metrics  

## Usage

1. **Run the notebook**: Execute all cells sequentially
2. **Review results**: Check `data/best_models_summary.csv` for overview
3. **Load complete data**: Use `data/group2_all_targets_complete.csv`
4. **Deploy models**: Load models from `models/` directory

## Expected Results

- **High data availability targets** (Primary Ferrite ~8.4%): R² > 0.80
- **Medium data availability** (Microstructure phases ~6-7%): R² 0.70-0.85
- **Low data availability** (FATT ~1.9%): R² 0.50-0.70 (challenging)

## Advantages

1. **Data Efficiency**: Leverages ~1,500 unlabeled samples per target
2. **Scalability**: Single notebook for all targets
3. **Reproducibility**: Fixed random states (random_state=42)
4. **Transparency**: Detailed logging and metrics at each step
5. **Flexibility**: Easy to add new targets or modify hyperparameters

## Limitations

1. **FATT Challenge**: Extremely sparse data (31 samples) may limit accuracy
2. **Training Time**: Sequential training of 24 models takes ~30-60 minutes
3. **Pseudo-label Risk**: Incorrect pseudo-labels can degrade performance
4. **Memory Usage**: Multiple models stored in memory during execution



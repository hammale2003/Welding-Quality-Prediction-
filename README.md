# Welding Quality Prediction Using Machine Learning

## Project Overview

This project applies advanced machine learning methods to predict critical mechanical properties and microstructural characteristics of welded joints, enabling comprehensive weld quality assessment. The methodology addresses the challenge of sparse labeled data through supervised and semi-supervised learning approaches.

## Dataset

- **Total Samples**: 1,654 welded joints
- **Input Features**: 52 variables
  - Chemical composition (C, Si, Mn, P, S, Cr, Mo, Ni, etc.)
  - Welding parameters (Heat Input, Interpass Temperature, PWHT, etc.)
  - Process variables (Electrode Type, Polarity, Weld Type, etc.)
- **Target Properties**: 13 mechanical and microstructural properties
  - Group 1 (16-30% data availability): Supervised learning
  - Group 2 (2-8% data availability): Semi-supervised learning

---

## Methodology

### Two-Group Approach Rationale

The 13 target properties are divided into two groups based on **data availability**, requiring fundamentally different machine learning strategies:

| Group | Properties | Availability | Approach |
|-------|-----------|--------------|----------|
| **Group 1** | Yield Strength, UTS, Elongation, Reduction Area, Charpy Energy, Charpy Temperature | 16-30% | **Supervised Learning + PCA** |
| **Group 2** | Hardness, FATT 50%, Primary Ferrite, Ferrite 2nd Phase, Acicular Ferrite, Martensite, Ferrite Carbide | 2-8% | **Semi-Supervised Learning** |

---

## Group 1 Methodology: Supervised Learning with Dimensionality Reduction

### Objective
Predict mechanical properties with **abundant labeled data** (16-30% availability, ~264-500 samples per property).

### Why PCA for Group 1?

With 52 input features and sufficient labeled data:
- **Curse of dimensionality**: High-dimensional feature space can lead to overfitting
- **Multicollinearity**: Chemical composition features are highly correlated
- **Computational efficiency**: Reduced feature space accelerates training
- **Variance retention**: PCA preserves 95% of original variance while reducing dimensions

### Pipeline

#### Stage 1: PCA Analysis (`1_PCA_Analysis.ipynb`)

1. **Data Preprocessing**:
   - StandardScaler: Zero mean, unit variance normalization
   - KNNImputer: Missing value imputation (k=5 neighbors, distance weighting)

2. **Principal Component Analysis**:
   - Compute covariance matrix of standardized features
   - Extract eigenvectors and eigenvalues
   - Select components explaining ≥95% cumulative variance
   - Typical reduction: 52 features → 15-25 components

3. **Outputs**:
   - Transformed dataset: `welddb_pca_[property].csv`
   - PCA model: `pca_model/pca_transformer.pkl`
   - Scaler: `pca_model/scaler.pkl`
   - Explained variance plot

#### Stage 2: Model Training (`2_Model_Training.ipynb`)

1. **Models Evaluated** (9 algorithms):
   - **Linear**: Ridge, Lasso, ElasticNet
   - **Tree-based**: Decision Tree, Random Forest, Gradient Boosting, XGBoost, LightGBM
   - **Kernel**: Support Vector Regression (RBF kernel)

2. **Hyperparameter Optimization**:
   - GridSearchCV with 5-fold cross-validation
   - Scoring metric: R² (coefficient of determination)
   - Parallel processing (n_jobs=-1)

3. **Evaluation Metrics**:
   - **R²**: Proportion of variance explained
   - **Adjusted R²**: R² penalized for feature count
   - **RMSE**: Root Mean Squared Error
   - **MAE**: Mean Absolute Error

4. **Outputs**:
   - Best model: `trained_models/best_[property]_model.pkl`
   - Performance comparison: `trained_models/model_comparison.csv`
   - Prediction visualizations

### Results (Group 1)

| Property | Best Model | R² Score | RMSE | Status |
|----------|-----------|----------|------|--------|
| Yield Strength | XGBoost | ~0.92 | ~45 MPa | ✓ Excellent |
| UTS | Random Forest | ~0.90 | ~52 MPa | ✓ Excellent |
| Elongation | Gradient Boosting | ~0.88 | ~3.2% | ✓ Good |
| Reduction Area | XGBoost | ~0.86 | ~4.5% | ✓ Good |
| Charpy Energy | Random Forest | ~0.84 | ~18 J | ✓ Good |
| Charpy Temperature | LightGBM | ~0.82 | ~12°C | ✓ Good |

---

## Group 2 Methodology: Semi-Supervised Learning

### Objective
Predict properties with **extremely sparse labeled data** (2-8% availability, only 31-138 samples per property).

### Why Semi-Supervised Learning?

Traditional supervised learning fails with sparse data:
- **High variance**: Unreliable estimates with <10% labeled samples
- **Overfitting**: 52 features overwhelm limited training data
- **Poor generalization**: Cannot capture complex patterns

**Solution**: Leverage ~1,500 unlabeled samples through self-training.

### Why NO PCA for Group 2?

Unlike Group 1, we **do NOT apply PCA** because:
1. **Insufficient samples**: Cannot reliably estimate 52×52 covariance matrix with 31-138 samples
2. **Information preservation**: With limited labeled data, we cannot afford to discard any variance
3. **Implicit regularization**: Self-training using unlabeled data provides regularization
4. **Physical interpretability**: Original features maintain metallurgical meaning

### Core Approach: Self-Training Framework

#### Algorithm Overview

```
1. Train base model on labeled data L
2. Predict unlabeled samples U with confidence estimation
3. Select top 15% most confident predictions as pseudo-labels
4. Add pseudo-labels to training set: L' = L ∪ pseudo-labels
5. Retrain model on augmented set L'
6. Repeat for max 10 iterations
```

#### Confidence Estimation via Prediction Variance

**Random Forest**:
- Variance across ensemble trees → confidence score
- High variance = low confidence (uncertain prediction)
- Low variance = high confidence (reliable pseudo-label)

**Formula**: `confidence = 1 / (1 + prediction_variance)`

#### Custom Components

1. **SelfTrainingRegressor**:
   - Sklearn-compatible wrapper for any base regressor
   - Handles NaN targets (indicating unlabeled samples)
   - Supports GridSearchCV hyperparameter optimization
   - Logs iteration metrics for transparency

2. **CustomLabeledUnlabeledKFold**:
   - K-Fold only on labeled data
   - Training folds: labeled (fold) + ALL unlabeled samples
   - Validation folds: labeled (fold) ONLY
   - Ensures proper semi-supervised evaluation

### Preprocessing for Group 2

1. **Feature Selection**:
   - Exclude other Group 2 properties (prevent data leakage)
   - Exclude Group 1 properties (already predicted)
   - Retain all 52 original features

2. **Normalization**: MinMaxScaler (0-1 range)
   - Bounded range prevents outlier dominance
   - Compatible with distance-based imputation

3. **Imputation**: KNNImputer (k=5, distance weighting)
   - Preserves local feature space structure

### Training Pipeline

#### For Each Target Property:

1. **Baseline (Supervised)**:
   - Random Forest with GridSearchCV
   - XGBoost with GridSearchCV
   - Train only on labeled data

2. **Semi-Supervised**:
   - Random Forest + SelfTraining
   - XGBoost + SelfTraining
   - Train on labeled + unlabeled (with pseudo-labels)

3. **Comparison**:
   - Evaluate all 4 models on test set
   - Select best based on R² score
   - Typical improvement: +5% to +20% in R²

### Two Implementations

#### Hardness_2nd_Approach (Single Target)
- Comprehensive notebook for Hardness prediction
- Detailed iteration logging and visualization
- Distribution comparison (original vs predicted)

#### Group2_All_Targets (Unified Workflow)
- Sequential training for all 6 targets (excluding Hardness)
- Processes: FATT 50%, Primary Ferrite, Ferrite 2nd Phase, Acicular Ferrite, Martensite, Ferrite Carbide
- Generates 24 models (4 per target)
- Comprehensive performance summary across all targets

### Results (Group 2)

| Property | Labeled Samples | Best Model | R² Score | Improvement* |
|----------|----------------|-----------|----------|--------------|
| Hardness | 138 (8.4%) | RF Semi-Supervised | ~0.85 | +12% |
| Primary Ferrite | 138 (8.4%) | XGB Semi-Supervised | ~0.82 | +15% |
| Acicular Ferrite | 120 (7.3%) | RF Semi-Supervised | ~0.78 | +18% |
| Martensite | 110 (6.7%) | XGB Semi-Supervised | ~0.76 | +14% |
| Ferrite Carbide | 105 (6.4%) | RF Semi-Supervised | ~0.74 | +16% |
| Ferrite 2nd Phase | 100 (6.1%) | XGB Semi-Supervised | ~0.72 | +10% |
| FATT 50% | 31 (1.9%) | RF Semi-Supervised | ~0.58 | +8% |

*Improvement over supervised baseline

---

## Weld Quality Index (WQI) Calculation

### Physical Basis

Weld quality is not a single property but a **balance between multiple mechanical characteristics**:
- **Strength**: Ability to withstand stress (Yield Strength, UTS)
- **Ductility**: Ability to deform without fracture (Elongation, Reduction Area)
- **Toughness**: Energy absorption capacity (Charpy Energy)
- **Low-temperature performance**: Fracture behavior at cold temperatures (Charpy Temperature, FATT)
- **Hardness**: Resistance to deformation and cracking susceptibility

### Metallurgical Considerations

The fusion zone and heat-affected zone (HAZ) undergo complex phase transformations:
- **Rapid cooling** → **Martensite** (hard, strong, brittle)
- **Slow cooling** → **Acicular Ferrite** (ductile, tough)

**Trade-off**:
- Excessive martensite → high hardness, cracking risk
- Excessive ferrite → low strength

### WQI Formula (Proposed)

After extensive literature review and physical analysis, we propose:

```
WQI = α₁ × (Normalized_Strength) + 
      α₂ × (Normalized_Ductility) + 
      α₃ × (Normalized_Toughness) + 
      α₄ × (Normalized_Hardness_Factor) + 
      α₅ × (Normalized_Microstructure_Balance)
```

Where:
- **Normalized_Strength** = f(Yield_Strength, UTS)
- **Normalized_Ductility** = f(Elongation, Reduction_Area)
- **Normalized_Toughness** = f(Charpy_Energy, Charpy_Temperature)
- **Normalized_Hardness_Factor** = f(Hardness, FATT)
- **Normalized_Microstructure_Balance** = f(Primary_Ferrite, Acicular_Ferrite, Martensite, Ferrite_Carbide)

Weights (α₁, α₂, α₃, α₄, α₅) reflect the relative importance of each component based on application requirements.

### Physical Interpretation

1. **Strength Component**: 
   - High strength enables load-bearing capacity
   - Determined by dislocation density and hard phase content

2. **Ductility Component**:
   - Prevents brittle fracture
   - Reflects metal's plastic deformation capability

3. **Toughness Component**:
   - Energy absorption before failure
   - Critical for impact resistance and low-temperature applications

4. **Hardness Factor**:
   - Moderate hardness is desirable
   - Too high → cracking risk; Too low → wear resistance issues

5. **Microstructure Balance**:
   - Optimal mix of phases (acicular ferrite dominant)
   - Martensite controlled to acceptable levels
   - Phase proportions sum to ~100%

### Implementation Strategy

1. **Predict all 13 properties** using Group 1 & Group 2 methodologies
2. **Normalize** each predicted property to [0,1] scale
3. **Apply WQI formula** with application-specific weights
4. **Validate** against known good/bad welds
5. **Optimize weights** using validation dataset

---

## Project Structure

```
Welding-Quality-Prediction-Project/
├── welddatabase/
│   ├── welddb.csv                    # Original dataset
│   └── welddb_new.csv                # Processed dataset
│
├── Group_1_Supervised_Learning/
│   ├── Yield_Strength_UTS/
│   │   ├── 1_PCA_Analysis.ipynb
│   │   ├── 2_Model_Training.ipynb
│   │   ├── 3_UTS_Prediction.ipynb
│   │   ├── data/                     # PCA-transformed data
│   │   ├── pca_model/                # PCA transformer & scaler
│   │   ├── trained_models/           # Best models & comparisons
│   │   └── figures/                  # Visualizations
│   ├── Elongation/
│   ├── Reduction_Area/
│   ├── Charpy_Energy/
│   └── Charpy_Temperature/
│
├── Group_2_Semi_Supervised_Learning/
│   ├── Hardness/                     # Original PCA-based approach
│   │   ├── 1_PCA_Analysis.ipynb
│   │   ├── 2_Model_Training_Reduced_Data.ipynb
│   │   └── Semi_Supervised_Training.ipynb
│   │
│   ├── Hardness_2nd_Approach/        # Self-training approach
│   │   ├── Hardness_Semi_Supervised_Learning.ipynb
│   │   ├── README.md
│   │   ├── data/                     # Complete predictions
│   │   ├── models/                   # Best model
│   │   └── figures/                  # Distributions
│   │
│   ├── Group2_All_Targets/           # Unified workflow
│   │   ├── Group2_Targets_Semi_Supervised_Learning.ipynb
│   │   ├── README.md
│   │   ├── data/                     # All targets complete
│   │   ├── models/                   # 6 best models
│   │   └── figures/                  # Comprehensive visualizations
│   
│  
├         
└── README.md                         # This file
```

---

## Key Results & Achievements

### Group 1 (Supervised with PCA)
✓ **Excellent prediction accuracy**: R² > 0.80 for all properties  
✓ **Dimensionality reduction**: 52 → ~15-25 features (95% variance retained)  
✓ **Computational efficiency**: Faster training with reduced features  
✓ **Model comparison**: 9 algorithms evaluated per property  

### Group 2 (Semi-Supervised)
✓ **Overcame data sparsity**: Reliable predictions with only 2-8% labeled data  
✓ **Leveraged unlabeled samples**: ~1,500 samples per property  
✓ **Significant improvements**: +5% to +20% R² over supervised baseline  
✓ **Unified workflow**: All 6 targets in single notebook  
✓ **Physical validation**: Predictions consistent with metallurgical relationships  

### Overall Impact
✓ **Complete dataset**: All 13 properties predicted for 1,654 samples  
✓ **Weld quality assessment**: Foundation for WQI calculation  
✓ **Reproducible framework**: Documented methodology with LaTeX reports  
✓ **Extensible codebase**: Sklearn-compatible custom implementations  

---

## Technologies & Libraries

- **Python 3.x**
- **Core ML**: scikit-learn, XGBoost, LightGBM
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Model Persistence**: joblib
- **Documentation**: Jupyter Notebooks, LaTeX

---

## Usage

### For Group 1 Properties:

1. Run `1_PCA_Analysis.ipynb` to generate PCA-transformed dataset
2. Run `2_Model_Training.ipynb` to train and compare models
3. Best model automatically saved for deployment

### For Group 2 Properties:

**Option 1 - Single Target (Hardness)**:
- Run `Hardness_2nd_Approach/Hardness_Semi_Supervised_Learning.ipynb`

**Option 2 - All Targets**:
- Run `Group2_All_Targets/Group2_Targets_Semi_Supervised_Learning.ipynb`

### For WQI Calculation:

1. Load all predicted properties from saved models
2. Apply normalization to [0,1] scale
3. Compute WQI using weighted formula
4. Validate against known quality benchmarks

---

## Future Enhancements

- **Multi-task learning**: Jointly predict correlated properties
- **Deep learning**: Autoencoders for semi-supervised regression
- **Active learning**: Strategic sample selection for labeling
- **Transfer learning**: Adapt models across welding processes
- **Uncertainty quantification**: Conformal prediction intervals
- **WQI optimization**: Empirical validation and weight tuning


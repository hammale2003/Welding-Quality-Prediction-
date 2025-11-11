Semi-Supervised Learning for Hardness Prediction (2nd Approach)
================================================================

## Overview

This approach uses **semi-supervised learning with self-training** to predict the hardness of welding samples by leveraging unlabeled data.

## Methodology

### 1. Data
- **Labeled**: ~138 samples with hardness measurements (8.4%)
- **Unlabeled**: ~1514 samples without measurements (91.6%)
- **Features**: 52 variables (chemical composition, welding parameters, etc.)

### 2. Preprocessing
- **Normalization**: MinMaxScaler (0-1 range)
- **Imputation**: KNN Imputer (k=5 neighbors, distance weighting)
- No PCA reduction - using original features

### 3. Custom Self-Training
**Principle**: Iteratively expand the training set with confident pseudo-labels

**Process**:
1. Train the model on current labeled data
2. Predict unlabeled samples
3. Estimate confidence (tree variance for RF, staged prediction variance for XGBoost)
4. Select top 15% most confident predictions
5. Add these pseudo-labels to the training set
6. Repeat (max 10 iterations)

**Confidence formula**: `confidence = 1 / (1 + prediction_variance)`

### 4. Models Compared
- **Random Forest** (supervised vs semi-supervised)
- **XGBoost** (supervised vs semi-supervised)

GridSearchCV optimization with custom cross-validation:
- Training: labeled data (fold) + all unlabeled data
- Validation: labeled data (fold) only

### 5. Results
The best model is selected based on R² score and used to predict all missing hardness values.

## Advantages of this Approach

✅ **Exploits unlabeled data** without dimensionality reduction  
✅ **Progressive selection** of pseudo-labels based on confidence  
✅ **Rigorous validation** separating semi-supervised training and supervised evaluation  
✅ **Sklearn compatibility** enabling GridSearchCV usage  
✅ **Transparency** with detailed metrics at each iteration

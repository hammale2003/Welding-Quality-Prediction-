Semi-Supervised Learning with Conformal Prediction for Hardness Prediction
==========================================================================

Overview
--------
This notebook implements a semi-supervised learning (SSL) approach to predict the hardness of welded samples. 
The method leverages both labeled and unlabeled data to improve predictive performance when the labeled dataset is limited.

The key idea is to iteratively expand the labeled dataset by adding predictions on unlabeled samples that the model is confident about. 
Conformal prediction intervals are used to quantify uncertainty and guide the selection of pseudo-labeled samples.

Approach
--------

1. Dataset and Feature Processing
   - The initial labeled dataset contains ~138 samples with hardness measurements.
   - The unlabeled dataset contains samples without hardness measurements.
   - Features are PCA-reduced and missing values are imputed using a pre-trained pipeline.
   - This reduces dimensionality, mitigates multicollinearity, and ensures the model can generalize well from limited labeled data.

2. Semi-Supervised Iterative Learning
   The iterative process is as follows:
   1. Train a Gradient Boosting Regressor on the current labeled set.
   2. Split labeled data to create a conformalization set for uncertainty estimation.
   3. Use MAPIE (Model Agnostic Prediction Interval Estimator) to generate conformal prediction intervals for the unlabeled samples.
   4. Compute the relative interval width of predictions:
      - Only predictions with narrow intervals (high confidence) are selected as pseudo-labels.
   5. Add pseudo-labeled samples to the labeled dataset and remove them from the unlabeled pool.
   6. Repeat until either no high-confidence pseudo-labels remain or the maximum number of iterations is reached.

   This ensures the model exploits unlabeled data carefully while controlling for prediction uncertainty.

3. Final Model
   After the iterative pseudo-labeling, the final Gradient Boosting Regressor is trained on the augmented labeled dataset. 
   The trained model is saved for future predictions.

Why This Approach is Effective
------------------------------
- Small labeled dataset: Traditional supervised learning is prone to overfitting on ~138 samples. SSL allows leveraging the unlabeled pool to effectively increase the training set.
- Uncertainty-aware selection: Only high-confidence predictions (narrow conformal intervals) are added as pseudo-labels, reducing the risk of introducing noisy labels.
- Statistical guarantees: Conformal prediction intervals provide a formal confidence level (e.g., 90%) for each prediction, ensuring pseudo-labeling is principled rather than heuristic.
- Iterative refinement: The process progressively improves the labeled set, allowing the model to better generalize.

Visualizing the Dataset
-----------------------
The notebook includes a distribution plot comparing the original labeled hardness values with the augmented labeled dataset after pseudo-labeling:

- Original labeled data: Red
- Augmented labeled data: Orange

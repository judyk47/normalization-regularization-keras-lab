# Normalization & Regularization in Keras

A hands-on lab comparing three neural network setups for binary classification, using scikit-learn's Breast Cancer dataset as a stand-in for tabular risk data such as loan default prediction.

## Goal

Test how batch normalization, L2 regularization, and dropout affect training stability, overfitting, and accuracy compared with a plain baseline network.

## Workflow

1. Load the data and split it into training, validation, and test sets
2. Standardize features with `StandardScaler`, fit on the training data only to avoid leakage
3. Train a baseline model (Dense 32 → 16 → 1, ReLU and sigmoid)
4. Add batch normalization (Dense → BatchNorm → ReLU)
5. Add L2 regularization (1e-4) and dropout (0.3)
6. Evaluate the regularized model on the held-out test set and plot accuracy over epochs

## Models

| Model | Key technique |
|---|---|
| Baseline | Dense layers with ReLU |
| Batch norm | Linear Dense, then BatchNormalization, then ReLU |
| Regularized | L2 weight penalty + Dropout |

## Results

Baseline validation accuracy:   95.60%
Batch norm validation accuracy: 95.60%
Regularized validation accuracy: 96.70%
Regularized test accuracy:      95.61%

## Tech stack

Python, TensorFlow/Keras, scikit-learn, pandas, NumPy, Matplotlib, Seaborn


## Notes

The dataset is small (569 samples), so differences between the three models are within noise. The point of the lab is to practice the techniques and the evaluation workflow, not to claim one method wins.

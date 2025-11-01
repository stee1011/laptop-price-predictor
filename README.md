# 🧠 Robust Regression with Outlier Detection

> Advanced regression workflow combining **Isolation Forest** for anomaly removal and **Huber Regressor** for robust modeling.  
> Designed for high-variance, real-world data where classical OLS fails under outlier pressure.

---

## 📂 Project Overview

This project demonstrates a **robust regression pipeline** on a dataset with 658 samples and 18 features.  
The workflow focuses on improving model stability and variance capture by intelligently eliminating outliers and applying a resistant regression technique.

---

## ⚙️ Workflow Summary

1. **Data Preparation**
   - Loaded and cleaned dataset (`df`)
   - Encoded categorical features using `OneHotEncoder`
   - Scaled numerical features for consistency

2. **Outlier Elimination**
   - Used `IsolationForest` to detect and drop anomalies
   - Tuned `contamination` rate between 5–20%
   - Achieved optimal results at **15% contamination removal**

3. **Robust Regression**
   - Implemented `HuberRegressor` with tuned `epsilon` and `alpha`
   - Balanced L2 and L1 loss to resist remaining outliers

4. **Model Evaluation**
   - Compared pre- and post-outlier removal metrics:
     | Metric | Before | After |
     |:--|--:|--:|
     | **MAE** | 15,365.58 | **9,806.25** |
     | **MSE** | 627,852,443.90 | **209,623,197.67** |
     | **R² (Variance Capture)** | 0.678 | **0.805** |

---

## 🧩 Code Snippet

```python
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import HuberRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd

# === Outlier Removal ===
iso = IsolationForest(contamination=0.15, random_state=42)
mask = iso.fit_predict(X) != -1
X_clean, y_clean = X[mask], y[mask]

# === Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)

# === Robust Regression ===
model = HuberRegressor(epsilon=1.2, alpha=0.0001, max_iter=1000)
model.fit(X_train, y_train)

# === Evaluation ===
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"R²: {r2:.4f}")

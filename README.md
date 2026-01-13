# The Honest Imputer: Robustness and Calibration in Missing Data
**"Complexity is not a proxy for performance."**
This research demonstrates that for structured datasets—even small ones like the 150-row Iris set—the robustness of the modeling pipeline and proper calibration are far more critical than the complexity of the imputation algorithm.


## 🔬 Research Summary

How much "information tax" do we pay when data goes missing? This project analyzes the impact of **25% MAR (Missing At Random)** amputation on predictive performance. By benchmarking several architectures, we identify how different imputation processes affect model bias and uncertainty.

The findings prove that an optimized "simple" approach can nearly recover the performance of the original clean data by treating **"missingness" as a first-class feature** rather than just a hole to be filled.

---

## 📊 Final Performance Benchmark

| Strategy | Macro F1-Score | ROC-AUC (OvR) | Brier Score (Multiclass) | Status |
| --- | --- | --- | --- | --- |
| **Clean Data (Ground Truth)** | **0.93** | **0.9683** | **0.0403** | **Baseline** |
| **Simple Impute + Optuna + Calib** | **0.90** | **0.9450** | **0.0764** | **Winner** |
| **Simple Impute (No Optuna)** | 0.77 | - | - | Moderate |
| **Native LightGBM Handling** | 0.74 | - | - | Baseline |
| **MiceForest (Complex)** | **0.31** | **0.5200** | - | **Failure** |

---

## 🏛️ The Three Research Pillars

### 1. The Complexity Trap

This works proved that advanced tree-based imputers like `miceforest` can catastrophically fail on small, structured datasets. By "over-fitting" patterns in a low-sample environment, it dropped the F1-score from **0.93 to 0.31**.

### 2. The Signal of Absence

Using a custom transformer (`Flagnan`), we validated that creating missing indicators (**_NA flags**) acts as a "good equalizer"Missing Indicator (_NA flag),preventing information loss.
The model identified that the *absence* of `petal width (cm)` was highly predictive, assigning it a significant importance score (36), proving that **missingness itself is a signal**.

### 3. Honest Probabilities

Through **Isotonic Calibration** and the **Multiclass Brier Score**, we ensured the model understands its own uncertainty. Achieving a Brier Score of **0.0764** (compared to the 0.0403 clean baseline) proves that the model remained reliable despite the missing features.

---

## 🔬 Diagnostic Analysis

### Residual Stability

The **Residual Analysis** confirms that the "Honest" pipeline preserves the predictive signal without introducing new pathological errors.

* **Consistency:** Most "Imputed Residuals" closely overlap with "Clean Residuals."
* **Integrity:** The error magnitude did not explode, maintaining high model reliability despite 25% data loss.

### Probability Calibration

The Brier Score calculation proves that the probabilities assigned by the model are well-calibrated, meaning a 90% confidence actually corresponds to a 90% empirical frequency.

---

## 🏁 Key Conclusions

1. **Scale Matters:** On small datasets (150 rows), simple imputation + flags beats complex multivariate methods.
2. **Calibration is Mandatory:** The jump from 0.77 to 0.90 F1-score was driven by **Optuna tuning** and **Calibration**, not by the imputer itself.
3. **The "Honest" Strategy:** For production, always use **Missing Flags** and **Post-hoc Calibration** to ensure trustworthy decision-making.

---

### 🛠️ Tech Stack

* **Modeling:** lightGBM, scikit-Learn, feature-engine
* **Optimization:** Optuna
* **Calibration Curve:** Isotonic Regression
* **Metrics:** Multiclass Brier Score, ROC-AUC, Macro F1,classification report

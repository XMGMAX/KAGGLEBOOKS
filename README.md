# Kaggle Playground S6E3 — Customer Churn Prediction
> Target: **Top 50 leaderboard** | Metric: **ROC-AUC** | Final CV: ~0.922+

---

## Pipeline Architecture

```
Raw Data (train + original Telco)
        │
        ▼
Feature Engineering (35 features)
        │
        ▼
CV-Safe Target Encoding + Frequency Encoding
        │
        ▼
Optuna HPO (50 trials — LGB + XGB)
        │
        ▼
┌───────────────────────────────────────┐
│         Level-1 Base Models           │
│  LGB · XGB · CB · ET · RF · MLP      │
│  10-fold StratifiedKFold × 5 seeds   │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│         Level-2 Meta-Models           │
│  LogisticRegressionCV + LGB + XGB    │
│  Input: logit(OOF) + interactions    │
└───────────────────────────────────────┘
        │
        ▼
Nelder-Mead Optimal Blending
70% Meta blend + 30% Rank Average
        │
        ▼
2-Round Pseudo-Labeling (0.85/0.90 thresholds)
        │
        ▼
Final Submission
```

---

## Feature Engineering

### Charge-derived
| Feature | Formula |
|---|---|
| `AvgMonthlyCharge` | `TotalCharges / (tenure + 1e-5)` |
| `ChargeDiff` | `MonthlyCharges - AvgMonthlyCharge` |
| `ChargeRatio` | `MonthlyCharges / (TotalCharges + 1)` |
| `ChargePerYear` | `TotalCharges / (tenure//12 + 1)` |
| `TenureXMonthly` | `tenure × MonthlyCharges` |
| `TotalChargesLog` | `log1p(TotalCharges)` |

### Risk flags (binary interactions)
| Feature | Condition |
|---|---|
| `HighRisk` | `Month-to-month AND HasInternet AND tenure ≤ 12` |
| `DigitalRisk` | `PaperlessBilling AND Electronic check` |
| `UnsupportedInternet` | `HasInternet AND no Security AND no TechSupport` |
| `HighValueAtRisk` | `MonthlyCharges > median AND Month-to-month` |

### Tenure bucketing
```python
pd.cut(tenure, bins=[0,12,24,48,72,999],
       labels=[0,1,2,3,4], include_lowest=True)
```

---

## Encoding Strategy

### CV-Safe Target Encoding
Standard target encoding leaks label info. This implementation folds:
```
For each fold k in StratifiedKFold(10):
    fold_map = mean(Churn) on folds [1..K-1, K+1..10]
    train[col_te][fold_k] = row.map(fold_map)
Test encoding = mean(Churn) fit on full train
```

### Frequency Encoding
Added alongside TE to capture category popularity signal independent of target:
```python
freq_map = train[col].value_counts(normalize=True)
```

---

## Hyperparameter Optimization

Optuna TPE sampler, 50 trials each, 5-fold inner CV:

### LightGBM search space
```
learning_rate    : log-uniform [0.01, 0.1]
num_leaves       : int [31, 127]
max_depth        : int [4, 8]
colsample_bytree : uniform [0.5, 1.0]
subsample        : uniform [0.5, 1.0]
min_child_samples: int [10, 50]
reg_alpha        : log-uniform [1e-4, 10]
reg_lambda       : log-uniform [1e-4, 10]
```

### XGBoost search space
```
learning_rate    : log-uniform [0.01, 0.1]
max_depth        : int [3, 8]
colsample_bytree : uniform [0.5, 1.0]
subsample        : uniform [0.5, 1.0]
min_child_weight : int [1, 20]
gamma            : uniform [0, 1.0]
reg_alpha        : log-uniform [1e-4, 10]
reg_lambda       : log-uniform [1e-4, 10]
```

---

## Base Model Configuration

| Model | Key settings | Seed avg |
|---|---|---|
| LightGBM | Optuna-tuned, `n_estimators=2000`, early stopping 100 | 5 seeds |
| XGBoost | Optuna-tuned, `tree_method=hist`, early stopping | 5 seeds |
| CatBoost | `depth=6`, `l2_leaf_reg=3`, `bagging_temperature=1` | 5 seeds |
| ExtraTrees | `max_features=0.6`, `min_samples_leaf=5` | 5 seeds |
| RandomForest | `max_features=0.5`, `min_samples_leaf=5` | 5 seeds |
| MLP | `(256,128,64)`, relu, `StandardScaler` pipeline | 5 seeds |

**Seed averaging** — each model trains 10-fold CV across 5 independent seeds. OOF and test predictions are mean-averaged across seeds before stacking. Reduces variance by ~30% vs single seed.

---

## Stacking Layer

### Meta-features
```
Direct:       logit(oof_lgb), logit(oof_xgb), logit(oof_cb),
              logit(oof_et),  logit(oof_rf),  logit(oof_mlp)

Interactions: lgb×xgb, lgb×cb, xgb×cb, lgb×mlp, et×mlp

Rank-norm:    lgb/max(lgb), xgb/max(xgb), cb/max(cb)
```

All base predictions passed through `logit(clip(x, 1e-6, 1-1e-6))` before meta-model. This maps probabilities to the real line, making the linear meta-model more effective.

### Meta-models
- `LogisticRegressionCV` — Cs=20, cv=5
- `LGBMClassifier` — shallow (num_leaves=15, lr=0.02)
- `XGBClassifier` — max_depth=3, lr=0.02

Final meta blend = equal 1/3 weight across all three.

### Final blend
```
final = 0.7 × meta_blend + 0.3 × rank_average
```

Rank averaging converts each model's predictions to percentile ranks before averaging — robust to probability miscalibration across different model families.

### Nelder-Mead optimal weights
```python
minimize(neg_auc, x0=[1/6]*6, method='Nelder-Mead')
# weights clipped to [0,1] and L1-normalized
```

---

## Pseudo-Labeling

Two-round iterative approach with tightening confidence:

```
Round 1: threshold = (pred > 0.85) OR (pred < 0.15)
         Retrain LGB + CB on train ∪ pseudo-labeled test
         Seeds: [42, 123, 2024]

Round 2: threshold = (R1_pred > 0.90) OR (R1_pred < 0.10)
         Retrain LGB + CB on train ∪ tighter pseudo-labeled test

Final blend:
    0.50 × original_final
  + 0.30 × pseudo_round1
  + 0.20 × pseudo_round2
```

Tighter threshold in Round 2 avoids compounding noise from borderline labels.

---

## Reproduction

```bash
# Environment
Python 3.12
lightgbm==4.x  xgboost==2.x  catboost==1.x
scikit-learn==1.x  optuna==3.x  scipy==1.x

# Data
competition : kaggle competitions download -c playground-series-s6e3
original    : kaggle datasets download -d blastchar/telco-customer-churn
```

```
notebooks/
├── cell_00_install.ipynb
├── cell_01_imports.ipynb
├── cell_02_load_data.ipynb
├── cell_03_feature_engineering.ipynb
├── cell_04_encoding.ipynb
├── cell_05_optuna_tuning.ipynb
├── cell_06_cv_runner.ipynb
├── cell_07_base_models.ipynb
├── cell_08_blend_weights.ipynb
├── cell_09_stacking.ipynb
├── cell_10_pseudo_labeling.ipynb
└── cell_11_submission.ipynb
```

---

## Results

| Stage | OOF AUC |
|---|---|
| LGB single seed baseline | ~0.916 |
| LGB + seed averaging | ~0.918 |
| All 6 base models | ~0.919 |
| + Stacking (3 metas) | ~0.921 |
| + Pseudo-labeling R1+R2 | ~0.922+ |

---

## Key Design Decisions

**Why logit-transform before meta-model?**
Base model probabilities cluster near 0 and 1. Logit maps these to an unbounded scale where linear separation is more effective. Without this, LR meta-model underperforms by ~0.001 AUC.

**Why rank averaging alongside stacking?**
Stacking can overfit the meta-training set. Rank averaging is a zero-parameter ensemble immune to probability scale differences between model families (GBDT vs MLP). The 70/30 blend hedges against meta-model overfitting.

**Why CV-safe target encoding matters here?**
Dataset is synthetic and small (~7k competition rows + ~7k original). Standard target encoding on the full train set causes ~0.003 AUC inflation in CV that does not transfer to the leaderboard.

**Why 5-seed averaging instead of more folds?**
At 10 folds, variance from fold randomness is already low. Additional seeds reduce inter-run variance without increasing bias. 5 seeds × 10 folds = 50 models per base learner — sufficient to stabilize OOF estimates.

**Why two pseudo-label rounds with tightening thresholds?**
Round 1 at 0.85/0.15 adds high-confidence signal. Using Round 1 predictions for Round 2 at 0.90/0.10 avoids amplifying noise from borderline labels that were pseudo-labeled in Round 1. Decaying blend weights (0.30 → 0.20) reflect decreasing reliability per round.

---

*Competition closes March 31, 2026 · Kaggle Playground Series S6E3*

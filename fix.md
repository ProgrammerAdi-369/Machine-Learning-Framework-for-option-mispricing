This output tells a very clear story. Let me map it out before giving you the fix plan.Now let me explain exactly what is happening, why it is a serious problem, and how to fix every part of it with code.

---

## What the output is telling you — the real diagnosis

Every symptom in your output points to a single root cause: **distribution shift**. Your model trained on May–December 2025 and then the market moved into a different regime in early 2026. The model has no way to adapt because it is static — once trained, it never updates.

Look at the temporal stability table. The model is essentially perfect during training (R² = 0.997–0.998, bias ≈ 0.000). Then December starts deteriorating, January gets worse, February gets bad, and March collapses to R² = 0.88 with a bias of +0.33. This is not random noise — it is a clean monotonic degradation over time. That is the signature of a model that memorized the price distribution of a specific period and is now being asked to price contracts in a world that has moved on.

The +0.15 mean bias means the model is systematically predicting prices that are too low. In practical terms: it thinks options should be priced at ₹X, but the market is pricing them at ₹X + 15% more. This is because IV and spot levels likely rose in early 2026, making all options more expensive, but the model only knows what options cost in mid-2025.

The crushed z-scores (std = 0.42 instead of 1.0) are a direct consequence of this. Z-scores are computed as `(mispricing - daily_mean) / daily_std`. If the model is biased in the same direction for all contracts every day, the within-day variation is tiny — they all look wrong in the same way — so the standard deviation of mispricing within a day is small. When you divide by a small std, almost nothing crosses the ±2 threshold. This explains why you only got 21 SELL signals and zero BUY signals.

The R² gap of 0.051 (just over the 0.05 warning threshold) is mild overfitting on top of the distribution shift. It is a secondary issue — fix the distribution shift first.

---

## Fix 1 — Walk-forward retraining (highest priority)

This is the core architectural fix. Instead of training once and deploying forever, you retrain the model every month by adding new data. This is called a rolling or walk-forward approach and is standard practice in quant finance.

```python
def walk_forward_retrain(df, initial_train_months=7, retrain_every_months=1):
    """
    Trains a new model each month using all data up to that point.
    Returns a dataframe with out-of-sample predictions for every row.
    """
    import xgboost as xgb
    from dateutil.relativedelta import relativedelta

    FEATURE_COLS = [
        'option_type_encoded', 'DTE', 'moneyness',
        'IV', 'IV_relative', 'Skew', 'TS_Slope',
        'IV_HV_Spread', 'ATM_IV', 'OI', 'Volume',
        'OI_normalized', 'Volume_normalized', 'abs_moneyness', 'IV_rank'
    ]
    TARGET = 'log_price'

    df = df.copy().sort_values('date')
    df['wf_predicted_log_price'] = np.nan
    df['wf_model_version'] = ''

    all_months = sorted(df['date'].dt.to_period('M').unique())

    # Start predicting from month (initial_train_months + 1) onward
    for i in range(initial_train_months, len(all_months)):
        predict_month = all_months[i]

        # All data BEFORE this month is training data
        train_mask = df['date'].dt.to_period('M') < predict_month
        pred_mask  = df['date'].dt.to_period('M') == predict_month

        train_data = df[train_mask].dropna(subset=FEATURE_COLS + [TARGET])
        pred_data  = df[pred_mask].dropna(subset=FEATURE_COLS)

        if len(train_data) < 5000 or len(pred_data) == 0:
            continue

        X_train = train_data[FEATURE_COLS]
        y_train = train_data[TARGET]

        # Slightly more regularized than your original — reduces R² gap
        model = xgb.XGBRegressor(
            n_estimators=400,
            max_depth=5,            # reduced from 6
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.7,   # reduced from 0.8
            min_child_weight=15,    # new — prevents overfitting small segments
            gamma=0.1,              # new — minimum loss reduction for split
            reg_alpha=0.05,         # new — L1 regularization
            reg_lambda=1.5,         # new — L2 regularization
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )

        model.fit(X_train, y_train)

        # Predict the current month out-of-sample
        preds = model.predict(pred_data[FEATURE_COLS])
        df.loc[pred_mask & df.index.isin(pred_data.index),
               'wf_predicted_log_price'] = preds
        df.loc[pred_mask, 'wf_model_version'] = str(predict_month)

        # Save the model for this month
        model.save_model(f'outputs/model_{predict_month}.json')

        month_r2 = r2_score(pred_data[TARGET], preds)
        print(f"  {predict_month}: trained on {len(train_data):,} rows → "
              f"predict {len(pred_data):,} rows | R²={month_r2:.4f}")

    # Recompute predicted prices from walk-forward predictions
    df['wf_predicted_price'] = np.exp(df['wf_predicted_log_price']) - 1
    df['wf_mispricing'] = df['close_price'] - df['wf_predicted_price']

    # Recompute z-scores within each day using walk-forward predictions
    df['wf_misp_mean'] = df.groupby('date')['wf_mispricing'].transform('mean')
    df['wf_misp_std']  = df.groupby('date')['wf_mispricing'].transform('std').clip(lower=0.5)
    df['wf_z_score']   = (df['wf_mispricing'] - df['wf_misp_mean']) / df['wf_misp_std']

    return df

# Run it
df_wf = walk_forward_retrain(df, initial_train_months=7)

# Compare old vs new predictions on the test period
test_wf = df_wf[df_wf['date'] > cutoff].dropna(subset=['wf_predicted_log_price'])
print("\nWalk-forward vs static model on test period:")
print(f"  Static  R²: {r2_score(test_wf['log_price'], test_wf['predicted_log_price']):.4f}")
print(f"  WF      R²: {r2_score(test_wf['log_price'], test_wf['wf_predicted_log_price']):.4f}")
print(f"  Static  bias: {(test_wf['log_price']-test_wf['predicted_log_price']).mean():.4f}")
print(f"  WF      bias: {(test_wf['log_price']-test_wf['wf_predicted_log_price']).mean():.4f}")
print(f"  Static  z-score std: {test_wf['z_score'].std():.3f}")
print(f"  WF      z-score std: {test_wf['wf_z_score'].std():.3f}")
```

After this runs, you should see March's R² jump from 0.88 back toward 0.97+. Each monthly model will have been trained on data that includes recent market behavior, so it will not be surprised by the new price levels.

---

## Fix 2 — Regularization to close the R² gap

The 0.051 R² gap is mild but real. These hyperparameter changes, added to the model above, directly address it. The key changes are reducing `max_depth` from 6 to 5 (each tree is less complex), adding `min_child_weight=15` (requires at least 15 samples to make a split, prevents the model learning noise from sparse strikes), and adding L1+L2 penalties (`reg_alpha`, `reg_lambda`).

To verify these are the right settings for your data, run a quick validation:

```python
from sklearn.model_selection import cross_val_score

# Use only the training period for this search
train_only = df[df['date'] <= cutoff].dropna(subset=FEATURE_COLS_CLEAN + ['log_price'])

configs = [
    {'max_depth': 6, 'min_child_weight': 5,  'reg_alpha': 0,    'reg_lambda': 1,   'label': 'original'},
    {'max_depth': 5, 'min_child_weight': 15, 'reg_alpha': 0.05, 'reg_lambda': 1.5, 'label': 'regularized'},
    {'max_depth': 4, 'min_child_weight': 20, 'reg_alpha': 0.1,  'reg_lambda': 2.0, 'label': 'heavy_reg'},
]

# Time-series cross-validation: 3 folds, always predict forward
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=3)

for config in configs:
    label = config.pop('label')
    m = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.7,
                          random_state=42, n_jobs=-1, verbosity=0, **config)
    scores = cross_val_score(m, train_only[FEATURE_COLS_CLEAN],
                              train_only['log_price'],
                              cv=tscv, scoring='r2')
    print(f"  {label:<15} fold R²: {scores} → mean={scores.mean():.4f} std={scores.std():.4f}")
    config['label'] = label  # restore
```

Pick the config that has the highest mean R² with the lowest std across folds. High std means the model is sensitive to which time period it trains on — more regularization is needed.

---

## Fix 3 — IV-normalized target to remove price-level drift

This is the structural fix that makes the model inherently more robust to market-level changes. Instead of predicting `log_price` directly, predict `IV` as the target. IV is already expressed relative to the market — a 15% IV means the same thing whether BANKNIFTY is at 40,000 or 55,000. Log-price is not: a ₹500 option at one market level is very different from a ₹500 option at another level.

```python
# === OPTION A: Switch target to IV ===
# Much more stable across market regimes

FEATURE_COLS_FOR_IV = [
    'option_type_encoded', 'DTE', 'moneyness', 'abs_moneyness',
    'ATM_IV', 'Skew', 'TS_Slope', 'IV_HV_Spread',
    'OI_normalized', 'Volume_normalized', 'IV_rank'
    # Note: do NOT include IV itself as a feature when IV is the target
]
TARGET_IV = 'IV'

# Train
model_iv = xgb.XGBRegressor(
    n_estimators=400, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.7,
    min_child_weight=15, gamma=0.1,
    reg_alpha=0.05, reg_lambda=1.5,
    random_state=42, n_jobs=-1
)
model_iv.fit(
    train_df[FEATURE_COLS_FOR_IV].dropna(),
    train_df.loc[train_df[FEATURE_COLS_FOR_IV].dropna().index, TARGET_IV]
)

# Predict and compute IV mispricing
df['predicted_IV'] = model_iv.predict(df[FEATURE_COLS_FOR_IV].fillna(df[FEATURE_COLS_FOR_IV].median()))
df['IV_mispricing'] = df['IV'] - df['predicted_IV']

# Z-score of IV mispricing within each day
df['IV_misp_mean'] = df.groupby('date')['IV_mispricing'].transform('mean')
df['IV_misp_std']  = df.groupby('date')['IV_mispricing'].transform('std').clip(lower=0.001)
df['IV_z_score']   = (df['IV_mispricing'] - df['IV_misp_mean']) / df['IV_misp_std']

# === OPTION B: Use log_price but divide by ATM_IV (relative price) ===
# Simpler change, keeps original structure

df['relative_log_price'] = df['log_price'] / (df['ATM_IV'] + 0.001)

# Then train with 'relative_log_price' as the target
# This normalizes price predictions by the volatility regime
```

Option A (predicting IV directly) is the cleaner approach for a cross-sectional model and is what professional quant desks use. The IV surface is more stationary than the price surface. You can always convert back to a price signal: if `IV_z_score > 2`, the option's IV is high relative to similar options today, meaning it is expensive.

---

## Fix 4 — Recalibrate z-scores to fix the std=0.42 problem

Even before retraining, you can partially fix the crushed z-scores right now. The issue is that the within-day mispricing std is small because systematic bias is dominating. Removing the bias first, then computing z-scores, gives better-calibrated signals:

```python
def recalibrate_zscores(df):
    """
    Removes rolling bias before computing z-scores.
    Uses a 5-day rolling window to estimate current model bias,
    then subtracts it before the within-day normalization.
    """
    df = df.copy().sort_values('date')

    # Compute daily mean residual (bias per day)
    daily_bias = df.groupby('date').apply(
        lambda x: (x['log_price'] - x['predicted_log_price']).mean()
    ).reset_index()
    daily_bias.columns = ['date', 'daily_bias']

    # Rolling 5-day average bias — this is our bias estimate
    daily_bias['rolling_bias'] = daily_bias['daily_bias'].rolling(5, min_periods=1).mean()

    df = df.merge(daily_bias[['date', 'rolling_bias']], on='date', how='left')

    # Bias-corrected residual
    df['corrected_residual'] = (df['log_price'] - df['predicted_log_price']) - df['rolling_bias']

    # Recompute z-scores from corrected residual
    df['corr_misp_mean'] = df.groupby('date')['corrected_residual'].transform('mean')
    df['corr_misp_std']  = df.groupby('date')['corrected_residual'].transform('std').clip(lower=0.01)
    df['corrected_z_score'] = (df['corrected_residual'] - df['corr_misp_mean']) / df['corr_misp_std']

    # Check improvement
    liquid = df[(df['OI_normalized'] > 0.5) & (df['Volume_normalized'] > 0.5) & (df['DTE'] > 5)]
    print(f"  Original z-score std:   {liquid['z_score'].std():.3f}")
    print(f"  Corrected z-score std:  {liquid['corrected_z_score'].std():.3f}")
    print(f"  Original z-score skew:  {liquid['z_score'].skew():.3f}")
    print(f"  Corrected z-score skew: {liquid['corrected_z_score'].skew():.3f}")

    signals_old = (abs(liquid['z_score']) > 2).sum()
    signals_new = (abs(liquid['corrected_z_score']) > 2).sum()
    print(f"  Signals before: {signals_old}  |  Signals after: {signals_new}")

    return df

df = recalibrate_zscores(df)
```

This is a band-aid, not a cure — the real fix is retraining. But it will immediately improve your signal count from 21 to something more usable while you implement the walk-forward approach.

---

## Priority order and what to expect

Run the fixes in this order:

First, run `recalibrate_zscores()` on your existing predictions right now. This costs nothing and should immediately unstick your signals from the near-zero state. You should see z-score std rise from 0.42 toward 0.8–0.9.

Second, run `walk_forward_retrain()` over your full dataset. This replaces your static model with monthly-updated models. March's R² should recover from 0.88 to somewhere in the 0.93–0.96 range, and the bias should shrink from +0.33 to near zero for most months.

Third, add the regularization parameters to all future training runs to bring the R² gap below 0.02.

Fourth, consider switching to IV as the target (Option A in Fix 3) once the walk-forward version is working. This is a bigger change but makes the model fundamentally more regime-robust.

The scorecard after these fixes should show: zero critical failures, the bias warning resolved, z-score std at 0.95–1.05, and the signal count rising from 21 to somewhere in the 80–150 range with a healthier BUY/SELL balance.
    
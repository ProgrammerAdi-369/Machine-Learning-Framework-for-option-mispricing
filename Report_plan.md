Looking at your output, the model is actually performing well — but you're right that these numbers alone don't tell the full story. Let me build you a comprehensive accuracy analysis plan.First, here is a map of every analysis layer you need and what question each one answers — before a single line of code.Now here is the full guided plan — every layer as runnable code with explanations of what each metric means and what to do when numbers look wrong.

---

## Setup — load your results first

All analysis runs on top of the dataframe that already has predictions attached. Make sure you have this at the top of your analysis script:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Load the full dataset with predictions already attached
# (this is the df from the end of Phase 6 in your pipeline)
df = pd.read_parquet('outputs/full_predictions.parquet')

# Separate train and test using the same date cutoff your pipeline used
sorted_dates = sorted(df['date'].unique())
cutoff = sorted_dates[int(len(sorted_dates) * 0.70)]
train_df = df[df['date'] <= cutoff].copy()
test_df  = df[df['date'] > cutoff].copy()

print(f"Train rows: {len(train_df):,} | Test rows: {len(test_df):,}")
print(f"Train dates: {train_df['date'].min()} → {train_df['date'].max()}")
print(f"Test dates:  {test_df['date'].min()} → {test_df['date'].max()}")
```

If your pipeline does not already save `full_predictions.parquet` at the end of Phase 6, add this line right after computing z-scores: `df.to_parquet('outputs/full_predictions.parquet', index=False)`.

---

## Layer A — Global fit metrics

This answers the most basic question: is the model predicting in the right ballpark? RMSE alone (which you already have) is not enough because it is in log-price units, which are hard to interpret. You need MAE for a human-readable average error, and MAPE to understand the percentage error relative to the actual price.

```python
def compute_global_metrics(actual_log, predicted_log, actual_price, predicted_price, label=''):
    rmse  = np.sqrt(mean_squared_error(actual_log, predicted_log))
    mae   = mean_absolute_error(actual_log, predicted_log)
    r2    = r2_score(actual_log, predicted_log)
    
    # Price-level errors (more interpretable)
    mae_price  = mean_absolute_error(actual_price, predicted_price)
    rmse_price = np.sqrt(mean_squared_error(actual_price, predicted_price))
    
    # MAPE: avoid division by zero for very cheap options
    mask = actual_price > 1.0  # only contracts worth more than ₹1
    mape = np.mean(np.abs((actual_price[mask] - predicted_price[mask]) / actual_price[mask])) * 100
    
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    print(f"  Log-price RMSE : {rmse:.4f}")
    print(f"  Log-price MAE  : {mae:.4f}")
    print(f"  R²             : {r2:.4f}")
    print(f"  Price RMSE     : ₹{rmse_price:.2f}")
    print(f"  Price MAE      : ₹{mae_price:.2f}")
    print(f"  MAPE (price>1) : {mape:.2f}%")
    
    return dict(label=label, rmse=rmse, mae=mae, r2=r2,
                rmse_price=rmse_price, mae_price=mae_price, mape=mape)

train_metrics = compute_global_metrics(
    train_df['log_price'], train_df['predicted_log_price'],
    train_df['close_price'], train_df['predicted_price'], 'TRAIN SET')

test_metrics = compute_global_metrics(
    test_df['log_price'], test_df['predicted_log_price'],
    test_df['close_price'], test_df['predicted_price'], 'TEST SET')

# Overfitting gap check
r2_gap = train_metrics['r2'] - test_metrics['r2']
print(f"\n  R² gap (train-test): {r2_gap:.4f}")
if r2_gap > 0.05:
    print("  WARNING: gap > 0.05 suggests overfitting — add regularization")
elif r2_gap < 0.0:
    print("  NOTE: test R² > train R² is unusual — check for data issues")
else:
    print("  OK: gap is within healthy range")
```

**What the numbers mean for your model:** Your test R² of 0.9521 with RMSE 0.4417 is healthy. The price-level RMSE and MAE will give you more concrete intuition — if the model is off by an average of ₹15 on a ₹200 contract, that is 7.5%, which is meaningful for trading. If it is off by ₹15 on a ₹1,000 contract, that is fine.

**Red flags to watch for:** MAPE above 30% means the model is not useful for thin-premium OTM options. R² gap above 0.05 means overfitting. If test RMSE is more than 2× train RMSE, you have a leakage or distribution shift problem.

---

## Layer B — Residual diagnostics

RMSE tells you the size of errors. Residual diagnostics tell you whether those errors are random (good) or structured (bad). Structured residuals mean the model has a systematic blind spot.

```python
def plot_residual_diagnostics(df, split_label='Test set'):
    residuals = df['log_price'] - df['predicted_log_price']
    
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f'Residual diagnostics — {split_label}', fontsize=14, y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)
    
    # Plot 1: Residuals vs predicted (the most important plot)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(df['predicted_log_price'], residuals, alpha=0.05, s=4, color='steelblue')
    ax1.axhline(0, color='red', linewidth=1)
    ax1.set_xlabel('Predicted log-price')
    ax1.set_ylabel('Residual')
    ax1.set_title('Residuals vs predicted\n(should be a flat cloud)')
    
    # Plot 2: Residual histogram
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(residuals, bins=100, color='steelblue', alpha=0.7, density=True)
    x = np.linspace(residuals.min(), residuals.max(), 300)
    ax2.plot(x, stats.norm.pdf(x, residuals.mean(), residuals.std()), 
             'r-', linewidth=2, label='Normal fit')
    ax2.set_title('Residual distribution\n(should be ~normal)')
    ax2.legend(fontsize=9)
    
    # Plot 3: Q-Q plot
    ax3 = fig.add_subplot(gs[0, 2])
    stats.probplot(residuals, dist='norm', plot=ax3)
    ax3.set_title('Q-Q plot\n(points on line = normal errors)')
    
    # Plot 4: Residuals vs moneyness
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.scatter(df['moneyness'], residuals, alpha=0.05, s=4, color='darkorange')
    ax4.axhline(0, color='red', linewidth=1)
    # Add a smoothed trend line
    moneyness_sorted = df['moneyness'].sort_values()
    res_sorted = residuals[moneyness_sorted.index]
    from scipy.ndimage import uniform_filter1d
    window = max(1, len(res_sorted) // 200)
    trend = uniform_filter1d(res_sorted.values, size=window)
    ax4.plot(moneyness_sorted.values, trend, 'r-', linewidth=2)
    ax4.set_xlabel('Moneyness')
    ax4.set_ylabel('Residual')
    ax4.set_title('Residuals vs moneyness\n(red trend should be flat)')
    
    # Plot 5: Residuals vs DTE
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.scatter(df['DTE'], residuals, alpha=0.05, s=4, color='purple')
    ax5.axhline(0, color='red', linewidth=1)
    ax5.set_xlabel('DTE')
    ax5.set_title('Residuals vs DTE\n(should be flat)')
    
    # Plot 6: Residuals over time (rolling mean)
    ax6 = fig.add_subplot(gs[1, 2])
    daily_res = df.groupby('date')['log_price'].apply(
        lambda x: (x - df.loc[x.index, 'predicted_log_price']).mean()
    )
    daily_res.plot(ax=ax6, alpha=0.4, color='steelblue', linewidth=0.8)
    daily_res.rolling(10).mean().plot(ax=ax6, color='red', linewidth=2)
    ax6.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax6.set_title('Daily mean residual over time\n(red rolling avg should hug zero)')
    ax6.set_xlabel('')
    
    plt.savefig('outputs/residual_diagnostics.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Statistical tests
    print("\nStatistical tests on residuals:")
    stat, p = stats.shapiro(residuals.sample(min(5000, len(residuals)), random_state=42))
    print(f"  Shapiro-Wilk normality test: p={p:.4f} {'(normal)' if p > 0.05 else '(NOT normal — heavy tails exist)'}")
    
    skew = residuals.skew()
    kurt = residuals.kurtosis()
    print(f"  Skewness: {skew:.3f}  {'OK' if abs(skew) < 0.5 else 'WARNING: skewed — model biased in one direction'}")
    print(f"  Kurtosis: {kurt:.3f}  {'OK' if kurt < 5 else 'WARNING: fat tails — extreme errors are common'}")
    
    bias = residuals.mean()
    print(f"  Mean residual (bias): {bias:.4f}  {'OK' if abs(bias) < 0.01 else 'WARNING: systematic bias exists'}")

plot_residual_diagnostics(test_df, 'Test set')
```

**What to look for in each plot:**

The "residuals vs predicted" scatter should look like a horizontal band of noise centered on zero. If it fans out at high predicted values (heteroscedasticity), your model has worse percentage errors for high-priced options. If there is a curve in the cloud, the model is missing a non-linear relationship.

The "residuals vs moneyness" plot is the most critical for this project. If the red trend line curves upward on the OTM side (moneyness < 0.95 or > 1.05), your model is systematically underpricing OTM options. The fix is to add `moneyness^2` as a feature.

The "daily mean residual over time" plot will show you if the model's performance decays in the test period — a rising or falling red line in the test set means the model is drifting and you may need to retrain more frequently.

---

## Layer C — Segment-level breakdown

This extends the simple ATM/ITM/OTM table you already have into a full multi-dimensional breakdown. The goal is to find the specific conditions where the model struggles.

```python
def segment_analysis(df, label='Test set'):
    results = []
    
    # Define all segmentation schemes
    segments = {
        'moneyness_bucket': {
            'Deep OTM': (df['moneyness'] < 0.93) | (df['moneyness'] > 1.07),
            'OTM':      ((df['moneyness'] >= 0.93) & (df['moneyness'] < 0.97)) | 
                        ((df['moneyness'] > 1.03) & (df['moneyness'] <= 1.07)),
            'Near ATM': (df['moneyness'] >= 0.97) & (df['moneyness'] <= 1.03),
        },
        'DTE_bucket': {
            'Near (1-10d)':  (df['DTE'] >= 1)  & (df['DTE'] <= 10),
            'Mid (11-30d)':  (df['DTE'] >= 11) & (df['DTE'] <= 30),
            'Far (31-90d)':  (df['DTE'] >= 31) & (df['DTE'] <= 90),
        },
        'option_type': {
            'CE (Call)': df['option_type'] == 'CE',
            'PE (Put)':  df['option_type'] == 'PE',
        },
        'liquidity_tier': {
            'High OI (top 25%)':   df['OI_normalized'] >= df['OI_normalized'].quantile(0.75),
            'Mid OI':              (df['OI_normalized'] >= df['OI_normalized'].quantile(0.25)) & 
                                   (df['OI_normalized'] < df['OI_normalized'].quantile(0.75)),
            'Low OI (bot 25%)':   df['OI_normalized'] < df['OI_normalized'].quantile(0.25),
        }
    }
    
    print(f"\n{'='*70}")
    print(f"  SEGMENT ANALYSIS — {label}")
    print(f"{'='*70}")
    
    for seg_name, buckets in segments.items():
        print(f"\n  [{seg_name}]")
        print(f"  {'Bucket':<20} {'N':>7} {'RMSE':>8} {'MAE':>8} {'R²':>8} {'MAPE%':>8} {'Bias':>8}")
        print(f"  {'-'*67}")
        
        for bucket_name, mask in buckets.items():
            sub = df[mask]
            if len(sub) < 50:
                continue
            
            rmse = np.sqrt(mean_squared_error(sub['log_price'], sub['predicted_log_price']))
            mae  = mean_absolute_error(sub['log_price'], sub['predicted_log_price'])
            r2   = r2_score(sub['log_price'], sub['predicted_log_price'])
            bias = (sub['log_price'] - sub['predicted_log_price']).mean()
            
            price_mask = sub['close_price'] > 1.0
            mape = (np.abs(sub.loc[price_mask, 'close_price'] - 
                           sub.loc[price_mask, 'predicted_price']) / 
                    sub.loc[price_mask, 'close_price']).mean() * 100 if price_mask.sum() > 0 else np.nan
            
            flag = ''
            if r2 < 0.85:  flag = ' <-- WEAK'
            if abs(bias) > 0.05: flag += ' BIASED'
            
            print(f"  {bucket_name:<20} {len(sub):>7,} {rmse:>8.4f} {mae:>8.4f} {r2:>8.4f} "
                  f"{mape:>8.1f} {bias:>8.4f}{flag}")
            
            results.append(dict(seg=seg_name, bucket=bucket_name, n=len(sub),
                                rmse=rmse, mae=mae, r2=r2, mape=mape, bias=bias))
    
    return pd.DataFrame(results)

seg_results = segment_analysis(test_df)
```

**How to read the output:** Look at the "bias" column first. A consistently positive bias in a segment means the model is systematically underestimating prices there — it is not random error, it is a structural gap. Look at R² next. Any segment below 0.85 is one the model effectively cannot price and you should filter out of signal generation.

**The liquidity tier breakdown is the most actionable.** If your model has R²=0.97 in the high-OI tier but 0.80 in the low-OI tier, you have confirmation that your liquidity filters are doing the right job. If the model is poor even in the high-OI tier, you have a real problem.

---

## Layer D — Feature importance and SHAP analysis

This answers: what did the model actually learn? And is it learning the right things?

```python
def feature_importance_analysis(model, feature_cols, df, label='Test set'):
    
    # Part 1: XGBoost native importance (fast, three types)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Feature importance — three views', fontsize=13)
    
    importance_types = ['weight', 'gain', 'cover']
    importance_labels = {
        'weight': 'Frequency\n(how often used in splits)',
        'gain':   'Gain\n(how much each feature improves predictions)',
        'cover':  'Cover\n(how many samples each feature affects)'
    }
    
    for ax, imp_type in zip(axes, importance_types):
        scores = model.get_booster().get_score(importance_type=imp_type)
        scores_df = pd.DataFrame({'feature': list(scores.keys()), 
                                   'score': list(scores.values())})
        scores_df = scores_df.sort_values('score', ascending=True).tail(15)
        
        ax.barh(scores_df['feature'], scores_df['score'], color='steelblue', alpha=0.8)
        ax.set_title(importance_labels[imp_type], fontsize=10)
        ax.tick_params(labelsize=8)
    
    plt.tight_layout()
    plt.savefig('outputs/feature_importance.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Part 2: SHAP values (the gold standard — needs pip install shap)
    try:
        import shap
        
        # Use a sample to keep computation fast
        sample = df[feature_cols].sample(min(3000, len(df)), random_state=42)
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('SHAP analysis', fontsize=13)
        
        plt.sca(axes[0])
        shap.summary_plot(shap_values, sample, show=False, max_display=15)
        axes[0].set_title('SHAP summary: feature impact on predictions')
        
        plt.sca(axes[1])
        shap.summary_plot(shap_values, sample, plot_type='bar', show=False, max_display=15)
        axes[1].set_title('SHAP mean absolute values')
        
        plt.tight_layout()
        plt.savefig('outputs/shap_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Interaction: moneyness SHAP vs actual moneyness
        moneyness_idx = list(feature_cols).index('moneyness')
        moneyness_shap = shap_values[:, moneyness_idx]
        
        plt.figure(figsize=(8, 5))
        plt.scatter(sample['moneyness'], moneyness_shap, alpha=0.1, s=4)
        plt.axhline(0, color='red', linewidth=1)
        plt.xlabel('Moneyness')
        plt.ylabel('SHAP value (impact on log-price prediction)')
        plt.title('How moneyness drives predictions\n(should be monotone — higher moneyness = higher CE price)')
        plt.savefig('outputs/shap_moneyness.png', dpi=150, bbox_inches='tight')
        plt.show()
        
    except ImportError:
        print("\n  SHAP not installed. Run: pip install shap")
        print("  Skipping SHAP analysis — native importance plots saved above.")

# Load the model from wherever you saved it
model = xgb.XGBRegressor()
model.load_model('outputs/xgb_model.json')  # adjust path as needed

FEATURE_COLS = [
    'strike', 'option_type_encoded', 'DTE', 'moneyness',
    'log_price', 'IV', 'IV_relative', 'Skew', 'TS_Slope',
    'IV_HV_Spread', 'ATM_IV', 'OI', 'Volume',
    'OI_normalized', 'Volume_normalized', 'abs_moneyness', 'IV_rank'
]
# Note: log_price should NOT be in features if it IS the target
# If it leaked in, remove it here and retrain
FEATURE_COLS_CLEAN = [f for f in FEATURE_COLS if f != 'log_price']

feature_importance_analysis(model, FEATURE_COLS_CLEAN, test_df)
```

**What to look for:** `moneyness`, `DTE`, and `IV` should dominate the gain importance. If `strike` raw value has very high gain, your model may be memorizing price levels rather than learning relationships — this makes it fragile when market moves to new strike ranges. If `log_price` appears in your feature list with high importance, you have a leakage problem and must retrain.

The SHAP moneyness plot is particularly revealing. For CE options, SHAP values should increase monotonically as moneyness goes above 1.0 (deeper ITM = higher price). If the curve is jagged or non-monotone, the model is confused by moneyness.

---

## Layer E — Z-score and signal quality analysis

This is the layer that connects model accuracy directly to trading viability. A model can have good R² but generate terrible trading signals if the z-scores are miscalibrated.

```python
def signal_quality_analysis(df):
    signals_df = df[
        (df['Volume_normalized'] > 0.5) &
        (df['OI_normalized'] > 0.5) &
        (df['DTE'] > 5)
    ].copy()
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Z-score and signal quality analysis', fontsize=13)
    
    # Plot 1: Z-score distribution
    ax = axes[0, 0]
    ax.hist(signals_df['z_score'], bins=80, color='steelblue', alpha=0.7, density=True)
    x = np.linspace(-5, 5, 300)
    ax.plot(x, stats.norm.pdf(x, 0, 1), 'r-', linewidth=2, label='Standard normal')
    ax.axvline(2, color='orange', linewidth=1.5, linestyle='--', label='+2 sell threshold')
    ax.axvline(-2, color='green', linewidth=1.5, linestyle='--', label='-2 buy threshold')
    ax.set_xlabel('Z-score')
    ax.set_title('Z-score distribution (should match normal)')
    ax.legend(fontsize=8)
    ax.set_xlim(-6, 6)
    
    # Plot 2: Z-score vs moneyness (should show no structure)
    ax = axes[0, 1]
    ax.scatter(signals_df['moneyness'], signals_df['z_score'], 
               alpha=0.08, s=4, color='darkorange')
    ax.axhline(2, color='red', linewidth=1, linestyle='--')
    ax.axhline(-2, color='green', linewidth=1, linestyle='--')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Moneyness')
    ax.set_ylabel('Z-score')
    ax.set_title('Z-score vs moneyness\n(should be random scatter)')
    
    # Plot 3: Z-score vs DTE
    ax = axes[0, 2]
    ax.scatter(signals_df['DTE'], signals_df['z_score'], 
               alpha=0.08, s=4, color='purple')
    ax.axhline(2, color='red', linewidth=1, linestyle='--')
    ax.axhline(-2, color='green', linewidth=1, linestyle='--')
    ax.set_xlabel('DTE')
    ax.set_title('Z-score vs DTE\n(should be random scatter)')
    
    # Plot 4: Signal count per day over test period
    ax = axes[1, 0]
    daily_signals = signals_df[abs(signals_df['z_score']) > 2].groupby('date').size()
    daily_signals.plot(ax=ax, color='steelblue', alpha=0.7)
    daily_signals.rolling(5).mean().plot(ax=ax, color='red', linewidth=2)
    ax.set_title('Signals per day over time\n(red = 5-day rolling average)')
    ax.set_xlabel('')
    
    # Plot 5: Z-score percentile calibration
    ax = axes[1, 1]
    theoretical_quantiles = np.linspace(0.01, 0.99, 100)
    theoretical_z = stats.norm.ppf(theoretical_quantiles)
    actual_z = np.percentile(signals_df['z_score'].dropna(), 
                              np.linspace(1, 99, 100))
    ax.plot(theoretical_z, actual_z, 'b-', linewidth=2, label='Actual z-scores')
    ax.plot([-4, 4], [-4, 4], 'r--', linewidth=1, label='Perfect calibration')
    ax.set_xlabel('Theoretical normal quantile')
    ax.set_ylabel('Actual z-score quantile')
    ax.set_title('Z-score calibration\n(blue on red line = well-calibrated)')
    ax.legend(fontsize=9)
    
    # Plot 6: Mispricing magnitude by moneyness bucket
    ax = axes[1, 2]
    buckets = ['Deep OTM', 'OTM', 'Near ATM']
    masks = [
        (abs(signals_df['moneyness'] - 1) > 0.07),
        ((abs(signals_df['moneyness'] - 1) > 0.03) & (abs(signals_df['moneyness'] - 1) <= 0.07)),
        (abs(signals_df['moneyness'] - 1) <= 0.03)
    ]
    bp_data = [signals_df.loc[m, 'mispricing'].dropna().values for m in masks]
    bp = ax.boxplot(bp_data, labels=buckets, patch_artist=True, 
                    medianprops=dict(color='red', linewidth=2))
    colors = ['#F09595', '#85B7EB', '#9FE1CB']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax.set_title('Mispricing distribution by bucket\n(median should be near 0)')
    ax.set_ylabel('Mispricing (₹)')
    
    plt.tight_layout()
    plt.savefig('outputs/signal_quality.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print signal statistics
    sell_signals = signals_df[signals_df['z_score'] > 2]
    buy_signals  = signals_df[signals_df['z_score'] < -2]
    
    print("\nSignal quality summary:")
    print(f"  Total liquid contracts: {len(signals_df):,}")
    print(f"  SELL signals (z>+2):    {len(sell_signals):,} ({len(sell_signals)/len(signals_df)*100:.1f}%)")
    print(f"  BUY signals  (z<-2):    {len(buy_signals):,}  ({len(buy_signals)/len(signals_df)*100:.1f}%)")
    print(f"\n  Z-score stats:")
    print(f"    Mean: {signals_df['z_score'].mean():.3f}  (should be ~0)")
    print(f"    Std:  {signals_df['z_score'].std():.3f}  (should be ~1)")
    print(f"    Skew: {signals_df['z_score'].skew():.3f}  (should be ~0)")
    
    # Check: are SELL signals mostly OTM? (a red flag)
    sell_atm_pct = (abs(sell_signals['moneyness'] - 1) < 0.03).mean() * 100
    sell_otm_pct = (abs(sell_signals['moneyness'] - 1) > 0.05).mean() * 100
    print(f"\n  SELL signal moneyness breakdown:")
    print(f"    Near ATM: {sell_atm_pct:.1f}%  OTM: {sell_otm_pct:.1f}%")
    if sell_otm_pct > 70:
        print("    WARNING: >70% of SELL signals are deep OTM — liquidity filters may need tightening")

signal_quality_analysis(test_df)
```

**The calibration plot in position [1,1] is the key signal quality check.** If the blue line bows above the red line at the tails, your z-scores are under-dispersed — real extremes are not being flagged as extremes. If the blue line stays below the red line at the tails, your thresholds are too aggressive and most "signals" are noise. A perfectly calibrated model has the blue line sitting on the red diagonal.

---

## Layer F — Temporal stability

A model that worked in December might not work in March. This layer shows whether model performance is stable across time.

```python
def temporal_stability_analysis(df):
    # Rolling R² and RMSE — computed monthly
    df = df.copy()
    df['year_month'] = df['date'].dt.to_period('M')
    
    monthly_metrics = []
    for period, group in df.groupby('year_month'):
        if len(group) < 100:
            continue
        r2   = r2_score(group['log_price'], group['predicted_log_price'])
        rmse = np.sqrt(mean_squared_error(group['log_price'], group['predicted_log_price']))
        bias = (group['log_price'] - group['predicted_log_price']).mean()
        atm_iv_mean = group['ATM_IV'].mean()
        
        monthly_metrics.append(dict(
            period=str(period), r2=r2, rmse=rmse, bias=bias,
            atm_iv=atm_iv_mean, n=len(group)
        ))
    
    mdf = pd.DataFrame(monthly_metrics)
    mdf['period_dt'] = pd.to_datetime(mdf['period'])
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle('Temporal stability of model performance', fontsize=13)
    
    # Shade the train/test boundary
    cutoff_str = df[df['date'] <= cutoff]['year_month'].max()
    
    for ax in axes.flat:
        ax.axvspan(mdf['period_dt'].min(), 
                   pd.to_datetime(str(cutoff_str)),
                   alpha=0.08, color='steelblue', label='Train period')
        ax.axvspan(pd.to_datetime(str(cutoff_str)),
                   mdf['period_dt'].max(),
                   alpha=0.08, color='orange', label='Test period')
    
    axes[0, 0].plot(mdf['period_dt'], mdf['r2'], 'o-', color='steelblue', markersize=4)
    axes[0, 0].axhline(0.90, color='red', linewidth=1, linestyle='--', label='R²=0.90 floor')
    axes[0, 0].set_title('Monthly R²')
    axes[0, 0].set_ylim(0.7, 1.01)
    axes[0, 0].legend(fontsize=8)
    
    axes[0, 1].plot(mdf['period_dt'], mdf['rmse'], 'o-', color='darkorange', markersize=4)
    axes[0, 1].set_title('Monthly RMSE')
    
    axes[1, 0].plot(mdf['period_dt'], mdf['bias'], 'o-', color='purple', markersize=4)
    axes[1, 0].axhline(0, color='black', linewidth=0.5, linestyle='--')
    axes[1, 0].set_title('Monthly mean bias\n(should stay near zero)')
    
    # Overlay ATM_IV to see if bias correlates with volatility regime
    ax_twin = axes[1, 1]
    ax_twin.plot(mdf['period_dt'], mdf['atm_iv'], 'o-', color='steelblue', 
                 markersize=4, label='ATM IV (left)')
    ax_twin2 = ax_twin.twinx()
    ax_twin2.plot(mdf['period_dt'], mdf['rmse'], 's--', color='coral', 
                  markersize=4, label='RMSE (right)')
    ax_twin.set_title('ATM IV vs RMSE\n(does high vol = worse accuracy?)')
    ax_twin.legend(loc='upper left', fontsize=8)
    ax_twin2.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('outputs/temporal_stability.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Flag problematic months
    print("\nMonthly performance table:")
    print(f"  {'Period':<12} {'N':>6} {'R²':>8} {'RMSE':>8} {'Bias':>8} {'Flag'}")
    print(f"  {'-'*60}")
    for _, row in mdf.iterrows():
        flag = ''
        if row['r2'] < 0.90: flag += ' LOW_R2'
        if abs(row['bias']) > 0.02: flag += ' BIASED'
        if row['rmse'] > mdf['rmse'].mean() * 1.5: flag += ' HIGH_ERR'
        print(f"  {row['period']:<12} {row['n']:>6,} {row['r2']:>8.4f} "
              f"{row['rmse']:>8.4f} {row['bias']:>8.4f}{flag}")

temporal_stability_analysis(df)  # pass the full df, not just test
```

**What to look for:** R² should stay above 0.90 in every month. A sudden drop in a specific month means something changed in the data or market regime that the model was not trained on. If RMSE spikes exactly when ATM_IV spikes (the bottom-right chart), your model degrades during high-volatility periods — this is fixable by adding IV regime features or training a separate model for high-vol periods.

---

## Layer G — The model scorecard

Run this after all layers to get a single printout that summarizes the health of the entire model:

```python
def print_model_scorecard(test_metrics, seg_results, df):
    print("\n" + "="*60)
    print("  MODEL ACCURACY SCORECARD")
    print("="*60)
    
    checks = []
    
    def check(name, condition, good_msg, bad_msg, critical=False):
        symbol = "PASS" if condition else ("FAIL" if critical else "WARN")
        msg = good_msg if condition else bad_msg
        checks.append((symbol, name, msg))
        print(f"  [{symbol:4s}] {name:<35} {msg}")
    
    check("Global R² (test)",        test_metrics['r2'] >= 0.90,
          f"{test_metrics['r2']:.4f}", f"{test_metrics['r2']:.4f} — retrain with more regularization", critical=True)
    
    check("Overfitting (R² gap)",    (test_metrics['r2'] - train_metrics['r2']) > -0.05,
          "gap is acceptable", "gap > 0.05 — overfitting detected", critical=True)
    
    check("Bias (mean residual)",    abs((df['log_price'] - df['predicted_log_price']).mean()) < 0.01,
          "no systematic bias", "systematic bias present — check feature engineering")
    
    check("Z-score std",             abs(df['z_score'].std() - 1.0) < 0.2,
          f"std={df['z_score'].std():.2f}", f"std={df['z_score'].std():.2f} — z-scores miscalibrated")
    
    check("Z-score mean",            abs(df['z_score'].mean()) < 0.1,
          f"mean={df['z_score'].mean():.3f}", f"mean={df['z_score'].mean():.3f} — signal bias")
    
    if seg_results is not None:
        atm_r2 = seg_results[(seg_results['seg']=='moneyness_bucket') & 
                              (seg_results['bucket']=='Near ATM')]['r2'].values
        if len(atm_r2):
            check("ATM R²",          atm_r2[0] >= 0.90,
                  f"{atm_r2[0]:.4f}", f"{atm_r2[0]:.4f} — model unreliable near ATM", critical=True)
    
    signal_pct = (abs(df[df['OI_normalized']>0.5]['z_score']) > 2).mean() * 100
    check("Signal rate (liquid)",    1.0 <= signal_pct <= 15.0,
          f"{signal_pct:.1f}% of liquid contracts", 
          f"{signal_pct:.1f}% — {'too few' if signal_pct < 1 else 'too many'} signals")
    
    print("\n" + "="*60)
    fails = sum(1 for s, _, _ in checks if s == 'FAIL')
    warns = sum(1 for s, _, _ in checks if s == 'WARN')
    print(f"  Result: {fails} critical failures, {warns} warnings")
    if fails == 0 and warns <= 2:
        print("  Model is PRODUCTION READY for signal generation")
    elif fails == 0:
        print("  Model is USABLE but review warnings before live trading")
    else:
        print("  Model needs IMPROVEMENT before use — address FAIL items first")
    print("="*60)

print_model_scorecard(test_metrics, seg_results, test_df)
```

---

## Execution order

Run the layers in this sequence in a single notebook or script:

1. Setup and data loading
2. Layer A: `compute_global_metrics()` on both train and test
3. Layer B: `plot_residual_diagnostics()` on test set
4. Layer C: `segment_analysis()` on test set
5. Layer D: `feature_importance_analysis()` — install `shap` first with `pip install shap`
6. Layer E: `signal_quality_analysis()` on test set
7. Layer F: `temporal_stability_analysis()` on the full dataframe
8. Layer G: `print_model_scorecard()` to get the final verdict

All plots are saved to `outputs/` so you can review them even after the script finishes. Based on your current numbers (R²=0.9521, z-score std≈1.0), I expect you will pass most checks, but the segment analysis and temporal stability plots will show you exactly where to focus improvement effort next.
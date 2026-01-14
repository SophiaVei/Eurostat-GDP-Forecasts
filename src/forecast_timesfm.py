import torch
import timesfm
import pandas as pd
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from data_processor import (
    load_wide_data,
    prepare_indicator_dataset,
    get_base_indicators_from_wide,
    impute_missing_values,
    preprocess_to_wide_file,
)

# Shared metric functions (same as forecast_utils for consistency)
def robust_mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = np.abs(y_true) > 1e-3
    if np.sum(mask) == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))

def calc_rmse(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def calc_mae(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred))

def calc_r2(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return np.nan
    return 1 - (ss_res / ss_tot)

def main():
    # Use relative paths from script location
    SCRIPT_DIR = Path(__file__).resolve().parent
    REPO_DIR = SCRIPT_DIR.parent
    
    RAW_FILE = REPO_DIR / "data" / "economy_nuts2_all_columns.csv"
    WIDE_FILE = REPO_DIR / "data" / "economy_nuts2_wide.csv"
    OUTPUT_DIR = REPO_DIR / "results" / "timesfm"
    
    # Ensure preprocessed wide file exists
    if not WIDE_FILE.exists():
        print(f"Preprocessing raw data to wide format: {WIDE_FILE}")
        preprocess_to_wide_file(str(RAW_FILE), str(WIDE_FILE))
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plots_dir = os.path.join(OUTPUT_DIR, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    print("Loading TimesFM...")
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    model.compile(timesfm.ForecastConfig(max_context=128, max_horizon=1, normalize_inputs=True, use_continuous_quantile_head=True))
    
    # Load preprocessed wide-format data (same as all other models)
    wide_df = load_wide_data(str(WIDE_FILE))
    indicators = get_base_indicators_from_wide(wide_df)
    
    comparisons = []
    final_forecasts = []
    plot_data_dict = {}

    for indicator in indicators:
        train_eval_df, forecast_df, col_info = prepare_indicator_dataset(wide_df, indicator)
        if train_eval_df is None or len(train_eval_df) < 5: continue
        
        target_col = col_info['target']
        single_cols = col_info['single']
        
        # Consistent Split: 80/20 with random_state=42
        train_idx, test_idx = train_test_split(train_eval_df.index, test_size=0.2, random_state=42)
        
        # Apply shared imputation to single-feature columns (same preprocessing as other models)
        X_single_imputed = impute_missing_values(train_eval_df[single_cols])
        
        # 1. Single-Feature Evaluation (Univariate)
        # Use imputed values to build time series for TimesFM
        inputs_s = []
        valid_idx_s = []
        for idx in test_idx:
            # Extract imputed series (all values should be non-NaN after imputation)
            series = X_single_imputed.loc[idx, single_cols].values.tolist()
            if len(series) > 0:
                inputs_s.append(np.array(series))
                valid_idx_s.append(idx)
        
        if not inputs_s: continue
        
        fc_s, _ = model.forecast(horizon=1, inputs=inputs_s)
        preds_s = np.maximum(0, fc_s[:, 0])
        y_true_s = train_eval_df.loc[valid_idx_s, target_col].values
        mape_s = robust_mape(y_true_s, preds_s)
        rmse_s = calc_rmse(y_true_s, preds_s)
        mae_s = calc_mae(y_true_s, preds_s)
        r2_s = calc_r2(y_true_s, preds_s)

        # 2. Multi-Feature: TimesFM is univariate, so Multi = Single
        mape_m = mape_s
        rmse_m = rmse_s
        mae_m = mae_s
        r2_m = r2_s
        
        if np.isnan(mape_s): continue
        
        improvement = 0
        better = "Single"
        
        comparisons.append({
            'indicator': indicator,
            'Multi_MAPE': mape_m * 100,
            'Single_MAPE': mape_s * 100,
            'Multi_RMSE': rmse_m,
            'Single_RMSE': rmse_s,
            'Multi_MAE': mae_m,
            'Single_MAE': mae_s,
            'Multi_R2': r2_m,
            'Single_R2': r2_s,
            'MAPE_Improvement': improvement,
            'Better': better
        })
        
        # Forecast missing
        if not forecast_df.empty:
            # Apply shared imputation to forecast data
            X_fc_imputed = impute_missing_values(forecast_df[single_cols])
            
            fc_inputs = []
            valid_fc_geos = []
            for idx, row in forecast_df.iterrows():
                # Extract imputed series
                series = X_fc_imputed.loc[idx, single_cols].values.tolist()
                if len(series) > 0:
                    fc_inputs.append(np.array(series))
                    valid_fc_geos.append(row['geo'])
            
            if fc_inputs:
                fc_point, _ = model.forecast(horizon=1, inputs=fc_inputs)
                fc_preds = np.maximum(0, fc_point[:, 0])
                for i, val in enumerate(fc_preds):
                    final_forecasts.append({
                        'geo': valid_fc_geos[i], 
                        'indicator': indicator, 
                        'forecast_2023': val, 
                        'model_used': 'Single (Univariate)'
                    })

        # Plots: Top 5 from test set
        num_to_plot = min(5, len(valid_idx_s))
        sample_indices = valid_idx_s[:num_to_plot]
        for i, idx in enumerate(sample_indices):
            row = train_eval_df.loc[idx]
            hist_years = [int(c.split('_')[-1]) for c in single_cols]
            # Use imputed values for plotting (consistent with what TimesFM sees)
            hist_vals = X_single_imputed.loc[idx, single_cols].values.tolist()
            pred_2023 = preds_s[i]
            actual_2023 = row[target_col]
            
            if indicator not in plot_data_dict: plot_data_dict[indicator] = []
            plot_data_dict[indicator].append({
                'geo': row['geo'], 
                'hist_years': hist_years, 
                'hist_vals': hist_vals, 
                'actual_2023': actual_2023, 
                'pred_2023': pred_2023, 
                'better': 'Univariate'
            })

    # Save
    pd.DataFrame(comparisons).to_csv(os.path.join(OUTPUT_DIR, 'indicator_comparison.csv'), index=False)
    if final_forecasts: 
        pd.DataFrame(final_forecasts).to_csv(os.path.join(OUTPUT_DIR, 'forecasts_2023.csv'), index=False)
    
    # Generate Plots
    print(f"Generating plots for TimesFM...")
    for indicator, regions in plot_data_dict.items():
        plt.figure(figsize=(15, 10))
        for i, reg in enumerate(regions):
            plt.subplot(len(regions), 1, i+1)
            plt.plot(reg['hist_years'], reg['hist_vals'], marker='o', label='History')
            if not pd.isna(reg['actual_2023']): 
                plt.scatter([2023], [reg['actual_2023']], color='green', s=100, label='Actual 2023')
            plt.scatter([2023], [reg['pred_2023']], color='red', marker='X', s=150, label='TimesFM 2023')
            plt.title(f"{indicator} - {reg['geo']} (Model: timesfm)")
            plt.xticks(reg['hist_years'] + [2023])
            plt.grid(True, alpha=0.3)
            if i == 0: plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{indicator}_forecast.png"))
        plt.close()
    print(f"TimesFM Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from data_processor import (
    load_wide_data,
    prepare_indicator_dataset,
    get_base_indicators_from_wide,
    impute_missing_values,
)
import logging
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# Suppress cmdstanpy logs (if any Stan-based models are used in the future)
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)

def robust_mape(y_true, y_pred):
    """Calculates MAPE while handling zero values in y_true."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = np.abs(y_true) > 1e-3
    if np.sum(mask) == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))

def calc_rmse(y_true, y_pred):
    """Root Mean Squared Error."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def calc_mae(y_true, y_pred):
    """Mean Absolute Error."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred))

def calc_r2(y_true, y_pred):
    """R-squared (Coefficient of Determination)."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return np.nan
    return 1 - (ss_res / ss_tot)


class TimesFMRegionalWrapper:
    """Fits TimesFM for each region independently."""
    def __init__(self, target_prefix=None, **kwargs):
        import torch
        import timesfm
        self.kwargs = kwargs
        print("Loading TimesFM model...")
        try:
            self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
            self.model.compile(
                timesfm.ForecastConfig(
                    max_context=128,
                    max_horizon=1,
                    normalize_inputs=True,
                    use_continuous_quantile_head=True,
                )
            )
        except Exception as e:
            print(f"Error loading TimesFM: {e}")
            self.model = None

    def fit(self, X, y):
        return self

    def predict(self, X):
        if self.model is None:
            return np.zeros(len(X))
            
        inputs = []
        for idx in range(len(X)):
            row = X.iloc[idx]
            series = [row[col] for col in X.columns if not pd.isna(row[col]) and row[col] != 0]
            inputs.append(np.array(series))
            
        if not inputs:
            return np.zeros(len(X))

        point_forecast, _ = self.model.forecast(horizon=1, inputs=inputs)
        return point_forecast[:, 0]

class EnsembleRegionalWrapper:
    """Weighted ensemble of Ridge and XGBoost."""
    def __init__(self, target_prefix=None, **kwargs):
        from sklearn.linear_model import Ridge
        import xgboost as xgb
        self.m_linear = Ridge(alpha=1.0)
        self.m_xgb = xgb.XGBRegressor(
            n_estimators=200, 
            max_depth=4, 
            learning_rate=0.1, 
            random_state=42,
            objective='reg:absoluteerror'
        )
        self.weights = [0.5, 0.5]  # Equal weights for Linear and XGBoost

    def fit(self, X, y):
        X_imp = impute_missing_values(X)
        self.m_linear.fit(X_imp, y)
        self.m_xgb.fit(X_imp, y)
        return self

    def predict(self, X):
        X_imp = impute_missing_values(X)
        p_linear = np.maximum(0, self.m_linear.predict(X_imp))
        p_xgb = np.maximum(0, self.m_xgb.predict(X_imp))
        return self.weights[0] * p_linear + self.weights[1] * p_xgb

def run_model_comparison(model_class, model_name, data_file, output_dir, **model_kwargs):
    """Standardized loop to compare Single vs Multi features for a given model class."""
    print(f"\n--- Running Optimized Comparison for {model_name.upper()} ---")
    
    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Load preprocessed wide-format data (all models use the same preprocessed file)
    wide_df = load_wide_data(data_file)
    indicators = get_base_indicators_from_wide(wide_df)
    
    comparisons = []
    final_forecasts = []
    plot_data_dict = {}

    for indicator in indicators:
        train_eval_df, forecast_df, col_info = prepare_indicator_dataset(wide_df, indicator)
        if train_eval_df is None or len(train_eval_df) < 5:
            continue
            
        target_col = col_info['target']
        single_cols = col_info['single']
        multi_cols = col_info['multi']
        
        train_idx, test_idx = train_test_split(train_eval_df.index, test_size=0.2, random_state=42)
        
        # Instantiate model. Only pass target_prefix if the class accepts it.
        # Check if it is one of our custom wrappers
        if model_class in [TimesFMRegionalWrapper, EnsembleRegionalWrapper]:
            m_s = model_class(target_prefix=indicator, **model_kwargs)
            m_m = model_class(target_prefix=indicator, **model_kwargs)
        else:
            m_s = model_class(**model_kwargs)
            m_m = model_class(**model_kwargs)
        
        # 1. Single-Feature
        X_s_train = impute_missing_values(train_eval_df.loc[train_idx, single_cols])
        y_s_train = train_eval_df.loc[train_idx, target_col]
        X_s_test = impute_missing_values(train_eval_df.loc[test_idx, single_cols])
        y_s_test = train_eval_df.loc[test_idx, target_col]
        
        m_s.fit(X_s_train, y_s_train)
        pred_s = np.maximum(0, m_s.predict(X_s_test))
        
        # 2. Multi-Feature
        X_m_train = impute_missing_values(train_eval_df.loc[train_idx, multi_cols])
        y_m_train = train_eval_df.loc[train_idx, target_col]
        X_m_test = impute_missing_values(train_eval_df.loc[test_idx, multi_cols])
        y_m_test = train_eval_df.loc[test_idx, target_col]
        
        m_m.fit(X_m_train, y_m_train)
        pred_m = np.maximum(0, m_m.predict(X_m_test))
        
        # Metrics
        mape_s = robust_mape(y_s_test, pred_s)
        mape_m = robust_mape(y_m_test, pred_m)
        
        if np.isnan(mape_s) or np.isnan(mape_m):
            continue

        better = "Multi" if mape_m < mape_s else "Single"
        
        comparisons.append({
            'indicator': indicator,
            'Multi_MAPE': mape_m * 100,
            'Single_MAPE': mape_s * 100,
            'Multi_RMSE': calc_rmse(y_m_test, pred_m),
            'Single_RMSE': calc_rmse(y_s_test, pred_s),
            'Multi_MAE': calc_mae(y_m_test, pred_m),
            'Single_MAE': calc_mae(y_s_test, pred_s),
            'Multi_R2': calc_r2(y_m_test, pred_m),
            'Single_R2': calc_r2(y_s_test, pred_s),
            'MAPE_Improvement': (mape_s - mape_m) * 100,
            'Better': better
        })
        
        # 3. Final Forecast
        if not forecast_df.empty:
            best_model = m_m if better == "Multi" else m_s
            best_cols = multi_cols if better == "Multi" else single_cols
            X_fc = impute_missing_values(forecast_df[best_cols])
            forecast_vals = np.maximum(0, best_model.predict(X_fc))
            
            for i, val in enumerate(forecast_vals):
                final_forecasts.append({
                    'geo': forecast_df.iloc[i]['geo'],
                    'indicator': indicator,
                    'forecast_2023': val,
                    'model_used': better
                })
        
        # 4. Store Plot Data
        if len(test_idx) > 0:
            sample_indices = test_idx[:5]
            for idx in sample_indices:
                region_row = train_eval_df.loc[idx]
                geo = region_row['geo']
                hist_years = [int(c.split('_')[-1]) for c in single_cols]
                hist_vals = [region_row[c] for c in single_cols]
                actual_2023 = region_row[target_col]
                
                best_model = m_m if better == "Multi" else m_s
                best_cols = multi_cols if better == "Multi" else single_cols
                X_p = region_row[best_cols].to_frame().T
                X_p_imp = impute_missing_values(X_p)
                pred_2023 = np.maximum(0, best_model.predict(X_p_imp))[0]
                
                if indicator not in plot_data_dict:
                    plot_data_dict[indicator] = []
                
                plot_data_dict[indicator].append({
                    'geo': geo, 'hist_years': hist_years, 'hist_vals': hist_vals,
                    'actual_2023': actual_2023, 'pred_2023': pred_2023, 'better': better
                })

    # Save CSVs
    comp_df = pd.DataFrame(comparisons)
    comp_df.to_csv(os.path.join(output_dir, 'indicator_comparison.csv'), index=False)
    if final_forecasts:
        pd.DataFrame(final_forecasts).to_csv(os.path.join(output_dir, 'forecasts_2023.csv'), index=False)
    
    # Plots
    print(f"Generating optimized plots for {model_name}...")
    for indicator, regions in plot_data_dict.items():
        if not regions: continue
        plt.figure(figsize=(15, 12))
        for i, reg in enumerate(regions):
            plt.subplot(len(regions), 1, i+1)
            plt.plot(reg['hist_years'], reg['hist_vals'], marker='o', label='History')
            if not pd.isna(reg['actual_2023']):
                plt.scatter([2023], [reg['actual_2023']], color='green', s=100, label='Actual 2023', zorder=5)
            plt.scatter([2023], [reg['pred_2023']], color='red', s=150, marker='X', label=f'Forecast 2023 ({reg["better"]})', zorder=6)
            plt.title(f"{indicator} - {reg['geo']} (Model: {model_name})")
            plt.grid(True, alpha=0.3)
            if i == 0: plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{indicator}_forecast.png"))
        plt.close()

    print(f"Optimized results for {model_name} saved to {output_dir}")
    return comp_df

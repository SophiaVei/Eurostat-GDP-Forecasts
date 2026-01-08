import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from data_processor import load_economy_data, pivot_to_wide, prepare_indicator_dataset, get_indicators
import logging
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# Suppress prophet logs
logging.getLogger('prophet').setLevel(logging.ERROR)
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)

def robust_mape(y_true, y_pred):
    """
    Calculates MAPE while handling zero values in y_true.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = np.abs(y_true) > 1e-3
    if np.sum(mask) == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))

class ProphetRegionalWrapper:
    """
    Fits a Prophet model for each region (row) independently.
    """
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.models = {}

    def fit(self, X, y):
        # We don't actually 'train' a global model. 
        # We just store X and y to fit individual series later during predict.
        # This wrapper is a bit of a hack to fit the scikit-learn API.
        self.X_train = X
        self.y_train = y
        return self

    def predict(self, X):
        from prophet import Prophet
        
        def fit_predict_single(idx):
            row = X.iloc[idx]
            data = []
            for col in X.columns:
                try:
                    year = int(col.split('_')[-1])
                    val = row[col]
                    data.append({'ds': f"{year}-01-01", 'y': val})
                except:
                    continue
            
            df_p = pd.DataFrame(data)
            if len(df_p) < 2:
                return 0
                
            m = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False, **self.kwargs)
            m.fit(df_p)
            future = m.make_future_dataframe(periods=1, freq='Y')
            forecast = m.predict(future)
            return forecast.iloc[-1]['yhat']

        # Use 4 parallel workers to avoid overwhelming CPU but speed up 
        results = Parallel(n_jobs=4)(delayed(fit_predict_single)(i) for i in range(len(X)))
        return np.array(results)

class TimesFMRegionalWrapper:
    """
    Fits TimesFM for each region independently.
    Note: TimesFM is a zero-shot model, but we use the 'fit' to store context if needed.
    """
    def __init__(self, **kwargs):
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
        # Zero-shot, nothing to train on training set specifically here for the region 
        # but the wrapper API expects this.
        return self

    def predict(self, X):
        if self.model is None:
            return np.zeros(len(X))
            
        inputs = []
        for idx in range(len(X)):
            row = X.iloc[idx]
            series = [row[col] for col in X.columns if not pd.isna(row[col])]
            inputs.append(np.array(series))
            
        if not inputs:
            return np.zeros(len(X))

        # TimesFM expects a list of arrays
        point_forecast, _ = self.model.forecast(
            horizon=1,
            inputs=inputs
        )
        
        # Extract forecast for 2023 (index 0 of horizon)
        return point_forecast[:, 0]

class EnsembleRegionalWrapper:
    """
    Averages predictions from Linear, XGBoost, and Prophet.
    """
    def __init__(self, **kwargs):
        from sklearn.linear_model import LinearRegression
        import xgboost as xgb
        self.m_linear = LinearRegression()
        self.m_xgb = xgb.XGBRegressor(n_estimators=100, objective='reg:absoluteerror', random_state=42)
        self.m_prophet = ProphetRegionalWrapper()

    def fit(self, X, y):
        self.m_linear.fit(X.fillna(0), y)
        self.m_xgb.fit(X, y)
        self.m_prophet.fit(X, y)
        return self

    def predict(self, X):
        p_linear = np.maximum(0, self.m_linear.predict(X.fillna(0)))
        p_xgb = np.maximum(0, self.m_xgb.predict(X))
        p_prophet = self.m_prophet.predict(X)
        
        return (p_linear + p_xgb + p_prophet) / 3.0

def run_model_comparison(model_class, model_name, data_file, output_dir, **model_kwargs):
    """
    Standardized loop to compare Single vs Multi features for a given model class.
    
    Args:
        model_class: A class that implements .fit(X, y) and .predict(X)
        model_name: Name of the model (e.g., 'linear', 'xgboost')
    """
    print(f"\n--- Running Comparison for {model_name.upper()} ---")
    
    # Setup directories
    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    df = load_economy_data(data_file)
    wide_df = pivot_to_wide(df)
    indicators = get_indicators(df)
    
    comparisons = []
    final_forecasts = []
    plot_data_dict = {} # indicator -> list of region PlotData

    for indicator in indicators:
        train_eval_df, forecast_df, col_info = prepare_indicator_dataset(wide_df, indicator)
        
        if train_eval_df is None or len(train_eval_df) < 5:
            continue
            
        target_col = col_info['target']
        single_cols = col_info['single']
        multi_cols = col_info['multi']
        
        # Split
        train_idx, test_idx = train_test_split(train_eval_df.index, test_size=0.2, random_state=42)
        
        # Single-Feature
        X_s_train = train_eval_df.loc[train_idx, single_cols]
        y_s_train = train_eval_df.loc[train_idx, target_col]
        X_s_test = train_eval_df.loc[test_idx, single_cols]
        y_s_test = train_eval_df.loc[test_idx, target_col]
        
        # Fill NaNs with 0 for models that don't handle them
        X_s_train = X_s_train.fillna(0)
        X_s_test = X_s_test.fillna(0)
        
        m_s = model_class(**model_kwargs)
        m_s.fit(X_s_train, y_s_train)
        pred_s = np.maximum(0, m_s.predict(X_s_test))
        mape_s = robust_mape(y_s_test, pred_s)
        
        # Multi-Feature
        X_m_train = train_eval_df.loc[train_idx, multi_cols]
        y_m_train = train_eval_df.loc[train_idx, target_col]
        X_m_test = train_eval_df.loc[test_idx, multi_cols]
        y_m_test = train_eval_df.loc[test_idx, target_col]
        
        X_m_train = X_m_train.fillna(0)
        X_m_test = X_m_test.fillna(0)
        
        m_m = model_class(**model_kwargs)
        m_m.fit(X_m_train, y_m_train)
        pred_m = np.maximum(0, m_m.predict(X_m_test))
        mape_m = robust_mape(y_m_test, pred_m)
        
        if np.isnan(mape_s) or np.isnan(mape_m):
            continue

        improvement = mape_s - mape_m
        better = "Multi" if mape_m < mape_s else "Single"
        
        comparisons.append({
            'indicator': indicator,
            'Multi-Feature': mape_m * 100,
            'Single-Feature': mape_s * 100,
            'Improvement': improvement * 100,
            'Better': better
        })
        
        # Final Forecast
        if not forecast_df.empty:
            forecast_df_clean = forecast_df.fillna(0)
            best_model = m_m if better == "Multi" else m_s
            best_cols = multi_cols if better == "Multi" else single_cols
            X_fc = forecast_df_clean[best_cols]
            forecast_vals = np.maximum(0, best_model.predict(X_fc))
            
            for i, val in enumerate(forecast_vals):
                final_forecasts.append({
                    'geo': forecast_df.iloc[i]['geo'],
                    'indicator': indicator,
                    'forecast_2023': val,
                    'model_used': better
                })
        
        # Store some data for plotting (e.g., first 5 regions in test set)
        if len(test_idx) > 0:
            sample_indices = test_idx[:5]
            for idx in sample_indices:
                region_row = train_eval_df.loc[idx]
                geo = region_row['geo']
                
                # Historic
                hist_years = [int(c.split('_')[-1]) for c in single_cols]
                hist_vals = [region_row[c] for c in single_cols]
                
                # Actual 2023
                actual_2023 = region_row[target_col]
                
                # Predict 2023 (using better model)
                best_model = m_m if better == "Multi" else m_s
                best_cols = multi_cols if better == "Multi" else single_cols
                X_p = region_row[best_cols].to_frame().T.fillna(0)
                pred_2023 = np.maximum(0, best_model.predict(X_p))[0]
                
                if indicator not in plot_data_dict:
                    plot_data_dict[indicator] = []
                
                plot_data_dict[indicator].append({
                    'geo': geo,
                    'hist_years': hist_years,
                    'hist_vals': hist_vals,
                    'actual_2023': actual_2023,
                    'pred_2023': pred_2023,
                    'better': better
                })

    # Save
    comp_df = pd.DataFrame(comparisons)
    comp_df.to_csv(os.path.join(output_dir, 'indicator_comparison.csv'), index=False)
    
    if final_forecasts:
        fc_df = pd.DataFrame(final_forecasts)
        fc_df.to_csv(os.path.join(output_dir, 'forecasts_2023.csv'), index=False)
    
    # Generate Plots
    print(f"Generating plots for {model_name}...")
    for indicator, regions in plot_data_dict.items():
        if not regions: continue
        
        plt.figure(figsize=(15, 10))
        num_regions = len(regions)
        for i, reg in enumerate(regions):
            plt.subplot(num_regions, 1, i+1)
            
            # 1. Historical Line
            plt.plot(reg['hist_years'], reg['hist_vals'], marker='o', label='History')
            
            # 2. Actual 2023 Dot (Green if available)
            if not pd.isna(reg['actual_2023']):
                plt.scatter([2023], [reg['actual_2023']], color='green', s=100, label='Actual 2023', zorder=5)
            
            # 3. Forecast 2023 Dot (Red)
            plt.scatter([2023], [reg['pred_2023']], color='red', s=150, marker='X', label=f'Forecast 2023 ({reg["better"]})', zorder=6)
            
            plt.title(f"{indicator} - {reg['geo']} (Model: {model_name})")
            plt.xticks(reg['hist_years'] + [2023])
            plt.grid(True, alpha=0.3)
            if i == 0: plt.legend()
            
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{indicator}_forecast.png"))
        plt.close()

    print(f"Results for {model_name} saved to {output_dir}")
    return comp_df

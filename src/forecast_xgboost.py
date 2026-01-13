import os
import xgboost as xgb
from forecast_utils import run_model_comparison

def main():
    DATA_FILE = r'c:\Users\Sofia\Eurostat-GDP-Forecasts\data\economy_nuts2_all_columns.csv'
    OUTPUT_DIR = r'c:\Users\Sofia\Eurostat-GDP-Forecasts\results\xgboost'
    
    # Advanced XGBoost parameters for Phase 2 optimization
    model_params = {
        'n_estimators': 500,        # Increased trees for better depth
        'max_depth': 4,
        'learning_rate': 0.05,     # Smaller learning rate for stability
        'gamma': 0.5,              # Minimum loss reduction for a split
        'reg_lambda': 1.5,         # L2 regularization
        'min_child_weight': 3,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'objective': 'reg:absoluteerror'
    }
    
    run_model_comparison(
        model_class=xgb.XGBRegressor,
        model_name='xgboost',
        data_file=DATA_FILE,
        output_dir=OUTPUT_DIR,
        **model_params
    )

if __name__ == "__main__":
    main()

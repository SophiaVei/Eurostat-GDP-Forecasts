import os
import xgboost as xgb
from forecast_utils import run_model_comparison

def main():
    DATA_FILE = r'c:\Users\Sofia\Eurostat-GDP-Forecasts\data\economy_nuts2_all_columns.csv'
    OUTPUT_DIR = r'c:\Users\Sofia\Eurostat-GDP-Forecasts\results\xgboost'
    
    # Optimized XGBoost parameters for annual economic data
    model_params = {
        'n_estimators': 200,
        'max_depth': 4,
        'learning_rate': 0.1,
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

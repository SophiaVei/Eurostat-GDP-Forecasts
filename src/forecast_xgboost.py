import os
import xgboost as xgb
from forecast_utils import run_model_comparison

def main():
    DATA_FILE = r'c:\Users\Sofia\Eurostat-GDP-Forecasts\data\economy_nuts2_all_columns.csv'
    OUTPUT_DIR = r'c:\Users\Sofia\Eurostat-GDP-Forecasts\results\xgboost'
    
    # XGBoost handles NaNs naturally, but we'll use robust settings
    model_params = {
        'n_estimators': 100,
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

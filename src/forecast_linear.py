import os
from sklearn.linear_model import Ridge
from forecast_utils import run_model_comparison

def main():
    DATA_FILE = r'c:\Users\Sofia\Eurostat-GDP-Forecasts\data\economy_nuts2_all_columns.csv'
    OUTPUT_DIR = r'c:\Users\Sofia\Eurostat-GDP-Forecasts\results\linear'
    
    # Using Ridge regression to handle many features (Multi-Feature) and prevent overfitting
    # alpha=1.0 provides moderate regularization
    run_model_comparison(
        model_class=Ridge,
        model_name='linear',
        data_file=DATA_FILE,
        output_dir=OUTPUT_DIR,
        alpha=1.0
    )

if __name__ == "__main__":
    main()

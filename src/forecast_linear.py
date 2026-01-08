import os
from sklearn.linear_model import LinearRegression
from forecast_utils import run_model_comparison

def main():
    DATA_FILE = r'c:\Users\Sofia\Eurostat-GDP-Forecasts\data\economy_nuts2_all_columns.csv'
    OUTPUT_DIR = r'c:\Users\Sofia\Eurostat-GDP-Forecasts\results\linear'
    
    run_model_comparison(
        model_class=LinearRegression,
        model_name='linear',
        data_file=DATA_FILE,
        output_dir=OUTPUT_DIR
    )

if __name__ == "__main__":
    main()

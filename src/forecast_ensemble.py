import os
from forecast_utils import run_model_comparison, EnsembleRegionalWrapper

def main():
    DATA_FILE = r'c:\Users\Sofia\Eurostat-GDP-Forecasts\data\economy_nuts2_all_columns.csv'
    OUTPUT_DIR = r'c:\Users\Sofia\Eurostat-GDP-Forecasts\results\ensemble'
    
    run_model_comparison(
        model_class=EnsembleRegionalWrapper,
        model_name='ensemble',
        data_file=DATA_FILE,
        output_dir=OUTPUT_DIR
    )

if __name__ == "__main__":
    main()

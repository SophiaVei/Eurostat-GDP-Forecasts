import pandas as pd
import os
import glob

def consolidate_results():
    results_dir = 'results'
    comparison_files = glob.glob(os.path.join(results_dir, '*', 'indicator_comparison.csv'))
    
    all_data = []
    
    for f in comparison_files:
        model_name = os.path.basename(os.path.dirname(f))
        df = pd.read_csv(f)
        
        # Some models use 'Multi-Feature' and 'Single-Feature' columns, 
        # others (TimesFM) might be different.
        if 'Multi-Feature' in df.columns:
            # We take the best one 
            df['Best_MAPE'] = df[['Multi-Feature', 'Single-Feature']].min(axis=1)
            df['Model'] = model_name
            all_data.append(df[['indicator', 'Best_MAPE', 'Model']])
        elif 'MAPE' in df.columns:
            df['Best_MAPE'] = df['MAPE']
            df['Model'] = model_name
            all_data.append(df[['indicator', 'Best_MAPE', 'Model']])

    if not all_data:
        print("No data found to consolidate.")
        return

    master_df = pd.concat(all_data)
    
    # Pivot to see models side-by-side
    pivot_df = master_df.pivot(index='indicator', columns='Model', values='Best_MAPE')
    
    # Ensure numeric for calculations
    numeric_df = pivot_df.apply(pd.to_numeric, errors='coerce')
    
    pivot_df['Winner'] = numeric_df.idxmin(axis=1)
    pivot_df['Min_MAPE'] = numeric_df.min(axis=1)
    
    master_comp_path = os.path.join(results_dir, 'master_comparison.csv')
    pivot_df.to_csv(master_comp_path)
    print(f"Master comparison saved to {master_comp_path}")
    print("\n--- PERFORMANCE SUMMARY (MAPE %) ---")
    print(pivot_df)

if __name__ == "__main__":
    consolidate_results()

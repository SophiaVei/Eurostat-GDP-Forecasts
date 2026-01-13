import pandas as pd
import os
import glob

def consolidate_results():
    # Use absolute path based on script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    results_dir = os.path.join(project_dir, 'results')
    comparison_files = glob.glob(os.path.join(results_dir, '*', 'indicator_comparison.csv'))
    
    all_data = []
    
    for f in comparison_files:
        model_name = os.path.basename(os.path.dirname(f))
        df = pd.read_csv(f)
        
        # Handle new column naming (Multi_MAPE, Single_MAPE) and legacy (Multi-Feature, Single-Feature)
        if 'Multi_MAPE' in df.columns:
            df['Best_MAPE'] = df[['Multi_MAPE', 'Single_MAPE']].min(axis=1)
            df['Best_RMSE'] = df.apply(lambda r: r['Multi_RMSE'] if r['Better'] == 'Multi' else r['Single_RMSE'], axis=1)
            df['Best_MAE'] = df.apply(lambda r: r['Multi_MAE'] if r['Better'] == 'Multi' else r['Single_MAE'], axis=1)
            df['Best_R2'] = df.apply(lambda r: r['Multi_R2'] if r['Better'] == 'Multi' else r['Single_R2'], axis=1)
            df['Model'] = model_name
            all_data.append(df[['indicator', 'Best_MAPE', 'Best_RMSE', 'Best_MAE', 'Best_R2', 'Better', 'Model']])
        elif 'Multi-Feature' in df.columns:
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

    master_df = pd.concat(all_data, ignore_index=True)
    
    # Pivot MAPE to see models side-by-side
    pivot_mape = master_df.pivot(index='indicator', columns='Model', values='Best_MAPE')
    
    # Ensure numeric for calculations
    numeric_df = pivot_mape.apply(pd.to_numeric, errors='coerce')
    
    pivot_mape['Winner'] = numeric_df.idxmin(axis=1)
    pivot_mape['Min_MAPE'] = numeric_df.min(axis=1)
    
    master_comp_path = os.path.join(results_dir, 'master_comparison.csv')
    pivot_mape.to_csv(master_comp_path)
    print(f"Master MAPE comparison saved to {master_comp_path}")
    print("\n--- PERFORMANCE SUMMARY (MAPE %) ---")
    print(pivot_mape)
    
    # Also save detailed metrics if available
    if 'Best_RMSE' in master_df.columns:
        detailed_path = os.path.join(results_dir, 'detailed_metrics.csv')
        master_df.to_csv(detailed_path, index=False)
        print(f"\nDetailed metrics saved to {detailed_path}")

if __name__ == "__main__":
    consolidate_results()

import pandas as pd
import os
import glob

def consolidate_results():
    # Use absolute path based on script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    results_dir = os.path.join(project_dir, 'results')
    # Results structure is now: results/domain/model/indicator_comparison.csv
    # We want to consolidate per domain.

    # Get all domain directories in results/
    domain_dirs = [d for d in glob.glob(os.path.join(results_dir, '*')) if os.path.isdir(d)]

    for domain_path in domain_dirs:
        domain_name = os.path.basename(domain_path)
        print(f"\nProcessing domain: {domain_name}")

        comparison_files = glob.glob(os.path.join(domain_path, '*', 'indicator_comparison.csv'))
        
        if not comparison_files:
            print(f"  No comparison files found in {domain_path}")
            continue

        all_data = []

        for f in comparison_files:
            model_name = os.path.basename(os.path.dirname(f))
            try:
                df = pd.read_csv(f)
            except Exception as e:
                print(f"  Error reading {f}: {e}")
                continue
            
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
            print(f"  No valid data extracted for {domain_name}.")
            continue

        master_df = pd.concat(all_data, ignore_index=True)
        
        # Pivot MAPE to see models side-by-side
        pivot_mape = master_df.pivot(index='indicator', columns='Model', values='Best_MAPE')
        
        # Ensure numeric for calculations
        numeric_df = pivot_mape.apply(pd.to_numeric, errors='coerce')
        
        pivot_mape['Winner'] = numeric_df.idxmin(axis=1)
        pivot_mape['Min_MAPE'] = numeric_df.min(axis=1)
        
        master_comp_path = os.path.join(domain_path, 'master_comparison.csv')
        pivot_mape.to_csv(master_comp_path)
        print(f"  Master MAPE comparison saved to {master_comp_path}")
        # print(pivot_mape) # optional: reduce clutter
        
        # Also save detailed metrics if available
        if 'Best_RMSE' in master_df.columns:
            detailed_path = os.path.join(domain_path, 'detailed_metrics.csv')
            master_df.to_csv(detailed_path, index=False)
            print(f"  Detailed metrics saved to {detailed_path}")

if __name__ == "__main__":
    consolidate_results()

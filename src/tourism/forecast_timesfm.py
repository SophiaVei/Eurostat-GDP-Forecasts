import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Try to import timesfm, but handle import errors gracefully
try:
    import timesfm
    TIMESFM_AVAILABLE = True
except (ImportError, OSError) as e:
    print(f"Warning: TimesFM not available ({type(e).__name__}: {e})")
    print("TimesFM requires PyTorch. Install PyTorch or use other models (linear, xgboost, ensemble).")
    TIMESFM_AVAILABLE = False
    timesfm = None

# Ensure parent src directory is on sys.path so we can import shared modules
CURRENT_DIR = Path(__file__).resolve().parent
PARENT_SRC = CURRENT_DIR.parent
if str(PARENT_SRC) not in sys.path:
    sys.path.insert(0, str(PARENT_SRC))

from data_processor import (  # type: ignore  # noqa: E402
    load_wide_data,
    prepare_indicator_dataset,
    get_base_indicators_from_wide,
    impute_missing_values,
    preprocess_to_wide_file,
)
from download_data import download_tourism_default  # type: ignore  # noqa: E402


def robust_mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = np.abs(y_true) > 1e-3
    if np.sum(mask) == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))


def calc_rmse(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def calc_mae(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred))


def calc_r2(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return np.nan
    return 1 - (ss_res / ss_tot)


def main():
    # Resolve paths relative to repo root
    script_dir = Path(__file__).resolve().parent          # .../src/tourism
    repo_dir = script_dir.parent.parent                   # repo root

    raw_file = repo_dir / "data" / "tourism_nuts2_all_columns.csv"
    wide_file = repo_dir / "data" / "tourism_nuts2_wide.csv"
    output_dir = repo_dir / "results" / "tourism" / "timesfm"

    # Ensure raw data exists
    if not raw_file.exists():
        print("Raw tourism data not found, downloading from /tourism endpoint...")
        download_tourism_default()

    # Ensure preprocessed wide file exists
    if not wide_file.exists():
        print(f"Preprocessing raw data to wide format: {wide_file}")
        preprocess_to_wide_file(str(raw_file), str(wide_file))

    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    if not TIMESFM_AVAILABLE:
        print("TimesFM not available, skipping.")
        return

    print("Loading TimesFM...")
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch"
    )
    model.compile(
        timesfm.ForecastConfig(
            max_context=128,
            max_horizon=1,
            normalize_inputs=True,
            use_continuous_quantile_head=True,
        )
    )

    # Load preprocessed wide-format data (same as all other models)
    wide_df = load_wide_data(str(wide_file))
    indicators = get_base_indicators_from_wide(wide_df)

    comparisons = []
    final_forecasts = []
    plot_data_dict = {}

    for indicator in indicators:
        train_eval_df, forecast_df, col_info, target_year = prepare_indicator_dataset(
            wide_df, indicator
        )
        if train_eval_df is None or len(train_eval_df) < 5:
            continue

        target_col = col_info["target"]
        single_cols = col_info["single"]

        # Consistent Split: 80/20 with random_state=42
        train_idx, test_idx = train_test_split(
            train_eval_df.index, test_size=0.2, random_state=42
        )

        # Apply shared imputation to single-feature columns
        X_single_imputed = impute_missing_values(train_eval_df[single_cols])

        # 1. Single-Feature Evaluation (Univariate)
        # Use imputed values to build time series for TimesFM
        inputs_s = []
        valid_idx_s = []
        for idx in test_idx:
            # Extract imputed series (all values should be non-NaN after imputation)
            series = X_single_imputed.loc[idx, single_cols].values.tolist()
            if len(series) > 0:
                inputs_s.append(np.array(series))
                valid_idx_s.append(idx)

        if not inputs_s:
            continue

        fc_s, _ = model.forecast(horizon=1, inputs=inputs_s)
        preds_s = np.maximum(0, fc_s[:, 0])
        y_true_s = train_eval_df.loc[valid_idx_s, target_col].values
        mape_s = robust_mape(y_true_s, preds_s)
        rmse_s = calc_rmse(y_true_s, preds_s)
        mae_s = calc_mae(y_true_s, preds_s)
        r2_s = calc_r2(y_true_s, preds_s)

        # 2. Multi-Feature: TimesFM is univariate, so Multi = Single
        mape_m = mape_s
        rmse_m = rmse_s
        mae_m = mae_s
        r2_m = r2_s

        if np.isnan(mape_s):
            continue

        improvement = 0
        better = "Single"

        comparisons.append(
            {
                "indicator": indicator,
                "Multi_MAPE": mape_m * 100,
                "Single_MAPE": mape_s * 100,
                "Multi_RMSE": rmse_m,
                "Single_RMSE": rmse_s,
                "Multi_MAE": mae_m,
                "Single_MAE": mae_s,
                "Multi_R2": r2_m,
                "Single_R2": r2_s,
                "MAPE_Improvement": improvement,
                "Better": better,
            }
        )

        # Forecast missing
        if not forecast_df.empty:
            # Apply shared imputation to forecast data
            X_fc_imputed = impute_missing_values(forecast_df[single_cols])

            fc_inputs = []
            valid_fc_geos = []
            for idx, row in forecast_df.iterrows():
                # Extract imputed series
                series = X_fc_imputed.loc[idx, single_cols].values.tolist()
                if len(series) > 0:
                    fc_inputs.append(np.array(series))
                    valid_fc_geos.append(row["geo"])

            if fc_inputs:
                fc_point, _ = model.forecast(horizon=1, inputs=fc_inputs)
                fc_preds = np.maximum(0, fc_point[:, 0])
                for i, val in enumerate(fc_preds):
                    final_forecasts.append(
                        {
                            "geo": valid_fc_geos[i],
                            "indicator": indicator,
                            "forecast_year": target_year,
                            "forecast_value": val,
                            "model_used": "Single (Univariate)",
                        }
                    )

        # Plots: Top 5 from test set
        num_to_plot = min(5, len(valid_idx_s))
        sample_indices = valid_idx_s[:num_to_plot]
        for i, idx in enumerate(sample_indices):
            row = train_eval_df.loc[idx]
            hist_years = [int(c.split("_")[-1]) for c in single_cols]
            # Use imputed values for plotting (consistent with what TimesFM sees)
            hist_vals = X_single_imputed.loc[idx, single_cols].values.tolist()
            pred_val = preds_s[i]
            actual_val = row[target_col]

            if indicator not in plot_data_dict:
                plot_data_dict[indicator] = []
            plot_data_dict[indicator].append(
                {
                    "geo": row["geo"],
                    "hist_years": hist_years,
                    "hist_vals": hist_vals,
                    "actual_year": target_year,
                    "actual_val": actual_val,
                    "pred_val": pred_val,
                    "better": "Univariate",
                }
            )

    # Save
    pd.DataFrame(comparisons).to_csv(
        os.path.join(output_dir, "indicator_comparison.csv"), index=False
    )
    if final_forecasts:
        # Use target_year from the loop (note: assumes mostly same year, but file might be mixed if indicators vary)
        # We'll just name it based on the last target_year or generic
        fname = f"forecasts_{target_year}.csv" if target_year else "forecasts.csv"
        pd.DataFrame(final_forecasts).to_csv(
            os.path.join(output_dir, fname), index=False
        )

    # Generate Plots
    print(f"Generating plots for TimesFM (tourism)...")
    for indicator, regions in plot_data_dict.items():
        plt.figure(figsize=(15, 10))
        for i, reg in enumerate(regions):
            t_year = reg.get('actual_year', 2023)
            plt.subplot(len(regions), 1, i + 1)
            plt.plot(reg["hist_years"], reg["hist_vals"], marker="o", label="History")
            if not pd.isna(reg["actual_val"]):
                plt.scatter(
                    [t_year],
                    [reg["actual_val"]],
                    color="green",
                    s=100,
                    label=f"Actual {t_year}",
                )
            plt.scatter(
                [t_year],
                [reg["pred_val"]],
                color="red",
                marker="X",
                s=150,
                label=f"TimesFM {t_year}",
            )
            plt.title(f"{indicator} - {reg['geo']} (Model: timesfm, tourism)")
            plt.xticks(reg["hist_years"] + [t_year])
            plt.grid(True, alpha=0.3)
            if i == 0:
                plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{indicator}_forecast.png"))
        plt.close()

    print(f"TimesFM results for tourism saved to {output_dir}")


if __name__ == "__main__":
    main()

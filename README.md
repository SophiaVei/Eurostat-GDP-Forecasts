# Eurostat GDP & Economic Indicator Forecasts (2023)

This project provides a comprehensive cross-regional forecasting framework to predict **2023** values for 18 economic indicators across European NUTS 2 regions.

## Dataset: `economy_nuts2_all_columns.csv`
The raw dataset contains annual economic data for regions across Europe.
- **Time Range**: 2008 – 2023 (Target year is 2023).
- **Indicators**: 18 variables including GDP (Mio EUR), GDP per Inhabitant, Gross Fixed Capital Formation (GFCF), and sectoral Gross Value Added (GVA).
- **Regions**: Approximately 400 NUTS 2 regions.

### Preprocessing
We transform the long-format data into a **Wide-Format** (Cross-Sectional) structure:
- Each row represents a **Region** (`geo`).
- Columns are flattened to `[indicator]_[year]` (e.g., `gdp_mio_eur_2008`, `gdp_mio_eur_2022`).
- Missing values in historical features are handled by filling with `0` or through model-specific imputation.

## Methodology: Cross-Regional Strategy
Unlike standard per-series time-series forecasting, we train a single model instance across **all regions** to capture cross-regional patterns and dependencies.

### Train / Test / Evaluation Split
1. **Target**: Predict the indicator value for the year **2023**.
2. **Evaluation Set**: Regions that ALREADY have 2023 data are split **80% Training / 20% Evaluation** using a fixed `random_state=42`.
3. **Forecast Set**: Regions with missing 2023 data are used as the final application set.
4. **Logic**: For each indicator, we compare:
   - **Single-Feature**: Only the history of the target indicator (2008-2022).
   - **Multi-Feature**: History of the target + history of all other 17 indicators.

## Models Evaluated
- **Linear Regression**: Standard multivariate regression.
- **XGBoost**: Gradient boosted trees optimized for tabular patterns.
- **Ensemble**: A weighted average of Linear and XGBoost predictions.
- **TimesFM**: Google's 2.5B parameter zero-shot time-series foundation model.

## Results Structure
All results are organized in the `results/` directory by model:
- `results/[model]/indicator_comparison.csv`: MAPE comparison for Single vs Multi features.
- `results/[model]/forecasts_2023.csv`: Predictions for missing 2023 regional data.
- `results/[model]/plots/`: Visualizations showing history (line), actual 2023 (green dot), and forecast 2023 (red X).

A master comparative summary is available in `results/master_comparison.csv`.

## Performance Overview (Best Models)
- **Ensemble** outperforms others on total GDP volume (`gdp_mio_eur`).
- **Linear Regression** shows high stability for normalized metrics like `gdp_eur_hab`.
- **XGBoost** captures complex sectoral dependencies in GVA metrics.

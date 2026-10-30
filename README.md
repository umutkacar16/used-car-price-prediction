# 🚗 Second-Hand Car Price Prediction & Machine Learning Pipeline

This project is an end-to-end machine learning pipeline that accurately predicts second-hand market prices ($TL$) for vehicles using technical and historical usage specifications (model year, mileage, engine capacity, cosmetic body condition, insurance damage records, etc.).

---

## 📌 Repository Directory Structure

```text
ikinci-el-arac-fiyat-tahmini/
│
├── data/
│   └── used_car_price_dataset.csv     # Synthetic non-linear market dataset
│
├── reports/
│   └── figures/                      # Auto-generated analytical plots & charts
│
├── src/
│   ├── __init__.py
│   ├── data_generator.py             # Realistic non-linear synthetic data engine
│   ├── eda.py                        # Exploratory Data Analysis module (5 Plots)
│   ├── preprocessing.py              # Imputation, Feature Engineering & Scaling
│   └── train.py                      # Multi-model training, diagnostics & evaluation
│
├── main.py                           # Central execution script for full pipeline
├── requirements.txt                  # Python package dependencies
├── .gitignore                        # Git exclusion file
└── README.md                         # Project documentation

## 📊 Model Performance Comparison

Three distinct regression algorithms were trained and evaluated on realistic non-linear pricing dynamics:

| Model | MAE (TL) | RMSE (TL) | $R^2$ Score |
| :--- | :--- | :--- | :--- |
| 🏆 **XGBoost Regressor** | **38,643.40 TL** | **58,846.35 TL** | **0.9718** |
| 🥈 **Random Forest Regressor** | 46,461.94 TL | 77,638.21 TL | 0.9510 |
| 🥉 **Linear Regression** | 110,789.18 TL | 137,349.27 TL | 0.8466 |

* **Best Performing Model:** **XGBoost Regressor** achieved the highest accuracy, explaining **97.18%** ($R^2 = 0.9718$) of the variance in vehicle market prices.

📈 Key Figures & Diagnostic Artifacts
Running the pipeline automatically generates publication-grade visualizations saved under reports/figures/:

01_price_distribution.png: Right-skewed vehicle market price distribution.

02_mileage_vs_price.png: Non-linear price depreciation curve relative to mileage.

03_brand_price_boxplot.png: Market price variance across luxury vs. standard brands.

04_cosmetics_impact_bar.png: Financial impact analysis of painted and replaced body parts.

05_correlation_heatmap.png: Feature cross-correlation matrix.

06_scaling_effect_comparison.png: Preprocessing visual comparison (Raw vs. StandardScaler).

07_model_residuals.png: Residual error distribution analysis evaluating model bias.

08_feature_importance.png: Top 10 most influential features driving vehicle valuation.
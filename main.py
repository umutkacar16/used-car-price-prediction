"""Main Execution Pipeline for Second-Hand Car Price Prediction.

Integrates data generation, exploratory data analysis (EDA),
preprocessing, model training, evaluation, and visualization.
"""

import os
import pandas as pd
from src.data_generator import CarDataGenerator
from src.eda import ExploratoryDataAnalysis
from src.preprocessing import DataPreprocessor
from src.train import ModelTrainer


def main():
    """Executes the complete machine learning workflow end-to-end."""
    print("🚀 Starting Second-Hand Car Price Prediction Pipeline...\n")

    # -------------------------------------------------------------------------
    # STEP 1: DATA GENERATION & LOADING
    # -------------------------------------------------------------------------
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, "used_car_price_dataset.csv")

    # Force regenerate to ensure updated non-linear dataset is active
    print("📦 Generating updated synthetic second-hand car dataset...")
    generator = CarDataGenerator(n_samples=5000, random_state=42)
    df = generator.generate()
    df.to_csv(file_path, index=False)
    print(f"✅ Enhanced dataset created and saved to '{file_path}'!\n")

    # -------------------------------------------------------------------------
    # STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
    # -------------------------------------------------------------------------
    print("📊 Stage 1: Running Exploratory Data Analysis (EDA)...")
    eda = ExploratoryDataAnalysis(df=df)
    eda.run_full_eda()

    # -------------------------------------------------------------------------
    # STEP 3: DATA PREPROCESSING & FEATURE ENGINEERING
    # -------------------------------------------------------------------------
    print("\n⚙️ Stage 2: Executing Data Preprocessing & Feature Engineering Pipeline...")
    preprocessor = DataPreprocessor(df=df, target_col="price_tl")
    X_train, X_test, y_train, y_test = preprocessor.prepare_data()

    print(f"✅ Data Preprocessing Completed Successfully!")
    print(f"   - Training Features Shape : {X_train.shape}")
    print(f"   - Testing Features Shape  : {X_test.shape}")

    # -------------------------------------------------------------------------
    # STEP 4: MODEL TRAINING & EVALUATION
    # -------------------------------------------------------------------------
    print("\n🤖 Stage 3: Training & Evaluating Machine Learning Models...")
    trainer = ModelTrainer()
    results_df = trainer.train_and_evaluate(X_train, X_test, y_train, y_test)

    print("\n=================== MODEL EVALUATION METRICS ===================")
    print(results_df.to_string(index=False))
    print("=================================================================")

    # -------------------------------------------------------------------------
    # STEP 5: VISUALIZATIONS & DIAGNOSTICS
    # -------------------------------------------------------------------------
    print("\n📈 Stage 4: Generating Prediction & Diagnostic Plots...")
    trainer.generate_all_plots(y_test=y_test, feature_names=list(X_train.columns))

    print("\n✨ Complete Machine Learning Pipeline Executed Successfully!")


if __name__ == "__main__":
    main()
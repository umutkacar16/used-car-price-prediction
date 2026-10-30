import os
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


class ModelTrainer:
    """Class responsible for model training, evaluation, residual diagnostics,

    and feature importance analysis across multiple regression algorithms.
    """

    def __init__(self, output_dir: str = 'reports/figures'):
        """Initializes trainer with candidate regression models and output directory.

        Args:
            output_dir (str): Directory where model evaluation charts will be
              saved.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Candidate regression models pipeline
        self.models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(
                n_estimators=150, max_depth=12, random_state=42, n_jobs=-1
            ),
            'XGBoost': XGBRegressor(
                n_estimators=150,
                learning_rate=0.08,
                max_depth=6,
                random_state=42,
                n_jobs=-1,
            ),
        }

        self.results = {}
        self.best_model_name = None
        self.best_model = None

    def train_and_evaluate(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> pd.DataFrame:
        """Trains candidate regression algorithms and evaluates model metrics.

        Args:
            X_train (pd.DataFrame): Scaled training features.
            X_test (pd.DataFrame): Scaled testing features.
            y_train (pd.Series): Training target values.
            y_test (pd.Series): Testing target values.

        Returns:
            pd.DataFrame: Performance summary table (MAE, RMSE, R2).
        """
        evaluation_data = []

        for name, model in self.models.items():
            # 1. Model Training
            model.fit(X_train, y_train)

            # 2. Prediction on Unseen Test Set
            y_pred = model.predict(X_test)

            # 3. Evaluation Metric Computation
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)

            # Save detailed prediction dictionary
            self.results[name] = {
                'model': model,
                'y_pred': y_pred,
                'residuals': y_test - y_pred,
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
            }

            evaluation_data.append(
                {
                    'Model': name,
                    'MAE (TL)': round(mae, 2),
                    'RMSE (TL)': round(rmse, 2),
                    'R2 Score': round(r2, 4),
                }
            )

        # 4. Determine Best Performing Model based on R2 Score
        best_row = max(evaluation_data, key=lambda x: x['R2 Score'])
        self.best_model_name = best_row['Model']
        self.best_model = self.results[self.best_model_name]['model']

        results_df = pd.DataFrame(evaluation_data).sort_values(
            by='R2 Score', ascending=False
        )
        return results_df

    def plot_model_predictions(
        self,
        y_test: pd.Series,
        output_filename: str = '04_model_predictions.png',
    ) -> str:
        """Plots Actual vs Predicted price scatter plot with ideal prediction line."""
        best_pred = self.results[self.best_model_name]['y_pred']

        plt.figure(figsize=(9, 6))
        plt.scatter(
            y_test,
            best_pred,
            alpha=0.45,
            color='#2b5c8f',
            edgecolors='k',
            linewidths=0.5,
            s=40,
        )

        # Perfect prediction reference line y = x
        min_val = min(y_test.min(), best_pred.min())
        max_val = max(y_test.max(), best_pred.max())
        plt.plot(
            [min_val, max_val],
            [min_val, max_val],
            'r--',
            lw=2,
            label='Perfect Prediction (y = x)',
        )

        plt.title(
            f'Actual vs. Predicted Prices ({self.best_model_name})',
            fontsize=12,
            fontweight='bold',
            pad=12,
        )
        plt.xlabel('Actual Prices (TL)', fontsize=10)
        plt.ylabel('Predicted Prices (TL)', fontsize=10)
        plt.legend(loc='upper left')

        filepath = os.path.join(self.output_dir, output_filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300)
        plt.close()
        return filepath

    def plot_residuals_distribution(
        self, output_filename: str = '07_model_residuals.png'
    ) -> str:
        """Plots residual/error distribution to evaluate prediction bias and normality."""
        residuals = self.results[self.best_model_name]['residuals']

        plt.figure(figsize=(9, 5))
        sns.histplot(residuals, kde=True, color='#d9534f', bins=30)
        plt.axvline(
            0,
            color='black',
            linestyle='--',
            linewidth=1.5,
            label='Zero Error Baseline',
        )

        plt.title(
            f'Residual Error Distribution ({self.best_model_name})',
            fontsize=12,
            fontweight='bold',
            pad=12,
        )
        plt.xlabel('Prediction Error / Residuals (TL)', fontsize=10)
        plt.ylabel('Frequency', fontsize=10)
        plt.legend()

        filepath = os.path.join(self.output_dir, output_filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300)
        plt.close()
        return filepath

    def plot_feature_importance(
        self, feature_names: list, output_filename: str = '08_feature_importance.png'
    ) -> str:
        """Visualizes feature importances if the best model supports attribute scoring."""
        if not hasattr(self.best_model, 'feature_importances_'):
            print(
                f"ℹ️ Model '{self.best_model_name}' does not support feature importances."
            )
            return ''

        importances = self.best_model.feature_importances_
        feature_df = (
            pd.DataFrame(
                {'Feature': feature_names, 'Importance': importances}
            )
            .sort_values(by='Importance', ascending=False)
            .head(10)
        )

        plt.figure(figsize=(10, 5.5))
        sns.barplot(
            data=feature_df,
            x='Importance',
            y='Feature',
            palette='viridis',
            edgecolor='none',
        )

        plt.title(
            f'Top 10 Most Influential Features ({self.best_model_name})',
            fontsize=12,
            fontweight='bold',
            pad=12,
        )
        plt.xlabel('Relative Feature Importance Score', fontsize=10)
        plt.ylabel('Feature Name', fontsize=10)

        filepath = os.path.join(self.output_dir, output_filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300)
        plt.close()
        return filepath

    def generate_all_plots(self, y_test: pd.Series, feature_names: list):
        """Executes all visualization methods for the trained pipeline."""
        p1 = self.plot_model_predictions(y_test)
        p2 = self.plot_residuals_distribution()
        p3 = self.plot_feature_importance(feature_names)

        print(f"✅ Prediction Scatter Plot saved to: '{p1}'")
        print(f"✅ Residuals Error Distribution Plot saved to: '{p2}'")
        if p3:
            print(f"✅ Feature Importance Plot saved to: '{p3}'")
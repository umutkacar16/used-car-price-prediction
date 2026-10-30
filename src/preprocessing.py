import os
from typing import Tuple
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataPreprocessor:
    """Class responsible for comprehensive data preprocessing, feature engineering,

    missing value imputation, and feature scaling.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target_col: str = 'price_tl',
        output_dir: str = 'reports/figures',
    ):
        """Initializes preprocessor with target dataset and output directories.

        Args:
            df (pd.DataFrame): Raw vehicle dataset.
            target_col (str): Target regression column name. Defaults to
              'price_tl'.
            output_dir (str): Directory where preprocessing figures will be
              saved.
        """
        self.df = df.copy()
        self.target_col = target_col
        self.output_dir = output_dir
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        os.makedirs(self.output_dir, exist_ok=True)

    def handle_missing_values(self) -> pd.DataFrame:
        """Fills missing values (NaN) using median imputation for robust handling of skewed distributions."""
        if self.df['damage_record_tl'].isna().sum() > 0:
            self.df['damage_record_tl'] = self.imputer.fit_transform(
                self.df[['damage_record_tl']]
            )
        return self.df

    def engineer_features(self) -> pd.DataFrame:
        """Derives new logical domain features from raw vehicle specifications."""
        # 1. Vehicle age derivation
        self.df['vehicle_age'] = 2026 - self.df['model_year']

        # 2. Annual average mileage derivation
        self.df['annual_avg_km'] = self.df['mileage_km'] / (
            self.df['vehicle_age'] + 1
        )

        # 3. Total cosmetic defects combination (Painted + Changed parts)
        self.df['total_cosmetic_issues'] = (
            self.df['painted_parts'] + self.df['changed_parts']
        )

        return self.df

    def encode_categorical_features(self) -> pd.DataFrame:
        """Converts categorical columns ('brand', 'fuel_type', 'transmission') into dummy variables."""
        categorical_cols = ['brand', 'fuel_type', 'transmission']
        self.df = pd.get_dummies(
            self.df, columns=categorical_cols, drop_first=True
        )
        return self.df

    def plot_preprocessing_summary(
        self,
        raw_series: pd.Series,
        scaled_series: pd.Series,
        feature_name: str = 'mileage_km',
    ) -> str:
        """Generates a simple and intuitive 2-panel comparison plot showing scaling effect (Before vs After).

        Args:
            raw_series (pd.Series): Feature values before scaling.
            scaled_series (pd.Series): Feature values after StandardScaler.
            feature_name (str): Name of feature being visualized.

        Returns:
            str: Path to generated plot image.
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

        # Panel 1: Original Distribution
        sns.histplot(
            raw_series, ax=axes[0], color='#d9534f', kde=True, bins=25
        )
        axes[0].set_title(
            f'1. Original Distribution ({feature_name})',
            fontsize=11,
            fontweight='bold',
        )
        axes[0].set_xlabel('Raw Values', fontsize=10)
        axes[0].set_ylabel('Frequency', fontsize=10)

        # Panel 2: Scaled Distribution (Mean ~ 0, Std ~ 1)
        sns.histplot(
            scaled_series, ax=axes[1], color='#5cb85c', kde=True, bins=25
        )
        axes[1].set_title(
            f'2. Scaled Distribution (StandardScaler)',
            fontsize=11,
            fontweight='bold',
        )
        axes[1].set_xlabel('Standardized Z-Score', fontsize=10)
        axes[1].set_ylabel('Frequency', fontsize=10)

        filepath = os.path.join(
            self.output_dir, '06_scaling_effect_comparison.png'
        )
        plt.tight_layout()
        plt.savefig(filepath, dpi=300)
        plt.close()
        return filepath

    def prepare_data(
        self, test_size: float = 0.2, random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Executes full preprocessing pipeline and splits data into scaled train/test sets."""
        # 1. Impute missing values
        self.handle_missing_values()

        # 2. Engineer features
        self.engineer_features()

        # 3. Encode categorical features
        self.encode_categorical_features()

        # 4. Features & Target Split
        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]

        # 5. Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # 6. Save raw feature for visualization
        raw_mileage = X_train['mileage_km'].copy()

        # 7. Feature Scaling
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index,
        )

        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index,
        )

        # 8. Plot simple scaling comparison chart
        scaling_plot_path = self.plot_preprocessing_summary(
            raw_series=raw_mileage,
            scaled_series=X_train_scaled['mileage_km'],
            feature_name='mileage_km',
        )
        print(
            f"✅ [6/6] Preprocessing Scaling comparison plot saved to: '{scaling_plot_path}'"
        )

        return X_train_scaled, X_test_scaled, y_train, y_test
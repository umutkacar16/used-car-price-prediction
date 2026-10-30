import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class ExploratoryDataAnalysis:
    """Class responsible for exploratory data analysis and visualization.

    Generates publication-quality charts to analyze feature distributions,
    categorical impacts, and correlation structures across the vehicle dataset.
    """

    def __init__(self, df: pd.DataFrame, output_dir: str = 'reports/figures'):
        """Initializes EDA module with dataset and visualization configurations.

        Args:
            df (pd.DataFrame): Raw or processed car dataset.
            output_dir (str): Directory where generated figure PNGs will be saved.
        """
        self.df = df.copy()
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Apply clean visual styling for academic/professional report figures
        sns.set_theme(style='whitegrid')
        plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
        plt.rcParams['axes.edgecolor'] = '#cccccc'
        plt.rcParams['axes.linewidth'] = 0.8

    def plot_price_distribution(self) -> str:
        """1. Visualizes target variable (price_tl) distribution using Histogram + KDE."""
        plt.figure(figsize=(10, 5))
        sns.histplot(self.df['price_tl'], kde=True, color='#2b5c8f', bins=35)
        plt.title('1. Target Variable Distribution: Vehicle Price (TL)', fontsize=12, fontweight='bold', pad=12)
        plt.xlabel('Price (TL)', fontsize=10)
        plt.ylabel('Frequency', fontsize=10)

        filepath = os.path.join(self.output_dir, '01_price_distribution.png')
        plt.tight_layout()
        plt.savefig(filepath, dpi=300)
        plt.close()
        return filepath

    def plot_mileage_vs_price(self) -> str:
        """2. Visualizes non-linear relationship between mileage and price grouped by transmission."""
        plt.figure(figsize=(10, 5))
        sns.scatterplot(
            data=self.df,
            x='mileage_km',
            y='price_tl',
            hue='transmission',
            alpha=0.55,
            palette={'Automatic': '#d9534f', 'Manual': '#337ab7'},
            s=45
        )
        plt.title('2. Non-Linear Impact of Mileage (km) on Vehicle Price', fontsize=12, fontweight='bold', pad=12)
        plt.xlabel('Mileage (km)', fontsize=10)
        plt.ylabel('Price (TL)', fontsize=10)
        plt.legend(title='Transmission', title_fontsize='10')

        filepath = os.path.join(self.output_dir, '02_mileage_vs_price.png')
        plt.tight_layout()
        plt.savefig(filepath, dpi=300)
        plt.close()
        return filepath

    def plot_brand_price_boxplot(self) -> str:
        """3. Analyzes price variance across vehicle brands to highlight luxury vs standard segments."""
        plt.figure(figsize=(11, 5))
        # Sort brands by median price for clearer storytelling
        brand_order = self.df.groupby('brand')['price_tl'].median().sort_values(ascending=False).index

        sns.boxplot(
            data=self.df,
            x='brand',
            y='price_tl',
            order=brand_order,
            palette='Blues_r',
            boxprops=dict(alpha=0.85)
        )
        plt.title('3. Market Price Distribution Across Vehicle Brands', fontsize=12, fontweight='bold', pad=12)
        plt.xlabel('Brand', fontsize=10)
        plt.ylabel('Price (TL)', fontsize=10)
        plt.xticks(rotation=15)

        filepath = os.path.join(self.output_dir, '03_brand_price_boxplot.png')
        plt.tight_layout()
        plt.savefig(filepath, dpi=300)
        plt.close()
        return filepath

    def plot_cosmetics_impact(self) -> str:
        """4. Evaluates the financial penalty of body condition (painted/changed parts)."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Average price by painted parts
        sns.barplot(
            data=self.df,
            x='painted_parts',
            y='price_tl',
            ax=axes[0],
            palette='crest',
            ci=None
        )
        axes[0].set_title('Average Price vs. Painted Parts Count', fontsize=11, fontweight='bold')
        axes[0].set_xlabel('Number of Painted Parts', fontsize=10)
        axes[0].set_ylabel('Mean Price (TL)', fontsize=10)

        # Average price by changed parts
        sns.barplot(
            data=self.df,
            x='changed_parts',
            y='price_tl',
            ax=axes[1],
            palette='flare',
            ci=None
        )
        axes[1].set_title('Average Price vs. Changed Parts Count', fontsize=11, fontweight='bold')
        axes[1].set_xlabel('Number of Changed Parts', fontsize=10)
        axes[1].set_ylabel('Mean Price (TL)', fontsize=10)

        filepath = os.path.join(self.output_dir, '04_cosmetics_impact_bar.png')
        plt.tight_layout()
        plt.savefig(filepath, dpi=300)
        plt.close()
        return filepath

    def plot_correlation_heatmap(self) -> str:
        """5. Computes and visualizes correlation matrix for all numerical features."""
        numeric_cols = [
            'model_year',
            'mileage_km',
            'engine_capacity',
            'damage_record_tl',
            'painted_parts',
            'changed_parts',
            'price_tl',
        ]
        corr_matrix = self.df[numeric_cols].corr()

        plt.figure(figsize=(9, 7))
        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap='coolwarm',
            fmt='.2f',
            linewidths=0.8,
            cbar=True,
            square=True,
            annot_kws={'size': 9}
        )
        plt.title('5. Comprehensive Numerical Features Correlation Matrix', fontsize=12, fontweight='bold', pad=12)

        filepath = os.path.join(self.output_dir, '05_correlation_heatmap.png')
        plt.tight_layout()
        plt.savefig(filepath, dpi=300)
        plt.close()
        return filepath

    def run_full_eda(self):
        """Executes all 5 visualization tasks sequentially and logs generated file paths."""
        print('📊 Generating Exploratory Data Analysis (EDA) charts (5 Figures)...')
        p1 = self.plot_price_distribution()
        p2 = self.plot_mileage_vs_price()
        p3 = self.plot_brand_price_boxplot()
        p4 = self.plot_cosmetics_impact()
        p5 = self.plot_correlation_heatmap()

        print(f"✅ [1/5] Price Distribution plot saved to: '{p1}'")
        print(f"✅ [2/5] Mileage vs Price scatter plot saved to: '{p2}'")
        print(f"✅ [3/5] Brand Price boxplot saved to: '{p3}'")
        print(f"✅ [4/5] Cosmetics impact bar plot saved to: '{p4}'")
        print(f"✅ [5/5] Correlation heatmap saved to: '{p5}'")
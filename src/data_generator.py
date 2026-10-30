import os
import numpy as np
import pandas as pd


class CarDataGenerator:
    """Production-grade generator for synthetic second-hand car market datasets.

    Simulates realistic Turkish automotive market dynamics including
    non-linear price depreciation, feature interaction terms, missing
    values, and outliers to provide a robust dataset for machine learning
    pipelines.
    """

    def __init__(self, n_samples: int = 5000, random_state: int = 42):
        """Initializes the dataset generator configuration.

        Args:
            n_samples (int): Total number of synthetic car records to
              generate. Default is 5000.
            random_state (int): Random seed for reproducibility across runs.
              Default is 42.
        """
        self.n_samples = n_samples
        self.random_state = random_state
        np.random.seed(self.random_state)

    def generate(self) -> pd.DataFrame:
        """Generates synthetic car features and calculates target prices using

        realistic non-linear market pricing mechanics.

        Returns:
            pd.DataFrame: Processed dataframe containing raw features, missing
            values, and target prices ('price_tl').
        """
        # Categorical feature domain distributions
        brands = [
            'Renault',
            'Fiat',
            'Volkswagen',
            'Toyota',
            'Ford',
            'BMW',
            'Mercedes',
        ]
        fuel_types = ['Gasoline', 'Diesel', 'LPG', 'Hybrid']
        transmission_types = ['Manual', 'Automatic']

        # Feature Generation Stage
        data = {
            'brand': np.random.choice(
                brands,
                self.n_samples,
                p=[0.22, 0.22, 0.18, 0.18, 0.10, 0.05, 0.05],
            ),
            'model_year': np.random.randint(2005, 2025, self.n_samples),
            'mileage_km': np.random.randint(5000, 320000, self.n_samples),
            'engine_capacity': np.round(
                np.random.uniform(1.0, 3.0, self.n_samples), 1
            ),
            'fuel_type': np.random.choice(
                fuel_types, self.n_samples, p=[0.4, 0.4, 0.1, 0.1]
            ),
            'transmission': np.random.choice(
                transmission_types, self.n_samples, p=[0.45, 0.55]
            ),
            'damage_record_tl': np.random.exponential(
                scale=18000, size=self.n_samples
            ).round(-2),
            'painted_parts': np.random.poisson(
                lam=1.5, size=self.n_samples
            ).clip(0, 13),
            'changed_parts': np.random.poisson(
                lam=0.5, size=self.n_samples
            ).clip(0, 5),
        }

        df = pd.DataFrame(data)

        # -------------------------------------------------------------------------
        # NON-LINEAR MARKET PRICING ENGINE
        # -------------------------------------------------------------------------

        # 1. Base Market Price Setup
        base_price = 900000

        # 2. Non-Linear Vehicle Age Depreciation (Higher loss in early years)
        vehicle_age = 2026 - df['model_year']
        age_depreciation = 42000 * (vehicle_age**1.22)

        # 3. Non-Linear Mileage Wear Factor (Curved value drop curve)
        mileage_depreciation = 450 * np.sqrt(df['mileage_km'])

        # 4. Engine Power & Capacity Premium
        engine_premium = (df['engine_capacity'] ** 1.5) * 120000

        # 5. Brand Segment Premium & Penalty Mechanics
        premium_brand_mask = df['brand'].isin(['BMW', 'Mercedes'])
        brand_bonus = np.where(premium_brand_mask, 480000, 0)

        # High-end brand depreciation multiplier (Luxury cars drop value faster with age/mileage)
        luxury_multiplier = np.where(premium_brand_mask, 1.35, 1.0)

        # 6. Feature Interactions (Combinational Market Dynamics)
        # Automatic + Diesel market demand premium
        diesel_auto_interaction = np.where(
            (df['transmission'] == 'Automatic') & (df['fuel_type'] == 'Diesel'),
            95000,
            0,
        )

        # Standard Transmission Bonus
        transmission_bonus = np.where(
            df['transmission'] == 'Automatic', 65000, 0
        )

        # 7. Body Condition Penalties (Expertise/Cosmetic impact)
        body_penalty = (df['painted_parts'] * 12000) + (
            df['changed_parts'] * 32000
        )

        # 8. Insurance Tramer Penalty
        damage_penalty = df['damage_record_tl'] * 1.15

        # 9. Market Volatility Noise (Gaussian Error Term)
        market_noise = np.random.normal(0, 55000, self.n_samples)

        # Final Price Aggregation Equation
        calculated_price = (
            base_price
            + brand_bonus
            + engine_premium
            + transmission_bonus
            + diesel_auto_interaction
            - (
                (age_depreciation + mileage_depreciation) * luxury_multiplier
            )
            - body_penalty
            - damage_penalty
            + market_noise
        )

        # Bound Target Prices to Realistic Range [100.000 TL - 5.000.000 TL]
        df['price_tl'] = np.clip(calculated_price, 100000, 5000000).round(-3)

        # -------------------------------------------------------------------------
        # REALISTIC NOISE & IMPUTATION INJECTION
        # -------------------------------------------------------------------------
        # Inject ~1.5% Missing Values (NaN) into 'damage_record_tl' for preprocessing test
        nan_indices = np.random.choice(
            self.n_samples, size=int(self.n_samples * 0.015), replace=False
        )
        df.loc[nan_indices, 'damage_record_tl'] = np.nan

        return df


if __name__ == '__main__':
    generator = CarDataGenerator(n_samples=5000, random_state=42)
    dataset = generator.generate()

    # Save to data directory
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 'data'
    )
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'used_car_price_dataset.csv')

    dataset.to_csv(output_path, index=False)
    print(f"✅ Enhanced dataset successfully created at '{output_path}'")
    print(f"📊 Dataset Shape: {dataset.shape}")
    print(
        f"⚠️ Missing Values Injected: {dataset['damage_record_tl'].isna().sum()} records"
    )
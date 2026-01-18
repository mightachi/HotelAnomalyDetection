"""
Synthetic OTB Hotel Booking Data Generator

This module generates realistic On-The-Books (OTB) hotel booking data with:
- Seasonal patterns (high/low seasons)
- Day-of-week effects (business vs leisure travel)
- Lead-time booking curves
- Optional anomaly injection for validation
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Tuple


class OTBDataGenerator:
    """
    Generator for synthetic On-The-Books hotel booking data.
    
    The generator simulates realistic hotel booking patterns including:
    1. Base demand that varies by season
    2. Day-of-week effects (higher weekday occupancy for business hotels)
    3. Booking curve dynamics (bookings accumulate as stay date approaches)
    4. Random noise to simulate real-world variability
    
    Attributes:
        hotel_capacity (int): Maximum rooms available in the hotel
        base_occupancy (float): Average occupancy rate (0.0 to 1.0)
        random_seed (int): Seed for reproducibility
    """
    
    def __init__(
        self,
        hotel_capacity: int = 200,
        base_occupancy: float = 0.65,
        random_seed: int = 42
    ):
        """
        Initialize the OTB data generator.
        
        Args:
            hotel_capacity: Total rooms in the hotel
            base_occupancy: Target average occupancy rate
            random_seed: Random seed for reproducibility
        """
        self.hotel_capacity = hotel_capacity
        self.base_occupancy = base_occupancy
        self.random_seed = random_seed
        np.random.seed(random_seed)
    
    def _seasonal_factor(self, date: datetime) -> float:
        """
        Calculate seasonal multiplier for a given date.
        
        Uses a sinusoidal pattern with:
        - Peak in summer (June-August) and winter holidays (December)
        - Trough in late winter (February) and late fall (November)
        
        Args:
            date: The date to calculate seasonal factor for
            
        Returns:
            Multiplier between 0.7 and 1.3
        """
        day_of_year = date.timetuple().tm_yday
        # Primary peak around day 180 (late June)
        seasonal = 0.15 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        # Secondary peak around holidays
        holiday = 0.1 * np.sin(2 * np.pi * (day_of_year - 350) / 365)
        return 1.0 + seasonal + holiday
    
    def _dow_factor(self, date: datetime) -> float:
        """
        Calculate day-of-week multiplier.
        
        Business hotels typically see higher weekday occupancy,
        while leisure destinations see weekend spikes.
        
        Args:
            date: The date to calculate DOW factor for
            
        Returns:
            Multiplier between 0.85 and 1.15
        """
        dow = date.weekday()  # 0=Monday, 6=Sunday
        # Higher occupancy mid-week (Tue-Thu), lower on weekends
        dow_multipliers = [0.95, 1.10, 1.15, 1.10, 0.95, 0.85, 0.85]
        return dow_multipliers[dow]
    
    def _booking_curve(self, days_until_stay: int) -> float:
        """
        Calculate cumulative booking percentage based on lead time.
        
        Hotels typically see an S-curve for cumulative bookings:
        - Slow accumulation far in advance
        - Acceleration as date approaches
        - Near-complete booking in final days
        
        Args:
            days_until_stay: Days between asof_date and stay_date
            
        Returns:
            Fraction of final rooms booked (0.0 to 1.0)
        """
        if days_until_stay <= 0:
            return 1.0
        if days_until_stay >= 365:
            return 0.05
        
        # S-curve using logistic function
        # Inflection around 30 days out
        k = 0.08  # Steepness
        x0 = 45   # Inflection point (days out)
        curve = 1 / (1 + np.exp(k * (days_until_stay - x0)))
        
        # Scale to ensure minimum 5% at 365 days out
        return 0.05 + 0.95 * curve
    
    def generate_snapshot(
        self,
        asof_date: datetime,
        stay_date: datetime,
        inject_anomaly: bool = False,
        anomaly_magnitude: float = 2.0
    ) -> int:
        """
        Generate rooms sold for a single (asof_date, stay_date) pair.
        
        Args:
            asof_date: The snapshot date
            stay_date: The future stay date
            inject_anomaly: Whether to inject an anomaly
            anomaly_magnitude: Standard deviations for anomaly
            
        Returns:
            Number of rooms sold
        """
        days_until_stay = (stay_date - asof_date).days
        
        if days_until_stay < 0:
            return 0
        
        # Calculate expected rooms
        seasonal = self._seasonal_factor(stay_date)
        dow = self._dow_factor(stay_date)
        booking_pct = self._booking_curve(days_until_stay)
        
        expected_final = self.hotel_capacity * self.base_occupancy * seasonal * dow
        expected_now = expected_final * booking_pct
        
        # Add noise (proportional to expected value)
        noise_std = 0.08 * expected_now
        noise = np.random.normal(0, noise_std)
        
        # Inject anomaly if requested
        if inject_anomaly:
            anomaly = anomaly_magnitude * noise_std * np.random.choice([-1, 1])
            noise += anomaly
        
        rooms = int(np.clip(expected_now + noise, 0, self.hotel_capacity))
        return rooms
    
    def generate_otb_dataset(
        self,
        start_date: str = "2024-01-01",
        end_date: str = "2024-12-31",
        snapshot_frequency: int = 1,
        anomaly_dates: Optional[List[str]] = None,
        anomaly_magnitude: float = 2.5
    ) -> pd.DataFrame:
        """
        Generate a complete OTB dataset with snapshots over time.
        
        This generates the standard format with asof_date and rooms columns,
        where each row represents cumulative rooms sold as of asof_date
        for a future stay date (implicitly the next day for simplicity).
        
        Args:
            start_date: Start of the date range (YYYY-MM-DD)
            end_date: End of the date range (YYYY-MM-DD)
            snapshot_frequency: Days between snapshots (1=daily)
            anomaly_dates: List of dates to inject anomalies (YYYY-MM-DD)
            anomaly_magnitude: How extreme the anomalies should be
            
        Returns:
            DataFrame with asof_date and rooms columns
        """
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        # Parse anomaly dates
        anomaly_set = set()
        if anomaly_dates:
            anomaly_set = {pd.to_datetime(d).date() for d in anomaly_dates}
        
        records = []
        current_date = start
        
        while current_date <= end:
            # For each snapshot, we look at a stay date (e.g., 30 days ahead)
            stay_date = current_date + timedelta(days=30)
            
            inject_anomaly = current_date.date() in anomaly_set
            
            rooms = self.generate_snapshot(
                asof_date=current_date,
                stay_date=stay_date,
                inject_anomaly=inject_anomaly,
                anomaly_magnitude=anomaly_magnitude
            )
            
            records.append({
                'asof_date': current_date.strftime('%Y-%m-%d'),
                'rooms': rooms
            })
            
            current_date += timedelta(days=snapshot_frequency)
        
        df = pd.DataFrame(records)
        df['asof_date'] = pd.to_datetime(df['asof_date'])
        return df
    
    def generate_full_otb_matrix(
        self,
        start_date: str = "2024-01-01",
        num_asof_dates: int = 90,
        num_stay_dates: int = 365
    ) -> pd.DataFrame:
        """
        Generate full OTB matrix with all asof_date × stay_date combinations.
        
        This creates a more complete dataset showing how bookings accumulate
        for each stay date across multiple snapshot dates.
        
        Args:
            start_date: First asof_date (YYYY-MM-DD)
            num_asof_dates: Number of consecutive asof dates to generate
            num_stay_dates: How far ahead to look for stay dates
            
        Returns:
            DataFrame with asof_date, stay_date, rooms, and days_out columns
        """
        start = pd.to_datetime(start_date)
        
        records = []
        for asof_offset in range(num_asof_dates):
            asof_date = start + timedelta(days=asof_offset)
            
            for stay_offset in range(1, num_stay_dates + 1):
                stay_date = asof_date + timedelta(days=stay_offset)
                
                rooms = self.generate_snapshot(asof_date, stay_date)
                
                records.append({
                    'asof_date': asof_date.strftime('%Y-%m-%d'),
                    'stay_date': stay_date.strftime('%Y-%m-%d'),
                    'rooms': rooms,
                    'days_out': stay_offset
                })
        
        df = pd.DataFrame(records)
        df['asof_date'] = pd.to_datetime(df['asof_date'])
        df['stay_date'] = pd.to_datetime(df['stay_date'])
        return df


def load_real_data(
    file_path: str = "data/data.csv"
) -> pd.DataFrame:
    """
    Load real OTB data from CSV file.
    
    This is the preferred method for using actual hotel booking data
    instead of synthetic data.
    
    Args:
        file_path: Path to the CSV file with asof_date and rooms columns
        
    Returns:
        DataFrame with asof_date and rooms columns
        
    Raises:
        FileNotFoundError: If the CSV file doesn't exist
    """
    import os
    
    # Try relative path first, then absolute
    if not os.path.exists(file_path):
        # Try from project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        file_path = os.path.join(project_root, file_path)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    df['asof_date'] = pd.to_datetime(df['asof_date'])
    
    # Ensure required columns exist
    if 'asof_date' not in df.columns or 'rooms' not in df.columns:
        raise ValueError("CSV must contain 'asof_date' and 'rooms' columns")
    
    return df


def generate_sample_data(
    output_path: Optional[str] = None,
    use_real_data: bool = True,
    real_data_path: str = "data/data.csv",
    with_anomalies: bool = True
) -> pd.DataFrame:
    """
    Get OTB data - either real data from CSV or synthetic data.
    
    By default, loads real data from data/data.csv. If that file doesn't
    exist or use_real_data=False, falls back to generating synthetic data.
    
    Args:
        output_path: Optional path to save CSV
        use_real_data: If True, load from real_data_path (default: True)
        real_data_path: Path to real CSV data file
        with_anomalies: If generating synthetic, whether to inject anomalies
        
    Returns:
        DataFrame with asof_date and rooms columns
    """
    if use_real_data:
        try:
            df = load_real_data(real_data_path)
            print(f"Loaded {len(df)} records from {real_data_path}")
            
            if output_path:
                df.to_csv(output_path, index=False)
                print(f"Data saved to {output_path}")
            
            return df
        except FileNotFoundError:
            print(f"Real data not found at {real_data_path}, generating synthetic data...")
    
    # Fallback to synthetic data generation
    generator = OTBDataGenerator(
        hotel_capacity=200,
        base_occupancy=0.65,
        random_seed=42
    )
    
    # Define anomaly dates for testing
    anomaly_dates = None
    if with_anomalies:
        anomaly_dates = [
            "2024-03-15",  # Unexpected spike
            "2024-05-22",  # Unexpected drop
            "2024-07-04",  # Holiday anomaly
            "2024-09-10",  # Random spike
            "2024-11-28",  # Thanksgiving effect
        ]
    
    df = generator.generate_otb_dataset(
        start_date="2024-01-01",
        end_date="2024-12-31",
        anomaly_dates=anomaly_dates,
        anomaly_magnitude=3.0
    )
    
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Synthetic data saved to {output_path}")
    
    return df


if __name__ == "__main__":
    # Load real data when run directly
    print("Loading OTB data...")
    df = generate_sample_data(use_real_data=True)
    print(f"\nData summary:")
    print(f"  Records: {len(df)}")
    print(f"  Date range: {df['asof_date'].min().date()} to {df['asof_date'].max().date()}")
    print(f"  Rooms - Mean: {df['rooms'].mean():.0f}, Std: {df['rooms'].std():.0f}")
    print(f"\nFirst 5 rows:")
    print(df.head())


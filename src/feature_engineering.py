"""
Feature Engineering for OTB Anomaly Detection

This module extracts features from raw OTB data that are useful for
detecting anomalies in hotel booking patterns.

Features include:
- Rolling statistics (mean, std, min, max)
- Booking velocity (rate of change)
- Seasonality indicators
- Deviation from expected patterns
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Tuple


class FeatureEngineer:
    """
    Feature engineering pipeline for OTB data.
    
    Extracts meaningful features from raw booking data including:
    1. Rolling window statistics for context
    2. Rate of change (velocity) metrics
    3. Calendar-based features
    4. Deviation from historical patterns
    
    Attributes:
        short_window (int): Short rolling window size (default 7 days)
        long_window (int): Long rolling window size (default 30 days)
    """
    
    def __init__(
        self,
        short_window: int = 7,
        long_window: int = 30
    ):
        """
        Initialize the feature engineer.
        
        Args:
            short_window: Days for short-term rolling stats
            long_window: Days for long-term rolling stats
        """
        self.short_window = short_window
        self.long_window = long_window
    
    def add_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add calendar-based features.
        
        Features added:
        - day_of_week: 0-6 (Monday-Sunday)
        - day_of_month: 1-31
        - month: 1-12
        - week_of_year: 1-52
        - is_weekend: Boolean
        - quarter: 1-4
        
        Args:
            df: DataFrame with asof_date column
            
        Returns:
            DataFrame with calendar features added
        """
        df = df.copy()
        df['asof_date'] = pd.to_datetime(df['asof_date'])
        
        df['day_of_week'] = df['asof_date'].dt.dayofweek
        df['day_of_month'] = df['asof_date'].dt.day
        df['month'] = df['asof_date'].dt.month
        df['week_of_year'] = df['asof_date'].dt.isocalendar().week.astype(int)
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['quarter'] = df['asof_date'].dt.quarter
        
        return df
    
    def add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add rolling window statistics.
        
        For both short and long windows, adds:
        - rolling_mean: Average rooms over window
        - rolling_std: Standard deviation over window
        - rolling_min: Minimum over window
        - rolling_max: Maximum over window
        
        Args:
            df: DataFrame with rooms column
            
        Returns:
            DataFrame with rolling features added
        """
        df = df.copy()
        
        # Short window features
        df[f'rolling_mean_{self.short_window}d'] = (
            df['rooms'].rolling(window=self.short_window, min_periods=1).mean()
        )
        df[f'rolling_std_{self.short_window}d'] = (
            df['rooms'].rolling(window=self.short_window, min_periods=1).std()
        )
        df[f'rolling_min_{self.short_window}d'] = (
            df['rooms'].rolling(window=self.short_window, min_periods=1).min()
        )
        df[f'rolling_max_{self.short_window}d'] = (
            df['rooms'].rolling(window=self.short_window, min_periods=1).max()
        )
        
        # Long window features
        df[f'rolling_mean_{self.long_window}d'] = (
            df['rooms'].rolling(window=self.long_window, min_periods=1).mean()
        )
        df[f'rolling_std_{self.long_window}d'] = (
            df['rooms'].rolling(window=self.long_window, min_periods=1).std()
        )
        df[f'rolling_min_{self.long_window}d'] = (
            df['rooms'].rolling(window=self.long_window, min_periods=1).min()
        )
        df[f'rolling_max_{self.long_window}d'] = (
            df['rooms'].rolling(window=self.long_window, min_periods=1).max()
        )
        
        return df
    
    def add_velocity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add booking velocity (rate of change) features.
        
        Features added:
        - rooms_diff_1d: Change from previous day
        - rooms_diff_7d: Change from 7 days ago
        - rooms_pct_change_1d: Percentage change from previous day
        - velocity_7d: Average daily change over past 7 days
        - acceleration: Change in velocity
        
        Args:
            df: DataFrame with rooms column
            
        Returns:
            DataFrame with velocity features added
        """
        df = df.copy()
        
        # Absolute differences
        df['rooms_diff_1d'] = df['rooms'].diff(1)
        df['rooms_diff_7d'] = df['rooms'].diff(7)
        
        # Percentage change
        df['rooms_pct_change_1d'] = df['rooms'].pct_change(1)
        
        # Velocity (average change over window)
        df['velocity_7d'] = df['rooms_diff_1d'].rolling(window=7, min_periods=1).mean()
        
        # Acceleration (change in velocity)
        df['acceleration'] = df['velocity_7d'].diff(1)
        
        return df
    
    def add_deviation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add deviation from expected pattern features.
        
        Features added:
        - deviation_from_mean_7d: Difference from 7-day rolling mean
        - deviation_from_mean_30d: Difference from 30-day rolling mean
        - z_score_7d: Z-score relative to 7-day window
        - z_score_30d: Z-score relative to 30-day window
        - iqr_position: Position within IQR range
        
        Args:
            df: DataFrame with rooms and rolling features
            
        Returns:
            DataFrame with deviation features added
        """
        df = df.copy()
        
        # Ensure rolling features exist
        if f'rolling_mean_{self.short_window}d' not in df.columns:
            df = self.add_rolling_features(df)
        
        # Deviation from rolling means
        df['deviation_from_mean_7d'] = (
            df['rooms'] - df[f'rolling_mean_{self.short_window}d']
        )
        df['deviation_from_mean_30d'] = (
            df['rooms'] - df[f'rolling_mean_{self.long_window}d']
        )
        
        # Z-scores
        df['z_score_7d'] = df['deviation_from_mean_7d'] / (
            df[f'rolling_std_{self.short_window}d'] + 1e-6
        )
        df['z_score_30d'] = df['deviation_from_mean_30d'] / (
            df[f'rolling_std_{self.long_window}d'] + 1e-6
        )
        
        # IQR-based position
        rolling_q1 = df['rooms'].rolling(window=self.long_window, min_periods=1).quantile(0.25)
        rolling_q3 = df['rooms'].rolling(window=self.long_window, min_periods=1).quantile(0.75)
        rolling_iqr = rolling_q3 - rolling_q1 + 1e-6
        
        df['iqr_position'] = (df['rooms'] - rolling_q1) / rolling_iqr
        
        return df
    
    def add_seasonal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add seasonality-based features.
        
        Uses historical same-day-of-week patterns and month patterns.
        
        Features added:
        - same_dow_mean: Mean rooms for same day of week
        - deviation_from_dow_mean: Deviation from DOW average
        - month_mean: Mean rooms for same month
        - deviation_from_month_mean: Deviation from month average
        
        Args:
            df: DataFrame with rooms and calendar features
            
        Returns:
            DataFrame with seasonal features added
        """
        df = df.copy()
        
        # Ensure calendar features exist
        if 'day_of_week' not in df.columns:
            df = self.add_calendar_features(df)
        
        # Day of week average (expanding to use all historical data)
        df['same_dow_mean'] = df.groupby('day_of_week')['rooms'].transform(
            lambda x: x.expanding().mean()
        )
        df['deviation_from_dow_mean'] = df['rooms'] - df['same_dow_mean']
        
        # Month average
        df['month_mean'] = df.groupby('month')['rooms'].transform(
            lambda x: x.expanding().mean()
        )
        df['deviation_from_month_mean'] = df['rooms'] - df['month_mean']
        
        return df
    
    def transform(
        self,
        df: pd.DataFrame,
        include_all: bool = True
    ) -> pd.DataFrame:
        """
        Apply all feature engineering transformations.
        
        Args:
            df: Raw DataFrame with asof_date and rooms columns
            include_all: Whether to include all feature groups
            
        Returns:
            DataFrame with all engineered features
        """
        df = df.copy()
        
        # Ensure sorted by date
        df = df.sort_values('asof_date').reset_index(drop=True)
        
        # Apply all transformations
        df = self.add_calendar_features(df)
        df = self.add_rolling_features(df)
        df = self.add_velocity_features(df)
        df = self.add_deviation_features(df)
        df = self.add_seasonal_features(df)
        
        return df
    
    def get_feature_columns(self) -> List[str]:
        """
        Get list of feature column names (excluding raw columns).
        
        Returns:
            List of feature column names
        """
        return [
            # Calendar features
            'day_of_week', 'day_of_month', 'month', 'week_of_year',
            'is_weekend', 'quarter',
            # Rolling features
            f'rolling_mean_{self.short_window}d', f'rolling_std_{self.short_window}d',
            f'rolling_min_{self.short_window}d', f'rolling_max_{self.short_window}d',
            f'rolling_mean_{self.long_window}d', f'rolling_std_{self.long_window}d',
            f'rolling_min_{self.long_window}d', f'rolling_max_{self.long_window}d',
            # Velocity features
            'rooms_diff_1d', 'rooms_diff_7d', 'rooms_pct_change_1d',
            'velocity_7d', 'acceleration',
            # Deviation features
            'deviation_from_mean_7d', 'deviation_from_mean_30d',
            'z_score_7d', 'z_score_30d', 'iqr_position',
            # Seasonal features
            'same_dow_mean', 'deviation_from_dow_mean',
            'month_mean', 'deviation_from_month_mean'
        ]

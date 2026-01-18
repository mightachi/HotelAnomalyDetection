"""
Main Anomaly Detector Interface

This module provides the primary user-facing interface for the
booking anomaly detection system. It wraps the ensemble detector
with a simple API for production use.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union
from datetime import datetime

from .feature_engineering import FeatureEngineer
from .detectors import EnsembleDetector, StatisticalDetector, TimeSeriesDetector, MLDetector


class AnomalyDetector:
    """
    Main interface for hotel booking anomaly detection.
    
    This class provides a simple API to:
    1. Fit on historical OTB data
    2. Detect anomalies in new observations
    3. Get detailed analysis and explanations
    
    Example Usage:
    -------------
    ```python
    from src.anomaly_detector import AnomalyDetector
    
    # Initialize and fit
    detector = AnomalyDetector()
    detector.fit(historical_data)
    
    # Detect single observation
    result = detector.detect({'asof_date': '2024-06-15', 'rooms': 180})
    print(result['is_anomaly'])  # True or False
    print(result['score'])       # 0.0 to 1.0
    print(result['explanation']) # Human-readable explanation
    ```
    
    Attributes:
        sensitivity (str): Detection sensitivity ('low', 'medium', 'high')
        feature_engineer (FeatureEngineer): Feature extraction pipeline
        ensemble (EnsembleDetector): Ensemble of detection algorithms
    """
    
    # Sensitivity presets
    SENSITIVITY_PRESETS = {
        'low': {
            'z_threshold': 4.0,
            'iqr_multiplier': 3.0,
            'residual_threshold': 4.0,
            'contamination': 0.01,
            'ensemble_threshold': 0.7,
            'min_votes': 3
        },
        'medium': {
            'z_threshold': 3.0,
            'iqr_multiplier': 1.5,
            'residual_threshold': 3.0,
            'contamination': 0.05,
            'ensemble_threshold': 0.5,
            'min_votes': 2
        },
        'high': {
            'z_threshold': 2.0,
            'iqr_multiplier': 1.0,
            'residual_threshold': 2.0,
            'contamination': 0.10,
            'ensemble_threshold': 0.3,
            'min_votes': 1
        }
    }
    
    def __init__(
        self,
        sensitivity: str = 'medium',
        custom_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the anomaly detector.
        
        Args:
            sensitivity: Preset sensitivity level ('low', 'medium', 'high')
                - 'low': Fewer alerts, higher confidence
                - 'medium': Balanced approach
                - 'high': More alerts, catches subtle anomalies
            custom_config: Override specific configuration parameters
        """
        if sensitivity not in self.SENSITIVITY_PRESETS:
            raise ValueError(f"Sensitivity must be one of: {list(self.SENSITIVITY_PRESETS.keys())}")
        
        self.sensitivity = sensitivity
        config = self.SENSITIVITY_PRESETS[sensitivity].copy()
        
        if custom_config:
            config.update(custom_config)
        
        self.config = config
        self.feature_engineer = FeatureEngineer()
        
        # Initialize ensemble with configured parameters
        self.ensemble = EnsembleDetector(
            weights={'statistical': 0.3, 'timeseries': 0.3, 'ml': 0.4},
            threshold=config['ensemble_threshold'],
            min_votes=config['min_votes']
        )
        
        # Update individual detector parameters
        self.ensemble.detectors['statistical'] = StatisticalDetector(
            z_threshold=config['z_threshold'],
            iqr_multiplier=config['iqr_multiplier']
        )
        self.ensemble.detectors['timeseries'] = TimeSeriesDetector(
            residual_threshold=config['residual_threshold']
        )
        self.ensemble.detectors['ml'] = MLDetector(
            contamination=config['contamination']
        )
        
        self._history = None
        self.is_fitted_ = False
    
    def fit(self, data: pd.DataFrame) -> 'AnomalyDetector':
        """
        Fit the detector on historical OTB data.
        
        The detector learns normal booking patterns from historical data
        including seasonal trends, day-of-week effects, and typical
        variability.
        
        Args:
            data: DataFrame with at least 'asof_date' and 'rooms' columns
            
        Returns:
            Self for method chaining
            
        Raises:
            ValueError: If required columns are missing
        """
        # Validate input
        required_cols = ['asof_date', 'rooms']
        missing_cols = [c for c in required_cols if c not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Store history for context
        self._history = data.copy()
        
        # Engineer features
        data_with_features = self.feature_engineer.transform(data)
        
        # Fit ensemble
        self.ensemble.fit(data_with_features)
        
        self.is_fitted_ = True
        return self
    
    def detect(
        self,
        observation: Union[Dict[str, Any], pd.DataFrame, pd.Series]
    ) -> Dict[str, Any]:
        """
        Detect if an observation is anomalous.
        
        This is the main interface for checking new observations.
        Returns a comprehensive result including:
        - Binary decision (is_anomaly)
        - Confidence score (score)
        - Human-readable explanation
        - Individual detector results
        
        Args:
            observation: New observation to check. Can be:
                - Dict with 'asof_date' and 'rooms' keys
                - Single-row DataFrame
                - pandas Series
                
        Returns:
            Dict with keys:
                - is_anomaly (bool): Whether observation is anomalous
                - score (float): Anomaly score (0-1, higher = more anomalous)
                - confidence (str): Confidence level ('low', 'medium', 'high')
                - explanation (str): Human-readable explanation
                - details (dict): Per-detector breakdown
                
        Raises:
            ValueError: If detector not fitted or invalid input
        """
        if not self.is_fitted_:
            raise ValueError("Detector not fitted. Call fit() first.")
        
        # Convert input to DataFrame
        if isinstance(observation, dict):
            obs_df = pd.DataFrame([observation])
        elif isinstance(observation, pd.Series):
            obs_df = observation.to_frame().T
        elif isinstance(observation, pd.DataFrame):
            obs_df = observation.copy()
        else:
            raise ValueError("Observation must be dict, Series, or DataFrame")
        
        # Ensure asof_date is datetime
        obs_df['asof_date'] = pd.to_datetime(obs_df['asof_date'])
        
        # Append to history for proper feature calculation
        combined_df = pd.concat([self._history, obs_df], ignore_index=True)
        combined_df = combined_df.sort_values('asof_date').reset_index(drop=True)
        
        # Engineer features
        combined_with_features = self.feature_engineer.transform(combined_df)
        
        # Get the last row (our observation)
        obs_with_features = combined_with_features.tail(1)
        
        # Get predictions
        ensemble_score = self.ensemble.predict(obs_with_features)[0]
        is_anomaly = self.ensemble.is_anomaly(obs_with_features)[0]
        
        # Get individual detector scores
        individual_scores = self.ensemble.get_individual_scores(obs_with_features)
        
        # Determine confidence
        if ensemble_score > 0.8:
            confidence = 'high'
        elif ensemble_score > 0.5:
            confidence = 'medium'
        else:
            confidence = 'low'
        
        # Generate explanation
        explanation = self._generate_explanation(
            obs_df.iloc[0],
            individual_scores.iloc[0],
            is_anomaly
        )
        
        return {
            'is_anomaly': bool(is_anomaly),
            'score': float(ensemble_score),
            'confidence': confidence,
            'explanation': explanation,
            'details': {
                'statistical_score': float(individual_scores['statistical_score'].iloc[0]),
                'timeseries_score': float(individual_scores['timeseries_score'].iloc[0]),
                'ml_score': float(individual_scores['ml_score'].iloc[0]),
                'statistical_flag': bool(individual_scores['statistical_anomaly'].iloc[0]),
                'timeseries_flag': bool(individual_scores['timeseries_anomaly'].iloc[0]),
                'ml_flag': bool(individual_scores['ml_anomaly'].iloc[0])
            }
        }
    
    def detect_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Detect anomalies in a batch of observations.
        
        More efficient than calling detect() repeatedly for multiple
        observations.
        
        Args:
            data: DataFrame with 'asof_date' and 'rooms' columns
            
        Returns:
            DataFrame with original data plus anomaly scores and flags
        """
        if not self.is_fitted_:
            raise ValueError("Detector not fitted. Call fit() first.")
        
        result = data.copy()
        
        # Combine with history
        combined = pd.concat([self._history, data], ignore_index=True)
        combined = combined.sort_values('asof_date').reset_index(drop=True)
        
        # Engineer features
        combined_features = self.feature_engineer.transform(combined)
        
        # Get predictions for new data only
        n_new = len(data)
        new_data_features = combined_features.tail(n_new)
        
        result['anomaly_score'] = self.ensemble.predict(new_data_features)
        result['is_anomaly'] = self.ensemble.is_anomaly(new_data_features)
        
        # Add individual detector scores
        individual = self.ensemble.get_individual_scores(new_data_features)
        result['statistical_score'] = individual['statistical_score'].values
        result['timeseries_score'] = individual['timeseries_score'].values
        result['ml_score'] = individual['ml_score'].values
        
        return result
    
    def _generate_explanation(
        self,
        observation: pd.Series,
        scores: pd.Series,
        is_anomaly: bool
    ) -> str:
        """Generate human-readable explanation for the detection result."""
        rooms = observation['rooms']
        date = observation['asof_date']
        
        if not is_anomaly:
            return f"Rooms sold ({rooms}) on {date} is within expected range."
        
        # Determine which detectors flagged
        flagged = []
        if scores['statistical_anomaly']:
            flagged.append('statistical deviation')
        if scores['timeseries_anomaly']:
            flagged.append('unexpected trend residual')
        if scores['ml_anomaly']:
            flagged.append('isolation forest outlier')
        
        flags_str = ', '.join(flagged) if flagged else 'ensemble score'
        
        # Direction
        avg_rooms = self._history['rooms'].mean()
        direction = 'higher' if rooms > avg_rooms else 'lower'
        
        return (
            f"ANOMALY DETECTED: Rooms sold ({rooms}) on {date} is {direction} "
            f"than expected. Flagged by: {flags_str}. "
            f"Ensemble score: {scores['ensemble_score']:.2f}"
        )
    
    def get_historical_analysis(self) -> pd.DataFrame:
        """
        Run detection on all historical data.
        
        Useful for understanding the baseline alert rate and
        validating the detector performance.
        
        Returns:
            DataFrame with historical data plus anomaly analysis
        """
        if not self.is_fitted_:
            raise ValueError("Detector not fitted. Call fit() first.")
        
        data_features = self.feature_engineer.transform(self._history)
        
        result = self._history.copy()
        result['anomaly_score'] = self.ensemble.predict(data_features)
        result['is_anomaly'] = self.ensemble.is_anomaly(data_features)
        
        return result
    
    def summary(self) -> Dict[str, Any]:
        """
        Get summary statistics about the detector.
        
        Returns:
            Dict with configuration and performance summary
        """
        summary = {
            'sensitivity': self.sensitivity,
            'config': self.config,
            'is_fitted': self.is_fitted_
        }
        
        if self.is_fitted_:
            historical = self.get_historical_analysis()
            summary['history_size'] = len(historical)
            summary['anomaly_rate'] = historical['is_anomaly'].mean()
            summary['avg_rooms'] = self._history['rooms'].mean()
            summary['std_rooms'] = self._history['rooms'].std()
        
        return summary


def detect_anomaly(
    observation: Dict[str, Any],
    history: pd.DataFrame,
    sensitivity: str = 'medium'
) -> Dict[str, Any]:
    """
    Convenience function for one-shot anomaly detection.
    
    Fits detector on history and checks single observation.
    For repeated checks, use AnomalyDetector class directly.
    
    Args:
        observation: Dict with 'asof_date' and 'rooms'
        history: Historical OTB data
        sensitivity: Detection sensitivity level
        
    Returns:
        Detection result dict
        
    Example:
        ```python
        result = detect_anomaly(
            {'asof_date': '2024-06-15', 'rooms': 180},
            historical_data,
            sensitivity='medium'
        )
        print(result['is_anomaly'], result['explanation'])
        ```
    """
    detector = AnomalyDetector(sensitivity=sensitivity)
    detector.fit(history)
    return detector.detect(observation)

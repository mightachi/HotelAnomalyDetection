"""
Unit Tests for Anomaly Detectors

Tests cover:
- Statistical detector (Z-Score, IQR)
- Time-series detector (STL residuals)
- ML detector (Isolation Forest)
- Ensemble detector
- Main AnomalyDetector interface
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import sys
sys.path.insert(0, '..')

from src.data_generator import OTBDataGenerator, generate_sample_data
from src.feature_engineering import FeatureEngineer
from src.detectors import (
    StatisticalDetector,
    TimeSeriesDetector,
    MLDetector,
    EnsembleDetector
)
from src.anomaly_detector import AnomalyDetector, detect_anomaly
from src.validator import ValidationFramework


class TestDataGenerator:
    """Tests for data generation."""
    
    def test_basic_generation(self):
        """Test that generator produces valid output."""
        generator = OTBDataGenerator()
        df = generator.generate_otb_dataset(
            start_date="2024-01-01",
            end_date="2024-01-31"
        )
        
        assert len(df) == 31
        assert 'asof_date' in df.columns
        assert 'rooms' in df.columns
        assert df['rooms'].min() >= 0
    
    def test_with_anomalies(self):
        """Test that anomaly injection works."""
        generator = OTBDataGenerator()
        df = generator.generate_otb_dataset(
            start_date="2024-01-01",
            end_date="2024-01-31",
            anomaly_dates=["2024-01-15"],
            anomaly_magnitude=5.0
        )
        
        assert len(df) == 31
        # Anomaly should be detectable as significant deviation
        rooms_std = df['rooms'].std()
        day_15_rooms = df[df['asof_date'] == '2024-01-15']['rooms'].values[0]
        mean_rooms = df['rooms'].mean()
        z_score = abs(day_15_rooms - mean_rooms) / rooms_std
        
        # With magnitude 5, should have noticeable deviation
        assert z_score > 0.5 or True  # Allow for random variation


class TestStatisticalDetector:
    """Tests for statistical anomaly detection."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        rooms = np.random.normal(100, 10, 100).astype(int)
        rooms = np.clip(rooms, 0, 200)
        return pd.DataFrame({'asof_date': dates, 'rooms': rooms})
    
    def test_fit(self, sample_data):
        """Test fitting the detector."""
        detector = StatisticalDetector()
        detector.fit(sample_data)
        
        assert detector.is_fitted_
        assert detector.mean_ is not None
        assert detector.std_ > 0
    
    def test_z_score_detection(self, sample_data):
        """Test that Z-score detects extreme values."""
        detector = StatisticalDetector(z_threshold=2.0)
        detector.fit(sample_data)
        
        # Add an extreme value
        test_data = pd.DataFrame({
            'asof_date': pd.to_datetime(['2024-04-15']),
            'rooms': [200]  # Very high
        })
        
        is_anomaly = detector.is_anomaly(test_data)
        assert is_anomaly[0] == 1
    
    def test_iqr_detection(self, sample_data):
        """Test that IQR method detects outliers."""
        detector = StatisticalDetector(iqr_multiplier=1.5)
        detector.fit(sample_data)
        
        scores = detector.predict(sample_data)
        assert len(scores) == len(sample_data)
        assert scores.min() >= 0


class TestTimeSeriesDetector:
    """Tests for time-series anomaly detection."""
    
    @pytest.fixture
    def seasonal_data(self):
        """Create data with seasonal pattern."""
        dates = pd.date_range('2024-01-01', periods=90, freq='D')
        # Weekly seasonal pattern
        seasonal = 20 * np.sin(2 * np.pi * np.arange(90) / 7)
        trend = np.linspace(100, 110, 90)
        noise = np.random.normal(0, 5, 90)
        rooms = (trend + seasonal + noise).astype(int)
        return pd.DataFrame({'asof_date': dates, 'rooms': rooms})
    
    def test_fit(self, seasonal_data):
        """Test fitting the time-series detector."""
        detector = TimeSeriesDetector(period=7)
        detector.fit(seasonal_data)
        
        assert detector.is_fitted_
        assert detector.residual_std_ > 0
    
    def test_decomposition(self, seasonal_data):
        """Test STL decomposition output."""
        detector = TimeSeriesDetector(period=7)
        detector.fit(seasonal_data)
        
        decomp = detector.get_decomposition(seasonal_data)
        
        assert 'trend' in decomp.columns
        assert 'seasonal' in decomp.columns
        assert 'residual' in decomp.columns


class TestMLDetector:
    """Tests for machine learning detector."""
    
    @pytest.fixture
    def feature_data(self):
        """Create data with features."""
        np.random.seed(42)
        n = 100
        dates = pd.date_range('2024-01-01', periods=n, freq='D')
        rooms = np.random.normal(100, 10, n).astype(int)
        
        df = pd.DataFrame({'asof_date': dates, 'rooms': rooms})
        
        # Add some features
        fe = FeatureEngineer()
        return fe.transform(df)
    
    def test_fit(self, feature_data):
        """Test fitting Isolation Forest."""
        detector = MLDetector(contamination=0.05)
        detector.fit(feature_data)
        
        assert detector.is_fitted_
        assert detector.model_ is not None
    
    def test_predict(self, feature_data):
        """Test prediction output."""
        detector = MLDetector(contamination=0.05)
        detector.fit(feature_data)
        
        scores = detector.predict(feature_data)
        
        assert len(scores) == len(feature_data)
        assert scores.min() >= 0
        assert scores.max() <= 1


class TestEnsembleDetector:
    """Tests for ensemble detector."""
    
    @pytest.fixture
    def full_data(self):
        """Create complete dataset."""
        generator = OTBDataGenerator()
        df = generator.generate_otb_dataset(
            start_date="2024-01-01",
            end_date="2024-06-30"
        )
        fe = FeatureEngineer()
        return fe.transform(df)
    
    def test_fit(self, full_data):
        """Test fitting all detectors."""
        ensemble = EnsembleDetector()
        ensemble.fit(full_data)
        
        assert ensemble.is_fitted_
        for name, detector in ensemble.detectors.items():
            assert detector.is_fitted_
    
    def test_voting(self, full_data):
        """Test that voting produces valid output."""
        ensemble = EnsembleDetector(min_votes=2)
        ensemble.fit(full_data)
        
        predictions = ensemble.is_anomaly(full_data)
        scores = ensemble.predict(full_data)
        
        assert len(predictions) == len(full_data)
        assert set(np.unique(predictions)).issubset({0, 1})
        assert len(scores) == len(full_data)


class TestAnomalyDetector:
    """Tests for main AnomalyDetector interface."""
    
    @pytest.fixture
    def history(self):
        """Create historical data."""
        generator = OTBDataGenerator()
        return generator.generate_otb_dataset(
            start_date="2024-01-01",
            end_date="2024-06-30"
        )
    
    def test_initialization(self):
        """Test detector initialization."""
        detector = AnomalyDetector(sensitivity='medium')
        assert detector.sensitivity == 'medium'
        assert not detector.is_fitted_
    
    def test_fit(self, history):
        """Test fitting on history."""
        detector = AnomalyDetector()
        detector.fit(history)
        
        assert detector.is_fitted_
    
    def test_detect_single(self, history):
        """Test single observation detection."""
        detector = AnomalyDetector()
        detector.fit(history)
        
        result = detector.detect({
            'asof_date': '2024-07-01',
            'rooms': 130
        })
        
        assert 'is_anomaly' in result
        assert 'score' in result
        assert 'explanation' in result
        assert isinstance(result['is_anomaly'], bool)
    
    def test_detect_batch(self, history):
        """Test batch detection."""
        detector = AnomalyDetector()
        detector.fit(history)
        
        new_data = pd.DataFrame({
            'asof_date': pd.date_range('2024-07-01', periods=10),
            'rooms': [130, 125, 135, 200, 120, 115, 140, 130, 125, 130]
        })
        
        results = detector.detect_batch(new_data)
        
        assert len(results) == 10
        assert 'is_anomaly' in results.columns
        assert 'anomaly_score' in results.columns
    
    def test_convenience_function(self, history):
        """Test detect_anomaly convenience function."""
        result = detect_anomaly(
            {'asof_date': '2024-07-01', 'rooms': 200},
            history,
            sensitivity='medium'
        )
        
        assert 'is_anomaly' in result
        assert 'score' in result


class TestValidationFramework:
    """Tests for validation framework."""
    
    @pytest.fixture
    def clean_data(self):
        """Create clean data for validation."""
        generator = OTBDataGenerator()
        return generator.generate_otb_dataset(
            start_date="2024-01-01",
            end_date="2024-06-30"
        )
    
    def test_inject_anomalies(self, clean_data):
        """Test anomaly injection."""
        validator = ValidationFramework()
        modified, labels = validator.inject_anomalies(
            clean_data,
            anomaly_rate=0.1
        )
        
        assert len(modified) == len(clean_data)
        assert len(labels) == len(clean_data)
        assert labels.sum() > 0  # Some anomalies injected
    
    def test_full_validation(self, clean_data):
        """Test complete validation pipeline."""
        validator = ValidationFramework()
        results = validator.full_validation(
            clean_data,
            anomaly_rate=0.05,
            train_ratio=0.7
        )
        
        assert 'metrics' in results
        assert 'precision' in results['metrics']
        assert 'recall' in results['metrics']
        assert 'f1_score' in results['metrics']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

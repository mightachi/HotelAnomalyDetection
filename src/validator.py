"""
Historical Validation Framework

This module provides tools to validate the anomaly detection approach
on historical data, including:
- Backtesting with synthetic anomaly injection
- Performance metrics (precision, recall, F1)
- Alert pattern analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta

from .anomaly_detector import AnomalyDetector


class ValidationFramework:
    """
    Framework for validating anomaly detection performance.
    
    Provides methods to:
    1. Inject synthetic anomalies into clean data
    2. Run backtesting on historical data
    3. Calculate precision, recall, and F1 score
    4. Analyze alert patterns
    
    Attributes:
        detector (AnomalyDetector): The detector to validate
        sensitivity (str): Sensitivity level for detector
    """
    
    def __init__(self, sensitivity: str = 'medium'):
        """
        Initialize validation framework.
        
        Args:
            sensitivity: Detector sensitivity level
        """
        self.sensitivity = sensitivity
        self.detector = AnomalyDetector(sensitivity=sensitivity)
    
    def inject_anomalies(
        self,
        data: pd.DataFrame,
        anomaly_rate: float = 0.05,
        anomaly_magnitude: float = 2.5,
        random_seed: int = 42
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Inject synthetic anomalies into data.
        
        Creates labeled data for validation by modifying random
        observations to be anomalous.
        
        Args:
            data: Original clean DataFrame
            anomaly_rate: Proportion of points to make anomalous
            anomaly_magnitude: Standard deviations for anomaly size
            random_seed: Random seed for reproducibility
            
        Returns:
            Tuple of (modified_data, ground_truth_labels)
            where labels are 1 for anomaly, 0 for normal
        """
        np.random.seed(random_seed)
        
        modified = data.copy()
        n = len(modified)
        n_anomalies = int(n * anomaly_rate)
        
        # Randomly select indices for anomalies
        anomaly_indices = np.random.choice(n, size=n_anomalies, replace=False)
        
        # Create ground truth labels
        labels = np.zeros(n, dtype=int)
        labels[anomaly_indices] = 1
        
        # Inject anomalies
        mean_rooms = modified['rooms'].mean()
        std_rooms = modified['rooms'].std()
        
        for idx in anomaly_indices:
            # Randomly spike up or down
            direction = np.random.choice([-1, 1])
            shift = direction * anomaly_magnitude * std_rooms
            modified.loc[modified.index[idx], 'rooms'] = int(
                max(0, modified.iloc[idx]['rooms'] + shift)
            )
        
        return modified, labels
    
    def backtest(
        self,
        data: pd.DataFrame,
        train_ratio: float = 0.7
    ) -> Dict[str, Any]:
        """
        Run backtesting on historical data.
        
        Splits data into training and test sets, fits on training,
        and evaluates on test data.
        
        Args:
            data: Full historical DataFrame
            train_ratio: Proportion of data for training
            
        Returns:
            Dict with test results including predictions and scores
        """
        n = len(data)
        train_size = int(n * train_ratio)
        
        train_data = data.iloc[:train_size].copy()
        test_data = data.iloc[train_size:].copy()
        
        # Fit on training data
        self.detector.fit(train_data)
        
        # Predict on test data
        test_results = self.detector.detect_batch(test_data)
        
        return {
            'train_size': train_size,
            'test_size': len(test_data),
            'test_data': test_data,
            'predictions': test_results['is_anomaly'].values,
            'scores': test_results['anomaly_score'].values,
            'anomaly_rate': test_results['is_anomaly'].mean()
        }
    
    def evaluate_with_labels(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate performance metrics with ground truth labels.
        
        Metrics calculated:
        - Precision: TP / (TP + FP) - how many alerts are real
        - Recall: TP / (TP + FN) - how many real anomalies caught
        - F1: Harmonic mean of precision and recall
        - Accuracy: Overall correctness
        
        Args:
            predictions: Predicted labels (0/1)
            ground_truth: True labels (0/1)
            
        Returns:
            Dict with precision, recall, f1, accuracy
        """
        predictions = np.asarray(predictions)
        ground_truth = np.asarray(ground_truth)
        
        # True positives, false positives, etc.
        tp = np.sum((predictions == 1) & (ground_truth == 1))
        fp = np.sum((predictions == 1) & (ground_truth == 0))
        fn = np.sum((predictions == 0) & (ground_truth == 1))
        tn = np.sum((predictions == 0) & (ground_truth == 0))
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / len(predictions) if len(predictions) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': accuracy,
            'true_positives': int(tp),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_negatives': int(tn)
        }
    
    def full_validation(
        self,
        clean_data: pd.DataFrame,
        anomaly_rate: float = 0.05,
        train_ratio: float = 0.7,
        random_seed: int = 42
    ) -> Dict[str, Any]:
        """
        Run complete validation with synthetic anomalies.
        
        Pipeline:
        1. Inject synthetic anomalies into data
        2. Split into train/test
        3. Fit detector on training
        4. Evaluate on test with ground truth
        
        Args:
            clean_data: Original clean data
            anomaly_rate: Rate of anomalies to inject
            train_ratio: Training data proportion
            random_seed: Random seed
            
        Returns:
            Complete validation results
        """
        # Inject anomalies
        data_with_anomalies, labels = self.inject_anomalies(
            clean_data, anomaly_rate, random_seed=random_seed
        )
        
        # Split data
        n = len(data_with_anomalies)
        train_size = int(n * train_ratio)
        
        train_data = data_with_anomalies.iloc[:train_size].copy()
        test_data = data_with_anomalies.iloc[train_size:].copy()
        test_labels = labels[train_size:]
        
        # Fit detector
        self.detector.fit(train_data)
        
        # Predict on test
        test_results = self.detector.detect_batch(test_data)
        predictions = test_results['is_anomaly'].values
        
        # Evaluate
        metrics = self.evaluate_with_labels(predictions, test_labels)
        
        return {
            'total_samples': n,
            'train_size': train_size,
            'test_size': len(test_data),
            'injected_anomalies': int(labels.sum()),
            'detected_anomalies': int(predictions.sum()),
            'metrics': metrics,
            'test_data': test_results
        }
    
    def analyze_alert_patterns(
        self,
        data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Analyze when the detector alerts vs stays quiet.
        
        Useful for understanding:
        - Alert frequency by day of week
        - Alert severity distribution
        - Patterns in alert timing
        
        Args:
            data: DataFrame with is_anomaly and anomaly_score columns
            
        Returns:
            Analysis of alert patterns
        """
        data = data.copy()
        data['asof_date'] = pd.to_datetime(data['asof_date'])
        data['day_of_week'] = data['asof_date'].dt.dayofweek
        data['month'] = data['asof_date'].dt.month
        
        anomalies = data[data['is_anomaly'] == 1]
        
        return {
            'total_observations': len(data),
            'total_alerts': len(anomalies),
            'alert_rate': len(anomalies) / len(data) if len(data) > 0 else 0,
            'alerts_by_dow': anomalies.groupby('day_of_week').size().to_dict(),
            'alerts_by_month': anomalies.groupby('month').size().to_dict(),
            'score_distribution': {
                'mean': data['anomaly_score'].mean(),
                'median': data['anomaly_score'].median(),
                'std': data['anomaly_score'].std(),
                'max': data['anomaly_score'].max()
            }
        }


def run_validation_report(
    data: pd.DataFrame,
    sensitivity: str = 'medium'
) -> Dict[str, Any]:
    """
    Run complete validation and generate report.
    
    Args:
        data: Historical OTB data
        sensitivity: Detector sensitivity
        
    Returns:
        Complete validation report
    """
    validator = ValidationFramework(sensitivity=sensitivity)
    
    # Full validation with synthetic anomalies
    validation_results = validator.full_validation(
        data,
        anomaly_rate=0.05,
        train_ratio=0.7
    )
    
    # Alert pattern analysis
    alert_patterns = validator.analyze_alert_patterns(
        validation_results['test_data']
    )
    
    return {
        'validation': validation_results,
        'alert_patterns': alert_patterns,
        'sensitivity': sensitivity
    }

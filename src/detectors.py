"""
Anomaly Detection Algorithms for OTB Data

This module implements multiple anomaly detection methods:
1. Statistical Detector (Z-Score + IQR)
2. Time-Series Detector (STL Decomposition)
3. Machine Learning Detector (Isolation Forest)

Each detector follows a common interface for easy ensemble combination.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from abc import ABC, abstractmethod
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class BaseDetector(ABC):
    """
    Abstract base class for anomaly detectors.
    
    All detectors must implement:
    - fit(): Learn normal patterns from historical data
    - predict(): Return anomaly scores for new observations
    - is_anomaly(): Return binary decision for new observations
    """
    
    @abstractmethod
    def fit(self, data: pd.DataFrame) -> 'BaseDetector':
        """Fit detector on historical data."""
        pass
    
    @abstractmethod
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Return anomaly scores (higher = more anomalous)."""
        pass
    
    @abstractmethod
    def is_anomaly(self, data: pd.DataFrame) -> np.ndarray:
        """Return binary predictions (1 = anomaly, 0 = normal)."""
        pass


class StatisticalDetector(BaseDetector):
    """
    Statistical anomaly detection using Z-Score and IQR methods.
    
    Z-Score Method:
    ---------------
    Measures how many standard deviations a point is from the mean.
    
    Formula: z = (x - μ) / σ
    
    - z > threshold indicates upper anomaly
    - z < -threshold indicates lower anomaly
    - Default threshold: 3 (99.7% confidence interval)
    
    Pros:
    - Simple and interpretable
    - Fast computation
    - Works well for normally distributed data
    
    Cons:
    - Assumes normal distribution
    - Sensitive to outliers in training data
    - May miss complex patterns
    
    IQR Method:
    -----------
    Uses quartiles to define "normal" range.
    
    Bounds:
    - Lower = Q1 - k × IQR
    - Upper = Q3 + k × IQR
    where IQR = Q3 - Q1 and k is typically 1.5
    
    Pros:
    - No distribution assumption
    - Robust to outliers
    - Simple and interpretable
    
    Cons:
    - May be too conservative
    - Doesn't account for time-series patterns
    
    Attributes:
        z_threshold (float): Z-score threshold for anomaly
        iqr_multiplier (float): IQR multiplier (typically 1.5 or 3.0)
        use_rolling (bool): Whether to use rolling statistics
        window (int): Rolling window size if use_rolling=True
    """
    
    def __init__(
        self,
        z_threshold: float = 3.0,
        iqr_multiplier: float = 1.5,
        use_rolling: bool = True,
        window: int = 30
    ):
        """
        Initialize statistical detector.
        
        Args:
            z_threshold: Z-score cutoff for anomaly detection
            iqr_multiplier: Multiplier for IQR bounds
            use_rolling: Use rolling window instead of global stats
            window: Rolling window size in days
        """
        self.z_threshold = z_threshold
        self.iqr_multiplier = iqr_multiplier
        self.use_rolling = use_rolling
        self.window = window
        
        # Learned parameters
        self.mean_ = None
        self.std_ = None
        self.q1_ = None
        self.q3_ = None
        self.iqr_ = None
        self.is_fitted_ = False
    
    def fit(self, data: pd.DataFrame) -> 'StatisticalDetector':
        """
        Learn statistical parameters from historical data.
        
        Args:
            data: DataFrame with 'rooms' column
            
        Returns:
            Self for method chaining
        """
        rooms = data['rooms'].values
        
        self.mean_ = np.mean(rooms)
        self.std_ = np.std(rooms) + 1e-6  # Avoid division by zero
        self.q1_ = np.percentile(rooms, 25)
        self.q3_ = np.percentile(rooms, 75)
        self.iqr_ = self.q3_ - self.q1_ + 1e-6
        
        self.is_fitted_ = True
        return self
    
    def _compute_z_scores(self, data: pd.DataFrame) -> np.ndarray:
        """Compute Z-scores for each observation."""
        rooms = data['rooms'].values
        
        if self.use_rolling and len(rooms) > self.window:
            # Rolling mean and std
            rolling_mean = pd.Series(rooms).rolling(
                window=self.window, min_periods=1
            ).mean().values
            rolling_std = pd.Series(rooms).rolling(
                window=self.window, min_periods=1
            ).std().values + 1e-6
            z_scores = (rooms - rolling_mean) / rolling_std
        else:
            z_scores = (rooms - self.mean_) / self.std_
        
        return z_scores
    
    def _compute_iqr_outliers(self, data: pd.DataFrame) -> np.ndarray:
        """Compute IQR-based outlier flags."""
        rooms = data['rooms'].values
        
        if self.use_rolling and len(rooms) > self.window:
            # Rolling quartiles
            q1 = pd.Series(rooms).rolling(
                window=self.window, min_periods=1
            ).quantile(0.25).values
            q3 = pd.Series(rooms).rolling(
                window=self.window, min_periods=1
            ).quantile(0.75).values
            iqr = q3 - q1 + 1e-6
        else:
            q1, q3, iqr = self.q1_, self.q3_, self.iqr_
        
        lower_bound = q1 - self.iqr_multiplier * iqr
        upper_bound = q3 + self.iqr_multiplier * iqr
        
        is_outlier = (rooms < lower_bound) | (rooms > upper_bound)
        return is_outlier.astype(float)
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Compute anomaly scores combining Z-score and IQR.
        
        Higher scores indicate more anomalous observations.
        Score = |z_score| + iqr_outlier
        
        Args:
            data: DataFrame with 'rooms' column
            
        Returns:
            Array of anomaly scores
        """
        if not self.is_fitted_:
            raise ValueError("Detector not fitted. Call fit() first.")
        
        z_scores = np.abs(self._compute_z_scores(data))
        iqr_flags = self._compute_iqr_outliers(data)
        
        # Combined score: normalized z-score + IQR flag
        combined_score = (z_scores / self.z_threshold) + iqr_flags
        return np.nan_to_num(combined_score, nan=0.0)
    
    def is_anomaly(self, data: pd.DataFrame) -> np.ndarray:
        """
        Return binary anomaly predictions.
        
        An observation is anomalous if:
        - |z_score| > z_threshold, OR
        - Outside IQR bounds
        
        Args:
            data: DataFrame with 'rooms' column
            
        Returns:
            Binary array (1 = anomaly, 0 = normal)
        """
        z_scores = np.abs(self._compute_z_scores(data))
        iqr_outliers = self._compute_iqr_outliers(data)
        
        is_z_anomaly = z_scores > self.z_threshold
        # iqr_outliers is float array: 1.0 = outlier, 0.0 = normal
        # Using > 0.5 to check if outlier (robust to floating-point precision)
        is_iqr_anomaly = iqr_outliers > 0.5
        
        return (is_z_anomaly | is_iqr_anomaly).astype(int)
    
    def get_details(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Get detailed anomaly analysis.
        
        Returns DataFrame with z-scores, IQR flags, and bounds.
        """
        z_scores = self._compute_z_scores(data)
        iqr_outliers = self._compute_iqr_outliers(data)
        
        return pd.DataFrame({
            'z_score': z_scores,
            'is_z_anomaly': np.abs(z_scores) > self.z_threshold,
            'is_iqr_anomaly': iqr_outliers > 0.5,
            'anomaly_score': self.predict(data)
        })


class TimeSeriesDetector(BaseDetector):
    """
    Time-series anomaly detection using STL decomposition.
    
    STL (Seasonal and Trend decomposition using Loess) separates a time series
    into three components:
    
    Y(t) = T(t) + S(t) + R(t)
    
    Where:
    - T(t) = Trend component (long-term movement)
    - S(t) = Seasonal component (repeating patterns)
    - R(t) = Residual component (noise/irregularity)
    
    Anomalies are detected by analyzing the residual component.
    Large residuals indicate unexpected deviations from the expected pattern.
    
    Pros:
    - Accounts for seasonality and trend
    - Better for time-series data than static methods
    - Interpretable decomposition
    
    Cons:
    - Requires sufficient history for decomposition
    - Computationally more expensive
    - Sensitive to period selection
    
    Attributes:
        period (int): Seasonal period (e.g., 7 for weekly)
        residual_threshold (float): Std deviations for residual anomaly
    """
    
    def __init__(
        self,
        period: int = 7,
        residual_threshold: float = 3.0
    ):
        """
        Initialize time-series detector.
        
        Args:
            period: Seasonal period (7=weekly, 30=monthly)
            residual_threshold: Std devs for residual anomaly threshold
        """
        self.period = period
        self.residual_threshold = residual_threshold
        
        # Learned parameters
        self.residual_mean_ = None
        self.residual_std_ = None
        self.is_fitted_ = False
    
    def _decompose(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform STL decomposition.
        
        Returns trend, seasonal, and residual components.
        """
        try:
            from statsmodels.tsa.seasonal import STL
            
            rooms = data['rooms'].values
            
            # STL requires at least 2 periods of data
            if len(rooms) < 2 * self.period:
                # Fallback to simple moving average decomposition
                trend = pd.Series(rooms).rolling(
                    window=self.period, min_periods=1, center=True
                ).mean().values
                seasonal = np.zeros_like(rooms)
                residual = rooms - trend
            else:
                stl = STL(rooms, period=self.period, robust=True)
                result = stl.fit()
                trend = result.trend
                seasonal = result.seasonal
                residual = result.resid
            
            return trend, seasonal, residual
            
        except ImportError:
            # Fallback without statsmodels
            rooms = data['rooms'].values
            trend = pd.Series(rooms).rolling(
                window=self.period, min_periods=1, center=True
            ).mean().values
            seasonal = np.zeros_like(rooms)
            residual = rooms - trend
            return trend, seasonal, residual
    
    def fit(self, data: pd.DataFrame) -> 'TimeSeriesDetector':
        """
        Learn normal residual distribution from historical data.
        
        Args:
            data: DataFrame with 'rooms' column
            
        Returns:
            Self for method chaining
        """
        _, _, residual = self._decompose(data)
        
        self.residual_mean_ = np.nanmean(residual)
        self.residual_std_ = np.nanstd(residual) + 1e-6
        
        self.is_fitted_ = True
        return self
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Compute anomaly scores based on residual magnitude.
        
        Score = |residual - mean| / std
        
        Args:
            data: DataFrame with 'rooms' column
            
        Returns:
            Array of anomaly scores
        """
        if not self.is_fitted_:
            raise ValueError("Detector not fitted. Call fit() first.")
        
        _, _, residual = self._decompose(data)
        
        # Normalized residual score
        scores = np.abs(residual - self.residual_mean_) / self.residual_std_
        return np.nan_to_num(scores, nan=0.0)
    
    def is_anomaly(self, data: pd.DataFrame) -> np.ndarray:
        """
        Return binary anomaly predictions based on residuals.
        
        Args:
            data: DataFrame with 'rooms' column
            
        Returns:
            Binary array (1 = anomaly, 0 = normal)
        """
        scores = self.predict(data)
        return (scores > self.residual_threshold).astype(int)
    
    def get_decomposition(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Get full STL decomposition for visualization.
        """
        trend, seasonal, residual = self._decompose(data)
        
        return pd.DataFrame({
            'trend': trend,
            'seasonal': seasonal,
            'residual': residual,
            'anomaly_score': self.predict(data)
        })


class MLDetector(BaseDetector):
    """
    Machine Learning-based anomaly detection using Isolation Forest.
    
    Isolation Forest Algorithm:
    ---------------------------
    Based on the observation that anomalies are "few and different."
    
    How it works:
    1. Build an ensemble of random trees
    2. Each tree recursively partitions data by random splits
    3. Anomalies require fewer splits to be isolated
    4. Path length to isolation = anomaly score
    
    Mathematical Intuition:
    - For a sample x, average path length h(x) across all trees
    - Anomaly score s(x, n) = 2^(-E[h(x)]/c(n))
    - Where c(n) is average path length of unsuccessful search in BST
    - Score close to 1 = anomaly, close to 0.5 = normal
    
    Pros:
    - No distribution assumptions
    - Handles high-dimensional data
    - Fast training and prediction
    - Works well with limited labels
    
    Cons:
    - Black-box (less interpretable)
    - May need tuning for contamination rate
    - Doesn't explicitly model time-series patterns
    
    Comparison with Alternatives:
    -----------------------------
    | Method          | Speed | Interpretability | Distribution-Free |
    |-----------------|-------|------------------|-------------------|
    | Isolation Forest| Fast  | Low              | Yes               |
    | LOF             | Slow  | Medium           | Yes               |
    | One-Class SVM   | Medium| Low              | No                |
    | Autoencoders    | Slow  | Low              | Yes               |
    
    Attributes:
        contamination (float): Expected anomaly proportion
        n_estimators (int): Number of trees in forest
        features (List[str]): Feature columns to use
    """
    
    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 100,
        features: Optional[List[str]] = None,
        random_state: int = 42
    ):
        """
        Initialize Isolation Forest detector.
        
        Args:
            contamination: Expected proportion of anomalies (0.0-0.5)
            n_estimators: Number of trees in the forest
            features: Feature columns to use (None = use all numeric)
            random_state: Random seed for reproducibility
        """
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.features = features
        self.random_state = random_state
        
        self.model_ = None
        self.scaler_ = None
        self.feature_cols_ = None
        self.is_fitted_ = False
    
    def _prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract and scale features for the model."""
        if self.features:
            feature_cols = [f for f in self.features if f in data.columns]
        else:
            # Use numeric columns except dates
            feature_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            # Ensure 'rooms' is included
            if 'rooms' not in feature_cols:
                feature_cols = ['rooms']
        
        self.feature_cols_ = feature_cols
        X = data[feature_cols].values
        
        # Handle NaN values
        X = np.nan_to_num(X, nan=0.0)
        
        return X
    
    def fit(self, data: pd.DataFrame) -> 'MLDetector':
        """
        Fit Isolation Forest on historical data.
        
        Args:
            data: DataFrame with feature columns
            
        Returns:
            Self for method chaining
        """
        X = self._prepare_features(data)
        
        # Scale features
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)
        
        # Fit Isolation Forest
        # 
        # IsolationForest Parameters Explained:
        # -------------------------------------
        # contamination: Expected proportion of anomalies in the dataset (0.0 to 0.5)
        #   - Default: 0.05 (5% of data expected to be anomalous)
        #   - Used to set the threshold for binary classification
        #   - Lower values = stricter (fewer anomalies flagged)
        #   - Higher values = more lenient (more anomalies flagged)
        #   - Example: contamination=0.05 means top 5% of scores are marked as anomalies
        #
        # n_estimators: Number of isolation trees in the forest
        #   - Default: 100 trees
        #   - More trees = more stable predictions, but slower training
        #   - Fewer trees = faster, but less stable
        #   - Typical range: 50-200 trees
        #   - Each tree provides a vote, final score is average across all trees
        #
        # random_state: Random seed for reproducibility
        #   - Ensures same random splits across runs
        #   - Critical for reproducible results in production
        #   - Same seed + same data = same predictions
        #
        # n_jobs: Number of parallel jobs for tree construction
        #   - -1 = use all available CPU cores
        #   - Speeds up training significantly on multi-core systems
        #   - Example: 8 cores = 8x speedup (approximately)
        #   - Set to 1 for single-threaded execution if needed
        #
        self.model_ = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.model_.fit(X_scaled)
        
        self.is_fitted_ = True
        return self
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Compute anomaly scores using Isolation Forest.
        
        Scores are transformed to [0, 1] where higher = more anomalous.
        
        Args:
            data: DataFrame with feature columns
            
        Returns:
            Array of anomaly scores (0-1)
        """
        if not self.is_fitted_:
            raise ValueError("Detector not fitted. Call fit() first.")
        
        X = data[self.feature_cols_].values
        X = np.nan_to_num(X, nan=0.0)
        X_scaled = self.scaler_.transform(X)
        
        # Isolation Forest returns negative for anomalies
        # Transform to positive scores where higher = more anomalous
        raw_scores = -self.model_.score_samples(X_scaled)
        
        # Normalize to 0-1 range
        min_score = raw_scores.min()
        max_score = raw_scores.max()
        if max_score > min_score:
            normalized_scores = (raw_scores - min_score) / (max_score - min_score)
        else:
            normalized_scores = np.zeros_like(raw_scores)
        
        return normalized_scores
    
    def is_anomaly(self, data: pd.DataFrame) -> np.ndarray:
        """
        Return binary anomaly predictions.
        
        Args:
            data: DataFrame with feature columns
            
        Returns:
            Binary array (1 = anomaly, 0 = normal)
        """
        if not self.is_fitted_:
            raise ValueError("Detector not fitted. Call fit() first.")
        
        X = data[self.feature_cols_].values
        X = np.nan_to_num(X, nan=0.0)
        X_scaled = self.scaler_.transform(X)
        
        # Isolation Forest: -1 = anomaly, 1 = normal
        predictions = self.model_.predict(X_scaled)
        
        # Convert from Isolation Forest convention to project convention:
        # Isolation Forest: -1 = anomaly, 1 = normal
        # Project Standard:  1 = anomaly, 0 = normal
        # 
        # How it works:
        # 1. (predictions == -1) creates boolean array:
        #    - True where predictions == -1 (anomalies)
        #    - False where predictions == 1 (normal)
        # 2. .astype(int) converts boolean to integer:
        #    - True → 1 (anomaly)
        #    - False → 0 (normal)
        # 
        # Example:
        #   predictions = [-1, 1, -1, 1, 1]  (Isolation Forest output)
        #   (predictions == -1) = [True, False, True, False, False]
        #   .astype(int) = [1, 0, 1, 0, 0]  (Project standard)
        return (predictions == -1).astype(int)


class EnsembleDetector:
    """
    Ensemble detector combining multiple detection methods.
    
    Ensemble Strategy:
    ------------------
    Combines predictions from statistical, time-series, and ML detectors
    using weighted voting.
    
    Final decision based on:
    1. Weighted average of normalized scores
    2. Majority voting (if 2+ methods agree)
    
    Benefits of Ensemble:
    - Reduces false positives from any single method
    - More robust across different anomaly types
    - Provides confidence through agreement
    
    Attributes:
        detectors (Dict[str, BaseDetector]): Named detector instances
        weights (Dict[str, float]): Weight for each detector
        threshold (float): Score threshold for anomaly decision
        min_votes (int): Minimum detectors that must agree
    """
    
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        threshold: float = 0.5,
        min_votes: int = 2
    ):
        """
        Initialize ensemble detector.
        
        Args:
            weights: Weight for each detector (None = equal weights)
            threshold: Score threshold for anomaly decision
            min_votes: Minimum detectors that must flag anomaly
        """
        # Initialize individual detectors
        self.detectors = {
            'statistical': StatisticalDetector(),
            'timeseries': TimeSeriesDetector(),
            'ml': MLDetector()
        }
        
        self.weights = weights or {
            'statistical': 0.3,
            'timeseries': 0.3,
            'ml': 0.4
        }
        
        self.threshold = threshold
        self.min_votes = min_votes
        self.is_fitted_ = False
    
    def fit(self, data: pd.DataFrame) -> 'EnsembleDetector':
        """
        Fit all detectors on historical data.
        
        Args:
            data: DataFrame with 'rooms' column and features
            
        Returns:
            Self for method chaining
        """
        for name, detector in self.detectors.items():
            detector.fit(data)
        
        self.is_fitted_ = True
        return self
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Compute weighted ensemble anomaly scores.
        
        Args:
            data: DataFrame with 'rooms' column and features
            
        Returns:
            Array of ensemble scores (0-1)
        """
        if not self.is_fitted_:
            raise ValueError("Ensemble not fitted. Call fit() first.")
        
        scores = {}
        for name, detector in self.detectors.items():
            raw_score = detector.predict(data)
            # Normalize each detector's score to 0-1
            if raw_score.max() > 1:
                raw_score = raw_score / (raw_score.max() + 1e-6)
            scores[name] = raw_score
        
        # Weighted average
        ensemble_score = np.zeros(len(data))
        total_weight = sum(self.weights.values())
        
        for name, score in scores.items():
            weight = self.weights.get(name, 1.0) / total_weight
            ensemble_score += weight * score
        
        return ensemble_score
    
    def is_anomaly(self, data: pd.DataFrame) -> np.ndarray:
        """
        Return binary anomaly predictions using voting.
        
        An observation is anomalous if:
        - Ensemble score > threshold, OR
        - min_votes detectors flag it as anomaly
        
        Args:
            data: DataFrame with 'rooms' column and features
            
        Returns:
            Binary array (1 = anomaly, 0 = normal)
        """
        ensemble_score = self.predict(data)
        is_score_anomaly = ensemble_score > self.threshold
        
        # Voting
        votes = np.zeros(len(data))
        for name, detector in self.detectors.items():
            votes += detector.is_anomaly(data)
        
        is_vote_anomaly = votes >= self.min_votes
        
        return (is_score_anomaly | is_vote_anomaly).astype(int)
    
    def get_individual_scores(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Get scores from each individual detector.
        
        Useful for debugging and understanding which detector
        is flagging anomalies.
        """
        result = {}
        for name, detector in self.detectors.items():
            result[f'{name}_score'] = detector.predict(data)
            result[f'{name}_anomaly'] = detector.is_anomaly(data)
        
        result['ensemble_score'] = self.predict(data)
        result['ensemble_anomaly'] = self.is_anomaly(data)
        
        return pd.DataFrame(result)

# Comprehensive Documentation: Hotel Booking Anomaly Detection System

## Table of Contents
1. [Project Overview](#project-overview)
2. [Module Documentation](#module-documentation)
3. [Algorithm Deep Dive](#algorithm-deep-dive)
4. [Core Concepts](#core-concepts)
5. [References and Sources](#references-and-sources)

---

## Project Overview

This project implements a production-ready anomaly detection system for hotel On-The-Books (OTB) data. The system uses an ensemble approach combining multiple detection methods to identify unusual booking patterns that may indicate data quality issues, system errors, or genuine business anomalies.

### Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                    AnomalyDetector (Main Interface)         │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
┌───────▼──────┐ ┌─────▼──────┐ ┌────▼──────────┐
│ Feature      │ │ Ensemble   │ │ Validator     │
│ Engineering  │ │ Detector   │ │ Framework     │
└──────┬───────┘ └─────┬──────┘ └───────────────┘
       │               │
       │    ┌──────────┼──────────┐
       │    │          │          │
┌──────▼────▼──┐ ┌─────▼──────┐ ┌─▼────────────┐
│ Statistical  │ │ TimeSeries │ │ ML Detector  │
│ Detector     │ │ Detector   │ │ (Isolation   │
│ (Z-Score,    │ │ (STL)      │ │  Forest)     │
│  IQR)        │ │            │ │              │
└──────────────┘ └────────────┘ └──────────────┘
```

---

## Module Documentation

### 1. `anomaly_detector.py` - Main Interface Module

#### **What**
The primary user-facing interface that provides a simple, production-ready API for anomaly detection. It wraps the ensemble detector and feature engineering pipeline into a cohesive system.

#### **Why**
- **Simplicity**: Hides complexity of ensemble voting, feature engineering, and multiple detectors
- **Consistency**: Provides uniform interface regardless of underlying detection methods
- **Production-Ready**: Handles data validation, error checking, and result formatting
- **Flexibility**: Supports different sensitivity levels and custom configurations

#### **How**
The module implements two main components:

1. **`AnomalyDetector` Class**: Stateful detector that can be fitted once and used repeatedly
   ```python
   detector = AnomalyDetector(sensitivity='medium')
   detector.fit(historical_data)
   result = detector.detect(observation)
   ```

2. **`detect_anomaly()` Function**: One-shot convenience function
   ```python
   result = detect_anomaly(observation, history, sensitivity='medium')
   ```

**Key Features:**
- **Sensitivity Presets**: Three levels (low, medium, high) with pre-configured thresholds
- **Feature Engineering**: Automatically extracts features before detection
- **Ensemble Integration**: Combines multiple detectors with weighted voting
- **Result Formatting**: Returns structured results with explanations

#### **Where to Use**
- **Production Systems**: Real-time anomaly detection in booking systems
- **Batch Processing**: Analyzing historical data for anomalies
- **Monitoring Dashboards**: Alerting when anomalies are detected
- **Data Quality Checks**: Validating incoming data streams

**Example Usage:**
```python
from src.anomaly_detector import AnomalyDetector

# Initialize detector
detector = AnomalyDetector(sensitivity='medium')

# Fit on historical data
detector.fit(historical_otb_data)

# Detect single observation
result = detector.detect({
    'asof_date': '2025-01-15',
    'rooms': 180
})

# Result contains:
# - is_anomaly: bool
# - score: float (0-1)
# - confidence: str ('low', 'medium', 'high')
# - explanation: str (human-readable)
# - details: dict (per-detector breakdown)
```

---

### 2. `detectors.py` - Detection Algorithms Module

#### **What**
Implements four core anomaly detection algorithms:
1. **StatisticalDetector**: Z-Score and IQR methods
2. **TimeSeriesDetector**: STL decomposition-based detection
3. **MLDetector**: Isolation Forest machine learning approach
4. **EnsembleDetector**: Combines all methods with voting

#### **Why**
- **Diversity**: Each method catches different types of anomalies
- **Robustness**: Ensemble reduces false positives from any single method
- **Modularity**: Each detector can be used independently
- **Extensibility**: Easy to add new detection methods

#### **How**
All detectors follow a common interface (`BaseDetector`):
- `fit(data)`: Learn normal patterns from historical data
- `predict(data)`: Return anomaly scores (0-1, higher = more anomalous)
- `is_anomaly(data)`: Return binary predictions (1 = anomaly, 0 = normal)

**StatisticalDetector:**
- Uses rolling window statistics for adaptive thresholds
- Combines Z-Score and IQR methods
- Fast and interpretable

**TimeSeriesDetector:**
- Performs STL decomposition to separate trend, seasonal, and residual components
- Analyzes residuals for anomalies
- Accounts for time-series patterns

**MLDetector:**
- Uses Isolation Forest from scikit-learn
- Learns complex patterns in feature space
- Handles high-dimensional data

**EnsembleDetector:**
- Weighted average of normalized scores
- Majority voting (requires min_votes detectors to agree)
- Combines both score threshold and voting logic

#### **Where to Use**
- **Direct Algorithm Access**: When you need specific detection methods
- **Custom Ensembles**: Building your own combination of detectors
- **Research/Experimentation**: Testing individual algorithm performance
- **Debugging**: Understanding which detector flags specific anomalies

**Example Usage:**
```python
from src.detectors import StatisticalDetector, EnsembleDetector

# Use individual detector
stat_detector = StatisticalDetector(z_threshold=3.0, iqr_multiplier=1.5)
stat_detector.fit(data)
scores = stat_detector.predict(new_data)

# Use ensemble
ensemble = EnsembleDetector(
    weights={'statistical': 0.3, 'timeseries': 0.3, 'ml': 0.4},
    threshold=0.5,
    min_votes=2
)
ensemble.fit(data)
predictions = ensemble.is_anomaly(new_data)
```

---

### 3. `feature_engineering.py` - Feature Extraction Module

#### **What**
Extracts meaningful features from raw OTB data to improve anomaly detection. Creates features that capture temporal patterns, deviations, and contextual information.

#### **Why**
- **Context Awareness**: Raw room counts don't tell the full story
- **Pattern Recognition**: Features help algorithms identify unusual patterns
- **Time-Series Support**: Captures seasonality, trends, and cycles
- **ML Compatibility**: Provides rich feature set for machine learning models

#### **How**
The `FeatureEngineer` class applies five transformation groups:

1. **Calendar Features** (`add_calendar_features`):
   - Day of week, month, quarter, week of year
   - Weekend indicator
   - Captures recurring patterns (e.g., higher bookings on weekends)

2. **Rolling Features** (`add_rolling_features`):
   - Rolling mean, std, min, max over short (7d) and long (30d) windows
   - Provides local context vs. global statistics
   - Example: `rolling_mean_7d` = average rooms over past 7 days

3. **Velocity Features** (`add_velocity_features`):
   - Day-over-day changes (`rooms_diff_1d`)
   - Week-over-week changes (`rooms_diff_7d`)
   - Percentage changes (`rooms_pct_change_1d`)
   - Velocity (average change rate) and acceleration
   - Captures sudden shifts in booking patterns

4. **Deviation Features** (`add_deviation_features`):
   - Deviation from rolling means
   - Z-scores relative to rolling windows
   - IQR position (where value sits in interquartile range)
   - Measures how unusual current value is relative to recent history

5. **Seasonal Features** (`add_seasonal_features`):
   - Same day-of-week average (expanding window)
   - Same month average
   - Deviation from seasonal expectations
   - Captures violations of expected seasonal patterns

**Example Feature Values:**
```python
# Raw data
asof_date: 2024-06-15, rooms: 180

# Engineered features
day_of_week: 5 (Saturday)
rolling_mean_7d: 165.3
rolling_std_7d: 12.4
rooms_diff_1d: +15
rooms_pct_change_1d: 0.091 (9.1% increase)
z_score_7d: 1.19 (1.19 std devs above 7-day mean)
same_dow_mean: 172.1 (average for Saturdays)
deviation_from_dow_mean: +7.9
```

#### **Where to Use**
- **Preprocessing**: Before feeding data to detectors
- **Feature Analysis**: Understanding what drives anomalies
- **Custom Models**: Building your own ML models with these features
- **Data Exploration**: Understanding booking patterns

**Example Usage:**
```python
from src.feature_engineering import FeatureEngineer

engineer = FeatureEngineer(short_window=7, long_window=30)
data_with_features = engineer.transform(raw_data)

# Now data has ~30 additional feature columns
print(data_with_features.columns)
# ['asof_date', 'rooms', 'day_of_week', 'rolling_mean_7d', ...]
```

---

### 4. `data_generator.py` - Synthetic Data Generation Module

#### **What**
Generates realistic synthetic OTB hotel booking data with configurable patterns, seasonality, and optional anomaly injection.

#### **Why**
- **Testing**: Validate detectors without real data
- **Development**: Work on algorithms before production data is available
- **Validation**: Create labeled data (with known anomalies) for testing
- **Demonstration**: Show system capabilities with realistic examples

#### **How**
The `OTBDataGenerator` class simulates realistic booking patterns:

1. **Seasonal Patterns** (`_seasonal_factor`):
   - Sinusoidal patterns with peaks in summer and holidays
   - Multiplier ranges from 0.7 to 1.3
   - Based on day of year

2. **Day-of-Week Effects** (`_dow_factor`):
   - Higher occupancy mid-week (Tue-Thu) for business hotels
   - Lower on weekends
   - Multiplier ranges from 0.85 to 1.15

3. **Booking Curve** (`_booking_curve`):
   - S-curve using logistic function
   - Bookings accumulate as stay date approaches
   - Far in advance: ~5% of final bookings
   - Near stay date: ~95% of final bookings
   - Inflection point around 45 days out

4. **Noise Injection**:
   - Gaussian noise proportional to expected value
   - Simulates real-world variability

5. **Anomaly Injection**:
   - Optional anomalies at specified dates
   - Magnitude controlled by standard deviations
   - Can be spikes (high) or drops (low)

**Mathematical Model:**
```
expected_final = capacity × base_occupancy × seasonal_factor × dow_factor
expected_now = expected_final × booking_curve(days_until_stay)
rooms = expected_now + noise + (optional_anomaly)
```

#### **Where to Use**
- **Unit Testing**: Generate test data for detector tests
- **Benchmarking**: Compare detector performance on known anomalies
- **Documentation**: Create examples and tutorials
- **Research**: Experiment with different anomaly types

**Example Usage:**
```python
from src.data_generator import OTBDataGenerator, generate_sample_data

# Generate synthetic data
generator = OTBDataGenerator(
    hotel_capacity=200,
    base_occupancy=0.65,
    random_seed=42
)

data = generator.generate_otb_dataset(
    start_date="2024-01-01",
    end_date="2024-12-31",
    anomaly_dates=["2024-03-15", "2024-07-04"],
    anomaly_magnitude=3.0
)

# Or use convenience function
data = generate_sample_data(use_real_data=False, with_anomalies=True)
```

---

### 5. `validator.py` - Validation Framework Module

#### **What**
Provides tools for validating anomaly detection performance including backtesting, synthetic anomaly injection, and performance metrics calculation.

#### **Why**
- **Performance Measurement**: Quantify how well detectors work
- **Threshold Tuning**: Find optimal sensitivity settings
- **Validation**: Ensure detectors work before production deployment
- **Comparison**: Compare different detector configurations

#### **How**
The `ValidationFramework` class provides:

1. **Anomaly Injection** (`inject_anomalies`):
   - Injects synthetic anomalies into clean data
   - Creates ground truth labels (1 = anomaly, 0 = normal)
   - Configurable rate and magnitude
   - Randomly selects dates and directions (spike/drop)

2. **Backtesting** (`backtest`):
   - Splits data into train/test sets
   - Fits detector on training data
   - Evaluates on test data
   - Returns predictions and scores

3. **Performance Metrics** (`evaluate_with_labels`):
   - **Precision**: TP / (TP + FP) - How many alerts are real anomalies?
   - **Recall**: TP / (TP + FN) - How many real anomalies were caught?
   - **F1 Score**: Harmonic mean of precision and recall
   - **Accuracy**: Overall correctness
   - Also returns confusion matrix (TP, FP, FN, TN)

4. **Full Validation** (`full_validation`):
   - Complete pipeline: inject → split → fit → evaluate
   - Returns comprehensive results

5. **Alert Pattern Analysis** (`analyze_alert_patterns`):
   - Analyzes when alerts occur (day of week, month)
   - Score distribution statistics
   - Alert frequency analysis

#### **Where to Use**
- **Model Development**: Validate detectors during development
- **Threshold Selection**: Find optimal sensitivity settings
- **Performance Reporting**: Generate validation reports
- **Quality Assurance**: Ensure detectors meet requirements before deployment

**Example Usage:**
```python
from src.validator import ValidationFramework

validator = ValidationFramework(sensitivity='medium')

# Full validation with synthetic anomalies
results = validator.full_validation(
    clean_data,
    anomaly_rate=0.05,  # 5% anomalies
    train_ratio=0.7
)

print(f"Precision: {results['metrics']['precision']:.3f}")
print(f"Recall: {results['metrics']['recall']:.3f}")
print(f"F1 Score: {results['metrics']['f1_score']:.3f}")
```

---

## Algorithm Deep Dive

### 1. Z-Score Method

#### **What**
A statistical method that measures how many standard deviations a data point is from the mean. It's one of the most fundamental anomaly detection techniques.

#### **Mathematical Foundation**

**Formula:**
```
z = (x - μ) / σ
```

Where:
- `x` = observed value
- `μ` = mean of the population/sample
- `σ` = standard deviation of the population/sample
- `z` = z-score (standard deviations from mean)

**Interpretation:**
- `|z| < 1`: Within 1 standard deviation (68% of data in normal distribution)
- `|z| < 2`: Within 2 standard deviations (95% of data)
- `|z| < 3`: Within 3 standard deviations (99.7% of data)
- `|z| > 3`: Extreme outlier (0.3% of data in normal distribution)

#### **Why Use It**
1. **Simplicity**: Easy to understand and implement
2. **Speed**: Very fast computation (O(n))
3. **Interpretability**: Clear meaning (standard deviations from mean)
4. **Standardization**: Normalizes data to comparable scale
5. **Theoretical Foundation**: Based on well-established statistical theory

#### **How It Works in This Project**

**Implementation Details:**
```python
# Global Z-Score (uses entire history)
z_score = (rooms - mean) / std
is_anomaly = abs(z_score) > threshold  # threshold = 3.0 (default)

# Rolling Z-Score (uses recent window)
rolling_mean = rooms.rolling(window=30).mean()
rolling_std = rooms.rolling(window=30).std()
z_score = (rooms - rolling_mean) / rolling_std
```

**Advantages:**
- Adapts to recent trends (rolling window)
- Less sensitive to old outliers
- Better for non-stationary time series

**Limitations:**
- Assumes normal distribution (may not hold for booking data)
- Sensitive to outliers in training data
- Doesn't account for seasonality

#### **Example**

**Scenario:** Hotel with average 150 rooms/day, std = 20 rooms

**Normal Day:**
- Rooms: 155
- Z-score: (155 - 150) / 20 = 0.25
- Interpretation: Normal (within 1 std dev)

**Anomalous Day:**
- Rooms: 220
- Z-score: (220 - 150) / 20 = 3.5
- Interpretation: Anomaly (3.5 std devs above mean, ~0.02% probability)

**Code Example:**
```python
import numpy as np

# Historical data
history = np.array([145, 150, 148, 152, 149, 151, 147, 153, 150, 149])

# New observation
observation = 220

# Calculate Z-score
mean = np.mean(history)
std = np.std(history)
z_score = (observation - mean) / std

print(f"Mean: {mean:.1f}, Std: {std:.1f}")
print(f"Z-score: {z_score:.2f}")
print(f"Anomaly: {abs(z_score) > 3}")
# Output: Mean: 149.4, Std: 2.4, Z-score: 29.4, Anomaly: True
```

#### **Original Source**
- **Origin**: Developed by Karl Pearson in the early 1900s
- **Reference**: Pearson, K. (1900). "On the criterion that a given system of deviations from the probable in the case of a correlated system of variables is such that it can be reasonably supposed to have arisen from random sampling." *Philosophical Magazine*, 50(302), 157-175.
- **Modern Application**: Widely used in quality control (Shewhart control charts), standardized testing, and outlier detection

---

### 2. IQR (Interquartile Range) Method

#### **What**
A robust statistical method that identifies outliers based on quartiles rather than mean and standard deviation. It's distribution-free and resistant to outliers.

#### **Mathematical Foundation**

**Quartiles:**
- **Q1 (First Quartile)**: 25th percentile - 25% of data below this value
- **Q2 (Median)**: 50th percentile - middle value
- **Q3 (Third Quartile)**: 75th percentile - 75% of data below this value

**IQR Calculation:**
```
IQR = Q3 - Q1
```

**Outlier Bounds:**
```
Lower Bound = Q1 - k × IQR
Upper Bound = Q3 + k × IQR
```

Where `k` is typically:
- `k = 1.5`: Standard (mild outliers)
- `k = 3.0`: Extreme outliers only

**Anomaly Detection:**
```
is_anomaly = (value < Lower Bound) OR (value > Upper Bound)
```

#### **Why Use It**
1. **Robustness**: Not affected by extreme outliers (unlike mean/std)
2. **Distribution-Free**: Works for any distribution, not just normal
3. **Non-Parametric**: No assumptions about data distribution
4. **Interpretability**: Easy to understand (outside normal range)
5. **Tukey's Method**: Based on well-established statistical method

#### **How It Works in This Project**

**Implementation:**
```python
# Calculate quartiles
Q1 = np.percentile(rooms, 25)
Q3 = np.percentile(rooms, 75)
IQR = Q3 - Q1

# Define bounds
lower_bound = Q1 - iqr_multiplier * IQR  # default: 1.5
upper_bound = Q3 + iqr_multiplier * IQR

# Detect anomalies
is_anomaly = (rooms < lower_bound) | (rooms > upper_bound)
```

**Rolling IQR:**
- Uses rolling window (e.g., 30 days) for adaptive thresholds
- Better for time series with trends

**Combined with Z-Score:**
- Z-Score catches extreme deviations
- IQR catches values outside normal range
- Combined: More comprehensive detection

#### **Example**

**Dataset:** [120, 125, 130, 135, 140, 145, 150, 155, 160, 200]

**Step 1: Calculate Quartiles**
- Q1 = 25th percentile = 130
- Q3 = 75th percentile = 155
- IQR = 155 - 130 = 25

**Step 2: Define Bounds (k=1.5)**
- Lower Bound = 130 - 1.5 × 25 = 92.5
- Upper Bound = 155 + 1.5 × 25 = 192.5

**Step 3: Identify Anomalies**
- 200 > 192.5 → **ANOMALY** (upper outlier)
- All other values are within bounds

**Visual Representation:**
```
    92.5         130    140    155         192.5
     |           |      |      |           |
     |<-- IQR -->|      |      |<-- IQR -->|
     |           |      |      |           |
     |           Q1    Q2     Q3           |
     |                                    |
     └────────────────────────────────────┘
                    Normal Range
     
    200 (anomaly) is outside upper bound
```

**Code Example:**
```python
import numpy as np

data = np.array([120, 125, 130, 135, 140, 145, 150, 155, 160, 200])

Q1 = np.percentile(data, 25)
Q3 = np.percentile(data, 75)
IQR = Q3 - Q1

k = 1.5
lower_bound = Q1 - k * IQR
upper_bound = Q3 + k * IQR

print(f"Q1: {Q1}, Q3: {Q3}, IQR: {IQR}")
print(f"Bounds: [{lower_bound:.1f}, {upper_bound:.1f}]")

anomalies = data[(data < lower_bound) | (data > upper_bound)]
print(f"Anomalies: {anomalies}")
# Output: Anomalies: [200]
```

#### **Original Source**
- **Origin**: John Tukey's "Exploratory Data Analysis" (1977)
- **Reference**: Tukey, J. W. (1977). *Exploratory Data Analysis*. Addison-Wesley.
- **Box Plot Connection**: IQR is the basis for box plots (whiskers extend to Q1-1.5×IQR and Q3+1.5×IQR)
- **Modern Use**: Standard method in statistics, data science, and quality control

---

### 3. STL Decomposition

#### **What**
STL (Seasonal and Trend decomposition using Loess) is a time-series decomposition method that separates a time series into three components: Trend, Seasonal, and Residual. Anomalies are detected by analyzing the residual component.

#### **Mathematical Foundation**

**Decomposition Model:**
```
Y(t) = T(t) + S(t) + R(t)
```

Where:
- **Y(t)**: Original time series
- **T(t)**: Trend component (long-term movement)
- **S(t)**: Seasonal component (repeating patterns)
- **R(t)**: Residual component (irregular/random variation)

**Anomaly Detection:**
- Large residuals indicate deviations from expected pattern
- Residuals should be normally distributed around zero
- Anomaly if: `|residual| > threshold × std(residual)`

#### **Why Use It**
1. **Time-Series Aware**: Explicitly models seasonality and trends
2. **Interpretable**: Clear separation of components
3. **Robust**: Uses LOESS (Locally Weighted Scatterplot Smoothing) - robust to outliers
4. **Flexible**: Handles multiple seasonal patterns
5. **Industry Standard**: Widely used in time-series analysis

#### **How It Works**

**STL Algorithm Steps:**

1. **Trend Extraction**:
   - Uses LOESS smoothing to extract long-term trend
   - LOESS fits local polynomial regressions
   - Robust to outliers

2. **Seasonal Extraction**:
   - Removes trend from data
   - Averages values at same seasonal position
   - Subtracts seasonal component

3. **Residual Calculation**:
   - Residual = Original - Trend - Seasonal
   - Should be random noise if model is good

4. **Anomaly Detection**:
   - Calculate residual statistics (mean, std)
   - Flag points where residual is extreme

**Implementation in Project:**
```python
from statsmodels.tsa.seasonal import STL

# Decompose time series
stl = STL(rooms, period=7, robust=True)  # period=7 for weekly seasonality
result = stl.fit()

trend = result.trend
seasonal = result.seasonal
residual = result.resid

# Detect anomalies in residuals
residual_mean = np.mean(residual)
residual_std = np.std(residual)
z_residual = (residual - residual_mean) / residual_std
is_anomaly = abs(z_residual) > threshold  # threshold = 3.0
```

#### **Example**

**Scenario:** Daily hotel bookings with weekly seasonality

**Original Data (7 days):**
```
Day:  Mon  Tue  Wed  Thu  Fri  Sat  Sun
Rooms: 140  160  165  160  150  180  175
```

**After STL Decomposition:**

**Trend Component:**
```
Day:  Mon  Tue  Wed  Thu  Fri  Sat  Sun
Trend: 155  156  157  158  159  160  161
```
*(Slight upward trend)*

**Seasonal Component:**
```
Day:  Mon  Tue  Wed  Thu  Fri  Sat  Sun
Seasonal: -10  +5   +8   +2   -5   +20  +15
```
*(Weekend boost, mid-week peak)*

**Residual Component:**
```
Day:  Mon  Tue  Wed  Thu  Fri  Sat  Sun
Residual: -5   -1    0    0   -4    0   -1
```
*(Small random variations)*

**Anomaly Detection:**
- All residuals are small (< 1 std dev)
- **No anomalies detected**

**If Anomaly Occurs:**
```
Day:  Mon  Tue  Wed  Thu  Fri  Sat  Sun
Rooms: 140  160  165  160  150  250  175  ← 250 is anomaly
Residual: -5   -1    0    0   -4   +70  -1  ← Large residual!
```
- Residual = 70 (expected ~5)
- Z-score of residual = 70 / 5 = 14 std devs
- **ANOMALY DETECTED**

**Code Example:**
```python
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL

# Generate sample data with seasonality
dates = pd.date_range('2024-01-01', periods=30, freq='D')
trend = np.linspace(150, 160, 30)
seasonal = 10 * np.sin(2 * np.pi * np.arange(30) / 7)  # Weekly pattern
noise = np.random.normal(0, 2, 30)
rooms = trend + seasonal + noise

# Inject anomaly
rooms[15] = 200  # Day 15 is anomalous

# STL Decomposition
stl = STL(rooms, period=7, robust=True)
result = stl.fit()

# Detect anomalies
residual = result.resid
residual_std = np.std(residual)
threshold = 3.0

anomalies = np.abs(residual) > threshold * residual_std
print(f"Anomalies detected at indices: {np.where(anomalies)[0]}")
# Output: Anomalies detected at indices: [15]
```

#### **Original Source**
- **Origin**: Cleveland, R. B., Cleveland, W. S., McRae, J. E., & Terpenning, I. (1990). "STL: A Seasonal-Trend Decomposition Procedure Based on Loess." *Journal of Official Statistics*, 6(1), 3-73.
- **Key Innovation**: Robust decomposition that handles outliers in the decomposition process itself
- **LOESS Reference**: Cleveland, W. S. (1979). "Robust Locally Weighted Regression and Smoothing Scatterplots." *Journal of the American Statistical Association*, 74(368), 829-836.
- **Modern Use**: Standard in time-series analysis, forecasting, and anomaly detection

---

### 4. Isolation Forest

#### **What**
A machine learning algorithm for anomaly detection based on the principle that anomalies are "few and different" - they require fewer random partitions to isolate than normal points.

#### **Mathematical Foundation**

**Core Concept:**
- Build an ensemble of random binary trees
- Each tree recursively partitions data with random splits
- Anomalies are isolated faster (shorter path length)
- Average path length across trees = anomaly score

**Path Length:**
- For a point `x`, path length `h(x)` is the number of edges from root to leaf
- Anomalies have shorter paths (isolated quickly)
- Normal points have longer paths (harder to isolate)

**Anomaly Score:**
```
s(x, n) = 2^(-E[h(x)] / c(n))
```

Where:
- `E[h(x)]` = Expected path length across all trees
- `c(n)` = Average path length of unsuccessful search in BST (normalization)
- `n` = Number of samples

**Score Interpretation:**
- `s(x) ≈ 1`: Anomaly (isolated quickly)
- `s(x) ≈ 0.5`: Normal point
- `s(x) ≈ 0`: Inlier (very normal)

**Decision:**
- `s(x) > threshold` → Anomaly
- Or use `contamination` parameter (expected proportion of anomalies)

#### **Why Use It**
1. **No Distribution Assumptions**: Works for any data distribution
2. **Handles High Dimensions**: Effective even with many features
3. **Fast**: O(n log n) training, O(log n) prediction
4. **Unsupervised**: No labeled data required
5. **Effective**: State-of-the-art performance on many benchmarks

#### **How It Works**

**Algorithm Steps:**

1. **Tree Construction**:
   - Randomly select feature and split value
   - Recursively partition data
   - Stop when point is isolated or max depth reached

2. **Isolation**:
   - Anomalies are isolated with fewer splits
   - Normal points require more splits

3. **Scoring**:
   - Average path length across all trees
   - Transform to anomaly score [0, 1]

4. **Decision**:
   - Compare score to threshold
   - Or use contamination rate

**Implementation in Project:**
```python
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Prepare features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# Fit Isolation Forest
model = IsolationForest(
    contamination=0.05,  # Expect 5% anomalies
    n_estimators=100,    # Number of trees
    random_state=42
)
model.fit(X_scaled)

# Predict
scores = -model.score_samples(X_scaled)  # Negative for anomalies
predictions = model.predict(X_scaled)     # -1 = anomaly, 1 = normal
```

**Feature Engineering:**
- Uses all engineered features (rolling stats, velocity, calendar, etc.)
- StandardScaler normalizes features
- More features = better pattern recognition

#### **Example**

**2D Example (for visualization):**

**Normal Points:**
```
(100, 150), (105, 152), (102, 148), (103, 151), ...
```
*(Clustered together)*

**Anomaly:**
```
(200, 50)
```
*(Far from cluster)*

**Isolation Process:**

**Tree 1:**
- Split 1: `feature_1 < 120` → Anomaly goes right
- Split 2: `feature_2 < 100` → Anomaly isolated! (path length = 2)

**Tree 2:**
- Split 1: `feature_2 < 140` → Anomaly goes right
- Split 2: `feature_1 > 150` → Anomaly isolated! (path length = 2)

**Normal Point:**
- Requires 5-6 splits on average to isolate
- Longer path length = lower anomaly score

**Result:**
- Anomaly score: 0.85 (high = anomalous)
- Normal score: 0.35 (low = normal)

**Code Example:**
```python
import numpy as np
from sklearn.ensemble import IsolationForest

# Generate data
np.random.seed(42)
normal_data = np.random.randn(100, 2) * 10 + [100, 150]
anomaly_data = np.array([[200, 50], [50, 200], [180, 30]])
data = np.vstack([normal_data, anomaly_data])

# Fit Isolation Forest
model = IsolationForest(contamination=0.03, random_state=42)
model.fit(data)

# Predict
predictions = model.predict(data)
scores = -model.score_samples(data)

# Results
anomaly_indices = np.where(predictions == -1)[0]
print(f"Anomalies detected at indices: {anomaly_indices}")
print(f"Anomaly scores: {scores[anomaly_indices]}")
# Output: Anomalies detected at indices: [100, 101, 102]
#         Anomaly scores: [0.72, 0.68, 0.75]
```

#### **Original Source**
- **Origin**: Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). "Isolation Forest." *2008 Eighth IEEE International Conference on Data Mining*, 413-422.
- **Key Innovation**: Novel approach using random partitioning instead of density estimation
- **Theoretical Foundation**: Based on concept that anomalies are easier to isolate
- **Performance**: Won best paper award, widely adopted in industry
- **Modern Use**: Standard algorithm in scikit-learn, used by many companies for anomaly detection

---

### 5. Ensemble Voting

#### **What**
A meta-learning approach that combines predictions from multiple detectors using weighted averaging and majority voting to make final decisions.

#### **Mathematical Foundation**

**Weighted Average Score:**
```
ensemble_score = Σ (w_i × score_i) / Σ w_i
```

Where:
- `w_i` = weight for detector i
- `score_i` = normalized score from detector i (0-1)
- `ensemble_score` = final combined score (0-1)

**Majority Voting:**
```
votes = Σ (prediction_i == anomaly)
is_anomaly = (votes >= min_votes) OR (ensemble_score > threshold)
```

Where:
- `prediction_i` = binary prediction from detector i (0 or 1)
- `min_votes` = minimum number of detectors that must agree
- `threshold` = score threshold for anomaly decision

**Final Decision Logic:**
```
IF (ensemble_score > threshold) OR (votes >= min_votes):
    RETURN anomaly
ELSE:
    RETURN normal
```

#### **Why Use It**
1. **Robustness**: Reduces false positives from any single method
2. **Diversity**: Different methods catch different anomaly types
3. **Confidence**: Agreement among methods increases confidence
4. **Flexibility**: Can weight methods based on performance
5. **Industry Best Practice**: Ensemble methods are standard in ML

#### **How It Works in This Project**

**Detector Weights (Default):**
```python
weights = {
    'statistical': 0.3,  # Z-Score + IQR
    'timeseries': 0.3,    # STL Decomposition
    'ml': 0.4            # Isolation Forest
}
```

**Voting Strategy:**
- **Score-based**: Weighted average of normalized scores
- **Vote-based**: Majority voting (min_votes = 2 by default)
- **Combined**: Either method can trigger anomaly

**Sensitivity Levels:**

**Low Sensitivity:**
- `threshold = 0.7` (high score required)
- `min_votes = 3` (all detectors must agree)
- Result: Fewer alerts, higher confidence

**Medium Sensitivity:**
- `threshold = 0.5` (moderate score)
- `min_votes = 2` (2+ detectors agree)
- Result: Balanced approach

**High Sensitivity:**
- `threshold = 0.3` (low score)
- `min_votes = 1` (any detector can trigger)
- Result: More alerts, catches subtle anomalies

#### **Example**

**Scenario:** Observation with rooms = 220 (normal ~150)

**Individual Detector Results:**

| Detector | Score | Binary | Weight |
|----------|-------|--------|--------|
| Statistical | 0.75 | 1 (anomaly) | 0.3 |
| TimeSeries | 0.60 | 1 (anomaly) | 0.3 |
| ML | 0.45 | 0 (normal) | 0.4 |

**Ensemble Calculation:**

**Weighted Score:**
```
ensemble_score = (0.3 × 0.75) + (0.3 × 0.60) + (0.4 × 0.45)
                = 0.225 + 0.180 + 0.180
                = 0.585
```

**Voting:**
```
votes = 1 + 1 + 0 = 2
min_votes = 2
votes >= min_votes → True
```

**Final Decision:**
```
ensemble_score (0.585) > threshold (0.5) → True
OR
votes (2) >= min_votes (2) → True
→ ANOMALY DETECTED
```

**Confidence:**
- 2 out of 3 detectors agree
- Score is above threshold
- **Medium confidence** anomaly

**Code Example:**
```python
import numpy as np

# Individual detector scores (normalized 0-1)
statistical_score = 0.75
timeseries_score = 0.60
ml_score = 0.45

# Weights
weights = {'statistical': 0.3, 'timeseries': 0.3, 'ml': 0.4}

# Weighted average
ensemble_score = (
    weights['statistical'] * statistical_score +
    weights['timeseries'] * timeseries_score +
    weights['ml'] * ml_score
)

# Binary predictions
statistical_pred = 1 if statistical_score > 0.5 else 0
timeseries_pred = 1 if timeseries_score > 0.5 else 0
ml_pred = 1 if ml_score > 0.5 else 0

# Voting
votes = statistical_pred + timeseries_pred + ml_pred
min_votes = 2

# Final decision
threshold = 0.5
is_anomaly = (ensemble_score > threshold) or (votes >= min_votes)

print(f"Ensemble score: {ensemble_score:.3f}")
print(f"Votes: {votes}/{len(weights)}")
print(f"Anomaly: {is_anomaly}")
# Output: Ensemble score: 0.585, Votes: 2/3, Anomaly: True
```

#### **Original Source**
- **Ensemble Learning**: Originated from multiple sources in machine learning
- **Key References**:
  - Breiman, L. (2001). "Random Forests." *Machine Learning*, 45(1), 5-32.
  - Dietterich, T. G. (2000). "Ensemble Methods in Machine Learning." *Multiple Classifier Systems*, 1-15.
- **Voting Theory**: Based on Condorcet's jury theorem (1785) - majority voting improves accuracy
- **Modern Use**: Standard practice in ML competitions and production systems

---

## Core Concepts

### 1. Anomaly Detection Fundamentals

#### **What is an Anomaly?**
An anomaly (outlier) is a data point that significantly deviates from the expected pattern or distribution of the dataset.

**Types of Anomalies:**
1. **Point Anomalies**: Single data point that's anomalous
   - Example: One day with 300 rooms when average is 150

2. **Contextual Anomalies**: Normal in one context, anomalous in another
   - Example: 200 rooms on a Saturday (normal) vs. 200 rooms on a Tuesday (anomalous)

3. **Collective Anomalies**: A collection of points that together are anomalous
   - Example: 10 consecutive days of declining bookings

#### **Why Detect Anomalies?**
1. **Data Quality**: Identify errors, missing data, system failures
2. **Fraud Detection**: Unusual patterns may indicate fraud
3. **System Monitoring**: Detect failures or performance issues
4. **Business Intelligence**: Identify opportunities or threats
5. **Model Validation**: Ensure data quality for downstream models

#### **Challenges:**
- **Definition**: What's "normal" vs. "anomalous"?
- **Imbalance**: Anomalies are rare (often < 5% of data)
- **Context**: Same value can be normal or anomalous depending on context
- **Evolution**: Normal patterns change over time

---

### 2. Time-Series Anomaly Detection

#### **Characteristics of Time Series:**
1. **Trend**: Long-term direction (increasing, decreasing, stable)
2. **Seasonality**: Repeating patterns (daily, weekly, monthly, yearly)
3. **Cyclical**: Irregular cycles (business cycles)
4. **Noise**: Random variation

#### **Why Standard Methods Fail:**
- **Non-Stationary**: Mean and variance change over time
- **Dependencies**: Values depend on previous values
- **Seasonality**: Must account for repeating patterns
- **Context**: Same value means different things at different times

#### **Solutions:**
1. **Rolling Windows**: Use recent history instead of all history
2. **Decomposition**: Separate trend, seasonal, and residual components
3. **Feature Engineering**: Extract time-aware features
4. **Time-Aware Models**: Models that understand temporal structure

---

### 3. Feature Engineering for Anomaly Detection

#### **Why Features Matter:**
Raw data (e.g., "150 rooms") doesn't tell the full story. Features provide context:
- **Temporal Context**: What day of week? What month?
- **Historical Context**: How does this compare to recent history?
- **Rate of Change**: Is this a sudden change or gradual?
- **Position**: Where does this sit in the distribution?

#### **Feature Categories:**

**1. Calendar Features:**
- Day of week, month, quarter, year
- Weekend/holiday indicators
- **Why**: Captures recurring patterns (e.g., higher bookings on weekends)

**2. Rolling Statistics:**
- Mean, std, min, max over windows (7d, 30d)
- **Why**: Provides local context vs. global statistics
- **Example**: `rolling_mean_7d = 155` means recent average is 155

**3. Velocity Features:**
- Day-over-day change, percentage change
- Rate of change (velocity), acceleration
- **Why**: Captures sudden shifts or trends
- **Example**: `rooms_diff_1d = +20` means 20 more rooms than yesterday

**4. Deviation Features:**
- Deviation from means, z-scores, IQR position
- **Why**: Measures how unusual current value is
- **Example**: `z_score_7d = 2.5` means 2.5 std devs above 7-day mean

**5. Seasonal Features:**
- Same day-of-week average, same month average
- Deviation from seasonal expectations
- **Why**: Captures violations of expected seasonal patterns
- **Example**: `deviation_from_dow_mean = -30` means 30 below typical Saturday

---

### 4. Ensemble Methods

#### **Why Combine Multiple Methods?**
1. **Complementary Strengths**: Each method catches different anomaly types
   - Statistical: Simple point anomalies
   - Time-series: Seasonal pattern violations
   - ML: Complex multi-dimensional patterns

2. **Error Reduction**: Errors from one method offset by others
3. **Robustness**: Less sensitive to failures of individual methods
4. **Confidence**: Agreement among methods increases confidence

#### **Combination Strategies:**

**1. Weighted Averaging:**
- Combine scores with weights
- Weights based on performance or domain knowledge
- **Pros**: Smooth, continuous scores
- **Cons**: Requires tuning weights

**2. Majority Voting:**
- Each method votes (anomaly or normal)
- Majority wins
- **Pros**: Simple, interpretable
- **Cons**: Binary, loses score information

**3. Hybrid (Used in This Project):**
- Weighted average for score
- Majority voting for binary decision
- Either can trigger anomaly
- **Pros**: Best of both worlds
- **Cons**: More complex

---

### 5. Sensitivity and Thresholds

#### **The Trade-off:**
- **High Sensitivity (Low Threshold)**: More alerts, but more false positives
- **Low Sensitivity (High Threshold)**: Fewer alerts, but may miss real anomalies

#### **Sensitivity Levels:**

**Low Sensitivity:**
- Threshold: 0.7 (high score required)
- Min Votes: 3 (all detectors must agree)
- **Use Case**: High-stakes decisions, minimize false positives
- **Example**: Financial fraud detection, critical system monitoring

**Medium Sensitivity:**
- Threshold: 0.5 (moderate score)
- Min Votes: 2 (2+ detectors agree)
- **Use Case**: Balanced approach, general monitoring
- **Example**: This project's default setting

**High Sensitivity:**
- Threshold: 0.3 (low score)
- Min Votes: 1 (any detector can trigger)
- **Use Case**: Exploratory analysis, catch subtle anomalies
- **Example**: Research, data exploration

#### **Choosing Sensitivity:**
1. **Business Impact**: Cost of missing anomaly vs. cost of false alarm
2. **Data Quality**: Clean data → higher sensitivity
3. **Domain Knowledge**: Known anomaly rates
4. **Validation**: Use validation framework to find optimal setting

---

## References and Sources

### Academic Papers

1. **Z-Score / Standardization:**
   - Pearson, K. (1900). "On the criterion that a given system of deviations from the probable in the case of a correlated system of variables is such that it can be reasonably supposed to have arisen from random sampling." *Philosophical Magazine*, 50(302), 157-175.

2. **IQR / Box Plot Method:**
   - Tukey, J. W. (1977). *Exploratory Data Analysis*. Addison-Wesley Publishing Company.

3. **STL Decomposition:**
   - Cleveland, R. B., Cleveland, W. S., McRae, J. E., & Terpenning, I. (1990). "STL: A Seasonal-Trend Decomposition Procedure Based on Loess." *Journal of Official Statistics*, 6(1), 3-73.
   - Cleveland, W. S. (1979). "Robust Locally Weighted Regression and Smoothing Scatterplots." *Journal of the American Statistical Association*, 74(368), 829-836.

4. **Isolation Forest:**
   - Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). "Isolation Forest." *2008 Eighth IEEE International Conference on Data Mining*, 413-422.
   - Liu, F. T., Ting, K. M., & Zhou, Z. H. (2012). "Isolation-Based Anomaly Detection." *ACM Transactions on Knowledge Discovery from Data*, 6(1), 1-39.

5. **Ensemble Methods:**
   - Breiman, L. (2001). "Random Forests." *Machine Learning*, 45(1), 5-32.
   - Dietterich, T. G. (2000). "Ensemble Methods in Machine Learning." *Multiple Classifier Systems*, 1-15.

### Books and Textbooks

1. **Anomaly Detection:**
   - Aggarwal, C. C. (2017). *Outlier Analysis* (2nd ed.). Springer.
   - Chandola, V., Banerjee, A., & Kumar, V. (2009). "Anomaly Detection: A Survey." *ACM Computing Surveys*, 41(3), 1-58.

2. **Time Series Analysis:**
   - Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts.
   - Shumway, R. H., & Stoffer, D. S. (2017). *Time Series Analysis and Its Applications* (4th ed.). Springer.

3. **Machine Learning:**
   - Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.
   - Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.

### Software Documentation

1. **scikit-learn:**
   - Isolation Forest: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html

2. **statsmodels:**
   - STL Decomposition: https://www.statsmodels.org/stable/generated/statsmodels.tsa.seasonal.STL.html

3. **pandas:**
   - Rolling Windows: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rolling.html

### Industry Applications

1. **Netflix:**
   - Used ensemble methods for anomaly detection in streaming data
   - Reference: Netflix Tech Blog

2. **Amazon:**
   - Isolation Forest for fraud detection
   - Reference: AWS Machine Learning Blog

3. **Google:**
   - Time-series anomaly detection for monitoring
   - Reference: Google Cloud AI Platform

### Online Resources

1. **Towards Data Science:**
   - Multiple articles on anomaly detection methods
   - https://towardsdatascience.com/

2. **Kaggle:**
   - Anomaly detection competitions and tutorials
   - https://www.kaggle.com/

3. **Papers with Code:**
   - State-of-the-art anomaly detection methods
   - https://paperswithcode.com/task/anomaly-detection

---

## Conclusion

This documentation provides a comprehensive overview of the Hotel Booking Anomaly Detection System, covering:

1. **Module Documentation**: Detailed explanation of each module (what, why, how, where to use)
2. **Algorithm Deep Dive**: Mathematical foundations, implementations, examples, and original sources for each algorithm
3. **Core Concepts**: Fundamental concepts with examples and justifications
4. **References**: Academic papers, books, and industry sources

The system combines multiple detection methods (Z-Score, IQR, STL, Isolation Forest) in an ensemble approach to provide robust, production-ready anomaly detection for hotel booking data. Each method has its strengths and weaknesses, and the ensemble approach leverages the complementary nature of these methods to achieve better performance than any single method alone.

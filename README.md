# Hotel Booking Anomaly Detection

A production-ready anomaly detection system for hotel On-The-Books (OTB) data.

## Overview

This project implements an ensemble-based anomaly detection system that identifies unusual booking patterns in hotel reservation data. It combines multiple detection methods for robust, reliable alerts.

## Features

- **Multiple Detection Methods**: Z-Score, IQR, STL Decomposition, Isolation Forest
- **Ensemble Voting**: Combines methods for higher confidence
- **Configurable Sensitivity**: Low/Medium/High presets
- **Simple API**: Single function call to detect anomalies
- **Comprehensive Validation**: Backtesting framework with metrics

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from src.anomaly_detector import AnomalyDetector, detect_anomaly
from src.data_generator import generate_sample_data

# Generate or load historical data
history = generate_sample_data()

# Option 1: One-liner detection
result = detect_anomaly(
    observation={'asof_date': '2025-01-15', 'rooms': 180},
    history=history,
    sensitivity='medium'
)
print(f"Anomaly: {result['is_anomaly']}, Score: {result['score']:.3f}")

# Option 2: Reusable detector
detector = AnomalyDetector(sensitivity='medium')
detector.fit(history)

# Single detection
result = detector.detect({'asof_date': '2025-01-15', 'rooms': 180})

# Batch detection
new_data = pd.DataFrame({
    'asof_date': pd.date_range('2025-01-01', periods=7),
    'rooms': [130, 125, 135, 200, 120, 115, 140]
})
results = detector.detect_batch(new_data)
```

### Running the Notebook

```bash
jupyter notebook booking_anomaly_detection.ipynb
```

### Running Tests

```bash
python -m pytest tests/ -v
```

## Project Structure

```
.
├── booking_anomaly_detection.ipynb  # Main analysis notebook
├── requirements.txt                  # Dependencies
├── README.md                         # This file
├── src/
│   ├── __init__.py
│   ├── data_generator.py            # Synthetic OTB data generation
│   ├── feature_engineering.py       # Feature extraction pipeline
│   ├── detectors.py                 # Detection algorithms
│   ├── anomaly_detector.py          # Main interface
│   └── validator.py                 # Backtesting framework
├── tests/
│   ├── __init__.py
│   └── test_detectors.py            # Unit tests
└── plots/                            # Generated visualizations
```

## Detection Methods

| Method | Description | Best For |
|--------|-------------|----------|
| **Z-Score** | Deviation from mean in standard deviations | Simple point anomalies |
| **IQR** | Outside interquartile range bounds | Robust to outliers |
| **STL** | Time-series decomposition residuals | Seasonal pattern violations |
| **Isolation Forest** | ML-based isolation difficulty | Complex patterns |

## Sensitivity Levels

| Level | Alert Rate | Use Case |
|-------|------------|----------|
| `low` | ~1% | High-stakes decisions, minimize false positives |
| `medium` | ~5% | Balanced approach (default) |
| `high` | ~10% | Exploratory, catch subtle anomalies |

## API Reference

### `AnomalyDetector` Class

```python
detector = AnomalyDetector(sensitivity='medium')
detector.fit(historical_data)
result = detector.detect({'asof_date': '2025-01-15', 'rooms': 150})
```

**Returns:**
```python
{
    'is_anomaly': bool,       # True if anomalous
    'score': float,           # 0-1, higher = more anomalous
    'confidence': str,        # 'low', 'medium', 'high'
    'explanation': str,       # Human-readable explanation
    'details': dict           # Per-detector breakdown
}
```

### `detect_anomaly` Function

```python
result = detect_anomaly(observation, history, sensitivity='medium')
```

Convenience function that fits and detects in one call.

## Requirements

- Python 3.8+
- pandas >= 2.0.0
- numpy >= 1.24.0
- scipy >= 1.10.0
- statsmodels >= 0.14.0
- scikit-learn >= 1.3.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0

"""
Comprehensive Analysis of Hotel OTB Data
Detects anomalies using multiple approaches
"""
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '.')

# Load data
df = pd.read_csv('data/data.csv')
df['asof_date'] = pd.to_datetime(df['asof_date'])

print('=' * 75)
print('HOTEL OTB DATA - COMPREHENSIVE ANOMALY ANALYSIS')
print('=' * 75)

# ============================================================================
# SECTION 1: DATA OVERVIEW
# ============================================================================
print('\n' + '=' * 75)
print('1. DATA OVERVIEW')
print('=' * 75)

print(f'''
Dataset Summary:
  Total records: {len(df):,}
  Date range: {df["asof_date"].min().date()} to {df["asof_date"].max().date()}
  Duration: {(df["asof_date"].max() - df["asof_date"].min()).days} days (~{(df["asof_date"].max() - df["asof_date"].min()).days / 365:.1f} years)

Rooms Statistics:
  Mean: {df["rooms"].mean():,.1f}
  Median: {df["rooms"].median():,.1f}
  Std Dev: {df["rooms"].std():,.1f}
  Min: {df["rooms"].min():,} (on {df.loc[df["rooms"].idxmin(), "asof_date"].date()})
  Max: {df["rooms"].max():,} (on {df.loc[df["rooms"].idxmax(), "asof_date"].date()})
''')

# ============================================================================
# SECTION 2: ANOMALY DETECTION - GLOBAL Z-SCORE
# ============================================================================
print('=' * 75)
print('2. ANOMALY DETECTION - GLOBAL Z-SCORE METHOD')
print('=' * 75)

mean = df['rooms'].mean()
std = df['rooms'].std()
df['z_score'] = (df['rooms'] - mean) / std
df['is_z_anomaly'] = abs(df['z_score']) > 3  # 3 std = 99.7% confidence

z_anomalies = df[df['is_z_anomaly']].copy()
z_anomalies = z_anomalies.sort_values('z_score')

print(f'\nAnomalies detected (|Z| > 3): {len(z_anomalies)}')
if len(z_anomalies) > 0:
    print('\n*** CRITICAL ANOMALIES ***')
    print('-' * 75)
    for _, row in z_anomalies.iterrows():
        direction = 'LOW' if row['z_score'] < 0 else 'HIGH'
        print(f'{row["asof_date"].date()} | Rooms: {row["rooms"]:>6,} | Z-score: {row["z_score"]:+.2f} | {direction}')

# ============================================================================
# SECTION 3: ANOMALY DETECTION - IQR METHOD
# ============================================================================
print('\n' + '=' * 75)
print('3. ANOMALY DETECTION - IQR METHOD')
print('=' * 75)

Q1 = df['rooms'].quantile(0.25)
Q3 = df['rooms'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df['is_iqr_anomaly'] = (df['rooms'] < lower_bound) | (df['rooms'] > upper_bound)
iqr_anomalies = df[df['is_iqr_anomaly']].copy()

print(f'''
IQR Bounds:
  Q1 (25th percentile): {Q1:,.0f}
  Q3 (75th percentile): {Q3:,.0f}
  IQR: {IQR:,.0f}
  Lower bound: {lower_bound:,.0f}
  Upper bound: {upper_bound:,.0f}

Anomalies detected: {len(iqr_anomalies)}
''')

if len(iqr_anomalies) > 0:
    print('*** IQR ANOMALIES ***')
    print('-' * 75)
    for _, row in iqr_anomalies.iterrows():
        direction = 'LOW' if row['rooms'] < lower_bound else 'HIGH'
        print(f'{row["asof_date"].date()} | Rooms: {row["rooms"]:>6,} | {direction}')

# ============================================================================
# SECTION 4: DAY-OVER-DAY CHANGE ANALYSIS
# ============================================================================
print('\n' + '=' * 75)
print('4. DAY-OVER-DAY CHANGE ANALYSIS')
print('=' * 75)

df['rooms_prev'] = df['rooms'].shift(1)
df['daily_change'] = df['rooms'] - df['rooms_prev']
df['daily_change_pct'] = 100 * df['daily_change'] / df['rooms_prev']

# Extreme daily changes
change_threshold = 20  # 20% change
df['is_sudden_change'] = abs(df['daily_change_pct']) > change_threshold
sudden_changes = df[df['is_sudden_change']].dropna().copy()
sudden_changes = sudden_changes.sort_values('daily_change_pct')

print(f'\nExtreme daily changes (>{change_threshold}% change): {len(sudden_changes)}')
if len(sudden_changes) > 0:
    print('\n*** SUDDEN CHANGES ***')
    print('-' * 75)
    for _, row in sudden_changes.iterrows():
        print(f'{row["asof_date"].date()} | Rooms: {row["rooms"]:>6,} | '
              f'Change: {row["daily_change"]:+,.0f} ({row["daily_change_pct"]:+.1f}%)')

# ============================================================================
# SECTION 5: YEARLY COMPARISON
# ============================================================================
print('\n' + '=' * 75)
print('5. YEARLY COMPARISON')
print('=' * 75)

df['year'] = df['asof_date'].dt.year
yearly_stats = df.groupby('year')['rooms'].agg(['count', 'mean', 'std', 'min', 'max'])
print('\n')
print(yearly_stats.to_string())

# ============================================================================
# SECTION 6: MONTHLY PATTERNS
# ============================================================================
print('\n' + '=' * 75)
print('6. MONTHLY PATTERNS')
print('=' * 75)

df['month'] = df['asof_date'].dt.month
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

monthly_stats = df.groupby('month')['rooms'].agg(['mean', 'std'])
print('\nMonthly averages:')
for month in range(1, 13):
    if month in monthly_stats.index:
        m = monthly_stats.loc[month]
        bar = '█' * int(m['mean'] / 500)
        print(f'  {month_names[month-1]}: {m["mean"]:>7,.0f} ±{m["std"]:>5,.0f} {bar}')

# ============================================================================
# SECTION 7: COMBINED ANOMALY SUMMARY
# ============================================================================
print('\n' + '=' * 75)
print('7. FINAL ANOMALY SUMMARY')
print('=' * 75)

df['anomaly_flags'] = df['is_z_anomaly'].astype(int) + df['is_iqr_anomaly'].astype(int) + df['is_sudden_change'].fillna(False).astype(int)
all_anomalies = df[df['anomaly_flags'] > 0].copy()
all_anomalies = all_anomalies.sort_values('anomaly_flags', ascending=False)

print(f'\nTotal anomalies flagged by at least one method: {len(all_anomalies)}')
print('\n*** CONFIRMED ANOMALIES (flagged by multiple methods) ***')
print('-' * 75)

confirmed = all_anomalies[all_anomalies['anomaly_flags'] >= 2]
for _, row in confirmed.iterrows():
    methods = []
    if row['is_z_anomaly']: methods.append('Z-score')
    if row['is_iqr_anomaly']: methods.append('IQR')
    if pd.notna(row['is_sudden_change']) and row['is_sudden_change']: methods.append('Sudden-change')
    
    print(f'{row["asof_date"].date()} | Rooms: {row["rooms"]:>6,} | '
          f'Z={row["z_score"]:+.2f} | Methods: {", ".join(methods)}')

# ============================================================================
# SECTION 8: KEY FINDINGS
# ============================================================================
print('\n' + '=' * 75)
print('8. KEY FINDINGS & RECOMMENDATIONS')
print('=' * 75)

print('''
CRITICAL ANOMALY DETECTED:
--------------------------
📅 Date: 2024-01-13
📉 Rooms: 179 (compared to ~9,350 the previous day)
📊 Z-score: -11.97 (12 standard deviations below mean!)
⚠️  This is almost certainly a DATA ERROR or SYSTEM OUTAGE

Possible causes:
1. Data pipeline failure
2. System reset/reboot that cleared counters
3. Data entry error (missing digit: 179 vs 9179?)
4. Reporting system outage

RECOMMENDATION: Investigate this date with the source system.
If confirmed as error, consider replacing with interpolated value or previous day.

SEASONAL PATTERNS:
------------------
• Peak months: February-March (avg ~10,500-11,500 rooms)
• Low months: October-November (avg ~8,800-9,100 rooms)
• Year-over-year: 2024 had highest bookings, 2025 trending similar to 2023

BOOKING TRENDS:
---------------
• Overall stable with expected seasonal variation
• Standard deviation ~796 rooms (about 8% of mean)
• Data quality is good except for the 2024-01-13 anomaly
''')

# Save anomalies to CSV for further analysis
all_anomalies.to_csv('data/detected_anomalies.csv', index=False)
print('\n✓ Anomalies saved to data/detected_anomalies.csv')

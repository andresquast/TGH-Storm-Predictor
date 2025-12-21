#!/usr/bin/env python3
"""
Statistical Summary for TGH Storm Predictor
Calculates comprehensive statistics from the dataset
"""

import pandas as pd
import numpy as np
from data_fetcher import StormDataFetcher

def print_statistical_summary():
    """Print comprehensive statistical summary"""
    
    print("="*70)
    print("STATISTICAL SUMMARY - TGH STORM PREDICTOR")
    print("="*70)

    # Load data
    fetcher = StormDataFetcher()
    fetcher.load_data()
    df = fetcher.get_storms_dataframe(include_no_closure=False)

    print(f"\n📊 DATASET OVERVIEW")
    print("-"*70)
    print(f"Total storms with closure data: {len(df)}")
    print(f"Date range: {df['year'].min()} - {df['year'].max()} ({df['year'].max() - df['year'].min()} years)")
    print(f"Data quality breakdown:")
    print(f"  - High quality: {len(df[df['data_quality']=='high'])} storms")
    print(f"  - Medium quality: {len(df[df['data_quality']=='medium'])} storms")

    print(f"\n⏱️  BRIDGE CLOSURE DURATION STATISTICS")
    print("-"*70)
    print(f"Mean closure duration: {df['closure_hours'].mean():.1f} hours ({df['closure_hours'].mean()/24:.1f} days)")
    print(f"Median closure duration: {df['closure_hours'].median():.1f} hours ({df['closure_hours'].median()/24:.1f} days)")
    print(f"Standard deviation: {df['closure_hours'].std():.1f} hours")
    print(f"Minimum: {df['closure_hours'].min():.0f} hours ({df.loc[df['closure_hours'].idxmin(), 'name']})")
    print(f"Maximum: {df['closure_hours'].max():.0f} hours ({df.loc[df['closure_hours'].idxmax(), 'name']})")
    print(f"25th percentile: {df['closure_hours'].quantile(0.25):.1f} hours")
    print(f"75th percentile: {df['closure_hours'].quantile(0.75):.1f} hours")
    print(f"IQR: {df['closure_hours'].quantile(0.75) - df['closure_hours'].quantile(0.25):.1f} hours")

    print(f"\n🌪️  STORM CHARACTERISTICS")
    print("-"*70)
    print(f"Wind Speed (mph):")
    print(f"  Mean: {df['max_wind'].mean():.1f} | Median: {df['max_wind'].median():.1f} | Range: {df['max_wind'].min():.0f}-{df['max_wind'].max():.0f}")
    print(f"Storm Surge (feet):")
    print(f"  Mean: {df['storm_surge'].mean():.2f} | Median: {df['storm_surge'].median():.2f} | Range: {df['storm_surge'].min():.1f}-{df['storm_surge'].max():.1f}")
    print(f"Track Distance (miles):")
    print(f"  Mean: {df['track_distance'].mean():.1f} | Median: {df['track_distance'].median():.1f} | Range: {df['track_distance'].min():.0f}-{df['track_distance'].max():.0f}")
    print(f"Forward Speed (mph):")
    print(f"  Mean: {df['forward_speed'].mean():.1f} | Median: {df['forward_speed'].median():.1f} | Range: {df['forward_speed'].min():.0f}-{df['forward_speed'].max():.0f}")

    print(f"\n📅 TEMPORAL DISTRIBUTION")
    print("-"*70)
    print("By Month:")
    month_counts = df.groupby('month')['name'].count()
    month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    for month, count in month_counts.items():
        print(f"  {month_names[month-1]:3s} ({month:2d}): {count:2d} storms")
    print(f"\n🔗 CORRELATIONS WITH CLOSURE DURATION")
    print("-"*70)
    features = ['max_wind', 'storm_surge', 'track_distance', 'forward_speed', 'month']
    correlations = df[features + ['closure_hours']].corr()['closure_hours'].drop('closure_hours')
    correlations = correlations.sort_values(key=abs, ascending=False)
    for feat, corr in correlations.items():
        direction = "↑" if corr > 0 else "↓"
        print(f"  {feat:20s}: {corr:+.3f} {direction}")

    # Model performance (if validation was run)
    try:
        from validate_model import leave_one_out_validation
        print(f"\n🎯 MODEL PERFORMANCE (Leave-One-Out Cross-Validation)")
        print("-"*70)
        results_df = leave_one_out_validation()
        mae = results_df['error_hours'].mean()
        print(f"\nMean Absolute Error: ±{mae:.1f} hours")
        print(f"Average % Error: {results_df['percent_error'].mean():.1f}%")
        print(f"Best case error: ±{results_df['error_hours'].min():.1f} hours")
        print(f"Worst case error: ±{results_df['error_hours'].max():.1f} hours")
        print(f"Standard deviation of errors: {results_df['error_hours'].std():.1f} hours")
    except Exception as e:
        print(f"\n⚠️  Model validation not available: {e}")

    print("\n" + "="*70)

if __name__ == '__main__':
    print_statistical_summary()


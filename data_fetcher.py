"""
Data fetcher for TGH Storm Prediction Model
Loads historical storm data from JSON file
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime

class StormDataFetcher:
    """Fetches and processes historical storm data from JSON"""
    
    def __init__(self, json_path='data/historical_storms.json'):
        """
        Initialize fetcher with path to JSON file
        
        Args:
            json_path: Path to historical_storms.json file
        """
        self.json_path = Path(json_path)
        self.data = None
        self.metadata = None
        
    def load_data(self):
        """Load data from JSON file"""
        with open(self.json_path, 'r') as f:
            raw_data = json.load(f)
        
        self.metadata = raw_data['metadata']
        self.data = raw_data
        
        print(f"Loaded dataset: {self.metadata['dataset_name']}")
        print(f"Created: {self.metadata['created_date']}")
        print(f"Total storms: {len(raw_data['storms'])}")
        
        return self.data
    
    def get_storms_dataframe(self, include_no_closure=False):
        """
        Get storms as pandas DataFrame for modeling
        
        Args:
            include_no_closure: If True, include storms with no bridge closures
                               If False, only include storms with closure data
        
        Returns:
            pandas.DataFrame with storm features and closure hours
        """
        if self.data is None:
            self.load_data()
        
        storms = []
        for storm in self.data['storms']:
            # Skip storms without closure data unless requested
            if not include_no_closure and storm['closure_hours'] is None:
                continue
            
            storms.append({
                'name': storm['name'],
                'date': storm['date'],
                'year': storm['year'],
                'month': storm['month'],
                'category': storm['category_at_tampa'],
                'max_wind': storm['max_wind_tampa'],
                'storm_surge': storm['storm_surge_tampa'],
                'track_distance': storm['track_distance_miles'],
                'forward_speed': storm['forward_speed_mph'],
                'closure_hours': storm['closure_hours'],
                'data_quality': storm['data_quality'],
                'notes': storm['notes']
            })
        
        df = pd.DataFrame(storms)
        
        print(f"\nDataFrame created with {len(df)} storms")
        print(f"Date range: {df['year'].min()} - {df['year'].max()}")
        print(f"Closure range: {df['closure_hours'].min():.0f} - {df['closure_hours'].max():.0f} hours")
        
        return df
    
    def get_high_quality_data(self):
        """Get only high-quality data points for training"""
        df = self.get_storms_dataframe(include_no_closure=False)
        return df[df['data_quality'] == 'high']
    
    def get_storms_by_year_range(self, start_year, end_year):
        """Get storms within a specific year range"""
        df = self.get_storms_dataframe(include_no_closure=False)
        return df[(df['year'] >= start_year) & (df['year'] <= end_year)]
    
    def get_summary_statistics(self):
        """Print summary statistics of the dataset"""
        df = self.get_storms_dataframe(include_no_closure=False)
        
        print("\n" + "="*60)
        print("DATASET SUMMARY STATISTICS")
        print("="*60)
        
        print(f"\nTotal storms with closure data: {len(df)}")
        print(f"Date range: {df['year'].min()} - {df['year'].max()}")
        
        print("\n--- Bridge Closure Duration ---")
        print(f"Mean: {df['closure_hours'].mean():.1f} hours")
        print(f"Median: {df['closure_hours'].median():.1f} hours")
        print(f"Min: {df['closure_hours'].min():.0f} hours ({df.loc[df['closure_hours'].idxmin(), 'name']})")
        print(f"Max: {df['closure_hours'].max():.0f} hours ({df.loc[df['closure_hours'].idxmax(), 'name']})")
        
        print("\n--- Storm Characteristics ---")
        print(f"Wind speed range: {df['max_wind'].min():.0f} - {df['max_wind'].max():.0f} mph")
        print(f"Storm surge range: {df['storm_surge'].min():.0f} - {df['storm_surge'].max():.0f} feet")
        print(f"Track distance range: {df['track_distance'].min():.0f} - {df['track_distance'].max():.0f} miles")
        print(f"Forward speed range: {df['forward_speed'].min():.0f} - {df['forward_speed'].max():.0f} mph")
        
        print("\n--- By Month ---")
        print(df.groupby('month')['name'].count().to_string())
        
        print("\n--- By Category ---")
        print(df.groupby('category')['name'].count().to_string())
        
        print("\n--- Data Quality ---")
        print(df.groupby('data_quality')['name'].count().to_string())
        
        return df.describe()
    
    def export_for_modeling(self, output_path='data/modeling_data.csv'):
        """Export clean data for modeling"""
        df = self.get_storms_dataframe(include_no_closure=False)
        
        # Select only features needed for modeling
        modeling_df = df[[
            'name', 'date', 'category', 'max_wind', 'storm_surge',
            'track_distance', 'forward_speed', 'month', 'closure_hours'
        ]]
        
        modeling_df.to_csv(output_path, index=False)
        print(f"\nExported {len(modeling_df)} storms to {output_path}")
        
        return modeling_df


if __name__ == '__main__':
    # Example usage
    fetcher = StormDataFetcher()
    
    # Load and display summary
    fetcher.load_data()
    fetcher.get_summary_statistics()
    
    # Get DataFrame for modeling
    df = fetcher.get_storms_dataframe(include_no_closure=False)
    print("\n" + "="*60)
    print("STORMS INCLUDED IN MODELING")
    print("="*60)
    print(df[['name', 'year', 'closure_hours', 'max_wind', 'forward_speed', 'data_quality']].to_string(index=False))
    
    # Export for modeling
    fetcher.export_for_modeling()

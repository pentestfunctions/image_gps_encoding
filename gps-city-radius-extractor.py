#!/usr/bin/env python3
"""
Comprehensive GPS Grid Generator

This script generates a dense grid of GPS coordinates within a 20km radius
of each city with 15,000+ population, creating a comprehensive coverage
for matching geotagged images.
"""

import os
import math
import zipfile
import requests
import pandas as pd
import numpy as np
from tqdm import tqdm
from geopy.distance import geodesic

# Constants
EARTH_RADIUS_KM = 6371.0  # Earth radius in kilometers
CITY_RADIUS_KM = 20.0     # Radius around cities in kilometers
GRID_SPACING_KM = 1.0     # Spacing between grid points in km
OUTPUT_DIR = "data"       # Directory for output
OUTPUT_FILE = "city_gps_points.csv"  # Output filename

def download_city_data():
    """Download and extract the cities15000.zip file"""
    cities_url = "http://download.geonames.org/export/dump/cities15000.zip"
    zip_path = os.path.join(OUTPUT_DIR, "cities15000.zip")
    
    # Create directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Check if the file already exists
    if not os.path.exists(zip_path):
        print("Downloading cities15000.zip...")
        response = requests.get(cities_url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(zip_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
    
    # Extract the zip file
    csv_path = os.path.join(OUTPUT_DIR, "cities15000.txt")
    if not os.path.exists(csv_path):
        print("Extracting cities15000.zip...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(OUTPUT_DIR)
    
    return csv_path

def parse_city_data(city_file):
    """Parse city data from geonames format into a pandas DataFrame"""
    # Column names according to geonames.org documentation
    columns = [
        'geonameid', 'name', 'asciiname', 'alternatenames', 'latitude', 'longitude',
        'feature_class', 'feature_code', 'country_code', 'cc2', 'admin1_code',
        'admin2_code', 'admin3_code', 'admin4_code', 'population', 'elevation',
        'dem', 'timezone', 'modification_date'
    ]
    
    # Read the tab-delimited file
    df = pd.read_csv(city_file, sep='\t', header=None, names=columns, encoding='utf-8')
    
    # Convert latitude and longitude to float
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    
    # Convert population to int
    df['population'] = pd.to_numeric(df['population'], errors='coerce').astype('Int64')
    
    # Drop rows with missing lat, lon, or population
    df = df.dropna(subset=['latitude', 'longitude', 'population'])
    
    # Sort by population in descending order
    df = df.sort_values('population', ascending=False)
    
    print(f"Loaded {len(df)} cities with population ≥ 15,000")
    return df

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    distance = EARTH_RADIUS_KM * c
    
    return distance

def create_dense_grid(lat, lon, radius_km=CITY_RADIUS_KM, spacing_km=GRID_SPACING_KM):
    """Create a dense grid of points within a specified radius"""
    # Approximate degrees per km (varies by latitude)
    km_per_degree_lat = 111.0  # 1 degree of latitude is approximately 111km
    degrees_per_km_lat = 1.0 / km_per_degree_lat
    
    km_per_degree_lon = 111.0 * math.cos(math.radians(lat))  # Longitude degrees per km varies with latitude
    degrees_per_km_lon = 1.0 / km_per_degree_lon
    
    # Calculate step sizes in degrees
    lat_step = degrees_per_km_lat * spacing_km
    lon_step = degrees_per_km_lon * spacing_km
    
    # Calculate grid boundaries
    lat_radius_in_degrees = radius_km * degrees_per_km_lat
    lon_radius_in_degrees = radius_km * degrees_per_km_lon
    
    min_lat = lat - lat_radius_in_degrees
    max_lat = lat + lat_radius_in_degrees
    min_lon = lon - lon_radius_in_degrees
    max_lon = lon + lon_radius_in_degrees
    
    # Generate grid points
    grid_points = []
    current_lat = min_lat
    
    while current_lat <= max_lat:
        current_lon = min_lon
        while current_lon <= max_lon:
            # Check if the point is within the radius
            distance = haversine(lat, lon, current_lat, current_lon)
            if distance <= radius_km:
                grid_points.append((current_lat, current_lon, distance))
            
            current_lon += lon_step
        current_lat += lat_step
    
    return grid_points

def process_cities(df, spacing_km=GRID_SPACING_KM):
    """Process city data and generate a dense grid of GPS points"""
    # Create a list to store all grid points
    all_points = []
    
    # Process each city
    total_points = 0
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing cities"):
        city_id = row['geonameid']
        city_name = row['name']
        lat = row['latitude']
        lon = row['longitude']
        population = row['population']
        country = row['country_code']
        
        # Create dense grid around the city
        grid_points = create_dense_grid(lat, lon, CITY_RADIUS_KM, spacing_km)
        
        # Add each point to the list with metadata
        for point_lat, point_lon, distance in grid_points:
            all_points.append({
                'city_id': int(city_id),
                'city_name': city_name,
                'country': country,
                'population': int(population),
                'city_lat': float(lat),
                'city_lon': float(lon),
                'point_lat': float(point_lat),
                'point_lon': float(point_lon),
                'distance_km': float(distance)
            })
        
        total_points += len(grid_points)
        
        # Display progress after each city
        #if (idx + 1) % 100 == 0 or idx == len(df) - 1:
        #    print(f"Processed {idx + 1}/{len(df)} cities, generated {total_points} points so far")
    
    # Convert to DataFrame
    return pd.DataFrame(all_points)

def main():
    # Download and parse city data
    city_file = download_city_data()
    city_df = parse_city_data(city_file)
        
    # Process cities and generate GPS points
    print(f"\nGenerating dense grid of GPS points within {CITY_RADIUS_KM}km of each city (spacing: {GRID_SPACING_KM}km)...")
    
    # Estimate the number of points per city for a rough progress indicator
    estimated_points_per_city = int(math.pi * (CITY_RADIUS_KM / GRID_SPACING_KM) ** 2)
    print(f"Estimated points per city: ~{estimated_points_per_city} (total may be {estimated_points_per_city * len(city_df):,} points)")
    
    points_df = process_cities(city_df, GRID_SPACING_KM)
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save to CSV
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    points_df.to_csv(output_path, index=False)
    print(f"\nGenerated {len(points_df):,} GPS points around {len(city_df):,} cities")
    print(f"Results saved to {output_path}")
    
    # Clean up temporary files
    try:
        os.remove(os.path.join(OUTPUT_DIR, "cities15000.zip"))
        os.remove(os.path.join(OUTPUT_DIR, "cities15000.txt"))
        print("Temporary files cleaned up")
    except:
        pass

if __name__ == "__main__":
    main()

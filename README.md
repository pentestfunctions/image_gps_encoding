# Geotagged Image Dataset Project Plan

## Project Overview
Create a balanced dataset of geotagged images from populated areas by:
1. Using GPS coordinates from city population data
2. Establishing 20km radius boundaries around cities
3. Filtering a large geotagged image dataset based on these boundaries

## Data Sources
- **World-Wide Scale Geotagged Image Dataset**: 79.3 GB tar file
  - Source: https://skulddata.cs.umass.edu/traces/mmsys/2014/user03.tar
  - Contains images with GPS coordinates encoded in EXIF data
  
- **Global Cities Database**: Population centers with 15,000+ inhabitants
  - Source: http://download.geonames.org/export/dump/cities15000.zip
  - Provides city names, populations, and GPS coordinates


Practical Importance of GPS distance in a GPS string
5 decimal places (1-meter accuracy) is generally sufficient for most applications like mapping and navigation.
6-7 decimal places are needed for high-precision applications such as land surveying or drone navigation.
8+ decimal places go beyond practical GPS accuracy due to real-world limitations like atmospheric interference.

## Implementation Steps
First python script:
1. Download the dataset
2. Extract all the cities15000 so we can find the GPS coordinates of all cities with the populations in it.
3. Extract all GPS locations use the haversine formula to calculate all GPS locations that can exist within a 20km radius of EACH GPS location. Save to file.

Grid-based Sampling: The script divides each city's 20km radius into a grid of 1km×1km cells.
Limited Images Per Cell: It selects a maximum of 5 images per grid cell, preventing clusters of images from the same exact location.
City-based Organization: Images are organized by their nearest city, with a maximum of 2,000 images per city.
# python gps-city-radius-extractor.py
Generated 37,135,862 GPS points around 29,828 cities
Results saved to data\city_gps_points.csv
Temporary files cleaned up
3.52 GB (3,783,621,189 bytes)

Second python sript:
(Notes: Due to 79GB of images we should process them cleanly to avoid memory issues such as moving them)
1. Load data\city_gps_points.csv and for each GPS location (Sample data )
```
city_id,city_name,country,population,city_lat,city_lon,point_lat,point_lon,distance_km
1796236,Shanghai,CN,24874500,31.22222,121.45806,31.05104882882883,121.3948509566066,19.961484833886768
```
2. Find the nearest image that match the location. Then do the same again for the next image. Ensuring at most 2,000 images per city. 
3. All images will be moved to data/extracted_images/city_name/images


Third python script:
1. Train the model using the GPS encoded data in exif from each image. We would want to encode the city name AND gps location into the model for future predictions.
2. Maybe need to use clip-vit-large-patch14-336, geoclip_utils, location_encoder_weights or something else.
3. Save the model after each epoch.
Dual-output prediction model:

Model could simultaneously predict the city name (classification task) and GPS coordinates (regression task)
clip-vit-large-patch14-336 would be a strong backbone since it's pre-trained on diverse image data
The model would have two output heads: one for city classification and one for coordinate regression

Input/output structure:
Input: Images from your balanced dataset
Output 1: City name prediction (categorical)
Output 2: GPS coordinates prediction (continuous values for lat/long)

Loss functions:
City prediction: Cross-entropy loss for the classification task
GPS prediction: Haversine distance loss or MSE loss on normalized coordinates
Combined loss: Weighted sum of both losses (you can tune these weights as hyperparameters)

In Evaluation, between epochs, use Haversine to calculate distance away from where the validation image was actually taken from the models guess. We can print this nicely. Green for within 20km, Yellow for within 100km and Red for anything beyond that range.


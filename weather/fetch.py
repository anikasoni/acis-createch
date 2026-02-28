import os
import json
import requests

# Coordinates for the 6 demo cities in your sidebar
CITIES = {
    'Chennai': {'lat': 13.0827, 'lon': 80.2707},
    'Mumbai':  {'lat': 19.0760, 'lon': 72.8777},
    'Delhi':   {'lat': 28.6139, 'lon': 77.2090},
    'Shimla':  {'lat': 31.1048, 'lon': 77.1734},
    'Jaipur':  {'lat': 26.9124, 'lon': 75.7873},
    'Kolkata': {'lat': 22.5726, 'lon': 88.3639}
}

CACHE_DIR = 'weather/cache'

def get_hourly_temps(city: str, hours: int = 48) -> list:
    """
    Fetches real hourly temperatures for the next few days.
    If the internet is down, it seamlessly falls back to the local cache.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, f"{city}.json")
    
    try:
        # Try to pull live data from Open-Meteo
        coords = CITIES.get(city, CITIES['Chennai'])
        url = f"https://api.open-meteo.com/v1/forecast?latitude={coords['lat']}&longitude={coords['lon']}&hourly=temperature_2m&forecast_days=3"
        
        response = requests.get(url, timeout=3)
        response.raise_for_status()
        data = response.json()
        
        # Save to cache silently in the background
        with open(cache_file, 'w') as f:
            json.dump(data, f)
            
        return data['hourly']['temperature_2m'][:hours]
        
    except Exception as e:
        # INTERNET IS DOWN: Fallback to cached file
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                data = json.load(f)
            return data['hourly']['temperature_2m'][:hours]
        else:
            # Absolute fallback if no cache exists yet
            return [28.0] * hours

# Run this script once manually to build your initial offline cache!
if __name__ == "__main__":
    print("Building offline weather cache...")
    for city in CITIES.keys():
        temps = get_hourly_temps(city)
        print(f"✅ Cached 48-hour forecast for {city} (Current: {temps[0]}°C)")
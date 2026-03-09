import asyncio
from main import get_weather_data, get_soil_data, get_nasa_power_data, get_soilgrids_data

async def test_all():
    lat, lon = 9.0820, 8.6753
    print("Testing NASA POWER directly...")
    nasa = await get_nasa_power_data(lat, lon)
    print("NASA POWER Temp:", nasa.temperature)
    
    print("Testing SoilGrids directly...")
    soil = await get_soil_data(lat, lon)
    print("SoilGrids pH:", soil.ph, "Moisture:", soil.moisture)
    
    print("Testing combined Weather Data...")
    weather = await get_weather_data(lat, lon)
    print("Combined Weather Data Temp:", weather.temperature, "Humidity:", weather.humidity)

if __name__ == "__main__":
    asyncio.run(test_all())

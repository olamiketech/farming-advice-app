import asyncio
import httpx
from datetime import datetime
import json
import sys

async def test_soilgrids():
    print("Testing SoilGrids...")
    lat, lon = 9.0820, 8.6753
    url = f"https://rest.soilgrids.org/query?lon={lon}&lat={lat}&property=phh2o&depth=0-30cm&value=mean"
    async with httpx.AsyncClient() as client:
        resp = await client.get(url)
        print("SoilGrids phh2o:")
        print(json.dumps(resp.json(), indent=2))

async def test_nasa_power():
    print("\nTesting NASA POWER...")
    lat, lon = 9.0820, 8.6753
    now = datetime.now()
    date_str = now.strftime("%Y%m%d")
    url = f"https://power.larc.nasa.gov/api/temporal/daily/point?parameters=T2M,RH2M,WS2M,ALLSKY_SFC_SW_DWN&start={date_str}&end={date_str}&latitude={lat}&longitude={lon}&community=AG&format=JSON"
    async with httpx.AsyncClient() as client:
        resp = await client.get(url)
        print("NASA POWER STATUS:", resp.status_code)
        try:
            print("NASA POWER DATA:")
            print(json.dumps(resp.json(), indent=2))
        except Exception as e:
            print("Failed to decode JSON:", e)
            print("Raw text:", resp.text[:200])

async def test_openweather():
    print("\nTesting OpenWeatherMap...")
    lat, lon = 9.0820, 8.6753
    import os
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("OPENWEATHERMAP_API_KEY")
    if not api_key:
        print("No OPENWEATHERMAP_API_KEY in environment")
        return
    url = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    async with httpx.AsyncClient() as client:
        resp = await client.get(url)
        print("OWM STATUS:", resp.status_code)
        if resp.status_code == 200:
            data = resp.json()
            print("OWM Data sample:")
            print(json.dumps(data["list"][0]["main"], indent=2))
        else:
            print("OWM error:", resp.text)

async def main():
    await test_soilgrids()
    await test_nasa_power()
    await test_openweather()

if __name__ == "__main__":
    asyncio.run(main())

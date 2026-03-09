import asyncio
import httpx
import json

async def test_endpoint():
    url = "http://localhost:8000/predict"
    payload = {
        "crop_type": "maize",
        "location": {
            "lat": 9.0820,
            "lon": 8.6753
        }
    }
    
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json=payload, timeout=30.0)
            print(f"Status: {resp.status_code}")
            if resp.status_code == 200:
                data = resp.json()
                print("Weather:", json.dumps(data.get("weather"), indent=2))
                print("Soil:", json.dumps(data.get("soil"), indent=2))
                print("Yield Forecast:", data.get("yield_forecast"))
            else:
                print("Error:", resp.text)
    except httpx.RequestError as exc:
         print(f"An error occurred while requesting {exc.request.url!r}. Make sure the server is running.")

if __name__ == "__main__":
    asyncio.run(test_endpoint())

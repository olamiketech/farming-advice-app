from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import httpx
import os
from dotenv import load_dotenv
import logging
import json
from typing import Optional, Dict, Any, List
from gtts import gTTS
import threading
import re  # Used for stripping HTML tags from crop names
from functools import lru_cache
from fastapi.middleware.gzip import GZipMiddleware

# Attempt to import pygame for audio playback; if unavailable, the app will still run without audio support.
try:
    import pygame
    pygame_available = True
except ImportError:
    pygame_available = False


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
import asyncio
import requests
from datetime import datetime, timedelta
import openai
from openai import AsyncOpenAI, OpenAIError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# -----------------------------------------------------------
# Performance enhancements: pooled HTTP client and cached static files
# -----------------------------------------------------------
class CachedStaticFiles(StaticFiles):
    """Static file handler that adds long-lived Cache-Control headers."""

    def __init__(self, *args, cache_max_age: int = 60 * 60 * 24 * 30, **kwargs):
        self.cache_max_age = cache_max_age
        super().__init__(*args, **kwargs)

    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        if response.status_code == 200:
            response.headers["Cache-Control"] = f"public, max-age={self.cache_max_age}"
        return response


# Re-use a single AsyncClient for external API calls to minimise TCP/TLS handshakes
# and reduce latency on poor connections.
async_client = httpx.AsyncClient(timeout=10.0)
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
import pathlib
import os
import io

load_dotenv()

# Sensitive debug prints removed for security

# Get the directory where this script is located
BASE_DIR = pathlib.Path(__file__).parent

# Configure CORS
app = FastAPI(title="AgroPredict AI API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)
# Enable automatic GZip compression for faster responses
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Serve static files (CSS, JS, images, etc.)
app.mount(
    "/static",
    CachedStaticFiles(directory=str(BASE_DIR / "static"), cache_max_age=60*60*24*30),
    name="static",
)

# Gracefully close the shared HTTPX client on shutdown.
@app.on_event("shutdown")
async def _shutdown_async_client():
    await async_client.aclose()

# Serve the main HTML file
@app.get("/", response_class=HTMLResponse)
async def read_root():
    html_path = BASE_DIR / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    with open(html_path, 'r') as f:
        return HTMLResponse(content=f.read(), status_code=200)

# ---------------------------------------------------------------------------
# Serve the frontend (index.html) and any static assets so that the SPA and
# API are delivered from the same origin (http://<host>:8000). This prevents
# browser requests to the API from being blocked when the user opens the page
# after a reboot.
# ---------------------------------------------------------------------------
BASE_DIR = pathlib.Path(__file__).resolve().parent
# Duplicate mount removed to prevent overriding correct static directory
# app.mount("/static", StaticFiles(directory=BASE_DIR), name="static")

@app.get("/", include_in_schema=False)
async def serve_frontend():
    """Return the Single-Page Application entry point."""
    return FileResponse(BASE_DIR / "index.html")

# ---------------- Additional static HTML pages ----------------
@app.get("/leafdoctor.html", response_class=HTMLResponse, include_in_schema=False)
async def leafdoctor_page():
    html_path = BASE_DIR / "leafdoctor.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(), status_code=200)
    raise HTTPException(status_code=404, detail="Page not found")

@app.get("/about.html", response_class=HTMLResponse, include_in_schema=False)
async def about_page():
    html_path = BASE_DIR / "about.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(), status_code=200)
    raise HTTPException(status_code=404, detail="Page not found")

# Initialize OpenAI client (v1+)
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Configure OpenAI settings
openai_settings = {
    "model": "gpt-3.5-turbo",
    "temperature": 0.7,
    "max_tokens": 500,
    "top_p": 1.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0
}

# Crop-specific requirements
CROP_DATA = {
    "maize": {
        "optimal_ph": (5.5, 6.5),
        "optimal_temp": (25, 30),
        "optimal_rainfall": 15,
        "optimal_soil_moisture": 0.3,
        "optimal_wind_speed": 10,
        "growing_season": [6, 7, 8],  # June, July, August
        "maturity_days": 90,
        "baseline_yield": 4.0  # tons/ha
    },
    "cassava": {
        "optimal_ph": (5.0, 6.0),
        "optimal_temp": (24, 28),
        "optimal_rainfall": 10,
        "optimal_soil_moisture": 0.4,
        "optimal_wind_speed": 12,
        "growing_season": [3, 4, 5],  # March, April, May
        "maturity_days": 270,
        "baseline_yield": 25.0
    },
    "rice": {
        "optimal_ph": (6.0, 7.0),
        "optimal_temp": (26, 32),
        "optimal_rainfall": 20,
        "optimal_soil_moisture": 0.5,
        "optimal_wind_speed": 8,
        "growing_season": [9, 10, 11],  # September, October, November
        "maturity_days": 120,
        "baseline_yield": 5.0
    }
}


# Load extended crop requirements from crops.json if available
CUSTOM_CROP_DATA: Dict[str, Dict[str, Any]] = {}
CUSTOM_CROPS_PATH = BASE_DIR / "crops.json"
if CUSTOM_CROPS_PATH.exists():
    try:
        with open(CUSTOM_CROPS_PATH, "r", encoding="utf-8") as cf:
            raw_records = json.load(cf)
        for rec in raw_records:
            # Attempt to support different key conventions from dataset
            raw_name = (rec.get("name") or rec.get("Crop Name") or rec.get("crop_name") or "").strip()
            if not raw_name:
                continue
            # Strip any HTML tags to ensure clean dropdown display
            name_clean = re.sub(r'<[^>]+>', '', raw_name).strip()
            key = name_clean.lower()
            # Extract ranges with sensible defaults if missing
            ph_min = float(rec.get("ph_min") or rec.get("pH min") or rec.get("pH Range (min)") or 5.5)
            ph_max = float(rec.get("ph_max") or rec.get("pH max") or rec.get("pH Range (max)") or 7.0)
            t_min = float(rec.get("temp_min") or rec.get("Temperature min") or rec.get("Temp Range (min)") or 18)
            t_max = float(rec.get("temp_max") or rec.get("Temperature max") or rec.get("Temp Range (max)") or 32)
            r_min = float(rec.get("rain_min") or rec.get("Rainfall min") or 400)
            r_max = float(rec.get("rain_max") or rec.get("Rainfall max") or 1200)
            category = rec.get("category") or rec.get("Crop Category") or ""
            CUSTOM_CROP_DATA[key] = {
                "category": category,
                "optimal_ph": (ph_min, ph_max),
                "optimal_temp": (t_min, t_max),
                # Use the midpoint as representative optimal rainfall if range provided
                "optimal_rainfall": (r_min + r_max) / 2,
                # Reasonable generic defaults for values not present in the dataset
                "optimal_soil_moisture": 0.35,
                "optimal_wind_speed": 10,
                "growing_season": [],
                "maturity_days": 120,
                "baseline_yield": 3.0
            }
    except Exception as exc:
        logger.warning(f"Failed to load crops.json: {exc}")

# Merge custom crops into built-in requirements:
# Custom values override built-in ones, but if the custom entry leaves a field empty/
# missing (e.g., growing_season), we fallback to the built-in default to avoid losing
# important data used elsewhere (calendar generation, etc.).
for crop_key, custom in CUSTOM_CROP_DATA.items():
    base = CROP_DATA.get(crop_key, {})
    merged = {**base, **{k: v for k, v in custom.items() if v not in (None, [], {}, "")}}
    CROP_DATA[crop_key] = merged


def get_crop_info(crop_type: str) -> Optional[Dict[str, Any]]:
    """Return crop requirements dict for given crop name (case-insensitive)."""
    return CROP_DATA.get(crop_type.lower())

# ---------------------------------------------------------------------------
# Helper: Generate dynamic crop calendar and local alerts
# ---------------------------------------------------------------------------

def generate_crop_calendar(crop_type: str, location: 'Location', weather: 'WeatherData'):
    """Generate a basic crop calendar and alerts.

    Returns a tuple (calendar, alerts) where:
    - calendar: List[dict] with keys: date (ISO), task, note
    - alerts: List[str] of upcoming task reminders within 14 days
    """
    try:
        crop_info = get_crop_info(crop_type.lower())
        if not crop_info:
            return [], []

        today = datetime.now().date()
        growing_months = crop_info.get("growing_season", [])

        # If no specific growing season is provided for this crop, assume it can be planted
        # immediately (today). This prevents the calendar from being empty for crops that
        # come from the extended `crops.json` file but lack seasonality metadata.
        if not growing_months:
            planting_date = today  # Use today's date as the planting date
        else:
            # Determine the next suitable planting month
            month_today = today.month
            # Compute month offsets (0-11). If the crop can be planted this month, use 0 instead of forcing 12.
            offsets = [(m - month_today) % 12 for m in growing_months]
            next_offset = min(offsets)
            planting_year = today.year if month_today + next_offset <= 12 else today.year + 1
            planting_month = ((month_today + next_offset - 1) % 12) + 1
            planting_date = datetime(planting_year, planting_month, 1).date()


        maturity_days = crop_info.get("maturity_days", 90)
        calendar = [
            {"date": planting_date.isoformat(), "task": "Planting", "note": "Start sowing seeds."},
            {"date": (planting_date + timedelta(days=7)).isoformat(), "task": "Fertilizing", "note": "Apply recommended fertilizer."},
            {"date": (planting_date + timedelta(days=30)).isoformat(), "task": "Weeding", "note": "Carry out first weeding."},
            {"date": (planting_date + timedelta(days=maturity_days)).isoformat(), "task": "Harvesting", "note": "Expected harvest window."}
        ]

        alerts: List[str] = []
        for entry in calendar:
            task_date = datetime.fromisoformat(entry["date"]).date()
            days_until = (task_date - today).days
            if 0 <= days_until <= 14:
                alerts.append(f"{entry['task']} in {days_until} day(s) on {entry['date']}")
        return calendar, alerts
    except Exception as exc:
        logger.error(f"generate_crop_calendar error: {exc}")
        return [], []

# ---------------------------------------------------------------------------
# Helper: Yield estimation
# ---------------------------------------------------------------------------

def estimate_yield(crop_type: str, weather: 'WeatherData', soil: 'SoilData') -> Dict[str, Any]:
    """Estimate crop yield and provide explanation.

    Returns dict with:
    - value: float (t/ha)
    - breakdown: List[str] describing each penalty/bonus applied
    """
    crop_info = get_crop_info(crop_type.lower())
    if not crop_info:
        return {"value": 0.0, "breakdown": ["Unknown crop"]}

    yield_est = crop_info.get("baseline_yield", 1.0)
    breakdown: List[str] = [f"Baseline yield: {yield_est} t/ha"]

    # pH adjustment
    ph_low, ph_high = crop_info["optimal_ph"]
    if not (ph_low <= soil.ph <= ph_high):
        yield_est *= 0.85
        breakdown.append(f"Soil pH {soil.ph:.1f} outside optimal ({ph_low}-{ph_high}) → −15%")

    # Temperature adjustment
    temp_low, temp_high = crop_info["optimal_temp"]
    if not (temp_low <= weather.temperature <= temp_high):
        yield_est *= 0.90
        breakdown.append(f"Temperature {weather.temperature:.1f}°C outside optimal ({temp_low}-{temp_high}) → −10%")

    # Soil moisture adjustment
    if soil.moisture < crop_info["optimal_soil_moisture"]:
        yield_est *= 0.90
        breakdown.append("Soil moisture below optimal → −10%")

    # Humidity / rainfall proxy
    if weather.humidity < 50:
        yield_est *= 0.90
        breakdown.append("Low humidity / rainfall → −10%")

    # Wind speed adjustment
    if weather.wind_speed > crop_info["optimal_wind_speed"]:
        yield_est *= 0.95
        breakdown.append("High wind speed → −5%")

    floor = 0.3 * crop_info.get("baseline_yield", 1.0)
    if yield_est < floor:
        yield_est = floor
        breakdown.append("Applied minimum floor (30% of baseline)")

    yield_final = round(yield_est, 1)
    breakdown.append(f"Estimated yield: {yield_final} t/ha")
    return {"value": yield_final, "breakdown": breakdown}

class Location(BaseModel):
    lat: float
    lon: float

class FarmInput(BaseModel):
    crop_type: str
    location: Location
    soil_type: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            crop_type=data["crop_type"],
            location=Location(
                lat=data["location"]["lat"],
                lon=data["location"]["lon"]
            ),
            soil_type=data.get("soil_type")
        )

class WeatherData(BaseModel):
    temperature: float
    humidity: float
    wind_speed: float
    solar_radiation: float
    soil_moisture: float
    
    def get_weather_summary(self) -> dict:
        """Generate a weather summary for the advisory"""
        return {
            "temperature": f"{self.temperature:.1f}" if self.temperature is not None else "N/A",
            "rain": self._get_rain_level() if self.humidity is not None else "N/A",
            "humidity": f"{self.humidity:.0f}" if self.humidity is not None else "N/A",
            "wind_speed": self._get_wind_speed_level() if self.wind_speed is not None else "N/A",
            "sunshine": self._get_sunshine_level() if self.solar_radiation is not None else "N/A",
            "soil_moisture": f"{self.soil_moisture*100:.0f}" if self.soil_moisture is not None else "N/A"
        }
    
    def _get_sunshine_level(self) -> str:
        """Determine sunshine level based on solar radiation"""
        if self.solar_radiation > 25:
            return "Strong"
        elif self.solar_radiation > 15:
            return "Moderate"
        else:
            return "Low"
    
    def _get_wind_speed_level(self) -> str:
        """Determine wind speed level"""
        if self.wind_speed > 10:
            return "High"
        elif self.wind_speed > 5:
            return "Moderate"
        else:
            return "Low"
    
    def _get_rain_level(self) -> str:
        """Determine rain level"""
        # Currently we don't have direct rainfall data, so we infer from humidity
        if self.humidity is None:
            return "Unknown"
        if self.humidity >= 80:
            return "High"
        elif self.humidity >= 50:
            return "Moderate"
        else:
            return "Low"

class SoilData(BaseModel):
    ph: float
    moisture: float
    type: Optional[str] = None

async def get_nasa_power_data(lat: float, lon: float):
    """Get historical climate data from NASA POWER API"""
    try:
        logger.info(f"Fetching NASA POWER data for lat: {lat}, lon: {lon}")
        
        # Get current date
        now = datetime.now()
        start_date = now.strftime("%Y%m%d")
        end_date = now.strftime("%Y%m%d")  # Single day data
        
        # Parameters for agriculture community
        params = {
            "parameters": "T2M,RH2M,WS2M,ALLSKY_SFC_SW_DWN",  # Temperature, Humidity, Wind Speed, Solar Radiation
            "start": start_date,
            "end": end_date,
            "latitude": lat,
            "longitude": lon,
            "community": "AG",  # Agriculture community
            "format": "JSON"
        }
        
        url = "https://power.larc.nasa.gov/api/temporal/daily/point"
        logger.info(f"Making NASA POWER API request with params: {params}")
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, params=params)
            logger.info(f"NASA POWER API response status: {response.status_code}")
            response.raise_for_status()
            data = response.json()
            logger.info(f"NASA POWER API response headers: {dict(response.headers)}")
            logger.debug(f"NASA POWER API raw response: {data}")
        
        # Check for error in response
        if "error" in data:
            raise Exception(f"NASA POWER API error: {data['error']}")
            
        # Get the properties
        if "properties" not in data:
            raise Exception("Invalid NASA POWER API response format: missing properties")
            
        properties = data["properties"]
        logger.info(f"Properties from response: {properties}")
        
        # Get the parameter data
        if "parameter" not in properties:
            raise Exception(f"Invalid NASA POWER API response format: missing parameter data. Response: {properties}")
            
        parameters = properties["parameter"]
        logger.info(f"Parameters from response: {parameters}")
        
        # Get temperature data with fallbacks
        try:
            temp_data = parameters.get("T2M", {})
            if not temp_data:
                raise ValueError("No temperature data found in response")
                
            mean_data = temp_data.get("MEAN", {})
            if not mean_data:
                raise ValueError("No MEAN temperature data found")
                
            all_data = mean_data.get("all", {})
            if not all_data:
                raise ValueError("No all-time temperature data found")
                
            value_data = all_data.get("value", {})
            if not value_data:
                raise ValueError("No temperature value data found")
                
            date_keys = list(value_data.keys())
            if not date_keys:
                raise ValueError("No available dates in temperature data")
                
            date_key = date_keys[0]
            logger.info(f"Using date key: {date_key}")
            temperature = value_data.get(date_key, 25.0)  # Default to 25°C
            
        except ValueError as e:
            logger.error(f"Error extracting temperature data: {e}")
            temperature = 25.0  # Fallback to default temperature
            
        # Get humidity data with fallbacks
        try:
            humidity_data = parameters.get("RH2M", {})
            if not humidity_data:
                raise ValueError("No humidity data found in response")
                
            mean_data = humidity_data.get("MEAN", {})
            if not mean_data:
                raise ValueError("No MEAN humidity data found")
                
            all_data = mean_data.get("all", {})
            if not all_data:
                raise ValueError("No all-time humidity data found")
                
            value_data = all_data.get("value", {})
            if not value_data:
                raise ValueError("No humidity value data found")
                
            humidity = value_data.get(date_key, 70.0)  # Default to 70%
            
        except ValueError as e:
            logger.error(f"Error extracting humidity data: {e}")
            humidity = 70.0  # Fallback to default humidity
            
        # Get wind speed data with fallbacks
        try:
            wind_data = parameters.get("WS2M", {})
            if not wind_data:
                raise ValueError("No wind speed data found in response")
                
            mean_data = wind_data.get("MEAN", {})
            if not mean_data:
                raise ValueError("No MEAN wind speed data found")
                
            all_data = mean_data.get("all", {})
            if not all_data:
                raise ValueError("No all-time wind speed data found")
                
            value_data = all_data.get("value", {})
            if not value_data:
                raise ValueError("No wind speed value data found")
                
            wind_speed = value_data.get(date_key, 2.0)  # Default to 2m/s
            
        except ValueError as e:
            logger.error(f"Error extracting wind speed data: {e}")
            wind_speed = 2.0  # Fallback to default wind speed
            
        # Get solar radiation data with fallbacks
        try:
            solar_data = parameters.get("ALLSKY_SFC_SW_DWN", {})
            if not solar_data:
                raise ValueError("No solar radiation data found in response")
                
            mean_data = solar_data.get("MEAN", {})
            if not mean_data:
                raise ValueError("No MEAN solar radiation data found")
                
            all_data = mean_data.get("all", {})
            if not all_data:
                raise ValueError("No all-time solar radiation data found")
                
            value_data = all_data.get("value", {})
            if not value_data:
                raise ValueError("No solar radiation value data found")
                
            solar_radiation = value_data.get(date_key, 18.0)  # Default to 18 MJ/m²/day
            
        except ValueError as e:
            logger.error(f"Error extracting solar radiation data: {e}")
            solar_radiation = 18.0  # Fallback to default solar radiation
            
        logger.info(f"Extracted values: temperature={temperature}, humidity={humidity}, wind_speed={wind_speed}, solar_radiation={solar_radiation}")
        
        # Calculate soil moisture from humidity (simplified approximation)
        soil_moisture = humidity / 100.0  # Convert percentage to fraction
            
        return WeatherData(
            temperature=float(temperature),
            humidity=float(humidity),
            wind_speed=float(wind_speed),
            solar_radiation=float(solar_radiation),
            soil_moisture=float(soil_moisture)
        )
        
    except httpx.RequestError as e:
        logger.error(f"Network error with NASA POWER API: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to fetch weather data: {str(e)}")
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error with NASA POWER API: {e}")
        raise HTTPException(status_code=e.response.status_code, detail=f"HTTP error fetching weather data: {str(e)}")
    except KeyError as e:
        logger.error(f"Missing key in NASA POWER response: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid NASA POWER API response format: {str(e)}")
    except Exception as e:
        logger.error(f"Error processing NASA POWER data: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to process weather data: {str(e)}")

async def get_soilgrids_data(lat: float, lon: float):
    """Get soil data from SoilGrids API (parallel fetch for responsiveness)"""
    try:
        logger.info(f"Fetching soil data for lat: {lat}, lon: {lon}")

        # Validate coordinates
        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            raise ValueError("Invalid coordinates")

        base_url = "https://rest.soilgrids.org/query?"

        async def make_request(prop: str, depth: str = "0-30cm") -> Dict[str, Any]:
            params = {
                "lon": lon,
                "lat": lat,
                "property": prop,
                "depth": depth,
                "value": "mean",
                "download": False
            }
            url = base_url + "&".join([f"{k}={v}" for k, v in params.items()])
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                return resp.json()

        # Launch SoilGrids requests concurrently
        ph_data, moisture_data, texture_data = await asyncio.gather(
            make_request("phh2o"),
            make_request("bdod"),
            make_request("sltp")
        )

        moisture_percentage = moisture_data["properties"]["value"] / 100.0
        texture_value = texture_data["properties"]["value"]
        if texture_value < 20:
            texture_type = "Clay"
        elif texture_value < 40:
            texture_type = "Sandy Clay"
        elif texture_value < 60:
            texture_type = "Loam"
        else:
            texture_type = "Sand"

        logger.info(f"Soil data fetched successfully: pH={ph_data['properties']['value']}, moisture={moisture_percentage}, type={texture_type}")
        return SoilData(
            ph=ph_data["properties"]["value"],
            moisture=moisture_percentage,
            type=texture_type
        )

    except ValueError as ve:
        logger.error(f"Invalid coordinates: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except (httpx.HTTPError, KeyError) as e:
        logger.error(f"SoilGrids fetch error: {e}")
        # Fallback to default soil values when API fails
        return SoilData(ph=6.0, moisture=0.30, type="Loam")

async def get_weather_data(lat: float, lon: float):
    """Get combined weather data from NASA POWER and OpenWeatherMap (parallel fetching for speed)."""
    try:
        api_key = os.getenv("OPENWEATHERMAP_API_KEY")
        if not api_key:
            logger.error("OpenWeatherMap API key not found")
            raise HTTPException(status_code=500, detail="OpenWeatherMap API key not configured")

        async def fetch_openweather() -> Dict[str, Any]:
            url = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units=metric"
            logger.info(f"Fetching OpenWeatherMap data from: {url}")
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                return resp.json()

        # Run NASA POWER and OpenWeatherMap calls concurrently
        nasa_task = asyncio.create_task(get_nasa_power_data(lat, lon))
        owm_task = asyncio.create_task(fetch_openweather())
        nasa_data, owm_data = await asyncio.gather(nasa_task, owm_task)

        if "list" not in owm_data:
            logger.error("Invalid OpenWeatherMap response format")
            raise HTTPException(status_code=500, detail="Invalid OpenWeatherMap response format")

        temp_sum = rain_sum = humidity_sum = wind_sum = count = 0
        for forecast in owm_data["list"][:8]:
            main = forecast.get("main", {})
            temp_sum += main.get("temp", 0)
            humidity_sum += main.get("humidity", 0)
            rain_sum += forecast.get("rain", {}).get("3h", 0)
            wind_sum += forecast.get("wind", {}).get("speed", 0)
            count += 1

        if count == 0:
            logger.error("No forecast data received from OpenWeatherMap")
            raise HTTPException(status_code=502, detail="No forecast data from OpenWeatherMap")

        logger.info(f"Calculated weather averages: temp={temp_sum/count}, rain={rain_sum}, humidity={humidity_sum/count}, wind={wind_sum/count}")

        return WeatherData(
            temperature=(temp_sum / count + nasa_data.temperature) / 2,
            humidity=humidity_sum / count,
            wind_speed=wind_sum / count,
            solar_radiation=nasa_data.solar_radiation,
            soil_moisture=nasa_data.soil_moisture
        )
    except (httpx.HTTPError, ValueError) as e:
        logger.error(f"HTTP error fetching weather data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch weather data: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error fetching weather data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process weather data: {str(e)}")

async def get_soil_data(lat: float, lon: float):
    """Get soil data from SoilGrids API"""
    try:
        logger.info(f"Fetching SoilGrids data for lat: {lat}, lon: {lon}")
        soil_data = await get_soilgrids_data(lat, lon)
        if not soil_data:
            logger.error("Failed to get SoilGrids data")
            raise HTTPException(status_code=500, detail="Failed to get SoilGrids data")
        
        logger.info(f"SoilGrids data received: {soil_data}")
        
        # If the response is already a SoilData instance (fallback case), return it directly
        if isinstance(soil_data, SoilData):
            return soil_data

        # Otherwise, expect a dict-like structure and convert to SoilData
        try:
            return SoilData(
                ph=soil_data["ph"],
                moisture=soil_data["moisture"],
                type=soil_data.get("type")
            )
        except (KeyError, TypeError) as e:
            logger.error(f"Unexpected soil data format: {e}")
            raise HTTPException(status_code=500, detail="Invalid soil data format")
    except Exception as e:
        logger.error(f"Error fetching soil data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch soil data: {str(e)}")

async def generate_advice_with_gpt(crop_type: str, weather: WeatherData, soil: SoilData, location: Optional[Location] = None):
    """Generate farming advice using GPT (fallback to rule-based)"""
    try:
        # Build comprehensive prompt that includes user's location and uses farmer-friendly language
        location_str = f"Latitude: {location.lat:.2f}, Longitude: {location.lon:.2f}" if location else "Unknown location"
        prompt = f"""
        You are an agricultural expert. Provide detailed farming advice for {crop_type} in Nigeria based on these conditions:
        
        Location: {location_str}
        
        Current Weather Conditions:
        - Temperature: {weather.temperature:.1f}°C
        - Humidity: {weather.humidity:.0f}%
        - Wind Speed: {weather.wind_speed:.1f} km/h
        - Solar Radiation: {weather.solar_radiation:.1f} MJ/m²/day
        - Soil Moisture: {weather.soil_moisture:.1%}
        
        Soil Conditions:
        - pH: {soil.ph:.1f}
        - Type: {soil.type or 'Unknown'}
        
        
        Please provide:
        1. A concise summary of current conditions
        2. Specific farming actions to take
        3. Any warnings or concerns
        4. A brief explanation of why these actions are recommended
        5. Voice guidance for farmers (imagine you're speaking to them)
        
        Format your response in markdown following the template below for excellent readability:\n        \n        ## 1. Current Conditions\n        - <list each condition on its own line>\n        \n        ## 2. Specific Farming Actions\n        - <one action per line>\n        \n        ## 3. Warnings or Concerns\n        - <one concern per line>\n        \n        ## 4. Explanation of Recommended Actions\n        - <one explanation per line>\n        \n        ## 5. Voice Guidance (Farmer Friendly)\n        - <one or two short sentences>
        """
        
        # Make API call with proper settings
        response = await openai_client.chat.completions.create(
            model=openai_settings["model"],
            messages=[{"role": "user", "content": prompt}],
            temperature=openai_settings["temperature"],
            max_tokens=openai_settings["max_tokens"],
            top_p=openai_settings["top_p"],
            frequency_penalty=openai_settings["frequency_penalty"],
            presence_penalty=openai_settings["presence_penalty"]
        )
        
        # Extract and format the response
        advice = response.choices[0].message.content
        logger.info(f"GPT advice generated successfully for {crop_type}")
        return advice
    except OpenAIError as e:
        logger.error(f"OpenAI API error while generating advice: {str(e)}")
        # Fallback to rule-based advice
        crop_info = get_crop_info(crop_type.lower())
        if not crop_info:
            logger.error(f"Invalid crop type: {crop_type}")
            raise HTTPException(status_code=400, detail=f"Invalid crop type: {crop_type}")
        
        logger.info(f"Falling back to rule-based advice for {crop_type}")
        return generate_advice(crop_type, weather, soil, crop_info)
    except Exception as e:
        logger.error(f"Error generating advice with GPT: {str(e)}")
        # Fallback to rule-based advice
        crop_info = get_crop_info(crop_type.lower())
        if not crop_info:
            logger.error(f"Invalid crop type: {crop_type}")
            raise HTTPException(status_code=400, detail=f"Invalid crop type: {crop_type}")
        
        logger.info(f"Falling back to rule-based advice for {crop_type}")
        return generate_advice(crop_type, weather, soil, crop_info)

def generate_advice(crop_type: str, weather: WeatherData, soil: SoilData, crop_info: Dict[str, Any]):
    """Generate farming advice using simple rules"""
    
    # Get current month
    current_month = datetime.now().month
    
    # Check if it's planting season
    is_in_season = current_month in crop_info["growing_season"]
    
    # Check if conditions are optimal
    is_optimal_ph = soil.ph >= crop_info["optimal_ph"][0] and soil.ph <= crop_info["optimal_ph"][1]
    is_optimal_temp = weather.temperature >= crop_info["optimal_temp"][0] and weather.temperature <= crop_info["optimal_temp"][1]
    is_optimal_rain = weather.humidity >= crop_info["optimal_rainfall"]
    is_optimal_soil = soil.moisture >= crop_info["optimal_soil_moisture"]
    is_wind_safe = weather.wind_speed <= crop_info["optimal_wind_speed"]
    
    # Generate advice based on conditions
    if not is_in_season:
        return f"❌ Wait to plant {crop_type.title()}. It's not the right season."
    
    if not is_optimal_ph:
        return f"❌ Wait to plant {crop_type.title()}. Soil pH is not optimal."
    
    if not is_optimal_temp:
        return f"❌ Wait to plant {crop_type.title()}. Temperature is not optimal."
    
    if not is_optimal_rain:
        return f"❌ Wait to plant {crop_type.title()}. Not enough rainfall expected."
    
    if not is_optimal_soil:
        return f"❌ Wait to plant {crop_type.title()}. Soil moisture is too low."
    
    if not is_wind_safe:
        return f"❌ Wait to plant {crop_type.title()}. Wind speed is too high for planting."
    
    return f"✅ Plant {crop_type.title()} now. Conditions are ideal for planting."

# Initialize pygame mixer only if pygame is available
if pygame_available:
    try:
        pygame.mixer.init()
    except Exception as exc:
        logger.warning(f"pygame mixer init failed: {exc}. Disabling audio features.")
        pygame_available = False

# Function to convert text to speech and play it in a separate thread
def play_audio(text, lang='yo'):
    # Skip audio playback if pygame is not available (e.g., server environment)
    if not pygame_available:
        logger.info("pygame not available; skipping audio playback")
        return

    def _play():
        # Convert text to speech
        tts = gTTS(text=text, lang=lang)
        audio_file = 'output.mp3'
        tts.save(audio_file)
        
        # Play the audio file
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play()

        # Wait for the audio to finish
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

        # Remove the audio file after playing
        os.remove(audio_file)

    # Run the audio playback in a separate thread
    threading.Thread(target=_play).start()

@app.post("/predict")
async def predict_farming_actions(request: dict):
    logger.info("Received request for farming actions prediction")
    try:
        logger.info(f"Request data: {request}")
        farm_input = FarmInput.from_dict(request)
        logger.info(f"Parsed farm input: {farm_input}")
        
        # Fetch weather and soil data
        weather_data = await get_weather_data(farm_input.location.lat, farm_input.location.lon)
        logger.info(f"Fetched weather data: {weather_data}")
        soil_data = await get_soil_data(farm_input.location.lat, farm_input.location.lon)
        logger.info(f"Fetched soil data: {soil_data}")

        # Generate advice
        advice = await generate_advice_with_gpt(farm_input.crop_type, weather_data, soil_data, farm_input.location)
        logger.info(f"Generated advice: {advice}")

        # Generate dynamic crop calendar and alerts
        calendar, alerts = generate_crop_calendar(farm_input.crop_type, farm_input.location, weather_data)

        # Estimate yield and explanation
        yield_data = estimate_yield(farm_input.crop_type, weather_data, soil_data)

        # Prepare response
        response_data = {
            "advice": advice,
            "weather": weather_data.get_weather_summary(),
            "soil": {
                "ph": soil_data.ph,
                "moisture": soil_data.moisture,
                "type": soil_data.type
            },
            "crop": farm_input.crop_type,
            "calendar": calendar,
            "alerts": alerts,
            "yield_forecast": yield_data["value"],
            "yield_breakdown": yield_data["breakdown"]
        }
        logger.info(f"Final response data: {response_data}")
        
        # Play the advice as audio
        play_audio(advice, lang='yo')
        
        return response_data
    except HTTPException as e:
        logger.error(f"HTTP error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in predict_farming_actions: {str(e)}")
        logger.error("Traceback:", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.post("/translate")
async def translate_text(payload: dict):
    """Translate the provided text into a target language using OpenAI"""
    try:
        text = payload.get("text")
        target_language = payload.get("target_language", "yo")  # Default language code (Yoruba)
        # Map ISO codes to language names understood by GPT
        lang_map = {
            "yo": "Yoruba",
            "ha": "Hausa",
            "ig": "Igbo",
            "fr": "French",
            "es": "Spanish",
            "en": "English"
        }
        target_language_name = lang_map.get(target_language.lower(), target_language)
        if not text:
            raise HTTPException(status_code=400, detail="No text provided for translation")

        # Ensure OpenAI API key is configured
        if not openai_client.api_key:
            logger.error("OpenAI API key is not configured. Set the OPENAI_API_KEY environment variable.")
            raise HTTPException(status_code=500, detail="OpenAI API key not configured on server")

        prompt = (
            f"You are a professional translator. Translate the following farming advice into {target_language_name}. "
            "Preserve any existing markdown formatting, line breaks, and bullet points. "
            "Return ONLY the translated text without additional commentary.\n\n" + text
        )

        response = await openai_client.chat.completions.create(
            model=openai_settings["model"],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=700
        )
        translation = response.choices[0].message.content.strip()
        logger.info("Translation generated successfully")
        return {"translation": translation}
    except HTTPException:
        raise
    except OpenAIError as e:
        logger.error(f"OpenAI API error: {str(e)}")
        raise HTTPException(status_code=502, detail="Upstream OpenAI error during translation")
    except Exception as e:
        logger.error(f"Error translating text: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to translate text")

# ---------------------------------------------------------------------------
# Chat endpoint: Ask Me Anything for agriculture questions
# ---------------------------------------------------------------------------
@app.post("/chat")
async def chat_assistant(payload: dict):
    """Answer farmer questions related to agriculture only."""
    question = payload.get("question")
    if not question:
        raise HTTPException(status_code=400, detail="Question is required")

    try:
        # Ensure OpenAI API key is configured
        if not openai_client.api_key:
            logger.error("OpenAI API key is not configured. Set the OPENAI_API_KEY environment variable.")
            raise HTTPException(status_code=500, detail="OpenAI API key not configured on server")

        messages = [
            {
                "role": "system",
                "content": (
                    "You are AgroPredict AI, an expert agronomist specialising in smallholder farming in West Africa. "
                    "Answer ONLY questions related to agriculture—including crops, soil, climate, fertiliser, pest and disease management, post-harvest handling, and farm business management. "
                    "If the user's question is outside agriculture, politely respond that you can only answer agriculture-related questions."
                )
            },
            {"role": "user", "content": question}
        ]

        response = await openai_client.chat.completions.create(
            model=openai_settings["model"],
            messages=messages,
            temperature=openai_settings["temperature"],
            max_tokens=700
        )
        answer = response.choices[0].message.content.strip()
        return {"answer": answer}
    except HTTPException:
        raise
    except OpenAIError as e:
        logger.error(f"OpenAI API error: {str(e)}")
        raise HTTPException(status_code=502, detail="Upstream OpenAI error during chat")
    except Exception as e:
        logger.error(f"Error processing chat: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process chat")

# ---------------------------------------------------------------------------
# Text-to-Speech endpoint to generate audio for translated advice
# ---------------------------------------------------------------------------
@app.post("/tts")
async def text_to_speech(payload: dict):
    """Generate TTS MP3 for the provided text and language."""
    text = payload.get("text")
    lang = payload.get("lang", "yo")
    if not text:
        raise HTTPException(status_code=400, detail="Text is required for TTS")
    try:
        tts = gTTS(text=text, lang=lang)
        buffer = io.BytesIO()
        tts.write_to_fp(buffer)
        buffer.seek(0)
        # Return as streaming response so frontend can create an ObjectURL
        return StreamingResponse(buffer, media_type="audio/mpeg")
    except ValueError as ve:
        # gTTS raises ValueError for unsupported languages
        logger.error(f"Unsupported language for TTS: {lang} - {ve}")
        raise HTTPException(status_code=400, detail=f"Language '{lang}' not supported for audio")
    except Exception as e:
        logger.error(f"TTS generation failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate audio")


# Cached crop list for faster repeated calls
@lru_cache(maxsize=1)
def _cached_crop_list():
    """Compute and cache the crop dropdown list once at startup."""
    return sorted([
        {"name": k.title(), "category": v.get("category", "")} for k, v in CROP_DATA.items()
    ], key=lambda x: x["name"])

@app.get("/crops")
async def list_available_crops():
    """Return cached list of crop names for dropdown."""
    return _cached_crop_list()

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/test")
async def test_advice():
    """Test endpoint with sample Ekiti data for maize"""
    sample_input = {
        "crop_type": "maize",
        "location": {"lat": 7.7, "lon": 5.3},
        "soil_type": "sandy loam"
    }
    return await predict_farming_actions(sample_input)

# ---------------------------------------------------------------------------
# Entry point: allows `python main.py` to start a development server quickly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn, os
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True)

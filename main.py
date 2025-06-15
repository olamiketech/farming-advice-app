from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import httpx
import os
from dotenv import load_dotenv
import logging
from typing import Optional, Dict, Any

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

load_dotenv()

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
        "growing_season": [6, 7, 8]  # June, July, August
    },
    "cassava": {
        "optimal_ph": (5.0, 6.0),
        "optimal_temp": (24, 28),
        "optimal_rainfall": 10,
        "optimal_soil_moisture": 0.4,
        "optimal_wind_speed": 12,
        "growing_season": [3, 4, 5]  # March, April, May
    },
    "rice": {
        "optimal_ph": (6.0, 7.0),
        "optimal_temp": (26, 32),
        "optimal_rainfall": 20,
        "optimal_soil_moisture": 0.5,
        "optimal_wind_speed": 8,
        "growing_season": [9, 10, 11]  # September, October, November
    }
}

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
    """Get soil data from SoilGrids API"""
    try:
        logger.info(f"Fetching soil data for lat: {lat}, lon: {lon}")
        
        # Validate coordinates
        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            logger.error(f"Invalid coordinates: lat={lat}, lon={lon}")
            raise ValueError("Invalid coordinates")
            
        # Base URL and common parameters
        base_url = "https://rest.soilgrids.org/query?"
        common_params = {
            "lon": lon,
            "lat": lat,
            "property": "bdod",
            "depth": "0-30cm",
            "value": "mean",
            "download": False
        }
        
        # Get pH data
        ph_params = common_params.copy()
        ph_params["property"] = "phh2o"
        ph_url = base_url + "&".join([f"{k}={v}" for k, v in ph_params.items()])
        
        logger.info(f"Making SoilGrids pH request to: {ph_url}")
        async with httpx.AsyncClient(timeout=10.0) as client:
            ph_response = await client.get(ph_url)
            logger.info(f"SoilGrids pH response status: {ph_response.status_code}")
            ph_response.raise_for_status()
            ph_data = ph_response.json()
            logger.debug(f"SoilGrids pH response: {ph_data}")
            
        # Get soil moisture data
        moisture_params = common_params.copy()
        moisture_params["property"] = "bdod"
        moisture_url = base_url + "&".join([f"{k}={v}" for k, v in moisture_params.items()])
        
        logger.info(f"Making SoilGrids moisture request to: {moisture_url}")
        async with httpx.AsyncClient(timeout=10.0) as client:
            moisture_response = await client.get(moisture_url)
            logger.info(f"SoilGrids moisture response status: {moisture_response.status_code}")
            moisture_response.raise_for_status()
            moisture_data = moisture_response.json()
            logger.debug(f"SoilGrids moisture response: {moisture_data}")
            
        # Calculate moisture percentage
        moisture_percentage = moisture_data["properties"]["value"] / 100.0
        
        # Get soil texture data
        texture_params = common_params.copy()
        texture_params["property"] = "sltp"
        texture_url = base_url + "&".join([f"{k}={v}" for k, v in texture_params.items()])
        
        logger.info(f"Making SoilGrids texture request to: {texture_url}")
        async with httpx.AsyncClient(timeout=10.0) as client:
            texture_response = await client.get(texture_url)
            logger.info(f"SoilGrids texture response status: {texture_response.status_code}")
            texture_response.raise_for_status()
            texture_data = texture_response.json()
            logger.debug(f"SoilGrids texture response: {texture_data}")
            
        # Map texture value to type
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
        logger.error(f"Invalid coordinates: {str(ve)}")
        raise HTTPException(
            status_code=400,
            detail=str(ve)
        )
    except Exception as e:
        logger.error(f"Error in get_soilgrids_data: {str(e)} - Returning default soil data fallback.")
        # Fallback to default reasonable soil values when API fails
        return SoilData(
            ph=6.0,
            moisture=0.30,
            type="Loam"
        )

async def get_weather_data(lat: float, lon: float):
    """Get combined weather data from NASA POWER and OpenWeatherMap"""
    try:
        print(f"Fetching NASA POWER data for lat: {lat}, lon: {lon}")
        nasa_data = await get_nasa_power_data(lat, lon)
        if not nasa_data:
            print("Failed to get NASA POWER data")
            return None
            
        print(f"NASA POWER data received: {nasa_data}")
        
        # Get OpenWeatherMap forecast
        api_key = os.getenv("OPENWEATHERMAP_API_KEY")
        if not api_key:
            raise Exception("OpenWeatherMap API key not found")
            
        url = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        print(f"Fetching OpenWeatherMap data from: {url}")
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()
        print(f"OpenWeatherMap response: {data}")
        
        if "list" not in data:
            raise Exception("Invalid OpenWeatherMap response format")
            
        # Get average values for the next 3 days
        temp_sum = 0
        rain_sum = 0
        humidity_sum = 0
        wind_sum = 0
        count = 0
        
        for forecast in data["list"][:8]:  # First 8 entries (every 3 hours)
            if "main" not in forecast:
                raise Exception("Invalid forecast data format")
                
            temp_sum += forecast["main"]["temp"]
            rain_sum += forecast["rain"]["3h"] if "rain" in forecast else 0
            humidity_sum += forecast["main"]["humidity"]
            wind_sum += forecast["wind"]["speed"]
            count += 1
        
        print(f"Calculated weather averages: temp={temp_sum/count}, rain={rain_sum}, humidity={humidity_sum/count}, wind={wind_sum/count}")
        
        # Combine NASA POWER and OpenWeatherMap data
        return WeatherData(
            temperature=(temp_sum / count + nasa_data.temperature) / 2,  # Average of forecast and historical
            humidity=humidity_sum / count,
            wind_speed=wind_sum / count,
            solar_radiation=nasa_data.solar_radiation,
            soil_moisture=nasa_data.soil_moisture
        )
    except requests.exceptions.RequestException as e:
        print(f"Network error fetching weather data: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to fetch weather data: {str(e)}")
    except KeyError as e:
        print(f"Missing key in weather data: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid weather data format")
    except Exception as e:
        print(f"Unexpected error fetching weather data: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to process weather data: {str(e)}")


async def get_soil_data(lat: float, lon: float):
    """Get soil data from SoilGrids API"""
    try:
        soil_data = await get_soilgrids_data(lat, lon)
        if not soil_data:
            raise Exception("Failed to get SoilGrids data")
            
        return SoilData(
            ph=soil_data["ph"],
            moisture=soil_data["moisture"],
            type=soil_data["type"]
        )
    except Exception as e:
        print(f"Error fetching soil data: {e}")
        return None

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
        crop_info = CROP_DATA.get(crop_type.lower())
        if not crop_info:
            logger.error(f"Invalid crop type: {crop_type}")
            raise HTTPException(status_code=400, detail=f"Invalid crop type: {crop_type}")
        
        logger.info(f"Falling back to rule-based advice for {crop_type}")
        return generate_advice(crop_type, weather, soil, crop_info)
    except Exception as e:
        logger.error(f"Error generating advice with GPT: {str(e)}")
        # Fallback to rule-based advice
        crop_info = CROP_DATA.get(crop_type.lower())
        if not crop_info:
            logger.error(f"Invalid crop type: {crop_type}")
            raise HTTPException(status_code=400, detail=f"Invalid crop type: {crop_type}")
        
        logger.info(f"Falling back to rule-based advice for {crop_type}")
        return generate_advice(crop_type, weather, soil, crop_info)

@app.post("/predict")
async def predict_farming_actions(request: dict):
    """Predict farming actions based on location and crop type"""
    try:
        logger.info(f"Received request: {request}")
        
        # Validate request
        if not isinstance(request, dict):
            raise HTTPException(status_code=400, detail="Invalid request format")
            
        # Convert dict to Pydantic model
        try:
            farm_input = FarmInput.from_dict(request)
            logger.info(f"Parsed farm input: {farm_input}")
        except Exception as e:
            logger.error(f"Error parsing farm input: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Invalid input data: {str(e)}")
        
        logger.info(f"Processing request for crop: {farm_input.crop_type} at lat: {farm_input.location.lat}, lon: {farm_input.location.lon}")
        
        # Get crop information
        crop_info = CROP_DATA.get(farm_input.crop_type.lower())
        if not crop_info:
            logger.warning(f"Crop not found: {farm_input.crop_type}")
            raise HTTPException(status_code=400, detail=f"Invalid crop type: {farm_input.crop_type}")
            
        # Get weather data with retry
        max_retries = 3
        retry_delay = 1  # seconds
        for attempt in range(max_retries):
            try:
                logger.info(f"Fetching weather data (attempt {attempt + 1}/{max_retries})...")
                weather_data = await get_weather_data(farm_input.location.lat, farm_input.location.lon)
                logger.info(f"Weather data received: {weather_data}")
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed to get weather data after {max_retries} attempts: {str(e)}")
                    raise HTTPException(status_code=500, detail=f"Failed to fetch weather data: {str(e)}")
                logger.warning(f"Weather data fetch failed, retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
        
        # Get soil data with retry
        for attempt in range(max_retries):
            try:
                logger.info(f"Fetching soil data (attempt {attempt + 1}/{max_retries})...")
                soil_data = await get_soilgrids_data(farm_input.location.lat, farm_input.location.lon)
                logger.info(f"Soil data received: {soil_data}")
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed to get soil data after {max_retries} attempts: {str(e)}")
                    raise HTTPException(status_code=500, detail=f"Failed to fetch soil data: {str(e)}")
                logger.warning(f"Soil data fetch failed, retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
        
        # Generate advice with retry
        for attempt in range(max_retries):
            try:
                logger.info(f"Generating advice (attempt {attempt + 1}/{max_retries})...")
                advice = await generate_advice_with_gpt(
                    farm_input.crop_type,
                    weather_data,
                    soil_data,
                    farm_input.location
                )
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed to generate advice after {max_retries} attempts: {str(e)}")
                    raise HTTPException(status_code=500, detail=f"Failed to generate advice: {str(e)}")
                logger.warning(f"Advice generation failed, retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
        
        # Prepare response
        response_data = {
            "advice": advice,
            "weather": weather_data.get_weather_summary(),
            "soil": {
                "ph": soil_data.ph,
                "moisture": soil_data.moisture,
                "type": soil_data.type
            },
            "crop": farm_input.crop_type
        }
        logger.info(f"Final response data: {response_data}")
        
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

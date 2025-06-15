# AgroPredict AI

A mobile-first platform that helps smallholder farmers in Africa make informed decisions about planting, fertilizing, and harvesting crops based on weather and soil data.

## Features

- Weather forecasting integration
- Soil condition analysis
- Crop-specific recommendations
- Mobile-friendly interface
- Voice support (coming soon)
- Multiple language support (coming soon)

## Tech Stack

- Backend: FastAPI
- Frontend: HTML + Tailwind CSS
- APIs: OpenWeatherMap, NASA POWER, SoilGrids
- AI: OpenAI (for future enhancements)

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file with your API keys:
```bash
cp .env.example .env
# Edit .env with your API keys
```

3. Run the server:
```bash
uvicorn main:app --reload
```

4. Open `index.html` in your browser

## Usage

1. Enter your location
2. Select your crop type
3. Click "Get Advice"
4. View weather, soil conditions, and farming recommendations

## Security

- API keys are stored in environment variables
- Input validation is implemented
- Error handling is in place

## Future Improvements

- Add voice support for low-literacy users
- Implement offline caching
- Add more crop types
- Improve weather forecasting accuracy
- Add fertilizer recommendations

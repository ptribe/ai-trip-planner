# AI Trip Planner

A fast, intelligent trip planning application powered by LangGraph, Groq, and Arize observability.

## 🚀 Performance Features

- **Groq Integration**: Uses Groq's lightning-fast inference for 10x faster responses
- **Parallel Processing**: Research, budget analysis, and local experiences run simultaneously
- **Optimized Graph**: Streamlined workflow eliminates unnecessary supervisor overhead
- **LiteLLM Instrumentation**: Comprehensive observability and prompt template tracking

## Architecture

### Frontend (React + TypeScript)
- Modern Material-UI interface
- Real-time trip planning requests
- Error handling and loading states

### Backend (FastAPI + LangGraph)
- **Parallel LangGraph Workflow**: 
  - Research Node: Destination analysis
  - Budget Node: Cost breakdown and recommendations  
  - Local Experiences Node: Authentic recommendations
  - Itinerary Node: Combines all data into day-by-day plan
- **Groq LLM**: Fast inference with `llama-3.1-70b-versatile`
- **Comprehensive Tracing**: LangChain + LiteLLM instrumentation

## Quick Start

### 1. Setup Environment

Create a `.env` file in the `backend/` directory:

```bash
# Required: Groq API Key (get from https://console.groq.com)
GROQ_API_KEY=your_groq_api_key_here

# Required: Arize observability (get from https://app.arize.com)
ARIZE_SPACE_ID=your_arize_space_id
ARIZE_API_KEY=your_arize_api_key

# Optional: For web search capabilities
TAVILY_API_KEY=your_tavily_api_key

# Optional: Fallback to OpenAI if Groq unavailable
OPENAI_API_KEY=your_openai_api_key

# LiteLLM Configuration
LITELLM_LOG=DEBUG
```

### 2. Install Dependencies

```bash
# Backend
cd backend
pip install -r requirements.txt

# Frontend  
cd ../frontend
npm install
```

### 3. Run the Application

```bash
# Start both services
./start.sh

# Or run separately:
# Backend: cd backend && python main.py
# Frontend: cd frontend && npm start
```

The application will be available at:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

## Performance Optimizations

### ⚡ Groq Integration
- **10x faster inference** compared to OpenAI
- Uses `llama-3.1-70b-versatile` model for optimal speed/quality balance
- 30-second timeout with 2000 max tokens

### 🔄 Parallel Graph Execution
- Research, budget, and local experience analysis run **simultaneously**
- Reduces total execution time from ~30-60 seconds to ~10-15 seconds
- Final itinerary creation waits for all parallel tasks to complete

### 📊 Observability
- **LangChain + LiteLLM instrumentation** for comprehensive tracing
- Prompt template tracking with proper variable separation
- Real-time performance monitoring via Arize platform

## API Endpoints

### POST `/plan-trip`
Creates a comprehensive trip plan.

**Request:**
```json
{
  "destination": "Tokyo, Japan",
  "duration": "7 days", 
  "budget": "$2000",
  "interests": "food, culture, temples",
  "travel_style": "cultural"
}
```

**Response:**
```json
{
  "result": "# 7-Day Tokyo Cultural Experience\n\n## Day 1: Arrival and Asakusa District..."
}
```

### GET `/health`
Health check endpoint.

## Development

### Graph Structure
```
START → [Research, Budget, Local] → Itinerary → END
       (parallel execution)
```

### Key Components
- `research_node()`: Destination research and weather analysis
- `budget_node()`: Cost breakdown and money-saving tips  
- `local_experiences_node()`: Authentic local recommendations
- `itinerary_node()`: Day-by-day planning with all data

### Prompt Templates
All tools use comprehensive prompt templates with proper variable tracking:
- `research-v1.0`: Destination analysis
- `budget-v1.0`: Cost breakdown  
- `local-v1.0`: Authentic experiences
- `itinerary-v1.0`: Day-by-day planning

## Troubleshooting

### Common Issues
1. **Slow responses**: Ensure you're using Groq API key, not OpenAI
2. **Empty results**: Check API key configuration in `.env`
3. **Graph errors**: Verify all dependencies are installed correctly

### Performance Monitoring
View detailed traces and performance metrics in your Arize dashboard to identify bottlenecks and optimize further.

## Tech Stack

- **Frontend**: React, TypeScript, Material-UI, Axios
- **Backend**: FastAPI, LangGraph, LangChain, Groq, LiteLLM
- **Observability**: Arize, OpenInference, OpenTelemetry
- **Infrastructure**: Docker, Docker Compose

# AI Trip Planner

Fast, sequential trip planning with FastAPI (backend), React (frontend), and LangGraph for orchestration. Optional Arize tracing is supported.

## Quickstart

1) Requirements
- Python 3.10+ (Docker optional)

2) Configure environment
- Copy `backend/.env.example` to `backend/.env`.
- Set one LLM key: `OPENAI_API_KEY=...` or `OPENROUTER_API_KEY=...`.
- Optional: `ARIZE_SPACE_ID` and `ARIZE_API_KEY` for tracing.

3) Install dependencies
```bash
cd backend
uv pip install -r requirements.txt   # faster, deterministic installs
# If uv is not installed: curl -LsSf https://astral.sh/uv/install.sh | sh
# Fallback: pip install -r requirements.txt
```

4) Run
```bash
# make sure you are back in the root directory of ai-trip-planner
cd ..
./start.sh                      # starts backend on 8000; serves minimal UI at '/'
# or
cd backend && uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

5) Open
- Frontend: http://localhost:3000
- API: http://localhost:8000
- Docs: http://localhost:8000/docs
 - Minimal UI: http://localhost:8000/

Docker (optional)
```bash
docker-compose up --build
```

## Project Structure
- `backend/`: FastAPI app (`main.py`), LangGraph agents, tracing hooks.
- `frontend/index.html`: Minimal static UI served by backend at `/`.
- `optional/airtable/`: Airtable integration (optional, not on critical path).
- `test scripts/`: `test_api.py`, `synthetic_data_gen.py` for quick checks/evals.
- Root: `start.sh`, `docker-compose.yml`, `README.md`.

## Development Commands
- Backend (dev): `uvicorn main:app --host 0.0.0.0 --port 8000 --reload`
- API smoke test: `python "test scripts"/test_api.py`
- Synthetic evals: `python "test scripts"/synthetic_data_gen.py --base-url http://localhost:8000 --count 12`

## API
- POST `/plan-trip` → returns a generated itinerary.
  Example body:
  ```json
  {"destination":"Tokyo, Japan","duration":"7 days","budget":"$2000","interests":"food, culture"}
  ```
- GET `/health` → simple status.

## Notes on Tracing (Optional)
- If `ARIZE_SPACE_ID` and `ARIZE_API_KEY` are set, OpenInference exports spans for agents/tools/LLM calls. View at https://app.arize.com.

## Optional Features

### RAG: Vector Search for Local Guides

The local agent can use vector search to retrieve curated local experiences from a database of 90+ real-world recommendations:

- **Enable**: Set `ENABLE_RAG=1` in your `.env` file
- **Requirements**: Requires `OPENAI_API_KEY` for embeddings
- **Data**: Uses curated experiences from `backend/data/local_guides.json`
- **Benefits**: Provides grounded, cited recommendations with sources
- **Learning**: Great example of production RAG patterns with fallback strategies

When disabled (default), the local agent uses LLM-generated responses.

See `RAG.md` for detailed documentation.

### Web Search: Real-Time Tool Data

Tools can call real web search APIs (Tavily or SerpAPI) for up-to-date travel information:

- **Enable**: Add `TAVILY_API_KEY` or `SERPAPI_API_KEY` to your `.env` file
- **Benefits**: Real-time data for weather, attractions, prices, customs, etc.
- **Fallback**: Without API keys, tools automatically fall back to LLM-generated responses
- **Learning**: Demonstrates graceful degradation and multi-tier fallback patterns

Recommended: Tavily (free tier: 1000 searches/month) - https://tavily.com

## Troubleshooting
- 401/empty results: verify `OPENAI_API_KEY` or `OPENROUTER_API_KEY` in `backend/.env`.
- No traces: ensure Arize credentials are set and reachable.
- Port conflicts: stop existing services on 3000/8000 or change ports.

## Deploy on Render
- This repo includes `render.yaml`. Connect your GitHub repo in Render and deploy as a Web Service.
- Render will run: `pip install -r backend/requirements.txt` and `uvicorn main:app --host 0.0.0.0 --port $PORT`.
- Set `OPENAI_API_KEY` (or `OPENROUTER_API_KEY`) and optional Arize vars in the Render dashboard.

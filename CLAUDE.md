# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Running the application
```bash
# Start backend server (from project root)
./start.sh

# Alternative: Start backend directly
cd backend && uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Install dependencies (using uv for faster, deterministic installs)
cd backend && uv pip install -r requirements.txt
# Fallback if uv not installed: pip install -r requirements.txt
```

### Testing
```bash
# Quick API smoke test
python "test scripts/test_api.py"

# Generate synthetic evaluation data
python "test scripts/synthetic_data_gen.py" --base-url http://localhost:8000 --count 12

# Test RAG functionality specifically
python "test scripts/synthetic_data_gen.py" --test-rag
```

### Development
```bash
# Backend development with auto-reload
cd backend && uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## Architecture

This is a **multi-agent system** using LangGraph for parallel agent orchestration. The system follows a specific flow:

1. **User Request** → FastAPI endpoint (`/plan-trip`)
2. **Parallel Agent Execution**: Three agents run simultaneously:
   - **Research Agent**: Gathers essential info (weather, visa, customs)
   - **Budget Agent**: Calculates costs and budget breakdowns
   - **Local Agent**: Finds local experiences (uses RAG if enabled)
3. **Synthesis**: **Itinerary Agent** combines all outputs into final plan
4. **Response**: Returns structured itinerary with tool call metadata

### Key Design Patterns

**Agent Pattern**: All agents in `backend/main.py` follow this structure:
```python
def agent_name(state: TripState) -> TripState:
    # 1. Extract request, build prompt
    # 2. Bind tools to LLM
    # 3. Instrument with tracing
    # 4. Process tool calls
    # 5. Return state update
```

**Tool Pattern**: Tools implement graceful degradation:
1. Try real API (Tavily/SerpAPI if keys available)
2. Fallback to LLM generation if API fails/unavailable

**State Management**: Uses `TripState` TypedDict with:
- `messages`: Agent conversation history
- `trip_request`: User input
- `research`, `budget`, `local`, `final`: Agent outputs
- `tool_calls`: Tracking for observability

### Graph Construction
The LangGraph workflow (`build_graph()` in main.py) creates parallel execution edges from START to three agents, then convergence edges to the itinerary agent.

## Environment Variables

Required in `backend/.env`:
- `OPENAI_API_KEY` or `OPENROUTER_API_KEY` (choose one LLM provider)

Optional features:
- `ENABLE_RAG=1`: Enable vector search for local guides
- `TAVILY_API_KEY` or `SERPAPI_API_KEY`: Enable real-time web search
- `ARIZE_SPACE_ID` + `ARIZE_API_KEY`: Enable tracing/observability

## Feature Flags

- **RAG disabled (default)**: Local agent uses LLM generation
- **RAG enabled** (`ENABLE_RAG=1`): Uses vector search over 540+ curated experiences in `backend/data/local_guides.json`
- **Web search disabled (default)**: Tools use LLM fallback
- **Web search enabled**: Tools query real APIs for current data

## Important Files

- `backend/main.py`: Core application - FastAPI app, agents, tools, LangGraph workflow
- `backend/data/local_guides.json`: Curated experiences database for RAG
- `frontend/index.html`: Single-page UI (served at `/`)
- `optional/airtable/`: Optional trace logging integration

## Common Modifications

### Adding a new agent
1. Define agent function following the pattern in main.py
2. Add node to graph: `g.add_node("new_agent", new_agent_func)`
3. Add parallel edge: `g.add_edge(START, "new_agent")`
4. Add convergence: `g.add_edge("new_agent", "itinerary_node")`
5. Update `TripState` if needed

### Adding a new tool
1. Use `@tool` decorator
2. Implement API → fallback pattern
3. Add to agent's tools list
4. Test with/without API keys

### Modifying RAG data
1. Edit `backend/data/local_guides.json`
2. Restart server (embeddings regenerate on startup)
3. Test with cities in the database

## Debugging

- Check Arize traces (if configured) for agent execution flow
- Tool calls logged with arguments and results
- Frontend at http://localhost:8000/ shows real-time progress
- API docs at http://localhost:8000/docs
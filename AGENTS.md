# Repository Guidelines

## Project Structure & Modules
- `backend/`: FastAPI + LangGraph service (`main.py`), optional Arize tracing. Env sample: `env_example.txt`.
- `frontend/`: React + TypeScript UI (`src/`, Material UI).
- `test scripts/`: API smoke tests and synthetic data tools.
- Root: `docker-compose.yml`, `start.sh`, `README.md`, sample `itinerary_results.json`.

## Build, Test, and Development
- Backend install: `cd backend && pip install -r requirements.txt`
- Backend run (dev): `uvicorn main:app --host 0.0.0.0 --port 8000 --reload`
  - Alternatively: `python backend/main.py`
- Frontend install: `cd frontend && npm install`
- Frontend run: `npm start`
- All via Docker: `docker-compose up --build`
- Convenience: `./start.sh` (starts backend on 8000 and frontend on 3000)

## Testing Guidelines
- Backend smoke tests: `python "test scripts"/test_api.py`
- Synthetic evals (tools/tone):
  `python "test scripts"/synthetic_data_gen.py --base-url http://localhost:8000 --count 12`
- Frontend tests: `cd frontend && npm test`
- No strict coverage requirement; include tests for new endpoints and core UI flows.

## Coding Style & Naming
- Python: PEP 8, 4‑space indent, `snake_case` for modules/functions, type hints where practical.
- React/TS: 2‑space indent, `PascalCase` components in `src/components`, `camelCase` variables/functions, shared types in `src/types`.
- API endpoints: `/health`, `/plan-trip`; keep responses JSON-serializable and stable.

## Commit & Pull Request Guidelines
- Conventional Commits: `feat:`, `fix:`, `docs:`, `refactor:`, `chore:`, with optional scopes, e.g. `feat(simple): …`, `fix(backend): …`.
- PRs must include: concise description, testing steps, screenshots for UI changes, and linked issues (e.g., `Closes #123`).
- Small, focused PRs preferred; update README or comments when behavior changes.

## Security & Configuration
- Copy `backend/env_example.txt` to `backend/.env`; set `OPENAI_API_KEY` or `OPENROUTER_API_KEY`. For tracing, set `ARIZE_SPACE_ID` and `ARIZE_API_KEY`.
- Local deterministic runs: `TEST_MODE=1` (backend uses a fake LLM).
- Do not commit secrets or `.env`; avoid logging sensitive data.


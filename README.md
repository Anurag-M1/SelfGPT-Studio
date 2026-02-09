# SelfGPT Studio

SelfGPT Studio is a local‑first, self‑hosted AI IDE that generates multi‑file full‑stack projects, runs them locally, and keeps everything in your own workspace. It supports multiple LLM providers, stack templates, file‑level editing, snapshots, diffs, dependency management, and export/import.

## What You Get
- **Landing page + Studio**: `/` (Get Started) → `/studio`
- **Multi‑file generation** (JSON‑based for real stacks)
- **Stacks**: Static Web, React, Next.js, Node, MERN, Python, FastAPI, Flask, Django
- **Agents**: Builder, Debugger, Refactor, UX, Performance, Security, Test, Docs
- **Run Profiles**: one‑click command suggestions in Terminal
- **Snapshots + Diff**: project snapshots and line diff view
- **Dependency manager**: edit `package.json` / `requirements.txt`
- **Terminal**: local shell with status + reconnect
- **Export/Import**: ZIP project files
- **Preview**: HTML preview with CSS/JS inlined (for static projects)

---

## Architecture
- **Frontend**: Next.js 14 (`/app`)
- **Backend**: FastAPI (`/backend/main.py`)
- **DB**: PostgreSQL (`projects`, `users`, `snapshots`)
- **Runtime Files**: stored in `data/projects/<project_id>`

---

## Requirements
- **Node.js** 18+ (20+ recommended)
- **Python** 3.12
- **PostgreSQL** 14+
- **Optional**: Docker (for containerized run profiles)

---

## Setup

### 1) Environment
Create or update `.env` in the repo root:

```
DATABASE_URL=postgresql+psycopg://<user>@localhost:5432/xaim_app
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000

# LLM Providers (add only the ones you use)
GROQ_API_KEY=...
OPENAI_API_KEY=...
MISTRAL_API_KEY=...
DEEPSEEK_API_KEY=...
GEMINI_API_KEY=...

DEFAULT_PROVIDER=groq
DEFAULT_MODEL=llama-3.3-70b-versatile
NEXT_PUBLIC_LLM_MODEL=llama-3.3-70b-versatile

# Optional: search
GOOGLE_API_KEY=...
GOOGLE_CSE_ID=...

# Runtime limits
RUN_MODE=local
RUN_CPU_SECONDS=10
RUN_MAX_MB=512
RUN_CONTAINER_CPUS=1
RUN_CONTAINER_MEMORY_MB=1024

JWT_SECRET=change-me
```

### 2) Backend
```
cd "SelfGPT Studio/backend"
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3) Frontend
```
cd "SelfGPT Studio"
npm install
```

---

## Running

**Backend**
```
cd "SelfGPT Studio/backend"
source .venv/bin/activate
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Frontend**
```
cd "SelfGPT Studio"
npm run dev
```

Open:
- **Landing**: http://localhost:3000
- **Studio**: http://localhost:3000/studio
- **Backend**: http://localhost:8000

---

## Core Workflows

### 1) Create a project
Choose a template and a stack. The stack informs multi‑file generation.

### 2) Generate with Agents
Pick an agent (Builder / Debugger / Refactor / etc.), then prompt.

### 3) Run & Debug
Use **Run Profile** or send a command directly in Terminal.

### 4) Preview
Static projects render in the preview panel. Framework/backends show a “Run it from Terminal” overlay.

### 5) Snapshot / Diff
Create snapshots and compare file diffs in the Diff panel.

### 6) Export / Import
Export a ZIP or import an existing ZIP into your project.

---

## Agents
Agents steer the LLM behavior:
- **Builder**: ships complete implementations
- **Debugger**: minimal fixes with root causes
- **Refactor**: improves structure without breaking behavior
- **UX**: UI polish & layout
- **Performance**: speed and runtime improvements
- **Security**: input validation & hardening
- **Test**: add tests and validation steps
- **Docs**: documentation

---

## Run Profiles
Profiles are shown in Terminal and are stack‑aware:
- Next.js: `npm run dev`
- React: `npm start`
- Node: `node index.js` / `node server.js`
- FastAPI: `uvicorn main:app --host 0.0.0.0 --port 8000`
- Flask: `python app.py`
- Django: `python manage.py runserver 0.0.0.0:8000`

---

## API Overview
- `GET /api/llm/options`
- `GET /api/llm/models?provider=groq|openai|mistral|deepseek|gemini`
- `GET /api/templates`
- `GET /api/run/profiles`
- `GET /api/agents`
- `POST /api/generate`
- `GET /api/projects/:id`
- `PUT /api/projects/:id`
- `POST /api/snapshots`
- `GET /api/snapshots?projectId=...`
- `GET /api/snapshots/:id`
- `POST /api/snapshots/:id/restore`
- `GET /api/projects/:id/export`
- `POST /api/projects/:id/import`

---

## Troubleshooting

### Uvicorn not found
Re‑create the venv:
```
cd backend
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Postgres connection refused
Ensure Postgres is running:
```
brew services start postgresql
createdb xaim_app
```
Update `DATABASE_URL` if needed.

### LLM not responding
Check the provider keys in `.env` and ensure the provider is enabled in the UI.

---

## Security Notes
- Keep `.env` private.
- Store API keys only on the local machine.
- JWT secret should be unique in production.

---

## License
Private / internal use (update as needed).

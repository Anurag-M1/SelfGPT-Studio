import base64
import hashlib
import hmac
import json
import os
import asyncio
import pty
import resource
import select
import subprocess
import threading
import time
import traceback
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple
from uuid import uuid4
from shutil import which

import bcrypt
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, Response
from groq import Groq
from sqlalchemy import Column, MetaData, String, Table, Text, create_engine, delete, select, text, update
from sqlalchemy.dialects.postgresql import JSONB

# --- Config ---
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

DATABASE_URL = os.getenv("DATABASE_URL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

JWT_SECRET = os.getenv("JWT_SECRET", "ai-builder-jwt-secret-2025")
DEFAULT_PROVIDER = os.getenv("DEFAULT_PROVIDER", "groq").lower()
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "llama-3.3-70b-versatile")

RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "60"))
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "24"))
MAX_MESSAGE_LEN = int(os.getenv("MAX_MESSAGE_LEN", "6000"))
RUN_CPU_SECONDS = int(os.getenv("RUN_CPU_SECONDS", "10"))
RUN_MAX_MB = int(os.getenv("RUN_MAX_MB", "512"))
RUN_MODE = os.getenv("RUN_MODE", "local").lower()
RUN_CONTAINER_CPUS = os.getenv("RUN_CONTAINER_CPUS", "1")
RUN_CONTAINER_MEMORY_MB = int(os.getenv("RUN_CONTAINER_MEMORY_MB", "1024"))

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set")

# --- FastAPI ---
app = FastAPI()

origins_env = os.getenv("CORS_ORIGINS", "*")
if origins_env.strip() == "*":
    allow_origins = ["*"]
    allow_credentials = False
else:
    allow_origins = [o.strip() for o in origins_env.split(",") if o.strip()]
    allow_credentials = True

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Groq Client ---
_groq_client: Optional[Groq] = None


def get_groq() -> Groq:
    global _groq_client
    if _groq_client is None:
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not set")
        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client


# --- Rate Limiting ---
_RATE_LIMIT_WINDOW = 60
_rate_limit_store: Dict[str, Deque[float]] = {}
_model_cache: Dict[str, Tuple[float, List[Dict[str, str]]]] = {}
_MODEL_CACHE_TTL = 300


def _client_id(request: Request) -> str:
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def enforce_rate_limit(request: Request) -> None:
    if RATE_LIMIT_PER_MIN <= 0:
        return
    now = time.time()
    client_id = _client_id(request)
    bucket = _rate_limit_store.get(client_id)
    if bucket is None:
        bucket = deque()
        _rate_limit_store[client_id] = bucket
    while bucket and now - bucket[0] > _RATE_LIMIT_WINDOW:
        bucket.popleft()
    if len(bucket) >= RATE_LIMIT_PER_MIN:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Please try again later.")
    bucket.append(now)


# --- Runtime / Process Management ---
PROJECTS_ROOT = Path(os.path.join(os.path.dirname(__file__), "..", "data", "projects"))
PROJECTS_ROOT.mkdir(parents=True, exist_ok=True)

_RUNS: Dict[str, Dict[str, Any]] = {}
_RUN_LOCK = threading.Lock()
_PROJECT_CONNECTIONS: Dict[str, List[WebSocket]] = {}
_TERMINAL_SESSIONS: Dict[str, Dict[str, Any]] = {}


def _project_dir(project_id: str) -> Path:
    path = PROJECTS_ROOT / project_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_files_to_disk(project_id: str, files: List[Dict[str, Any]], env_vars: Optional[List[Dict[str, Any]]] = None) -> Path:
    base = _project_dir(project_id)
    # Clear existing files (basic cleanup, keep root)
    for child in base.iterdir():
        if child.is_file():
            child.unlink()
        else:
            # Best-effort recursive cleanup
            for sub in child.rglob("*"):
                if sub.is_file():
                    sub.unlink()
            for sub in sorted(child.rglob("*"), reverse=True):
                if sub.is_dir():
                    sub.rmdir()
            child.rmdir()
    for f in files:
        rel = f.get("path") or ""
        if not rel or rel.startswith("..") or rel.startswith("/"):
            continue
        full_path = base / rel
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(f.get("content") or "", encoding="utf-8")
    if env_vars is not None:
        env_lines = []
        for item in env_vars:
            key = item.get("key")
            value = item.get("value")
            if key:
                env_lines.append(f"{key}={value or ''}")
        (base / ".env").write_text("\n".join(env_lines) + ("\n" if env_lines else ""), encoding="utf-8")
    return base


def _start_process(command: str, cwd: Path, env: Dict[str, str]) -> str:
    run_id = str(uuid4())
    def _limit_resources():
        try:
            if RUN_CPU_SECONDS > 0:
                resource.setrlimit(resource.RLIMIT_CPU, (RUN_CPU_SECONDS, RUN_CPU_SECONDS))
            if RUN_MAX_MB > 0:
                mem_bytes = RUN_MAX_MB * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        except Exception:
            pass

    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        shell=True,
        bufsize=1,
        preexec_fn=_limit_resources,
    )
    queue: Deque[str] = deque()
    run_info = {"process": process, "queue": queue, "done": False, "command": command}

    def reader():
        try:
            if process.stdout:
                for line in process.stdout:
                    queue.append(line)
        finally:
            code = process.wait()
            queue.append(f"\n[process exited with code {code}]\n")
            run_info["done"] = True

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()

    with _RUN_LOCK:
        _RUNS[run_id] = run_info
    return run_id


def _start_container(command: str, cwd: Path, env: Dict[str, str], runtime: str) -> str:
    run_id = str(uuid4())
    image = "node:20-alpine" if runtime == "node" else "python:3.12-slim"
    container_name = f"selfgpt-{run_id}"

    env_args: List[str] = []
    for key, value in env.items():
        env_args.extend(["-e", f"{key}={value}"])

    docker_cmd = [
        "docker", "run", "--rm",
        "--name", container_name,
        "-v", f"{cwd}:/workspace",
        "-w", "/workspace",
        "--cpus", str(RUN_CONTAINER_CPUS),
        "--memory", f"{RUN_CONTAINER_MEMORY_MB}m",
        *env_args,
        image,
        "sh", "-lc", command,
    ]

    process = subprocess.Popen(
        docker_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    queue: Deque[str] = deque()
    run_info = {
        "process": process,
        "queue": queue,
        "done": False,
        "command": command,
        "mode": "container",
        "container_name": container_name,
    }

    def reader():
        try:
            if process.stdout:
                for line in process.stdout:
                    queue.append(line)
        finally:
            code = process.wait()
            queue.append(f"\n[container exited with code {code}]\n")
            run_info["done"] = True

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()

    with _RUN_LOCK:
        _RUNS[run_id] = run_info
    return run_id

# --- Collaboration WebSocket ---
@app.websocket("/ws/projects/{project_id}")
async def project_ws(websocket: WebSocket, project_id: str):
    await websocket.accept()
    connections = _PROJECT_CONNECTIONS.setdefault(project_id, [])
    connections.append(websocket)
    try:
        while True:
            message = await websocket.receive_text()
            for conn in list(_PROJECT_CONNECTIONS.get(project_id, [])):
                if conn is websocket:
                    continue
                try:
                    await conn.send_text(message)
                except Exception:
                    try:
                        _PROJECT_CONNECTIONS[project_id].remove(conn)
                    except ValueError:
                        pass
    except WebSocketDisconnect:
        pass
    finally:
        try:
            _PROJECT_CONNECTIONS[project_id].remove(websocket)
        except ValueError:
            pass


@app.websocket("/ws/terminal/{project_id}")
async def terminal_ws(websocket: WebSocket, project_id: str):
    token = websocket.query_params.get("token")
    if token:
        auth_data = verify_token(token)
        if not auth_data:
            await websocket.close(code=1008)
            return
        # Ensure project belongs to user
        with engine.begin() as conn:
            project = conn.execute(
                select(projects_table).where(
                    (projects_table.c.id == project_id) & (projects_table.c.user_id == auth_data["userId"])
                )
            ).mappings().first()
        if not project:
            await websocket.close(code=1008)
            return
        # Sync files to disk for terminal context
        files = project.get("files") or []
        env_vars = project.get("env_vars") or []
        _write_files_to_disk(project_id, files, env_vars)
    else:
        # Allow local dev without token
        env_vars = []

    await websocket.accept()
    loop = asyncio.get_event_loop()

    pid, fd = pty.fork()
    if pid == 0:
        try:
            os.chdir(str(_project_dir(project_id)))
        except Exception:
            pass
        env = os.environ.copy()
        for item in env_vars:
            key = item.get("key")
            value = item.get("value")
            if key:
                env[key] = value or ""
        shell = os.environ.get("SHELL", "/bin/zsh")
        os.execvpe(shell, [shell], env)

    def reader():
        try:
            while True:
                r, _, _ = select.select([fd], [], [], 0.1)
                if fd in r:
                    data = os.read(fd, 1024)
                    if not data:
                        break
                    asyncio.run_coroutine_threadsafe(
                        websocket.send_text(data.decode(errors="ignore")),
                        loop,
                    )
        except Exception:
            pass

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()

    try:
        while True:
            msg = await websocket.receive_text()
            if msg:
                os.write(fd, msg.encode())
    except WebSocketDisconnect:
        pass
    finally:
        try:
            os.close(fd)
        except Exception:
            pass


# --- LLM Providers ---
def _openai_compat_chat(
    *,
    api_base: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
) -> str:
    url = f"{api_base}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    response = httpx.post(url, headers=headers, json=payload, timeout=60)
    if response.status_code >= 400:
        raise HTTPException(status_code=500, detail=f"LLM error: {response.text}")
    data = response.json()
    content = data.get("choices", [{}])[0].get("message", {}).get("content")
    if not content:
        raise HTTPException(status_code=500, detail="LLM returned empty response")
    return content


def _openai_compat_list_models(*, api_base: str, api_key: str) -> List[Dict[str, str]]:
    url = f"{api_base}/models"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    response = httpx.get(url, headers=headers, timeout=30)
    if response.status_code >= 400:
        raise HTTPException(status_code=500, detail=f"LLM models error: {response.text}")
    data = response.json()
    models: List[Dict[str, str]] = []
    if isinstance(data, dict) and isinstance(data.get("data"), list):
        for m in data["data"]:
            mid = m.get("id")
            if mid:
                models.append({"id": mid, "label": mid})
    elif isinstance(data, list):
        for m in data:
            mid = m.get("id")
            if mid:
                models.append({"id": mid, "label": mid})
    return models


def _gemini_generate(
    *,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
) -> str:
    if not model:
        model = "gemini-1.5-pro"
    model_path = model if model.startswith("models/") else f"models/{model}"
    url = f"https://generativelanguage.googleapis.com/v1beta/{model_path}:generateContent?key={api_key}"
    combined = "\n\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages])
    payload = {
        "contents": [{"role": "user", "parts": [{"text": combined}]}],
        "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
    }
    response = httpx.post(url, json=payload, timeout=60)
    if response.status_code >= 400:
        raise HTTPException(status_code=500, detail=f"LLM error: {response.text}")
    data = response.json()
    candidates = data.get("candidates") or []
    if not candidates:
        raise HTTPException(status_code=500, detail="LLM returned empty response")
    parts = candidates[0].get("content", {}).get("parts") or []
    if not parts or "text" not in parts[0]:
        raise HTTPException(status_code=500, detail="LLM returned empty response")
    return parts[0]["text"]


def _gemini_list_models(*, api_key: str) -> List[Dict[str, str]]:
    url = "https://generativelanguage.googleapis.com/v1beta/models"
    models: List[Dict[str, str]] = []
    page_token = None
    while True:
        params = {"pageSize": 1000, "key": api_key}
        if page_token:
            params["pageToken"] = page_token
        response = httpx.get(url, params=params, timeout=30)
        if response.status_code >= 400:
            raise HTTPException(status_code=500, detail=f"LLM models error: {response.text}")
        data = response.json()
        for m in data.get("models", []):
            name = m.get("name")
            if not name:
                continue
            # Prefer models that support generateContent
            methods = m.get("supportedGenerationMethods") or []
            if methods and "generateContent" not in methods:
                continue
            label = m.get("displayName") or name
            model_id = name.replace("models/", "", 1) if name.startswith("models/") else name
            models.append({"id": model_id, "label": label})
        page_token = data.get("nextPageToken")
        if not page_token:
            break
    return models


def _enabled_providers() -> Dict[str, bool]:
    return {
        "groq": bool(GROQ_API_KEY),
        "openai": bool(OPENAI_API_KEY),
        "mistral": bool(MISTRAL_API_KEY),
        "deepseek": bool(DEEPSEEK_API_KEY),
        "gemini": bool(GEMINI_API_KEY),
    }


def list_models(provider: str) -> List[Dict[str, str]]:
    provider = provider.lower()
    enabled = _enabled_providers()
    if provider not in enabled:
        raise HTTPException(status_code=400, detail="Unsupported provider")
    if not enabled[provider]:
        raise HTTPException(status_code=400, detail=f"Provider '{provider}' is not configured")

    cached = _model_cache.get(provider)
    now = time.time()
    if cached and now - cached[0] < _MODEL_CACHE_TTL:
        return cached[1]

    if provider == "groq":
        try:
            models = _openai_compat_list_models(
                api_base="https://api.groq.com/openai/v1",
                api_key=GROQ_API_KEY,
            )
        except HTTPException:
            models = [{"id": DEFAULT_MODEL, "label": DEFAULT_MODEL}]
    elif provider == "openai":
        models = _openai_compat_list_models(
            api_base="https://api.openai.com/v1",
            api_key=OPENAI_API_KEY,
        )
    elif provider == "mistral":
        models = _openai_compat_list_models(
            api_base="https://api.mistral.ai/v1",
            api_key=MISTRAL_API_KEY,
        )
    elif provider == "deepseek":
        try:
            models = _openai_compat_list_models(
                api_base="https://api.deepseek.com/v1",
                api_key=DEEPSEEK_API_KEY,
            )
        except HTTPException:
            # Fallback to known public model IDs if models endpoint is unavailable
            models = [
                {"id": "deepseek-chat", "label": "deepseek-chat"},
                {"id": "deepseek-reasoner", "label": "deepseek-reasoner"},
            ]
    elif provider == "gemini":
        models = _gemini_list_models(api_key=GEMINI_API_KEY)
    else:
        raise HTTPException(status_code=400, detail="Unsupported provider")

    _model_cache[provider] = (now, models)
    return models


def call_llm(
    *,
    provider: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
) -> Tuple[str, str]:
    provider = provider.lower()
    enabled = _enabled_providers()
    if provider not in enabled:
        raise HTTPException(status_code=400, detail="Unsupported provider")
    if not enabled[provider]:
        raise HTTPException(status_code=400, detail=f"Provider '{provider}' is not configured")

    if provider == "groq":
        groq = get_groq()
        completion = groq.chat.completions.create(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = completion.choices[0].message.content or ""
        if not content:
            raise HTTPException(status_code=500, detail="LLM returned empty response")
        return content, model

    if provider == "openai":
        return (
            _openai_compat_chat(
                api_base="https://api.openai.com/v1",
                api_key=OPENAI_API_KEY,
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            ),
            model,
        )

    if provider == "mistral":
        return (
            _openai_compat_chat(
                api_base="https://api.mistral.ai/v1",
                api_key=MISTRAL_API_KEY,
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            ),
            model,
        )

    if provider == "deepseek":
        return (
            _openai_compat_chat(
                api_base="https://api.deepseek.com/v1",
                api_key=DEEPSEEK_API_KEY,
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            ),
            model,
        )

    if provider == "gemini":
        return (
            _gemini_generate(
                api_key=GEMINI_API_KEY,
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            ),
            model,
        )

    raise HTTPException(status_code=400, detail="Unsupported provider")


# --- Run / Terminal API ---
@app.post("/api/run/start")
async def run_start(request: Request):
    auth_data = get_user_from_request(request)
    if not auth_data:
        raise HTTPException(status_code=401, detail="Unauthorized")

    body = await request.json()
    project_id = body.get("projectId")
    runtime = (body.get("runtime") or "node").lower()
    mode = (body.get("mode") or RUN_MODE).lower()
    command = body.get("command")

    if not project_id:
        raise HTTPException(status_code=400, detail="projectId is required")

    with engine.begin() as conn:
        project = conn.execute(
            select(projects_table).where(
                (projects_table.c.id == project_id) & (projects_table.c.user_id == auth_data["userId"])
            )
        ).mappings().first()

    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    files = project.get("files") or []
    env_vars = project.get("env_vars") or []

    if not command:
        file_paths = {f.get("path") for f in files}
        if runtime == "python":
            if "manage.py" in file_paths:
                command = "python manage.py runserver 0.0.0.0:8000"
            elif "main.py" in file_paths and any("fastapi" in (f.get("content") or "").lower() for f in files if f.get("path") == "requirements.txt"):
                command = "uvicorn main:app --host 0.0.0.0 --port 8000"
            elif "app.py" in file_paths:
                command = "python app.py"
            else:
                command = "python main.py"
        else:
            if "package.json" in file_paths:
                command = "npm run dev"
            elif "server.js" in file_paths:
                command = "node server.js"
            else:
                command = "node index.js"

    if runtime == "python":
        file_paths = {f.get("path") for f in files}
        if not ({"main.py", "app.py", "manage.py"} & file_paths):
            raise HTTPException(status_code=400, detail="No runnable Python entry found (main.py/app.py/manage.py)")
    if runtime == "node" and not any(f.get("path") == "index.js" for f in files) and not any(f.get("path") == "package.json" for f in files):
        # allow generic commands if user provided, otherwise warn
        if "node" in command:
            raise HTTPException(status_code=400, detail="index.js not found for node runtime")

    cwd = _write_files_to_disk(project_id, files, env_vars)
    env = os.environ.copy()
    for item in env_vars:
        key = item.get("key")
        value = item.get("value")
        if key:
            env[key] = value or ""

    if mode == "container":
        if which("docker") is None:
            raise HTTPException(status_code=400, detail="Docker is not installed or not in PATH")
        run_id = _start_container(command, cwd, env, runtime)
    else:
        run_id = _start_process(command, cwd, env)
    return {"runId": run_id, "command": command, "mode": mode}


@app.get("/api/run/stream")
async def run_stream(run_id: str):
    def event_stream():
        while True:
            with _RUN_LOCK:
                run = _RUNS.get(run_id)
            if not run:
                yield "event: done\ndata: {}\n\n"
                break
            queue = run["queue"]
            while queue:
                line = queue.popleft()
                payload = json.dumps({"line": line})
                yield f"data: {payload}\n\n"
            if run.get("done"):
                yield "event: done\ndata: {}\n\n"
                with _RUN_LOCK:
                    _RUNS.pop(run_id, None)
                break
            time.sleep(0.2)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/api/run/stop")
async def run_stop(request: Request):
    auth_data = get_user_from_request(request)
    if not auth_data:
        raise HTTPException(status_code=401, detail="Unauthorized")

    body = await request.json()
    run_id = body.get("runId")
    if not run_id:
        raise HTTPException(status_code=400, detail="runId is required")

    with _RUN_LOCK:
        run = _RUNS.get(run_id)
    if not run:
        return {"success": True}

    process = run.get("process")
    if process and process.poll() is None:
        if run.get("mode") == "container":
            container_name = run.get("container_name")
            if container_name:
                subprocess.run(["docker", "stop", container_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            process.terminate()
    with _RUN_LOCK:
        _RUNS.pop(run_id, None)
    return {"success": True}

# --- Postgres ---
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
metadata = MetaData()

users_table = Table(
    "users",
    metadata,
    Column("id", String, primary_key=True),
    Column("email", String, unique=True, index=True, nullable=False),
    Column("password", String, nullable=False),
    Column("name", String, nullable=False),
    Column("created_at", String, nullable=False),
)

projects_table = Table(
    "projects",
    metadata,
    Column("id", String, primary_key=True),
    Column("user_id", String, index=True, nullable=False),
    Column("name", String, nullable=False),
    Column("description", Text, nullable=False),
    Column("code", Text, nullable=False),
    Column("files", JSONB, nullable=False),
    Column("env_vars", JSONB, nullable=False),
    Column("settings", JSONB, nullable=False),
    Column("messages", JSONB, nullable=False),
    Column("created_at", String, nullable=False),
    Column("updated_at", String, nullable=False),
)

snapshots_table = Table(
    "snapshots",
    metadata,
    Column("id", String, primary_key=True),
    Column("project_id", String, index=True, nullable=False),
    Column("name", String, nullable=False),
    Column("files", JSONB, nullable=False),
    Column("env_vars", JSONB, nullable=False),
    Column("created_at", String, nullable=False),
)


@app.on_event("startup")
def on_startup():
    metadata.create_all(engine)
    # Ensure new columns exist on existing tables
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE projects ADD COLUMN IF NOT EXISTS files JSONB NOT NULL DEFAULT '[]'::jsonb"))
        conn.execute(text("ALTER TABLE projects ADD COLUMN IF NOT EXISTS env_vars JSONB NOT NULL DEFAULT '[]'::jsonb"))
        conn.execute(text("ALTER TABLE projects ADD COLUMN IF NOT EXISTS settings JSONB NOT NULL DEFAULT '{}'::jsonb"))


# --- Helpers ---

def utcnow_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())


def default_index_html() -> str:
    return (
        "<!DOCTYPE html>\n"
        "<html lang=\"en\">\n"
        "<head>\n"
        "  <meta charset=\"UTF-8\">\n"
        "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n"
        "  <title>My Website</title>\n"
        "  <style>\n"
        "    body { font-family: system-ui, sans-serif; display: flex; align-items: center; justify-content: center; min-height: 100vh; margin: 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }\n"
        "    h1 { font-size: 3rem; text-align: center; }\n"
        "  </style>\n"
        "</head>\n"
        "<body>\n"
        "  <h1>Start building with AI!</h1>\n"
        "</body>\n"
        "</html>"
    )


def extract_index_html(files: List[Dict[str, Any]]) -> str:
    for f in files:
        if f.get("path") == "index.html":
            return f.get("content") or ""
    return ""

def _prompt_suggests_multifile(prompt: str) -> bool:
    if not prompt:
        return False
    lower = prompt.lower()
    keywords = [
        "next.js", "nextjs", "react", "node", "express", "mern",
        "django", "fastapi", "flask", "python", "typescript", "tsx",
        "package.json", "requirements.txt",
    ]
    return any(k in lower for k in keywords)

def _detect_language_from_path(path: str) -> str:
    if not path:
        return "text"
    ext = path.split(".")[-1].lower()
    if ext in {"js", "jsx"}:
        return "javascript"
    if ext in {"ts", "tsx"}:
        return "typescript"
    if ext == "css":
        return "css"
    if ext in {"html", "htm"}:
        return "html"
    if ext == "json":
        return "json"
    if ext == "py":
        return "python"
    if ext == "md":
        return "markdown"
    return "text"


def _should_use_json_files(files: List[Dict[str, Any]]) -> bool:
    if not files:
        return False
    allowed = {"index.html", "styles.css", "script.js"}
    for f in files:
        if f.get("path") not in allowed:
            return True
    return False


def _parse_files_from_response(text: str) -> Optional[List[Dict[str, Any]]]:
    try:
        data = json.loads(text)
    except Exception:
        try:
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                return None
            data = json.loads(text[start:end + 1])
        except Exception:
            return None
    if isinstance(data, list):
        files = data
    elif isinstance(data, dict):
        files = data.get("files")
    else:
        return None
    if not isinstance(files, list):
        return None
    parsed: List[Dict[str, Any]] = []
    for f in files:
        if not isinstance(f, dict):
            continue
        path = f.get("path")
        content = f.get("content")
        if not path or content is None:
            continue
        parsed.append({
            "path": path,
            "content": content,
            "language": f.get("language") or _detect_language_from_path(path),
        })
    return parsed if parsed else None


def split_html_assets(html: str) -> Tuple[str, Optional[str], Optional[str]]:
    import re
    style_blocks = re.findall(r"<style[^>]*>(.*?)</style>", html, flags=re.DOTALL | re.IGNORECASE)
    script_blocks = re.findall(r"<script(?![^>]*src)[^>]*>(.*?)</script>", html, flags=re.DOTALL | re.IGNORECASE)

    css = "\n\n".join([s.strip() for s in style_blocks if s.strip()]) if style_blocks else None
    js = "\n\n".join([s.strip() for s in script_blocks if s.strip()]) if script_blocks else None

    if not css and not js:
        return html, None, None

    # Remove inline style/script blocks
    html_no_assets = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)
    html_no_assets = re.sub(r"<script(?![^>]*src)[^>]*>.*?</script>", "", html_no_assets, flags=re.DOTALL | re.IGNORECASE)

    # Insert link/script tags
    if css:
        if "</head>" in html_no_assets:
            html_no_assets = html_no_assets.replace("</head>", "  <link rel=\"stylesheet\" href=\"styles.css\" />\n</head>")
        else:
            html_no_assets = "  <link rel=\"stylesheet\" href=\"styles.css\" />\n" + html_no_assets
    if js:
        if "</body>" in html_no_assets:
            html_no_assets = html_no_assets.replace("</body>", "  <script src=\"script.js\"></script>\n</body>")
        else:
            html_no_assets = html_no_assets + "\n<script src=\"script.js\"></script>"

    return html_no_assets, css, js


RUN_PROFILES: List[Dict[str, Any]] = [
    {"id": "nextjs", "label": "Next.js Dev", "runtime": "node", "command": "npm run dev", "stack": "nextjs"},
    {"id": "react", "label": "React Dev", "runtime": "node", "command": "npm start", "stack": "react"},
    {"id": "node", "label": "Node App", "runtime": "node", "command": "node index.js", "stack": "node"},
    {"id": "express", "label": "Express Server", "runtime": "node", "command": "node server.js", "stack": "node"},
    {"id": "mern", "label": "MERN Dev", "runtime": "node", "command": "npm run dev", "stack": "mern"},
    {"id": "fastapi", "label": "FastAPI", "runtime": "python", "command": "uvicorn main:app --host 0.0.0.0 --port 8000", "stack": "fastapi"},
    {"id": "flask", "label": "Flask", "runtime": "python", "command": "python app.py", "stack": "flask"},
    {"id": "django", "label": "Django", "runtime": "python", "command": "python manage.py runserver 0.0.0.0:8000", "stack": "django"},
]

AGENTS: List[Dict[str, Any]] = [
    {"id": "builder", "label": "Builder", "description": "Scaffold features and ship complete solutions."},
    {"id": "debugger", "label": "Debugger", "description": "Find root causes and propose minimal fixes."},
    {"id": "refactor", "label": "Refactor", "description": "Improve structure, readability, and architecture."},
    {"id": "ux", "label": "UX Designer", "description": "Polish UI, layout, and interactions."},
    {"id": "performance", "label": "Performance", "description": "Speed, memory, and runtime improvements."},
    {"id": "security", "label": "Security", "description": "Harden auth, input validation, and secrets."},
    {"id": "tester", "label": "Test Engineer", "description": "Add tests and reproducible validations."},
    {"id": "docs", "label": "Docs", "description": "Write clear, developer-facing docs."},
]

AGENT_PROMPTS: Dict[str, str] = {
    "builder": "You are the Builder agent. Deliver complete, working implementations with minimal back-and-forth.",
    "debugger": "You are the Debugger agent. Identify root causes, show fixes, and keep changes minimal.",
    "refactor": "You are the Refactor agent. Improve structure and maintainability without changing behavior.",
    "ux": "You are the UX agent. Focus on layout, polish, and interaction details.",
    "performance": "You are the Performance agent. Optimize for speed, memory, and responsiveness.",
    "security": "You are the Security agent. Harden auth, validate inputs, and reduce attack surface.",
    "tester": "You are the Test agent. Add tests and verification steps.",
    "docs": "You are the Docs agent. Produce clear, concise developer documentation.",
}

def build_preview_html(files: List[Dict[str, Any]], fallback_html: Optional[str] = None) -> str:
    import re

    html = extract_index_html(files) or fallback_html or ""
    if not html:
        return (
            "<!DOCTYPE html><html><head><meta charset=\"UTF-8\" />"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />"
            "<title>Preview</title></head><body>"
            "<h2 style=\"font-family:system-ui;\">No preview available</h2>"
            "<p style=\"font-family:system-ui;\">This project does not include an index.html file.</p>"
            "</body></html>"
        )

    file_map: Dict[str, str] = {}
    for f in files:
        path = f.get("path")
        if path:
            file_map[path] = f.get("content") or ""

    def normalize_path(raw: str) -> str:
        cleaned = (raw or "").split("?")[0].split("#")[0]
        cleaned = cleaned.lstrip("./").lstrip("/")
        return cleaned

    css_inlined = False
    js_inlined = False

    def replace_link(match: re.Match) -> str:
        nonlocal css_inlined
        href = match.group(1) or ""
        path = normalize_path(href)
        if path in file_map:
            css_inlined = True
            return f"<style>{file_map[path]}</style>"
        return match.group(0)

    def replace_script(match: re.Match) -> str:
        nonlocal js_inlined
        src = match.group(1) or ""
        path = normalize_path(src)
        if path in file_map:
            js_inlined = True
            return f"<script>{file_map[path]}</script>"
        return match.group(0)

    html = re.sub(r"<link[^>]+href=[\"']([^\"']+)[\"'][^>]*>", replace_link, html, flags=re.IGNORECASE)
    html = re.sub(r"<script[^>]+src=[\"']([^\"']+)[\"'][^>]*>\\s*</script>", replace_script, html, flags=re.IGNORECASE)

    if not css_inlined and "styles.css" in file_map:
        style_tag = f"<style>{file_map['styles.css']}</style>"
        if "</head>" in html:
            html = html.replace("</head>", f"{style_tag}\n</head>")
        else:
            html = f"{style_tag}\n{html}"

    if not js_inlined and "script.js" in file_map:
        script_tag = f"<script>{file_map['script.js']}</script>"
        if "</body>" in html:
            html = html.replace("</body>", f"{script_tag}\n</body>")
        else:
            html = f"{html}\n{script_tag}"

    return html


TEMPLATES: List[Dict[str, Any]] = [
    {
        "id": "blank",
        "name": "Blank",
        "description": "Minimal starter with index.html",
        "runtime": "web",
        "files": [
            {"path": "index.html", "content": default_index_html(), "language": "html"},
        ],
    },
    {
        "id": "node-http",
        "name": "Node HTTP Server",
        "description": "Simple Node server with static HTML",
        "runtime": "node",
        "files": [
            {
                "path": "index.html",
                "content": "<!DOCTYPE html>\\n<html><head><meta charset=\\\"UTF-8\\\"><meta name=\\\"viewport\\\" content=\\\"width=device-width, initial-scale=1.0\\\"><title>Node App</title></head><body><h1>Node HTTP Server</h1><p>Edit index.html to change this page.</p></body></html>",
                "language": "html",
            },
            {
                "path": "index.js",
                "content": "const http = require('http');\\nconst fs = require('fs');\\nconst path = require('path');\\n\\nconst port = process.env.PORT || 3001;\\n\\nconst server = http.createServer((req, res) => {\\n  const filePath = path.join(__dirname, 'index.html');\\n  fs.readFile(filePath, (err, data) => {\\n    if (err) {\\n      res.writeHead(500);\\n      res.end('Server error');\\n      return;\\n    }\\n    res.writeHead(200, { 'Content-Type': 'text/html' });\\n    res.end(data);\\n  });\\n});\\n\\nserver.listen(port, () => {\\n  console.log(`Server running on http://localhost:${port}`);\\n});\\n",
                "language": "javascript",
            },
            {
                "path": "package.json",
                "content": "{\\n  \\\"name\\\": \\\"node-http\\\",\\n  \\\"version\\\": \\\"1.0.0\\\",\\n  \\\"main\\\": \\\"index.js\\\",\\n  \\\"scripts\\\": {\\n    \\\"start\\\": \\\"node index.js\\\"\\n  }\\n}\\n",
                "language": "json",
            },
        ],
    },
    {
        "id": "node-express",
        "name": "Node + Express",
        "description": "Express server with a basic API",
        "runtime": "node",
        "files": [
            {
                "path": "server.js",
                "content": "const express = require('express');\\nconst app = express();\\nconst port = process.env.PORT || 3001;\\n\\napp.get('/', (req, res) => res.send('Hello from Express!'));\\napp.get('/api/health', (req, res) => res.json({ status: 'ok' }));\\n\\napp.listen(port, () => console.log(`Server running on http://localhost:${port}`));\\n",
                "language": "javascript",
            },
            {
                "path": "package.json",
                "content": "{\\n  \\\"name\\\": \\\"node-express\\\",\\n  \\\"version\\\": \\\"1.0.0\\\",\\n  \\\"main\\\": \\\"server.js\\\",\\n  \\\"scripts\\\": {\\n    \\\"start\\\": \\\"node server.js\\\"\\n  },\\n  \\\"dependencies\\\": {\\n    \\\"express\\\": \\\"^4.19.2\\\"\\n  }\\n}\\n",
                "language": "json",
            },
        ],
    },
    {
        "id": "react-cdn",
        "name": "React (CDN)",
        "description": "React app without build tools",
        "runtime": "web",
        "files": [
            {
                "path": "index.html",
                "content": "<!DOCTYPE html>\\n<html lang=\\\"en\\\">\\n<head>\\n  <meta charset=\\\"UTF-8\\\">\\n  <meta name=\\\"viewport\\\" content=\\\"width=device-width, initial-scale=1.0\\\">\\n  <title>React CDN App</title>\\n  <script crossorigin src=\\\"https://unpkg.com/react@18/umd/react.development.js\\\"></script>\\n  <script crossorigin src=\\\"https://unpkg.com/react-dom@18/umd/react-dom.development.js\\\"></script>\\n  <script src=\\\"https://unpkg.com/@babel/standalone/babel.min.js\\\"></script>\\n  <link rel=\\\"stylesheet\\\" href=\\\"styles.css\\\" />\\n</head>\\n<body>\\n  <div id=\\\"root\\\"></div>\\n  <script type=\\\"text/babel\\\" src=\\\"script.js\\\"></script>\\n</body>\\n</html>\\n",
                "language": "html",
            },
            {
                "path": "styles.css",
                "content": "body { font-family: system-ui, sans-serif; padding: 24px; background: #0f172a; color: #e2e8f0; }",
                "language": "css",
            },
            {
                "path": "script.js",
                "content": "const App = () => (\\n  <div>\\n    <h1>React CDN Starter</h1>\\n    <p>Edit script.js to build your UI.</p>\\n  </div>\\n);\\n\\nconst root = ReactDOM.createRoot(document.getElementById('root'));\\nroot.render(<App />);\\n",
                "language": "javascript",
            },
        ],
    },
    {
        "id": "nextjs-basic",
        "name": "Next.js (App Router)",
        "description": "Next.js starter with app directory",
        "runtime": "node",
        "files": [
            {
                "path": "package.json",
                "content": "{\\n  \\\"name\\\": \\\"nextjs-basic\\\",\\n  \\\"version\\\": \\\"1.0.0\\\",\\n  \\\"private\\\": true,\\n  \\\"scripts\\\": {\\n    \\\"dev\\\": \\\"next dev\\\",\\n    \\\"build\\\": \\\"next build\\\",\\n    \\\"start\\\": \\\"next start\\\"\\n  },\\n  \\\"dependencies\\\": {\\n    \\\"next\\\": \\\"14.2.3\\\",\\n    \\\"react\\\": \\\"^18\\\",\\n    \\\"react-dom\\\": \\\"^18\\\"\\n  }\\n}\\n",
                "language": "json",
            },
            {
                "path": "app/layout.js",
                "content": "export const metadata = { title: 'Next App', description: 'Next.js starter' };\\n\\nexport default function RootLayout({ children }) {\\n  return (\\n    <html lang=\\\"en\\\">\\n      <body style={{ fontFamily: 'system-ui, sans-serif', margin: 0 }}>{children}</body>\\n    </html>\\n  );\\n}\\n",
                "language": "javascript",
            },
            {
                "path": "app/page.js",
                "content": "export default function Home() {\\n  return (\\n    <main style={{ padding: 32 }}><h1>Next.js Starter</h1><p>Edit app/page.js to build.</p></main>\\n  );\\n}\\n",
                "language": "javascript",
            },
        ],
    },
    {
        "id": "mern-basic",
        "name": "MERN Basic",
        "description": "Express API + React CDN client",
        "runtime": "node",
        "files": [
            {
                "path": "server.js",
                "content": "const express = require('express');\\nconst { MongoClient } = require('mongodb');\\nconst app = express();\\nconst port = process.env.PORT || 3001;\\n\\napp.use(express.json());\\n\\napp.get('/api/health', (req, res) => res.json({ status: 'ok' }));\\n\\napp.listen(port, () => console.log(`API running on http://localhost:${port}`));\\n\\n// Update MONGO_URL in .env for DB connectivity.\\n",
                "language": "javascript",
            },
            {
                "path": "client/index.html",
                "content": "<!DOCTYPE html>\\n<html lang=\\\"en\\\">\\n<head>\\n  <meta charset=\\\"UTF-8\\\">\\n  <meta name=\\\"viewport\\\" content=\\\"width=device-width, initial-scale=1.0\\\">\\n  <title>MERN Client</title>\\n  <script crossorigin src=\\\"https://unpkg.com/react@18/umd/react.development.js\\\"></script>\\n  <script crossorigin src=\\\"https://unpkg.com/react-dom@18/umd/react-dom.development.js\\\"></script>\\n  <script src=\\\"https://unpkg.com/@babel/standalone/babel.min.js\\\"></script>\\n</head>\\n<body>\\n  <div id=\\\"root\\\"></div>\\n  <script type=\\\"text/babel\\\">\\n    const App = () => <div><h1>MERN Starter</h1><p>Edit client/index.html</p></div>;\\n    ReactDOM.createRoot(document.getElementById('root')).render(<App />);\\n  </script>\\n</body>\\n</html>\\n",
                "language": "html",
            },
            {
                "path": "package.json",
                "content": "{\\n  \\\"name\\\": \\\"mern-basic\\\",\\n  \\\"version\\\": \\\"1.0.0\\\",\\n  \\\"private\\\": true,\\n  \\\"scripts\\\": {\\n    \\\"start\\\": \\\"node server.js\\\"\\n  },\\n  \\\"dependencies\\\": {\\n    \\\"express\\\": \\\"^4.19.2\\\",\\n    \\\"mongodb\\\": \\\"^6.6.0\\\"\\n  }\\n}\\n",
                "language": "json",
            },
        ],
    },
    {
        "id": "python-http",
        "name": "Python HTTP Server",
        "description": "Simple Python server",
        "runtime": "python",
        "files": [
            {
                "path": "main.py",
                "content": "import http.server\\nimport socketserver\\n\\nPORT = 3001\\n\\nclass Handler(http.server.SimpleHTTPRequestHandler):\\n    pass\\n\\nif __name__ == '__main__':\\n    with socketserver.TCPServer(('', PORT), Handler) as httpd:\\n        print(f'Serving at http://localhost:{PORT}')\\n        httpd.serve_forever()\\n",
                "language": "python",
            },
            {
                "path": "index.html",
                "content": "<!DOCTYPE html>\\n<html><head><meta charset=\\\"UTF-8\\\"><meta name=\\\"viewport\\\" content=\\\"width=device-width, initial-scale=1.0\\\"><title>Python App</title></head><body><h1>Python HTTP Server</h1><p>Edit index.html to change this page.</p></body></html>",
                "language": "html",
            },
        ],
    },
    {
        "id": "fastapi",
        "name": "FastAPI",
        "description": "Python FastAPI starter",
        "runtime": "python",
        "files": [
            {
                "path": "main.py",
                "content": "from fastapi import FastAPI\\n\\napp = FastAPI()\\n\\n@app.get('/')\\nasync def root():\\n    return { 'status': 'ok' }\\n",
                "language": "python",
            },
            {
                "path": "requirements.txt",
                "content": "fastapi\\nuvicorn\\n",
                "language": "text",
            },
        ],
    },
    {
        "id": "flask",
        "name": "Flask",
        "description": "Python Flask starter",
        "runtime": "python",
        "files": [
            {
                "path": "app.py",
                "content": "from flask import Flask\\n\\napp = Flask(__name__)\\n\\n@app.route('/')\\ndef index():\\n    return 'Hello from Flask!'\\n\\nif __name__ == '__main__':\\n    app.run(host='0.0.0.0', port=8000, debug=True)\\n",
                "language": "python",
            },
            {
                "path": "requirements.txt",
                "content": "flask\\n",
                "language": "text",
            },
        ],
    },
    {
        "id": "django",
        "name": "Django",
        "description": "Python Django starter",
        "runtime": "python",
        "files": [
            {
                "path": "manage.py",
                "content": "import os\\nimport sys\\n\\nif __name__ == '__main__':\\n    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mysite.settings')\\n    from django.core.management import execute_from_command_line\\n    execute_from_command_line(sys.argv)\\n",
                "language": "python",
            },
            {
                "path": "mysite/__init__.py",
                "content": "",
                "language": "python",
            },
            {
                "path": "mysite/settings.py",
                "content": "from pathlib import Path\\nBASE_DIR = Path(__file__).resolve().parent.parent\\nSECRET_KEY = 'dev-key'\\nDEBUG = True\\nALLOWED_HOSTS = ['*']\\nINSTALLED_APPS = ['django.contrib.contenttypes','django.contrib.staticfiles']\\nMIDDLEWARE = []\\nROOT_URLCONF = 'mysite.urls'\\nTEMPLATES = []\\nWSGI_APPLICATION = 'mysite.wsgi.application'\\nDATABASES = { 'default': { 'ENGINE': 'django.db.backends.sqlite3', 'NAME': BASE_DIR / 'db.sqlite3' } }\\nSTATIC_URL = 'static/'\\n",
                "language": "python",
            },
            {
                "path": "mysite/urls.py",
                "content": "from django.urls import path\\nfrom django.http import HttpResponse\\n\\nurlpatterns = [\\n    path('', lambda request: HttpResponse('Hello from Django!')),\\n]\\n",
                "language": "python",
            },
            {
                "path": "mysite/wsgi.py",
                "content": "import os\\nfrom django.core.wsgi import get_wsgi_application\\n\\nos.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mysite.settings')\\napplication = get_wsgi_application()\\n",
                "language": "python",
            },
            {
                "path": "requirements.txt",
                "content": "django\\n",
                "language": "text",
            },
        ],
    },
]


def get_template(template_id: str) -> Optional[Dict[str, Any]]:
    for t in TEMPLATES:
        if t["id"] == template_id:
            return t
    return None


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")


def _b64url_decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def create_token(user_id: str) -> str:
    payload = {"userId": user_id, "exp": int(time.time() * 1000) + 7 * 24 * 60 * 60 * 1000}
    payload_str = json.dumps(payload, separators=(",", ":"))
    signature = hmac.new(JWT_SECRET.encode("utf-8"), payload_str.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{_b64url_encode(payload_str.encode('utf-8'))}.{signature}"


def verify_token(token_str: str) -> Optional[Dict[str, Any]]:
    try:
        parts = token_str.split(".")
        if len(parts) != 2:
            return None
        payload_b64, signature = parts
        payload_bytes = _b64url_decode(payload_b64)
        payload_str = payload_bytes.decode("utf-8")
        expected_sig = hmac.new(JWT_SECRET.encode("utf-8"), payload_str.encode("utf-8"), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected_sig):
            return None
        data = json.loads(payload_str)
        if data.get("exp", 0) < int(time.time() * 1000):
            return None
        return data
    except Exception:
        return None


def get_user_from_request(request: Request) -> Optional[Dict[str, Any]]:
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return None
    token = auth_header[7:]
    return verify_token(token)


# --- Auth Handlers ---
@app.post("/api/auth/register")
async def auth_register(request: Request):
    try:
        body = await request.json()
        email = (body.get("email") or "").strip().lower()
        password = body.get("password") or ""
        name = (body.get("name") or "").strip()

        if not email or not password or not name:
            raise HTTPException(status_code=400, detail="Email, password, and name are required")
        if len(password) < 6:
            raise HTTPException(status_code=400, detail="Password must be at least 6 characters")

        with engine.begin() as conn:
            existing_user = conn.execute(select(users_table).where(users_table.c.email == email)).mappings().first()
            if existing_user:
                raise HTTPException(status_code=409, detail="Email already registered")

            hashed_password = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
            user = {
                "id": str(uuid4()),
                "email": email,
                "password": hashed_password,
                "name": name,
                "created_at": utcnow_iso(),
            }

            conn.execute(users_table.insert().values(**user))

        token = create_token(user["id"])

        return JSONResponse(
            {
                "token": token,
                "user": {"id": user["id"], "email": user["email"], "name": user["name"]},
            },
            status_code=201,
        )
    except HTTPException as e:
        raise e
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/api/auth/login")
async def auth_login(request: Request):
    try:
        body = await request.json()
        email = (body.get("email") or "").strip().lower()
        password = body.get("password") or ""

        if not email or not password:
            raise HTTPException(status_code=400, detail="Email and password are required")

        with engine.begin() as conn:
            user = conn.execute(select(users_table).where(users_table.c.email == email)).mappings().first()
            if not user:
                raise HTTPException(status_code=401, detail="Invalid credentials")

        if not bcrypt.checkpw(password.encode("utf-8"), user["password"].encode("utf-8")):
            raise HTTPException(status_code=401, detail="Invalid credentials")

        token = create_token(user["id"])
        return {"token": token, "user": {"id": user["id"], "email": user["email"], "name": user["name"]}}
    except HTTPException as e:
        raise e
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/auth/me")
async def auth_me(request: Request):
    auth_data = get_user_from_request(request)
    if not auth_data:
        raise HTTPException(status_code=401, detail="Unauthorized")

    with engine.begin() as conn:
        user = conn.execute(select(users_table).where(users_table.c.id == auth_data["userId"])).mappings().first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

    return {"user": {"id": user["id"], "email": user["email"], "name": user["name"]}}


# --- Project Handlers ---
@app.post("/api/projects")
async def create_project(request: Request):
    auth_data = get_user_from_request(request)
    if not auth_data:
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        body = await request.json()
        template_id = body.get("templateId")
        template = get_template(template_id) if template_id else None
        settings = body.get("settings") or {}
        if body.get("stack") and "stack" not in settings:
            settings["stack"] = body.get("stack")
        project = {
            "id": str(uuid4()),
            "user_id": auth_data["userId"],
            "name": (body.get("name") or "Untitled Project").strip(),
            "description": (body.get("description") or "").strip(),
            "code": (extract_index_html(template["files"]) or default_index_html()) if template else default_index_html(),
            "files": (template["files"] if template else [
                {"path": "index.html", "content": default_index_html(), "language": "html"},
            ]),
            "env_vars": [],
            "settings": settings,
            "messages": [],
            "created_at": utcnow_iso(),
            "updated_at": utcnow_iso(),
        }

        with engine.begin() as conn:
            conn.execute(projects_table.insert().values(**project))
        _write_files_to_disk(project["id"], project["files"], project["env_vars"])

        response_project = {
            "id": project["id"],
            "userId": project["user_id"],
            "name": project["name"],
            "description": project["description"],
            "code": project["code"],
            "files": project["files"],
            "envVars": project["env_vars"],
            "messages": project["messages"],
            "settings": project["settings"],
            "createdAt": project["created_at"],
            "updatedAt": project["updated_at"],
        }

        return JSONResponse({"project": response_project}, status_code=201)
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/projects")
async def list_projects(request: Request):
    auth_data = get_user_from_request(request)
    if not auth_data:
        raise HTTPException(status_code=401, detail="Unauthorized")

    with engine.begin() as conn:
        rows = conn.execute(
            select(projects_table)
            .where(projects_table.c.user_id == auth_data["userId"])
            .order_by(projects_table.c.updated_at.desc())
        ).mappings().all()

    project_list = []
    for p in rows:
        messages = p.get("messages") or []
        project_list.append(
            {
                "id": p.get("id"),
                "name": p.get("name"),
                "description": p.get("description"),
                "createdAt": p.get("created_at"),
                "updatedAt": p.get("updated_at"),
                "messageCount": len(messages),
            }
        )

    return {"projects": project_list}


@app.get("/api/projects/{project_id}")
async def get_project(project_id: str, request: Request):
    auth_data = get_user_from_request(request)
    if not auth_data:
        raise HTTPException(status_code=401, detail="Unauthorized")

    with engine.begin() as conn:
        project = conn.execute(
            select(projects_table).where(
                (projects_table.c.id == project_id) & (projects_table.c.user_id == auth_data["userId"])
            )
        ).mappings().first()

    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    response_project = {
        "id": project.get("id"),
        "userId": project.get("user_id"),
        "name": project.get("name"),
        "description": project.get("description"),
        "code": project.get("code"),
        "files": project.get("files") or [],
        "envVars": project.get("env_vars") or [],
        "settings": project.get("settings") or {},
        "messages": project.get("messages") or [],
        "createdAt": project.get("created_at"),
        "updatedAt": project.get("updated_at"),
    }

    return {"project": response_project}


@app.put("/api/projects/{project_id}")
async def update_project(project_id: str, request: Request):
    auth_data = get_user_from_request(request)
    if not auth_data:
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        updates = await request.json()
        allowed_fields = {"name", "description", "code", "messages", "files", "env_vars", "settings"}
        update_data = {k: v for k, v in updates.items() if k in allowed_fields}
        if "envVars" in updates:
            update_data["env_vars"] = updates["envVars"]
        if "settings" in updates and isinstance(updates["settings"], dict):
            update_data["settings"] = updates["settings"]
        if "files" in updates:
            files = updates["files"] or []
            update_data["files"] = files
            # Keep code in sync with index.html
            index_html = extract_index_html(files)
            if index_html:
                update_data["code"] = index_html
        update_data["updated_at"] = utcnow_iso()

        with engine.begin() as conn:
            result = conn.execute(
                update(projects_table)
                .where((projects_table.c.id == project_id) & (projects_table.c.user_id == auth_data["userId"]))
                .values(**update_data)
            )

            if result.rowcount == 0:
                raise HTTPException(status_code=404, detail="Project not found")

            project = conn.execute(select(projects_table).where(projects_table.c.id == project_id)).mappings().first()

        if project and project.get("files"):
            _write_files_to_disk(project_id, project.get("files") or [], project.get("env_vars") or [])

        response_project = {
            "id": project.get("id"),
            "userId": project.get("user_id"),
            "name": project.get("name"),
            "description": project.get("description"),
            "code": project.get("code"),
            "files": project.get("files") or [],
            "envVars": project.get("env_vars") or [],
            "settings": project.get("settings") or {},
            "messages": project.get("messages") or [],
            "createdAt": project.get("created_at"),
            "updatedAt": project.get("updated_at"),
        }
        return {"project": response_project}
    except HTTPException as e:
        raise e
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/projects/{project_id}/export")
async def export_project(project_id: str, request: Request):
    auth_data = get_user_from_request(request)
    if not auth_data:
        raise HTTPException(status_code=401, detail="Unauthorized")

    with engine.begin() as conn:
        project = conn.execute(
            select(projects_table).where(
                (projects_table.c.id == project_id) & (projects_table.c.user_id == auth_data["userId"])
            )
        ).mappings().first()

    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    import io
    import zipfile

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in project.get("files") or []:
            path = f.get("path")
            if not path or path.startswith("/") or ".." in path:
                continue
            zf.writestr(path, f.get("content") or "")
        meta = {
            "name": project.get("name"),
            "description": project.get("description"),
            "settings": project.get("settings") or {},
        }
        zf.writestr("project.json", json.dumps(meta, indent=2))

    filename = (project.get("name") or "project").replace(" ", "-").lower()
    headers = {
        "Content-Disposition": f"attachment; filename={filename}.zip"
    }
    return Response(content=buffer.getvalue(), media_type="application/zip", headers=headers)


@app.post("/api/projects/{project_id}/import")
async def import_project(project_id: str, request: Request, file: UploadFile = File(...)):
    auth_data = get_user_from_request(request)
    if not auth_data:
        raise HTTPException(status_code=401, detail="Unauthorized")

    with engine.begin() as conn:
        project = conn.execute(
            select(projects_table).where(
                (projects_table.c.id == project_id) & (projects_table.c.user_id == auth_data["userId"])
            )
        ).mappings().first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    import io
    import zipfile

    content = await file.read()
    files: List[Dict[str, Any]] = []
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            path = info.filename
            if not path or path.startswith("/") or ".." in path:
                continue
            raw = zf.read(info.filename)
            try:
                text = raw.decode("utf-8")
            except Exception:
                text = raw.decode("latin-1", errors="ignore")
            files.append({
                "path": path,
                "content": text,
                "language": _detect_language_from_path(path),
            })

    if not files:
        raise HTTPException(status_code=400, detail="No files found in archive")

    code = extract_index_html(files) or project.get("code") or default_index_html()
    with engine.begin() as conn:
        conn.execute(
            update(projects_table)
            .where((projects_table.c.id == project_id) & (projects_table.c.user_id == auth_data["userId"]))
            .values(
                files=files,
                code=code,
                updated_at=utcnow_iso(),
            )
        )
    _write_files_to_disk(project_id, files, project.get("env_vars") or [])

    return {"success": True, "files": files}


@app.delete("/api/projects/{project_id}")
async def delete_project(project_id: str, request: Request):
    auth_data = get_user_from_request(request)
    if not auth_data:
        raise HTTPException(status_code=401, detail="Unauthorized")

    with engine.begin() as conn:
        result = conn.execute(
            delete(projects_table).where(
                (projects_table.c.id == project_id) & (projects_table.c.user_id == auth_data["userId"])
            )
        )

    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="Project not found")

    return {"success": True}


# --- Snapshots ---
@app.get("/api/snapshots")
async def list_snapshots(request: Request, projectId: str):
    auth_data = get_user_from_request(request)
    if not auth_data:
        raise HTTPException(status_code=401, detail="Unauthorized")

    with engine.begin() as conn:
        project = conn.execute(
            select(projects_table).where(
                (projects_table.c.id == projectId) & (projects_table.c.user_id == auth_data["userId"])
            )
        ).mappings().first()
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        rows = conn.execute(
            select(snapshots_table).where(snapshots_table.c.project_id == projectId).order_by(snapshots_table.c.created_at.desc())
        ).mappings().all()

    snapshots = []
    for row in rows:
        snapshots.append({
            "id": row.get("id"),
            "projectId": row.get("project_id"),
            "name": row.get("name"),
            "createdAt": row.get("created_at"),
        })

    return {"snapshots": snapshots}


@app.get("/api/snapshots/{snapshot_id}")
async def get_snapshot(snapshot_id: str, request: Request):
    auth_data = get_user_from_request(request)
    if not auth_data:
        raise HTTPException(status_code=401, detail="Unauthorized")

    with engine.begin() as conn:
        snapshot = conn.execute(
            select(snapshots_table).where(snapshots_table.c.id == snapshot_id)
        ).mappings().first()
        if not snapshot:
            raise HTTPException(status_code=404, detail="Snapshot not found")

        project = conn.execute(
            select(projects_table).where(
                (projects_table.c.id == snapshot.get("project_id")) & (projects_table.c.user_id == auth_data["userId"])
            )
        ).mappings().first()
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

    return {
        "snapshot": {
            "id": snapshot.get("id"),
            "projectId": snapshot.get("project_id"),
            "name": snapshot.get("name"),
            "files": snapshot.get("files") or [],
            "envVars": snapshot.get("env_vars") or [],
            "createdAt": snapshot.get("created_at"),
        }
    }


@app.post("/api/snapshots")
async def create_snapshot(request: Request):
    auth_data = get_user_from_request(request)
    if not auth_data:
        raise HTTPException(status_code=401, detail="Unauthorized")

    body = await request.json()
    project_id = body.get("projectId")
    name = body.get("name") or f"Snapshot {utcnow_iso()}"
    if not project_id:
        raise HTTPException(status_code=400, detail="projectId is required")

    with engine.begin() as conn:
        project = conn.execute(
            select(projects_table).where(
                (projects_table.c.id == project_id) & (projects_table.c.user_id == auth_data["userId"])
            )
        ).mappings().first()
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        snapshot = {
            "id": str(uuid4()),
            "project_id": project_id,
            "name": name,
            "files": project.get("files") or [],
            "env_vars": project.get("env_vars") or [],
            "created_at": utcnow_iso(),
        }
        conn.execute(snapshots_table.insert().values(**snapshot))

    return {"snapshot": {"id": snapshot["id"], "name": snapshot["name"], "createdAt": snapshot["created_at"]}}


@app.post("/api/snapshots/{snapshot_id}/restore")
async def restore_snapshot(snapshot_id: str, request: Request):
    auth_data = get_user_from_request(request)
    if not auth_data:
        raise HTTPException(status_code=401, detail="Unauthorized")

    with engine.begin() as conn:
        snapshot = conn.execute(
            select(snapshots_table).where(snapshots_table.c.id == snapshot_id)
        ).mappings().first()
        if not snapshot:
            raise HTTPException(status_code=404, detail="Snapshot not found")

        project = conn.execute(
            select(projects_table).where(
                (projects_table.c.id == snapshot.get("project_id")) & (projects_table.c.user_id == auth_data["userId"])
            )
        ).mappings().first()
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        files = snapshot.get("files") or []
        env_vars = snapshot.get("env_vars") or []
        code = extract_index_html(files) or project.get("code") or default_index_html()

        conn.execute(
            update(projects_table)
            .where(projects_table.c.id == project.get("id"))
            .values(
                files=files,
                env_vars=env_vars,
                code=code,
                updated_at=utcnow_iso(),
            )
        )

    _write_files_to_disk(project.get("id"), files, env_vars)

    return {"project": {"id": project.get("id"), "files": files, "envVars": env_vars, "code": code}}


# --- LLM Options ---
@app.get("/api/llm/options")
async def llm_options():
    enabled = _enabled_providers()
    providers = [{"id": name, "enabled": enabled[name]} for name in sorted(enabled.keys())]
    return {
        "defaultProvider": DEFAULT_PROVIDER,
        "defaultModel": DEFAULT_MODEL,
        "providers": providers,
    }

@app.get("/api/llm/models")
async def llm_models(provider: str):
    models = list_models(provider)
    return {
        "provider": provider,
        "models": models,
        "count": len(models),
    }


@app.get("/api/templates")
async def templates():
    return {
        "templates": [
            {k: v for k, v in t.items() if k != "files"} for t in TEMPLATES
        ]
    }


@app.get("/api/run/profiles")
async def run_profiles():
    return {"profiles": RUN_PROFILES}


@app.get("/api/agents")
async def agents():
    return {"agents": AGENTS}


# --- AI Generation ---
@app.post("/api/generate")
async def generate(request: Request):
    auth_data = get_user_from_request(request)
    if not auth_data:
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        enforce_rate_limit(request)
        body = await request.json()
        prompt = body.get("prompt")
        current_code = body.get("currentCode")
        project_id = body.get("projectId")
        provider = (body.get("provider") or DEFAULT_PROVIDER).lower()
        model = body.get("model") or DEFAULT_MODEL
        stack = body.get("stack")
        agent = body.get("agent")

        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt is required")
        if MAX_MESSAGE_LEN and len(prompt) > MAX_MESSAGE_LEN:
            raise HTTPException(status_code=400, detail="Prompt is too long")

        chat_history = []
        project_messages = []
        project_files = []
        project_env_vars = []
        project_settings = {}
        project = None
        response_files = None
        if project_id:
            with engine.begin() as conn:
                project = conn.execute(
                    select(projects_table).where(
                        (projects_table.c.id == project_id) & (projects_table.c.user_id == auth_data["userId"])
                    )
                ).mappings().first()
                if project:
                    project_messages = project.get("messages") or []
                    project_files = project.get("files") or []
                    project_env_vars = project.get("env_vars") or []
                    project_settings = project.get("settings") or {}
                    if project_messages:
                        chat_history = project_messages[-MAX_HISTORY_MESSAGES:]
        if not stack and project_settings:
            stack = project_settings.get("stack")

        stack_hint = (stack or "").lower()
        use_json = (
            _should_use_json_files(project_files)
            or _prompt_suggests_multifile(prompt)
            or (stack_hint not in {"", "web", "static", "html"})
        )

        if use_json:
            stack_line = f"Stack: {stack}\n" if stack else ""
            system_prompt = (
                "You are a senior full-stack engineer. You must respond with strict JSON only.\n\n"
                "Output format (required):\n"
                "{ \"files\": [ { \"path\": \"...\", \"content\": \"...\", \"language\": \"optional\" } ] }\n\n"
                "Rules:\n"
                "- Output ONLY valid JSON. No markdown, no code fences, no extra keys.\n"
                "- All newlines inside content MUST be escaped as \\n.\n"
                "- Include ALL files needed to run the project for the requested stack.\n"
                "- Use relative paths (e.g., index.html, src/App.jsx, backend/main.py).\n"
                "- If updating existing files, modify them and keep unrelated files unless asked to delete.\n"
                "- For static websites, use separate files: index.html, styles.css, script.js.\n"
                "- If the request mentions Next.js, React, Node/Express, MERN, Django, FastAPI, or Flask, include the conventional file structure and config (package.json/requirements.txt, etc.).\n"
                + stack_line
            )
        else:
            system_prompt = (
                "You are an expert web developer and designer. Your task is to generate or modify a complete HTML file based on the user's request.\n\n"
                "CRITICAL RULES:\n"
                "- Output ONLY the complete HTML code. No markdown, no explanations, no code fences.\n"
                "- The very first characters must be <!DOCTYPE html>\n"
                "- Include ALL CSS inside a <style> tag in the <head>\n"
                "- Include ALL JavaScript inside a <script> tag before </body>\n"
                "- Use modern, responsive design with CSS Grid and Flexbox\n"
                "- Add smooth animations, transitions, and hover effects\n"
                "- Use a professional, cohesive color scheme\n"
                "- Ensure full mobile responsiveness\n"
                "- Use Google Fonts via CDN link in the head\n"
                "- Use semantic HTML5 elements\n"
                "- Make designs visually stunning with gradients, shadows, and modern UI patterns\n"
                "- Add proper meta viewport tag\n\n"
                + (
                    "IMPORTANT: The user has existing code. Modify it based on their new request. Preserve the overall structure unless asked to change it completely. Return the FULL modified HTML."
                    if current_code
                    else "Generate a complete, beautiful HTML page from scratch."
                )
            )

        agent_instruction = AGENT_PROMPTS.get((agent or "").lower())
        if agent_instruction:
            system_prompt = f"{agent_instruction}\n\n{system_prompt}"

        messages = [{"role": "system", "content": system_prompt}]

        if chat_history:
            context_messages = [m for m in chat_history if m.get("role") == "user"][-3:]
            for msg in context_messages:
                messages.append({"role": "user", "content": msg.get("content")})
                messages.append({"role": "assistant", "content": "Done. What would you like to change?"})

        if use_json:
            if project_files:
                file_context = "\n\n".join(
                    [f"FILE: {f.get('path')}\n{f.get('content')}" for f in project_files if f.get("path")]
                )
                messages.append(
                    {
                        "role": "user",
                        "content": f"Existing files:\n\n{file_context}\n\nTask: {prompt}\n\nReturn only JSON.",
                    }
                )
            else:
                messages.append(
                    {
                        "role": "user",
                        "content": f"Task: {prompt}\n\nReturn only JSON.",
                    }
                )
        else:
            if current_code:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Here is my current HTML code:\n\n"
                            f"{current_code}\n\n"
                            f"Please modify it based on this instruction: {prompt}\n\n"
                            "Remember: Output ONLY the complete modified HTML code starting with <!DOCTYPE html>. No explanations."
                        ),
                    }
                )
            else:
                messages.append(
                    {
                        "role": "user",
                        "content": f"{prompt}\n\nRemember: Output ONLY the complete HTML code starting with <!DOCTYPE html>. No explanations.",
                    }
                )

        generated_code, resolved_model = call_llm(
            provider=provider,
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=8192,
        )

        # Clean up markdown code blocks if present
        generated_code = generated_code.strip()
        if generated_code.lower().startswith("```html"):
            generated_code = generated_code[7:]
        if generated_code.startswith("```"):
            generated_code = generated_code[3:]
        if generated_code.endswith("```"):
            generated_code = generated_code[:-3]
        generated_code = generated_code.strip()

        if use_json:
            parsed_files = _parse_files_from_response(generated_code)
            if not parsed_files:
                if "<html" in generated_code.lower():
                    use_json = False
                else:
                    raise HTTPException(status_code=500, detail="LLM returned invalid JSON. Please try again.")
            else:
                # Normalize HTML assets if needed
                html_file = next((f for f in parsed_files if f.get("path") == "index.html"), None)
                if html_file:
                    separated_html, css_content, js_content = split_html_assets(html_file.get("content") or "")
                    html_file["content"] = separated_html
                    html_file["language"] = html_file.get("language") or "html"
                    if css_content and not any(f.get("path") == "styles.css" for f in parsed_files):
                        parsed_files.append({"path": "styles.css", "content": css_content, "language": "css"})
                    if js_content and not any(f.get("path") == "script.js" for f in parsed_files):
                        parsed_files.append({"path": "script.js", "content": js_content, "language": "javascript"})

                generated_code = extract_index_html(parsed_files)
                response_files = parsed_files

                if project_id:
                    updated_messages = (project_messages or []) + [
                        {"role": "user", "content": prompt, "timestamp": utcnow_iso()},
                        {"role": "assistant", "content": "Code generated successfully", "timestamp": utcnow_iso()},
                    ]
                    if MAX_HISTORY_MESSAGES and len(updated_messages) > MAX_HISTORY_MESSAGES:
                        updated_messages = updated_messages[-MAX_HISTORY_MESSAGES:]
                    with engine.begin() as conn:
                        conn.execute(
                            update(projects_table)
                            .where((projects_table.c.id == project_id) & (projects_table.c.user_id == auth_data["userId"]))
                            .values(
                                code=generated_code or (project.get("code") if project else ""),
                                updated_at=utcnow_iso(),
                                messages=updated_messages,
                                files=response_files,
                            )
                        )
                    _write_files_to_disk(project_id, response_files, project_env_vars)

                return {"code": generated_code, "model": resolved_model, "provider": provider, "files": response_files}

        # Ensure it starts with <!DOCTYPE html>
        lower_code = generated_code.lower()
        doctype_index = lower_code.find("<!doctype html>")
        if doctype_index > 0:
            generated_code = generated_code[doctype_index:]
        elif doctype_index == -1:
            html_index = lower_code.find("<html")
            if html_index > 0:
                generated_code = "<!DOCTYPE html>\n" + generated_code[html_index:]
            elif html_index == -1:
                generated_code = (
                    "<!DOCTYPE html>\n"
                    "<html lang=\"en\">\n"
                    "<head>\n"
                    "  <meta charset=\"UTF-8\">\n"
                    "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n"
                    "  <title>Generated Page</title>\n"
                    "</head>\n"
                    "<body>\n"
                    f"{generated_code}\n"
                    "</body>\n"
                    "</html>"
                )

        # Split inline CSS/JS into separate files if present
        separated_html, css_content, js_content = split_html_assets(generated_code)
        if css_content or js_content:
            generated_code = separated_html

        if project_id:
            updated_messages = (project_messages or []) + [
                {"role": "user", "content": prompt, "timestamp": utcnow_iso()},
                {"role": "assistant", "content": "Code generated successfully", "timestamp": utcnow_iso()},
            ]
            if MAX_HISTORY_MESSAGES and len(updated_messages) > MAX_HISTORY_MESSAGES:
                updated_messages = updated_messages[-MAX_HISTORY_MESSAGES:]
            updated_files = project_files or []
            if updated_files:
                found = False
                for f in updated_files:
                    if f.get("path") == "index.html":
                        f["content"] = generated_code
                        f["language"] = f.get("language") or "html"
                        found = True
                        break
                if not found:
                    updated_files.append({"path": "index.html", "content": generated_code, "language": "html"})
            else:
                updated_files = [{"path": "index.html", "content": generated_code, "language": "html"}]

            if css_content:
                css_found = False
                for f in updated_files:
                    if f.get("path") == "styles.css":
                        f["content"] = css_content
                        f["language"] = f.get("language") or "css"
                        css_found = True
                        break
                if not css_found:
                    updated_files.append({"path": "styles.css", "content": css_content, "language": "css"})

            if js_content:
                js_found = False
                for f in updated_files:
                    if f.get("path") == "script.js":
                        f["content"] = js_content
                        f["language"] = f.get("language") or "javascript"
                        js_found = True
                        break
                if not js_found:
                    updated_files.append({"path": "script.js", "content": js_content, "language": "javascript"})
            with engine.begin() as conn:
                conn.execute(
                    update(projects_table)
                    .where((projects_table.c.id == project_id) & (projects_table.c.user_id == auth_data["userId"]))
                    .values(
                        code=generated_code,
                        updated_at=utcnow_iso(),
                        messages=updated_messages,
                        files=updated_files,
                    )
                )
            _write_files_to_disk(project_id, updated_files, project_env_vars)
            response_files = updated_files
        else:
            response_files = [{"path": "index.html", "content": generated_code, "language": "html"}]
            if css_content:
                response_files.append({"path": "styles.css", "content": css_content, "language": "css"})
            if js_content:
                response_files.append({"path": "script.js", "content": js_content, "language": "javascript"})

        return {"code": generated_code, "model": resolved_model, "provider": provider, "files": response_files}

    except HTTPException as e:
        raise e
    except Exception as e:
        traceback.print_exc()
        message = str(e) or "Failed to generate code"
        raise HTTPException(status_code=500, detail=message)


# --- Preview Handler ---
@app.get("/api/preview/{project_id}")
async def preview(project_id: str):
    try:
        with engine.begin() as conn:
            project = conn.execute(select(projects_table).where(projects_table.c.id == project_id)).mappings().first()
        if not project:
            return HTMLResponse("<html><body><h1>Project not found</h1></body></html>", status_code=404)

        files = project.get("files") or []
        html = build_preview_html(files, project.get("code") or "")
        return HTMLResponse(html)
    except Exception:
        return HTMLResponse("<html><body><h1>Error loading preview</h1></body></html>", status_code=500)


# --- Health ---
@app.get("/api/health")
async def health():
    return {"status": "ok", "timestamp": utcnow_iso()}

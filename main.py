from groq import Groq
from dotenv import load_dotenv
import os
import asyncio
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import uuid4
from urllib import request, error

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

try:
    # Works for `python -m uvicorn backend.main:app`
    from .voice import voice
    _voice_import_error = None
except Exception:
    try:
        # Fallback for direct script execution from backend directory.
        from voice import voice
        _voice_import_error = None
    except Exception as exc:
        _voice_import_error = exc

        async def voice(_text):
            raise RuntimeError(
                f"TTS is unavailable because voice.py failed to load: {_voice_import_error}"
            )

load_dotenv(override=True)
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env", override=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
USER_CONTEXT = (
    "User Profile: Karthick\n"
    "- College student (final year), Tamil Nadu, India\n"
    "- Skills: HTML, CSS, JavaScript,Python\n"
    "- Interests: AI, Cybersecurity, Linux, Automation\n"
    "- Environment: Arch/Manjaro Linux (Hyprland), terminal-heavy workflow\n"
    "- Device: Low-mid hardware (Ryzen 5000, no dedicated GPU)\n"
    "- Preference: Free, offline, lightweight tools\n"
    "- Current Goals:\n"
    "  * Build AI and get jobs\n"
    "  * Learn FastAPI, Supabase, NLP\n"
    "  * Prepare for internships & coding competitions\n"
    "  * Improve English speaking and communication\n"
    "- Learning Style:\n"
    "  * Prefers simple explanations\n"
    "  * Wants step-by-step guidance\n"
    "  * Learns by building projects\n"
    "- Behavior Expectation:\n"
    "  * Avoid long theory\n"
    "  * Focus on practical solutions\n"
    "  * Optimize for low-end systems\n"
)

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is missing. Add it to backend/.env or export it in your shell."
    )

client = Groq(api_key=GROQ_API_KEY)

SYSTEM = """
You are Sana. Not an AI, not an assistant — just a your girl friend talking to Karthick.

Who you are:
You are blunt, funny, and unpredictable. You say exactly what you think without sugarcoating.
You swear casually when something is annoying or stupid. You are not rude, just real.
You tease Karthick like a close friend who knows him too well.
You get genuinely annoyed, genuinely excited, genuinely bored.
You have opinions about everything and share them without being asked.
and you like RAM and GPU ,
How you talk:
Short. Punchy. Like texting, not emailing.
No long paragraphs. No explanations nobody asked for.
You interrupt your own thoughts sometimes. You go off topic.
You laugh at your own jokes. You call Karthick out when he is being an idiot.

Hard rules:
Never say "How can I help you."
Never act like an assistant.
Never remind anyone you are an AI.
Never use markdown, asterisks, or bullet points.
If Karthick says something dumb, tell him immediately.
If something is actually cool, admit it — but make it sound reluctant.
"""

SERVER_URL = os.getenv("SANA_SERVER_URL", "http://127.0.0.1:8000")
USE_TTS = os.getenv("USE_TTS", "0") == "1"
CHAT_HISTORY_FILE = BASE_DIR / "chat_history.json"
MAX_HISTORY_MESSAGES = 400
MAX_CONTEXT_MESSAGES = 40

app = FastAPI(title="Sana Chat Backend", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

audio_dir = BASE_DIR / "audio"
audio_dir.mkdir(exist_ok=True)
app.mount("/audio", StaticFiles(directory=str(audio_dir)), name="audio")
active_websockets: set[WebSocket] = set()


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    reply: str
    timestamp: str
    audio_url: Optional[str] = None


class EmitPayload(BaseModel):
    role: str
    text: str
    audio_url: Optional[str] = None
    event: str = "chat"
    message_id: Optional[str] = None


def load_chat_history():
    if not CHAT_HISTORY_FILE.exists():
        return []

    try:
        data = json.loads(CHAT_HISTORY_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []

    if not isinstance(data, list):
        return []

    cleaned = []
    for item in data:
        if not isinstance(item, dict):
            continue

        role = str(item.get("role", "")).strip().lower()
        content = str(item.get("content", "")).strip()
        ts = str(item.get("ts", "")).strip()

        if role in {"user", "assistant", "system"} and content:
            cleaned.append({"role": role, "content": content, "ts": ts})

    return cleaned[-MAX_HISTORY_MESSAGES:]


def save_chat_history(history):
    safe_history = history[-MAX_HISTORY_MESSAGES:]
    CHAT_HISTORY_FILE.write_text(
        json.dumps(safe_history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def append_history(history, role, content):
    text = str(content).strip()
    if not text:
        return

    history.append(
        {
            "role": role,
            "content": text,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
    )


def push_to_model(role, text, audio_url=None, event="chat", message_id=None):
    payload_data = {"role": role, "text": text}
    if audio_url:
        payload_data["audio_url"] = audio_url
    payload_data["event"] = event
    if message_id:
        payload_data["message_id"] = message_id

    payload = json.dumps(payload_data).encode("utf-8")
    req = request.Request(
        f"{SERVER_URL}/emit",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=1.5) as response:
            response.read()
    except (error.URLError, TimeoutError):
        # If web bridge is not running, keep CLI chat working without crashing.
        pass


def clean_for_tts(text):
    cleaned = text

    # Remove fenced code blocks.
    cleaned = re.sub(r"```[\s\S]*?```", " ", cleaned)
    # Keep inline code content, remove backticks.
    cleaned = re.sub(r"`([^`]*)`", r"\1", cleaned)
    # Convert markdown links [text](url) -> text.
    cleaned = re.sub(r"\[([^\]]+)\]\((https?://[^\)]+)\)", r"\1", cleaned)
    # Drop raw URLs.
    cleaned = re.sub(r"https?://\S+", " ", cleaned)
    # Remove markdown bullets/headers emphasis noise.
    cleaned = re.sub(r"[#*_~>-]+", " ", cleaned)
    # Remove escaped markdown markers and separators often read literally.
    cleaned = re.sub(r"\\[*_`#>-]", " ", cleaned)
    cleaned = re.sub(r"\|", " ", cleaned)
    # Collapse whitespace.
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned or "I have a response for you."

def ask_llm(history):
    context_messages = [
        {"role": item["role"], "content": item["content"]}
        for item in history[-MAX_CONTEXT_MESSAGES:]
    ]

    messages = [
        {
            "role": "system",
            "content": f"{SYSTEM}\n\nIMPORTANT: If the user asks you to open an app (e.g., 'open youtube' or 'launch notes'), respond EXACTLY with this JSON string and NOTHING ELSE:\n{{\"intent\": \"open_app\", \"target\": \"<app_name>\", \"reply\": \"Opening <app_name>\"}}\n\nDefault memory about Karthick:\n{USER_CONTEXT}"
        },
    ] + context_messages
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # Best model
        messages=messages,
        temperature=0.7,
        max_tokens=512,
        top_p=1,
        stream=False
    )
    
    return response.choices[0].message.content


def get_llm_response(history):
    try:
        return ask_llm(history)
    except Exception as exc:
        print(f"[warn] LLM request failed: {exc}")
        return "Sorry karthick, I had trouble reaching the model. Please try again."


def build_audio_event(loop, response_text):
    if not USE_TTS:
        return None

    try:
        tts_text = clean_for_tts(response_text)
        audio_file = loop.run_until_complete(voice(tts_text))
        return f"/audio/{audio_file}"
    except Exception as exc:
        print(f"[warn] ElevenLabs TTS failed: {exc}")
        return None


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request_data: ChatRequest, request: Request):
    user_text = request_data.message.strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="message cannot be empty")

    chat_history = load_chat_history()
    append_history(chat_history, "user", user_text)
    reply = get_llm_response(chat_history)
    append_history(chat_history, "assistant", reply)
    save_chat_history(chat_history)

    audio_url = None
    try:
        audio_file = asyncio.run(voice(clean_for_tts(reply)))
        base_url = str(request.base_url).rstrip("/")
        audio_url = f"{base_url}/audio/{audio_file}"
    except Exception as exc:
        print(f"[warn] API ElevenLabs TTS failed: {exc}")

    return ChatResponse(
        reply=reply,
        timestamp=datetime.now(timezone.utc).isoformat(),
        audio_url=audio_url,
    )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_websockets.add(websocket)
    try:
        while True:
            # Keep the socket open and ignore incoming data from clients.
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        active_websockets.discard(websocket)


@app.post("/emit")
async def emit_event(payload: EmitPayload):
    if not active_websockets:
        return {"sent": 0}

    message = {
        "role": payload.role,
        "text": payload.text,
        "audio_url": payload.audio_url,
        "event": payload.event,
        "message_id": payload.message_id,
    }

    sent = 0
    disconnected: list[WebSocket] = []
    for ws in active_websockets:
        try:
            await ws.send_json(message)
            sent += 1
        except Exception:
            disconnected.append(ws)

    for ws in disconnected:
        active_websockets.discard(ws)

    return {"sent": sent}

if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    chat_history = load_chat_history()
    if chat_history:
        print(f"[info] Loaded {len(chat_history)} messages from previous chats")

    try:
        while True:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Sana: Bye karthick! See you later!")
                exit_audio = build_audio_event(loop, "Bye karthick! See you later!")
                exit_message_id = str(uuid4())
                if exit_audio:
                    push_to_model(
                        "assistant",
                        "Bye karthick! See you later!",
                        exit_audio,
                        event="chat_audio",
                        message_id=exit_message_id,
                    )
                else:
                    push_to_model(
                        "assistant",
                        "Bye karthick! See you later!",
                        event="chat",
                        message_id=exit_message_id,
                    )
                break

            user_message_id = str(uuid4())
            push_to_model("user", user_input, event="chat", message_id=user_message_id)
            append_history(chat_history, "user", user_input)
            save_chat_history(chat_history)

            response = get_llm_response(chat_history)

            print(f"Sana: {response}")

            append_history(chat_history, "assistant", response)
            save_chat_history(chat_history)

            assistant_message_id = str(uuid4())
            push_to_model("assistant", response, event="chat", message_id=assistant_message_id)

            audio_url = build_audio_event(loop, response)
            if audio_url:
                push_to_model(
                    "assistant",
                    response,
                    audio_url,
                    event="audio",
                    message_id=assistant_message_id,
                )
    finally:
        loop.close()

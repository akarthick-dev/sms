import os
from pathlib import Path
import time
import re
import asyncio
import aiohttp
from dotenv import load_dotenv

load_dotenv(override=True)

ELEVENLABS_TTS_URL = "https://api.elevenlabs.io/v1/text-to-speech"
DEFAULT_MODEL_ID = "eleven_turbo_v2_5"


def _clean_env(name: str) -> str:
    value = os.getenv(name)
    if value is None:
        return ""
    return value.strip().strip('"').strip("'")


def _normalize_text(text: str) -> str:
    cleaned = str(text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    # Keep requests comfortably within practical TTS bounds.
    return cleaned[:1200] if len(cleaned) > 1200 else cleaned


async def voice(text, output_file=None):
    api_key = _clean_env("ELEVEN_LABS_API_KEY") or _clean_env("ELEVENLABS_API_KEY")
    voice_id = "cgSgspJ2msm6clMCkdW9"
    if not api_key:
        raise ValueError("Please set ELEVEN_LABS_API_KEY in .env")
    if not voice_id:
        raise ValueError("Please set VOICE_ID in .env")

    safe_text = _normalize_text(text)
    if not safe_text:
        safe_text = "Sorry, I had trouble speaking that response."

    audio_dir = Path(__file__).resolve().parent / "audio"
    audio_dir.mkdir(exist_ok=True)

    if output_file is None:
        output_file = audio_dir / f"voice_{int(time.time() * 1000)}.mp3"
    else:
        output_file = Path(output_file)

    last_error = None
    url = f"{ELEVENLABS_TTS_URL}/{voice_id}"
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    }
    payload = {
        "text": safe_text,
        "model_id": os.getenv("ELEVEN_LABS_MODEL_ID", DEFAULT_MODEL_ID),
        "voice_settings": {
            "stability": 0.4,
            "similarity_boost": 0.75,
        },
    }

    timeout = aiohttp.ClientTimeout(total=30)
    for attempt in range(2):
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        # Do not retry for auth/permission failures; they are non-transient.
                        if response.status in {400, 401, 403}:
                            raise RuntimeError(
                                "ElevenLabs rejected this request. "
                                "Check API key scope (needs text_to_speech), voice id, and account access. "
                                f"Response {response.status}: {error_text[:300]}"
                            )
                        raise RuntimeError(
                            f"ElevenLabs API error {response.status}: {error_text[:300]}"
                        )

                    audio_bytes = await response.read()
                    if not audio_bytes:
                        raise RuntimeError("ElevenLabs returned empty audio data")

                    output_file.write_bytes(audio_bytes)
                    return output_file.name
        except Exception as exc:
            last_error = exc
            # Retry only transient failures.
            message = str(exc)
            if "Response 400:" in message or "Response 401:" in message or "Response 403:" in message:
                break
            await asyncio.sleep(0.3 * (attempt + 1))

    raise RuntimeError(f"ElevenLabs TTS failed after retries: {last_error}")

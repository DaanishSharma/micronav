"""Text to speech with ElevenLabs, for the launch video narration.

The API key is read from $ELEVENLABS_API_KEY or ~/.config/micronav/elevenlabs_key,
never from the command line, so it doesn't end up in shell history or logs.

    python tts_elevenlabs.py OUT.mp3 "text" [--voice VOICE_ID]
    python tts_elevenlabs.py --voices          (list the voices on your account)
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

KEY_FILE = Path.home() / ".config" / "micronav" / "elevenlabs_key"
DEFAULT_VOICE = "JBFqnCBsd6RMkjVDRZzb"   # "George": a default voice, usable on the free plan
API = "https://api.elevenlabs.io/v1"


def api_key():
    key = os.environ.get("ELEVENLABS_API_KEY") or (KEY_FILE.read_text().strip() if KEY_FILE.exists() else "")
    if not key:
        sys.exit(f"no ElevenLabs key: set ELEVENLABS_API_KEY or save it to {KEY_FILE}")
    return key


def request(path, body=None):
    req = urllib.request.Request(API + path, data=json.dumps(body).encode() if body else None,
                                 headers={"xi-api-key": api_key(), "Content-Type": "application/json",
                                          "Accept": "audio/mpeg" if body else "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        sys.exit(f"ElevenLabs {e.code}: {e.read().decode(errors='replace')[:500]}")


def speak(text, out, voice=DEFAULT_VOICE, model="eleven_multilingual_v2"):
    audio = request(f"/text-to-speech/{voice}", dict(
        text=text, model_id=model,
        voice_settings=dict(stability=0.5, similarity_boost=0.75, style=0.15, use_speaker_boost=True)))
    Path(out).write_bytes(audio)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?")
    ap.add_argument("text", nargs="?")
    ap.add_argument("--voice", default=DEFAULT_VOICE)
    ap.add_argument("--voices", action="store_true")
    a = ap.parse_args()
    if a.voices:
        for v in json.loads(request("/voices"))["voices"]:
            print(v["voice_id"], v["name"], v.get("labels", {}))
        return
    speak(a.text, a.out, a.voice)


if __name__ == "__main__":
    main()

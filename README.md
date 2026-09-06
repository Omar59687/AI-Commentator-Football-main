# AI Football Commentator

This app turns a football video into grounded English and Arabic commentary,
expressive speech, and commentary videos.

## What the commentary pipeline does

1. Gemini watches the complete clip at a football-friendly sampling rate and
   returns a timestamped event timeline.
2. It writes matching English and Arabic scripts from those visible events.
3. The script length is limited to the video duration so speech is not cut off.
4. ElevenLabs v3 uses restrained performance tags: calm during build-up and more
   excited only when the visible action earns it.
5. The original stadium sound is mixed quietly beneath the commentator.

The uploaded video is deleted from Gemini Files after the commentary request.

## Setup

Create a virtual environment, install the packages, and copy `.env.example` to
`.env`:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Fill in the API keys and voice IDs in `.env`. For the best Arabic pronunciation,
use an Arabic-native voice ID; use a separate English sports-style voice ID for
English. If only `ELEVENLABS_VOICE_ID` is supplied, both languages use that voice.

Run the web app:

```powershell
uvicorn api.main:app --reload
```

Then open `http://127.0.0.1:8000`.

Generated files are written to `media/output/`. The `*_tts_script.txt` files keep
the performance directions sent to ElevenLabs, while the normal commentary text
files contain clean text for display.

## Voice tuning

`ELEVENLABS_MODEL_ID=eleven_v3` is recommended because it understands emotional
audio tags. `ELEVENLABS_STABILITY=0.30` gives a lively delivery. Raise stability
toward `0.5` if a selected voice becomes inconsistent; lower it slightly if the
voice remains too flat.

## Run checks

```powershell
python -m unittest discover -s tests -v
```

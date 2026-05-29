# audio-book-but-for-git-hub-users-

## Audiobook TTS workflow

This repo now includes a GitHub Actions workflow at:

- `.github/workflows/audiobook-tts.yml`

### How to use

1. Upload/commit your source text files to:
   - `uploads/`
   - Supported formats: `.txt` and `.md`
   - Subdirectories are supported (the workflow scans recursively).
2. Trigger the **Build Audiobook Audio** workflow (or push files into `uploads/`).
3. Download the workflow artifact named `audiobook-output-<run_number>`.

### Voice and engine behavior

- Uses one consistent voice for the full run (no mid-chapter voice switches).
- Provider priority is fixed:
  1. ElevenLabs (`ELEVENLABS_API_KEY` + `ELEVENLABS_VOICE_ID`)
  2. OpenAI TTS (`OPENAI_API_KEY`, voice `onyx` or `nova`)
  3. Google Cloud TTS WaveNet (`GOOGLE_APPLICATION_CREDENTIALS_JSON`)
- If a segment fails, it retries that segment on the same provider/voice.
- It does **not** fall back to a different voice/provider mid-chapter when retries fail.

### Required secrets/variables

Configure in repo settings:

- Secrets:
  - `ELEVENLABS_API_KEY` (optional)
  - `ELEVENLABS_VOICE_ID` (required when ElevenLabs is used)
  - `OPENAI_API_KEY` (optional)
  - `GOOGLE_APPLICATION_CREDENTIALS_JSON` (optional, full service-account JSON)
- Variables:
  - `OPENAI_TTS_VOICE` (`onyx` or `nova`, default `onyx`)
  - `GOOGLE_CLOUD_TTS_VOICE` (must be WaveNet, default `en-US-Wavenet-D`)

### Notes

- The workflow script is at `scripts/generate_audiobook.py`.
- Output files:
  - `dist/audiobook.mp3`
  - `dist/audiobook.json` (provider/voice metadata)
- `ffmpeg` must be available (GitHub-hosted runners already include it).

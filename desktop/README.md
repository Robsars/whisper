Desktop Whisper Dictation App

Overview
- Hotkey-activated, low-latency dictation running locally on Windows.
- Captures microphone audio, performs Whisper inference, and inserts transcribed text into the active text field.
- Follows the buffering, VAD gating, and decoding cadence guidelines in SPEC.md and implementation conventions in AGENTS.md.

Run
1) Double-click `start.bat` from the repo root.
2) The script creates a virtual environment, installs dependencies (including this repository as a package), and launches the app.
3) Default hotkey: Ctrl+Shift+Space to start/stop dictation.
4) Open http://127.0.0.1:8765 to access the settings dashboard.

Notes
- GPU will use Whisper `turbo`; CPU will use `small.en` by default.
- No ffmpeg required (raw mic capture). Whisper file loading paths still require ffmpeg, but this app does not use file decoding.
- On first run, Whisper will download model weights (hundreds of MB) — allow time for this.

Dashboard
- A local dashboard is available at http://127.0.0.1:8765.
- You can configure: hotkey, language/task, thresholds, cadence, model selection, and start/stop.
- Model reload requires stopping dictation first; the dashboard will enforce this.

Troubleshooting
- If your device doesn’t support 16 kHz input, the app will fall back to the default device rate and resample.
- If the `keyboard` global hotkey needs elevated privileges, run the terminal as Administrator.

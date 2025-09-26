Agent Guidelines for This Repository

Scope
- This file applies to the entire repository.
- It provides conventions and practical tips for agents and contributors working on Whisper and on the real-time speech-to-text (STT) integration described in SPEC.md.

Baseline Conventions
- Language: Python 3.8+ (repo supports 3.8–3.13).
- Style/format: Use Black and isort (see pyproject.toml). Keep changes minimal and focused.
- Dependencies: Prefer editing pyproject.toml for runtime deps; avoid adding heavyweight libs unless justified.
- Tests: Use pytest. Be mindful: some tests download large Whisper models and need ffmpeg.
- Don’t change model URLs/weights or public APIs without explicit reason and documentation.

Local Setup
- Install: `pip install -r requirements.txt` (or `pip install -U openai-whisper`).
- System: Ensure `ffmpeg` is installed and on PATH.
- Optional GPU: CUDA for best latency. FP16 is auto-disabled on CPU.

Project Structure (key modules)
- `whisper/audio.py`: Audio IO, resampling (via ffmpeg in load path), mel spectrograms.
- `whisper/model.py`: Encoder/decoder, attention, kv-cache hooks.
- `whisper/decoding.py`: Decoding loop (greedy/beam), language ID, timestamp rules.
- `whisper/transcribe.py`: Sliding-window transcription and CLI.
- `whisper/tokenizer.py`: Tiktoken wrapper & special tokens.
- `whisper/timing.py`: Word-level timestamps via cross-attention + DTW (optional Triton).
- `tests/`: Unit and integration tests; `tests/test_transcribe.py` downloads models.

Working Patterns
- Prefer small, surgical changes. Follow existing patterns for options and argument parsing.
- If adding new writer formats, implement in `whisper/utils.py` and register in `get_writer`.
- For decoding behavior changes, add options via `DecodingOptions` and respect existing defaults.
- When touching `tokenizer.py`, keep special token ordering intact.
- For performance-sensitive code, keep CPU/GPU fallbacks and avoid breaking FP16 logic.

Real-Time STT Integration (for new code)
- Place server-side, web-facing code in a new top-level directory: `realtime/`.
  - Suggested layout:
    - `realtime/server.py` (FastAPI/Starlette WebSocket app)
    - `realtime/requirements.txt` (fastapi, uvicorn, webrtcvad, torchaudio/soxr, pyav if decoding opus)
    - `realtime/README.md` (run instructions)
- Do not modify core Whisper internals unless necessary; use public entry points:
  - `whisper.decode()` for 30s mel windows.
  - `whisper.transcribe.transcribe()` for batch.
  - `whisper.audio.pad_or_trim()` and `log_mel_spectrogram()` for streaming windows.
- Keep the model hot-loaded (singleton per worker) and pinned to device.
- Use `DecodingOptions(without_timestamps=True, temperature=0.0)` for partials; timestamps only on finalize.
- Employ VAD to gate decoding and detect segment boundaries (webrtcvad recommended).

Testing & Validation
- Unit tests: add targeted tests under `tests/` for any new utility functions.
- Avoid making CI download all models unless strictly needed. If adding new tests that need large downloads, mark them with a custom marker and skip by default.
- Manual verification for realtime: add a small script in `realtime/` that streams a short mic sample and prints partial/final outputs.

Performance & Limits
- Target partial update cadence of 250–500 ms; ensure each decode fits within cadence.
- Prefer `turbo` on GPU for interactive use; use `small.en`/`base` on CPU with slower cadence.
- Consider micro-batching multiple sessions for decoder forward pass if implementing a pool.

Documentation
- When adding realtime components, keep SPEC.md as the source of truth for protocols and update it with any changes.
- Document any public endpoints, message schemas, and required client-side behaviors in `realtime/README.md`.

Security & Ops Notes
- Require TLS for WebSocket endpoints in production.
- Enforce per-client rate limits and maximum session duration.
- Sanitize/validate any user-sent JSON control messages.


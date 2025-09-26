Real-Time Speech-to-Text (STT) Integration Spec (Whisper)

Summary
- Goal: Enable users to dictate into a browser text field and receive real-time (or near real-time) transcription from Whisper running server-side.
- Non-goals: On-device inference in the browser; use of browser-native STT (e.g., Web Speech API).

Non-Functional Requirements
- Latency: Partial updates within 250–500 ms under nominal load; finalization within 600–1000 ms after speech pause.
- Accuracy: Comparable to Whisper batch transcription, acknowledging a small degradation for streaming partials.
- Capacity: Configurable concurrency per GPU/CPU worker; graceful backpressure.
- Portability: Modern browsers; minimal client setup; server Linux/x86_64 with optional NVIDIA GPU.
- Security: TLS termination for WS; basic auth or token auth supported by hosting environment.

Architecture
- Frontend (Browser)
  - Microphone capture (Web Audio API).
  - Stream audio frames over a WebSocket to the backend.
  - Apply partial and final transcription updates into the focused input field.
- Backend (Python)
  - FastAPI/Starlette WebSocket endpoint (`/ws/stt`).
  - Per-connection audio ring buffer and VAD.
  - Periodic Whisper decoding for partials; finalize on VAD-detected pause.
  - Optional REST batch endpoint (`POST /api/transcribe`) for non-interactive uploads.

Transport
- Protocol: WebSocket over TLS (wss).
- Messages:
  - Client → Server
    - Text (JSON): initialization/control
      - `{"type":"init", "sampleRate": number, "lang?": string, "task?": "transcribe"|"translate"}`
      - Optional: `{"type":"stop"}` to finalize and close.
    - Binary: raw PCM float32 mono frames (little-endian) at the device sample rate (e.g., 48 kHz), chunked every ~10–40 ms.
    - Alternative (optional): `MediaRecorder` blobs (`audio/webm;codecs=opus`) every ~250 ms; requires server-side opus decode.
  - Server → Client
    - Text (JSON):
      - Partial hypothesis: `{"type":"partial", "text": string}` (incremental tail)
      - Finalized text: `{"type":"final", "text": string}` (up to current end)
      - Error/info: `{"type":"error"|"info", "message": string}`

Audio Handling
- Client captures mono or stereo; client should downmix to mono or server will downmix.
- Server resamples to 16 kHz for Whisper and VAD. Recommended libraries: torchaudio or soxr; fallback linear interpolation allowed for MVP.
- Ring buffer capacity: 30 s (Whisper window); keep last N_SAMPLES (480000 at 16 kHz).

Decoding Strategy
- Cadence: 250–500 ms decode tick per session.
- Gating: Use webrtcvad on recent 300 ms to skip decoding silence.
- Partials: `DecodingOptions(temperature=0.0, beam_size=1, without_timestamps=True, fp16=True on CUDA)`.
- Context: `condition_on_previous_text=True`; maintain previous committed tokens/text to stabilize output; reset if anomalies detected (optional).
- Finalization: After ~600–800 ms without speech, mark current text as final; optionally run a second pass on the segment audio with `word_timestamps=True` for highlighting.

Language & Tasks
- Default task `transcribe`. If `task=translate`, recommend `medium`/`large` models for quality; `turbo` is not trained for translation.
- Language detection: If client doesn’t set language, let Whisper detect on the first window.

Model Selection
- GPU: `turbo` (large-v3-turbo) for low latency interactive ASR; fp16 enabled.
- CPU: `small.en` (English) or `base` (multilingual) with slower cadence.
- Load model once per worker; do not reload per-connection.

Backend API (Batch)
- Endpoint: `POST /api/transcribe`
- Request: multipart/form-data with `file`, optional `model`, `language`, `task`, `word_timestamps`.
- Response: JSON `{ text, segments?, language }` compatible with Whisper CLI outputs.

Client Behavior
- Start: Open WS, send `init` with `sampleRate` (AudioContext rate), begin streaming frames.
- Update: On `partial`, append/replace the live tail in the text field; on `final`, replace the full dictated span.
- Stop: Send `stop`, close WS, or auto-close when the UI toggles off.

Performance Targets
- Partial round-trip: < 300–600 ms on GPU (turbo), < 800–1200 ms on CPU (small/base) per update.
- Throughput: Tune worker concurrency so decode time < cadence. Consider micro-batching across sessions as an optimization.

Resource Management
- Session limits: max concurrent sessions per worker; timeouts on idle/no-audio.
- Backpressure: Drop/skip decode ticks when decode backlog exists; apply VAD gating to shrink work.
- Memory: 30 s ring buffers per session; monitor GPU/CPU memory usage.

Security & Privacy
- TLS termination required; authenticate WS with session cookies or bearer token.
- Do not persist raw audio by default; log minimal metadata for ops.
- Provide an opt-in flag for saving audio/text for debugging.

Rollout Plan (Incremental)
1) Batch MVP: Implement `/api/transcribe`, validate environment and models.
2) WS Ingest: Add `/ws/stt`, stream PCM float32, build ring buffer; echo back counts for validation.
3) Periodic Decode: Add decode loop and partial/final messages (no VAD yet).
4) VAD & Finalization: Gate decoding and finalize on silence; improve UX stability.
5) Timestamps (optional): Second-pass timestamps on finalized segments for word highlighting.
6) Scaling & Observability: Worker pool, metrics (latency, tokens/sec), limits.

Open Questions / Future Work
- Diarization or speaker change detection (out of scope for MVP).
- Punctuation stabilization and re-segmentation heuristics for streaming text.
- Multimodal noise suppression (server-side) vs client-side constraints.
- Multi-language auto-switching in a single session.

References
- Core code paths: `whisper/transcribe.py`, `whisper/decoding.py`, `whisper/audio.py`, `whisper/timing.py`.
- Recommended libs: FastAPI/Starlette, Uvicorn, webrtcvad, torchaudio/soxr, pyav (if decoding opus).


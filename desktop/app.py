import argparse
import json
import asyncio
import sys
import threading
import time
import os
import webbrowser
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
import sounddevice as sd
import keyboard as kb
import pyperclip
import torch
import webrtcvad

# Ensure local repo import works even if not pip installed
try:
    import whisper  # installed via start.bat (pip install -e .)
except Exception:  # pragma: no cover
    # fallback to local path if running from source without pkg install
    import os
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    import whisper

from whisper.audio import N_SAMPLES, SAMPLE_RATE, pad_or_trim, log_mel_spectrogram
from whisper.utils import compression_ratio

# Dashboard (FastAPI)
try:
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse, JSONResponse
    import uvicorn
except Exception:
    FastAPI = None  # dashboard optional if deps missing

# Tray icon (optional)
try:
    import pystray
    from PIL import Image, ImageDraw
except Exception:
    pystray = None


DEFAULT_HOTKEY = "ctrl+shift+space"


@dataclass
class Config:
    hotkey: str = DEFAULT_HOTKEY
    paste_threshold: int = 100  # characters; use clipboard+paste if suffix longer than threshold
    silence_partial_ms: int = 300
    silence_final_ms: int = 800
    decode_cadence_ms: int = 300
    aggressiveness: int = 2  # webrtcvad 0-3
    model_gpu: str = "turbo"
    model_cpu: str = "small.en"
    language: Optional[str] = None  # None -> auto detect (multilingual models)
    task: str = "transcribe"
    logprob_threshold: float = -1.0
    compression_ratio_threshold: float = 2.4
    no_speech_threshold: float = 0.6
    device_preference: str = "auto"  # auto|cpu|cuda


class RingBuffer:
    def __init__(self, capacity_samples: int):
        self.capacity = capacity_samples
        self.buf = deque(maxlen=capacity_samples)
        self.lock = threading.Lock()

    def extend(self, samples: np.ndarray):
        with self.lock:
            self.buf.extend(samples.tolist())

    def snapshot(self) -> np.ndarray:
        with self.lock:
            if not self.buf:
                return np.zeros(self.capacity, dtype=np.float32)
            return np.fromiter(self.buf, dtype=np.float32, count=len(self.buf))


class Transcriber:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        if cfg.device_preference == "cpu":
            self.device = "cpu"
        elif cfg.device_preference == "cuda":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = cfg.model_gpu if self.device == "cuda" else cfg.model_cpu
        print(f"[Whisper] Loading model '{model_name}' on {self.device}...")
        self.model = whisper.load_model(model_name, device=self.device)
        self.running = False
        self.stream: Optional[sd.InputStream] = None
        self.vad = webrtcvad.Vad(cfg.aggressiveness)
        self.buffer = RingBuffer(N_SAMPLES)
        self.last_committed = ""
        self.emitted_text = ""  # what we've actually typed so far
        self._recent_texts = deque(maxlen=3)
        self.decode_thread: Optional[threading.Thread] = None
        self.decode_lock = threading.Lock()
        self.sample_rate = SAMPLE_RATE
        self._last_activity_ts = 0.0

    def _on_audio(self, indata, frames, time_info, status):
        if status:
            # print(status)  # noisy in production
            pass
        # indata is float32 arrays shape (frames, channels)
        if indata.ndim == 2 and indata.shape[1] > 1:
            mono = indata.mean(axis=1).astype(np.float32)
        else:
            mono = indata.reshape(-1).astype(np.float32)

        self.buffer.extend(mono)

    def _recent_has_speech(self, window_ms: int) -> bool:
        # Check last window_ms for speech via webrtcvad; expects 16k PCM16 10/20/30ms frames
        x = self.buffer.snapshot()
        if x.size == 0:
            return False
        w = int(self.sample_rate * window_ms / 1000)
        if w <= 0:
            return False
        x = x[-w:]
        # Convert to 16-bit PCM for VAD
        pcm16 = np.clip(x * 32768.0, -32768, 32767).astype(np.int16).tobytes()
        frame_len = int(0.02 * self.sample_rate) * 2  # 20ms frames, bytes
        if frame_len <= 0 or len(pcm16) < frame_len:
            return False
        any_speech = False
        for i in range(0, len(pcm16) - frame_len + 1, frame_len):
            frame = pcm16[i:i+frame_len]
            try:
                if self.vad.is_speech(frame, self.sample_rate):
                    any_speech = True
                    break
            except Exception:
                # If device rate unsupported by VAD, skip
                break
        return any_speech

    def _insert_text(self, suffix: str):
        if not suffix:
            return
        # Prefer direct typing for short suffix; use paste for long text
        try:
            if len(suffix) <= self.cfg.paste_threshold:
                kb.write(suffix)
            else:
                old_clip = None
                try:
                    old_clip = pyperclip.paste()
                except Exception:
                    old_clip = None
                pyperclip.copy(suffix)
                kb.press_and_release("ctrl+v")
                # restore clipboard if possible
                if old_clip is not None:
                    pyperclip.copy(old_clip)
        except Exception as e:
            print(f"[Insert] Failed to inject text: {e}")

    def _resample_to_16k(self, x: np.ndarray) -> np.ndarray:
        if self.sample_rate == SAMPLE_RATE:
            return x
        if x.size == 0:
            return x
        # Linear resample to 16k for mel computation
        src_len = x.shape[0]
        dst_len = int(round(src_len * SAMPLE_RATE / float(self.sample_rate)))
        if dst_len <= 0:
            return x
        src_idx = np.arange(src_len, dtype=np.float64)
        dst_idx = np.linspace(0, src_len - 1, dst_len, dtype=np.float64)
        y = np.interp(dst_idx, src_idx, x).astype(np.float32)
        return y

    def _decode_once(self):
        # Avoid overlapping decodes
        if not self.decode_lock.acquire(blocking=False):
            return
        try:
            audio = self.buffer.snapshot()
            if audio.size == 0:
                return
            # Resample to 16k for Whisper mel pipeline
            audio16 = self._resample_to_16k(audio)
            audio = pad_or_trim(torch.from_numpy(audio16), N_SAMPLES)
            mel = log_mel_spectrogram(audio, n_mels=self.model.dims.n_mels).to(self.model.device)
            language_opt = self.cfg.language
            if language_opt is None and not self.model.is_multilingual:
                language_opt = "en"

            opts = whisper.DecodingOptions(
                task=self.cfg.task if self.model.is_multilingual else "transcribe",
                language=language_opt,
                temperature=0.0,
                without_timestamps=True,
                beam_size=1,
                fp16=(self.device == "cuda"),
            )
            result = whisper.decode(self.model, mel, options=opts)
            text = result.text.strip()
            # Skip likely hallucinations or silence based on thresholds
            has_speech_now = self._recent_has_speech(self.cfg.silence_partial_ms)
            if (
                not text
                or (result.no_speech_prob > self.cfg.no_speech_threshold and not has_speech_now)
                or (result.avg_logprob is not None and result.avg_logprob < self.cfg.logprob_threshold)
                or compression_ratio(text) > self.cfg.compression_ratio_threshold
            ):
                return
            if text:
                # Maintain recent predictions to compute a stable prefix
                self._recent_texts.append(text)
                stable = self._recent_texts[0]
                for t in list(self._recent_texts)[1:]:
                    # longest common prefix
                    m = min(len(stable), len(t))
                    i = 0
                    while i < m and stable[i] == t[i]:
                        i += 1
                    stable = stable[:i]

                # Only emit if stable grows beyond what we already typed
                if stable.startswith(self.emitted_text):
                    suffix = stable[len(self.emitted_text):]
                    # Emit only complete words, or if suffix length grows (responsiveness)
                    if suffix and (suffix.endswith((" ", ",", ".", "!", "?", ";", ":")) or suffix.count(" ") >= 1 or len(suffix) >= 6):
                        self._insert_text(suffix)
                        self.emitted_text = stable
                # update activity timestamp if we saw speech recently
                if has_speech_now:
                    self._last_activity_ts = time.time()
                # finalize on pause
                if (not self._recent_has_speech(self.cfg.silence_final_ms)) and text:
                    # On finalization, emit any remaining difference once
                    if text.startswith(self.emitted_text):
                        tail = text[len(self.emitted_text):]
                        if tail:
                            self._insert_text(tail)
                    self.last_committed = text
                    self.emitted_text = text
                    self._recent_texts.clear()
                else:
                    # do not over-commit too aggressively to preserve ability to correct
                    pass
        finally:
            self.decode_lock.release()

    def _decode_loop(self):
        print("[Transcriber] Decode loop started; press hotkey again to stop.")
        self._last_activity_ts = time.time()
        while self.running:
            try:
                # Gate on recent speech to reduce wasted compute
                has_speech = self._recent_has_speech(self.cfg.silence_partial_ms)
                if has_speech:
                    self._decode_once()
                time.sleep(self.cfg.decode_cadence_ms / 1000)
            except Exception as e:
                # Keep loop alive on errors
                print(f"[Decode] Error: {e}")
                time.sleep(self.cfg.decode_cadence_ms / 1000)
        print("[Transcriber] Decode loop stopped.")

    def start(self):
        if self.running:
            return
        self.running = True
        self.last_committed = ""
        self.emitted_text = ""
        self._recent_texts.clear()
        # Try 16 kHz first for simpler pipeline
        preferred_sr = SAMPLE_RATE
        try:
            print(f"[Audio] Opening input stream at {preferred_sr} Hz mono...")
            self.stream = sd.InputStream(
                callback=self._on_audio,
                channels=1,
                samplerate=preferred_sr,
                dtype="float32",
                blocksize=int(preferred_sr * 0.02),  # 20ms blocks
            )
            self.sample_rate = preferred_sr
            self.stream.start()
        except Exception as e:
            # Fallback to default device SR
            print(f"[Audio] {e}; falling back to device default sample rate")
            self.stream = sd.InputStream(
                callback=self._on_audio,
                channels=1,
                dtype="float32",
            )
            self.sample_rate = int(self.stream.samplerate)
            self.stream.start()
        # Resize ring buffer capacity to preserve ~30s at current sample rate
        desired_capacity = int(round(N_SAMPLES * (self.sample_rate / float(SAMPLE_RATE))))
        self.buffer = RingBuffer(max(desired_capacity, N_SAMPLES))
        print("[Transcriber] Listening... (speak now)")
        self.decode_thread = threading.Thread(target=self._decode_loop, daemon=True)
        self.decode_thread.start()

    def stop(self):
        if not self.running:
            return
        print("[Transcriber] Stopping...")
        self.running = False
        try:
            if self.decode_thread:
                self.decode_thread.join(timeout=2.0)
        except Exception:
            pass
        try:
            if self.stream:
                self.stream.stop()
                self.stream.close()
        finally:
            self.stream = None
        print("[Transcriber] Stopped.")


CONFIG_PATH_DEFAULT = os.path.join(os.path.dirname(__file__), "config.json")


def load_config(path: Optional[str]) -> Config:
    cfg = Config()
    candidate = path or CONFIG_PATH_DEFAULT
    try:
        with open(candidate, "r", encoding="utf-8") as f:
            data = json.load(f)
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        print(f"[Config] Loaded settings from {candidate}")
    except FileNotFoundError:
        print(f"[Config] No config file found at {candidate}, using defaults")
    return cfg


def save_config(cfg: Config, path: Optional[str] = None):
    target = path or CONFIG_PATH_DEFAULT
    try:
        with open(target, "w", encoding="utf-8") as f:
            json.dump(vars(cfg), f, indent=2)
        print(f"[Config] Saved settings to {target}")
    except Exception as e:
        print(f"[Config] Failed to save settings: {e}")


def main():
    parser = argparse.ArgumentParser(description="Hotkey-activated local Whisper dictation")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print("Using config:")
    print(cfg)

    transcriber = Transcriber(cfg)

    toggle_state = {"on": False}

    def on_toggle():
        if toggle_state["on"]:
            toggle_state["on"] = False
            transcriber.stop()
        else:
            toggle_state["on"] = True
            transcriber.start()

    print(f"[Hotkey] Press {cfg.hotkey} to start/stop dictation.")
    kb.add_hotkey(cfg.hotkey, on_toggle)

    # Optional dashboard
    dash_thread = None
    if FastAPI is not None:
        app = FastAPI()

        @app.get("/", response_class=HTMLResponse)
        async def home():
            html = """
            <html><head><title>Whisper Dictation</title></head>
            <body style='font-family: sans-serif; max-width: 800px; margin: 2rem;'>
              <h2>Whisper Dictation – Dashboard</h2>
              <div id='status'></div>
              <hr/>
              <h3>Controls</h3>
              <button onclick='fetch("/api/toggle",{method:"POST"}).then(load)'>Start/Stop</button>
              <button onclick='fetch("/api/reload_model",{method:"POST"}).then(load)'>Reload Model</button>
              <hr/>
              <h3>Configuration</h3>
              <form onsubmit='save(event)'>
                <label>Hotkey: <input id='hotkey'/></label><br/>
                <label>Device: <select id='device_preference'><option>auto</option><option>cuda</option><option>cpu</option></select></label><br/>
                <label>Language (blank=auto): <input id='language'/></label><br/>
                <label>Task: <select id='task'><option>transcribe</option><option>translate</option></select></label><br/>
                <label>Cadence (ms): <input id='decode_cadence_ms' type='number'/></label><br/>
                <label>VAD partial (ms): <input id='silence_partial_ms' type='number'/></label><br/>
                <label>VAD final (ms): <input id='silence_final_ms' type='number'/></label><br/>
                <label>VAD aggressiveness (0-3): <input id='aggressiveness' type='number'/></label><br/>
                <label>Paste threshold: <input id='paste_threshold' type='number'/></label><br/>
                <label>Logprob threshold: <input id='logprob_threshold' type='number' step='0.1'/></label><br/>
                <label>Compression ratio threshold: <input id='compression_ratio_threshold' type='number' step='0.1'/></label><br/>
                <label>No-speech threshold: <input id='no_speech_threshold' type='number' step='0.1'/></label><br/>
                <label>Model (GPU): <input id='model_gpu'/></label><br/>
                <label>Model (CPU): <input id='model_cpu'/></label><br/>
                <button type='submit'>Save</button>
              </form>
              <script>
                async function load(){
                  const s = await fetch('/api/status').then(r=>r.json());
                  document.getElementById('status').innerText = JSON.stringify(s, null, 2);
                  for (const k of [
                    'hotkey','device_preference','language','task','decode_cadence_ms','silence_partial_ms','silence_final_ms','aggressiveness','paste_threshold','logprob_threshold','compression_ratio_threshold','no_speech_threshold','model_gpu','model_cpu'
                  ]){
                    const el = document.getElementById(k);
                    if (el) el.value = s.config[k] ?? '';
                  }
                }
                async function save(e){
                  e.preventDefault();
                  const body = {};
                  for (const k of [
                    'hotkey','device_preference','language','task','decode_cadence_ms','silence_partial_ms','silence_final_ms','aggressiveness','paste_threshold','logprob_threshold','compression_ratio_threshold','no_speech_threshold','model_gpu','model_cpu'
                  ]){
                    const el = document.getElementById(k);
                    if (el) body[k] = el.value;
                  }
                  await fetch('/api/config', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
                  await load();
                }
                load();
              </script>
            </body></html>
            """
            return HTMLResponse(html)

        @app.get("/api/status")
        async def status():
            return JSONResponse({
                "running": transcriber.running,
                "device": transcriber.device,
                "sample_rate": transcriber.sample_rate,
                "model": transcriber.model.dims.__dict__ if hasattr(transcriber.model, 'dims') else str(transcriber.model),
                "is_multilingual": getattr(transcriber.model, 'is_multilingual', False),
                "config": vars(cfg),
            })

        @app.post("/api/toggle")
        async def toggle():
            if transcriber.running:
                transcriber.stop()
            else:
                transcriber.start()
            return JSONResponse({"running": transcriber.running})

        @app.post("/api/reload_model")
        async def reload_model():
            if transcriber.running:
                return JSONResponse({"error": "Stop dictation before reloading model"}, status_code=400)
            # Reload model to pick up cfg.model_gpu/cfg.model_cpu
            transcriber.__init__(cfg)
            return JSONResponse({"ok": True, "device": transcriber.device})

        @app.post("/api/config")
        async def update_config(req: Request):
            body = await req.json()
            # Update config fields; handle hotkey separately
            old_hotkey = cfg.hotkey
            for k, v in body.items():
                if not hasattr(cfg, k):
                    continue
                # numeric fields
                if k in {"decode_cadence_ms","silence_partial_ms","silence_final_ms","aggressiveness","paste_threshold"}:
                    try: v = int(v)
                    except Exception: pass
                if k in {"logprob_threshold","compression_ratio_threshold","no_speech_threshold"}:
                    try: v = float(v)
                    except Exception: pass
                setattr(cfg, k, (None if v=="" else v))
            if cfg.hotkey != old_hotkey:
                try:
                    kb.remove_hotkey(old_hotkey)
                except Exception:
                    pass
                kb.add_hotkey(cfg.hotkey, on_toggle)
            # Persist settings
            save_config(cfg)
            return JSONResponse({"ok": True, "config": vars(cfg)})

        def run_dash():
            # Uvicorn server in a thread
            try:
                print("[Dashboard] Starting at http://127.0.0.1:8765")
                uvicorn.run(app, host="127.0.0.1", port=8765, log_level="warning")
            except Exception as e:
                print(f"[Dashboard] Failed to start: {e}")

        dash_thread = threading.Thread(target=run_dash, daemon=True)
        dash_thread.start()
        # Try to open the dashboard automatically after a short delay
        def open_soon():
            time.sleep(1.0)
            try:
                webbrowser.open("http://127.0.0.1:8765")
            except Exception:
                pass
        threading.Thread(target=open_soon, daemon=True).start()

    # System tray (optional)
    tray = None
    if pystray is not None:
        def make_icon(running: bool):
            img = Image.new("RGB", (64, 64), color=(50, 50, 50))
            d = ImageDraw.Draw(img)
            color = (0, 200, 0) if running else (180, 180, 180)
            d.ellipse((12, 12, 52, 52), fill=color)
            return img

        def on_open():
            webbrowser.open("http://127.0.0.1:8765")

        def on_toggle_tray():
            on_toggle()
            try:
                tray.icon = make_icon(transcriber.running)
            except Exception:
                pass

        def on_quit():
            try:
                transcriber.stop()
            except Exception:
                pass
            os._exit(0)

        menu = pystray.Menu(
            pystray.MenuItem("Start/Stop", lambda: on_toggle_tray()),
            pystray.MenuItem("Open Dashboard", lambda: on_open()),
            pystray.MenuItem("Quit", lambda: on_quit()),
        )
        tray = pystray.Icon("whisper-dictation", make_icon(False), "Whisper Dictation", menu)
        threading.Thread(target=tray.run, daemon=True).start()

    try:
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        transcriber.stop()


if __name__ == "__main__":
    main()

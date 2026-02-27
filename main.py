import argparse
import io
import queue
import threading
import time
import wave
import numpy as np
import cv2
import torch
import sounddevice as sd
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from pynput import keyboard
from piper.voice import PiperVoice


# --- Queue Helper ---
def clear_queue(q):
    """Empties the queue to prioritize new system announcements."""
    while not q.empty():
        try:
            q.get_nowait()
            q.task_done()
        except queue.Empty:
            break


def _synthesize_to_pcm(voice: PiperVoice, text: str) -> tuple[np.ndarray, int]:
    """
    Synthesizes *text* with *voice* and returns (float32_audio, sample_rate).

    Handles both piper-tts API variants automatically:

    - **rhasspy/piper** (original, most common): ``voice.synthesize(text, wav_file)``
      writes WAV data to a file-like object.  We wrap an in-memory BytesIO buffer
      as a ``wave.Wave_write`` object so nothing hits disk.

    - **OHF-Voice/piper1-gpl** (newer fork): ``voice.synthesize(text)`` returns an
      iterator of ``AudioChunk`` objects whose ``.audio_int16_bytes`` attribute
      contains raw PCM.  ``synthesize_stream_raw`` is also available here but was
      removed in the original piper-tts, hence the ``AttributeError`` you saw.
    """
    sample_rate = voice.config.sample_rate

    # ── Try the iterator / AudioChunk API first (piper1-gpl / newer builds) ──
    try:
        chunks = list(voice.synthesize(text))  # returns iterator of AudioChunk
        # AudioChunk objects have .audio_int16_bytes
        raw_bytes = b"".join(c.audio_int16_bytes for c in chunks)
        audio_i16 = np.frombuffer(raw_bytes, dtype=np.int16)
        return audio_i16.astype(np.float32) / 32768.0, sample_rate
    except TypeError:
        # synthesize() in the original piper-tts requires a wav_file argument
        pass

    # ── Fall back to the wav_file API (rhasspy/piper original) ──
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        voice.synthesize(text, wav_file)

    buf.seek(0)
    with wave.open(buf, "rb") as wf:
        sample_rate = wf.getframerate()
        raw_bytes = wf.readframes(wf.getnframes())

    audio_i16 = np.frombuffer(raw_bytes, dtype=np.int16)
    return audio_i16.astype(np.float32) / 32768.0, sample_rate


# --- TTS Worker Thread (Piper) ---
def tts_worker(tts_queue: "queue.Queue[str | None]", voice_model_path: str, use_cuda: bool = False):
    """
    Consumes texts from the queue and speaks them via Piper TTS.

    Args:
        tts_queue:        Queue of str items (or None as a poison pill).
        voice_model_path: Path to the Piper .onnx voice model file.
                          The matching .onnx.json config must sit next to it.
        use_cuda:         Run Piper ONNX inference on GPU (needs onnxruntime-gpu).
    """
    print(f"[TTS] Loading Piper voice from: {voice_model_path}")
    voice = PiperVoice.load(voice_model_path, use_cuda=use_cuda)
    print(f"[TTS] Voice loaded — sample rate: {voice.config.sample_rate} Hz")

    while True:
        item = tts_queue.get()
        if item is None:  # Poison pill — shut down cleanly
            break

        try:
            audio_f32, sample_rate = _synthesize_to_pcm(voice, item)
            # Blocking playback — the queue already serialises utterances
            sd.play(audio_f32, samplerate=sample_rate)
            sd.wait()
        except Exception as e:
            print(f"\n[TTS Error] {e}")
        finally:
            tts_queue.task_done()


# --- Helper: Multi-line text for OpenCV ---
def draw_multiline_text(img, text, x, y, font, scale, color, thickness, max_width):
    words = text.split()
    if not words:
        return
    lines, current_line = [], words[0]
    for word in words[1:]:
        size, _ = cv2.getTextSize(current_line + " " + word, font, scale, thickness)
        if size[0] < max_width:
            current_line += " " + word
        else:
            lines.append(current_line)
            current_line = word
    lines.append(current_line)
    line_height = cv2.getTextSize("Wg", font, scale, thickness)[0][1] + 10
    for i, line in enumerate(lines):
        cv2.putText(img, line, (x, y + i * line_height), font, scale, color, thickness)


# --- Main Application ---
def main():
    parser = argparse.ArgumentParser(description="Offline Real-time Camera Captioning")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument(
        "--model-path",
        type=str,
        default="Salesforce/blip-image-captioning-base",
        help="HuggingFace model ID or local path for BLIP",
    )
    parser.add_argument(
        "--voice",
        type=str,
        default="models/en_US-lessac-medium.onnx",
        help=(
            "Path to a Piper .onnx voice model file.  "
            "Download voices from https://github.com/rhasspy/piper/releases  "
            "Example: en_US-lessac-medium.onnx"
        ),
    )
    parser.add_argument(
        "--voice-cuda",
        action="store_true",
        help="Run Piper voice inference on CUDA (requires onnxruntime-gpu)",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=3.0,
        help="Seconds between captions in auto mode",
    )
    parser.add_argument(
        "--headless", action="store_true", help="Run without OpenCV window"
    )
    args = parser.parse_args()

    # 1. State Variables & Controls
    state = {
        "paused": False,
        "mode": "auto",       # 'auto' or 'manual'
        "trigger_cap": False, # Trigger for manual mode
        "quit": False,
    }

    # 2. Start TTS Thread
    tts_queue: queue.Queue[str | None] = queue.Queue()
    tts_thread = threading.Thread(
        target=tts_worker,
        args=(tts_queue, args.voice, args.voice_cuda),
        daemon=True,
    )
    tts_thread.start()

    def announce(text: str):
        """Clear pending captions and announce a system message immediately."""
        print(f"[*] {text}")
        clear_queue(tts_queue)
        tts_queue.put(text)

    # 3. Global Keyboard Listener (pynput)
    def on_press(key):
        try:
            if hasattr(key, "char") and key.char:
                k = key.char.lower()
                if k == "q":
                    state["quit"] = True
                    announce("Quitting")
                elif k == "p":
                    state["paused"] = True
                    announce("Paused")
                elif k == "r":
                    state["paused"] = False
                    announce("Resumed")
                elif k == "a":
                    state["mode"] = "auto"
                    announce("Auto mode activated")
                elif k == "m":
                    state["mode"] = "manual"
                    announce("Manual mode activated")
                elif k == "g" and state["mode"] == "manual":
                    state["trigger_cap"] = True
        except Exception:
            pass

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    # 4. Initialisation
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"[*] Loading BLIP model on {device.upper()} ({dtype})...")

    processor = BlipProcessor.from_pretrained(args.model_path)
    model = BlipForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=dtype
    ).to(device)
    model.eval()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[!] Could not open camera {args.camera}.")
        return

    latest_caption = "Waiting for first frame..."
    last_process_time = 0.0

    # FPS tracking
    prev_frame_time = time.time()
    fps_smooth = 0.0

    print("\n--- Controls ---")
    print("[a] Auto Mode   [m] Manual Mode   [g] Generate (Manual)")
    print("[p] Pause       [r] Resume        [q] Quit\n")
    announce("System initialized.")

    # 5. Main Loop
    while not state["quit"]:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()

        # Calculate FPS
        fps = 1.0 / (current_time - prev_frame_time)
        fps_smooth = (fps_smooth * 0.9) + (fps * 0.1)
        prev_frame_time = current_time

        # Logic for caption generation
        time_elapsed = (current_time - last_process_time) >= args.rate_limit
        should_generate = False
        if not state["paused"]:
            if state["mode"] == "auto" and time_elapsed:
                should_generate = True
            elif state["mode"] == "manual" and state["trigger_cap"]:
                should_generate = True
                state["trigger_cap"] = False

        if should_generate:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            with torch.inference_mode(), torch.autocast(device_type=device, dtype=dtype):
                inputs = processor(images=pil_image, return_tensors="pt").to(
                    device, dtype
                )
                outputs = model.generate(**inputs, max_new_tokens=128, num_beams=3)
                caption = processor.decode(
                    outputs[0], skip_special_tokens=True
                ).capitalize()

            latest_caption = caption
            last_process_time = time.time()

            if args.headless:
                print(f"Caption: {caption}")

            # Enqueue only when TTS is free to avoid a growing backlog
            if tts_queue.empty():
                tts_queue.put(caption)

        # 6. Display (if not disabled)
        if not args.headless:
            show_frame = frame.copy()

            cv2.rectangle(show_frame, (0, 0), (show_frame.shape[1], 100), (0, 0, 0), -1)

            status_text = f"[{state['mode'].upper()}] " + (
                "PAUSED" if state["paused"] else "RUNNING"
            )
            status_color = (0, 0, 255) if state["paused"] else (0, 255, 0)

            cv2.putText(
                show_frame, status_text, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2,
            )
            cv2.putText(
                show_frame, f"FPS: {fps_smooth:.1f}",
                (show_frame.shape[1] - 120, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2,
            )
            draw_multiline_text(
                show_frame, latest_caption, 10, 55,
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                show_frame.shape[1] - 20,
            )
            cv2.imshow("BLIP Real-time Captioner", show_frame)
            cv2.waitKey(1)

    # 7. Cleanup
    print("\n[*] Shutting down...")
    tts_queue.put(None)  # Poison pill
    tts_thread.join()
    cap.release()
    if not args.headless:
        cv2.destroyAllWindows()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
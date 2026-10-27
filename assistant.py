"""
Offline voice assistant: mic -> Whisper (faster-whisper, CPU) -> GGUF LLM (llama.cpp, Vulkan GPU) -> TTS (pyttsx3)

Install (Windows):
    pip install faster-whisper sounddevice numpy pyttsx3
    # llama-cpp-python with Vulkan (needs the LunarG Vulkan SDK + Visual Studio Build Tools + CMake):
    set CMAKE_ARGS=-DGGML_VULKAN=on
    pip install llama-cpp-python --no-cache-dir --force-reinstall --upgrade

Run:
    python voice_assistant.py
Press Enter to start talking; it stops automatically when you go silent. Ctrl+C to quit.
"""

import queue
import re
import threading
import time

import numpy as np
import pyttsx3
import sounddevice as sd
from faster_whisper import WhisperModel
from llama_cpp import Llama

# ------------------------- CONFIG -------------------------
MODEL_PATH = r"N:\models\HauhauCS\gemma-4-4b model uncensored\Gemma-4-E2B-Uncensored-HauhauCS-Aggressive-IQ3_M.gguf"

N_CTX = 4096            # lower this (2048) if you run out of VRAM
N_GPU_LAYERS = -1       # -1 = offload everything to the Vulkan GPU; lower it (e.g. 20) if VRAM overflows
MAX_TOKENS = 300

WHISPER_MODEL = "base.en"   # tiny.en / base.en / small.en  (use "base"/"small" for multilingual)
WHISPER_DEVICE = "cpu"      # keep CPU: no CUDA/ROCm on your GPU, and this saves VRAM for the LLM
WHISPER_COMPUTE = "int8"

SAMPLE_RATE = 16000
SILENCE_THRESHOLD = 0.012   # RMS level treated as silence; raise if your room is noisy
SILENCE_SECONDS = 1.0       # stop recording after this much silence
MAX_RECORD_SECONDS = 30

SYSTEM_PROMPT = (
    "You are a helpful voice assistant. Reply in short, natural spoken sentences. "
    "Do not use markdown, lists, emojis or special symbols."
)
# ----------------------------------------------------------


# ---------- Speech to text ----------
def record_until_silence() -> np.ndarray:
    """Record from the default mic; start on speech, stop after SILENCE_SECONDS of quiet."""
    block = int(SAMPLE_RATE * 0.1)  # 100 ms blocks
    q: "queue.Queue[np.ndarray]" = queue.Queue()

    def callback(indata, frames, time_info, status):
        q.put(indata[:, 0].copy())

    chunks, started, silent_blocks = [], False, 0
    max_blocks = int(MAX_RECORD_SECONDS / 0.1)
    silence_blocks_needed = int(SILENCE_SECONDS / 0.1)

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                        blocksize=block, callback=callback):
        for _ in range(max_blocks):
            data = q.get()
            rms = float(np.sqrt(np.mean(data ** 2)))
            if rms > SILENCE_THRESHOLD:
                started = True
                silent_blocks = 0
                chunks.append(data)
            elif started:
                chunks.append(data)
                silent_blocks += 1
                if silent_blocks >= silence_blocks_needed:
                    break
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)


def transcribe(whisper: WhisperModel, audio: np.ndarray) -> str:
    if audio.size < SAMPLE_RATE * 0.3:
        return ""
    segments, _ = whisper.transcribe(audio, beam_size=1, vad_filter=True, language=None
                                     if not WHISPER_MODEL.endswith(".en") else "en")
    return " ".join(s.text.strip() for s in segments).strip()


# ---------- Text to speech ----------
class Speaker:
    """Speaks sentences from a queue on a background thread so the LLM keeps generating meanwhile."""

    def __init__(self):
        self.q: "queue.Queue[str | None]" = queue.Queue()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        engine = pyttsx3.init()   # created inside the thread that uses it
        engine.setProperty("rate", 175)
        engine.startLoop(False)
        speaking = False
        while True:
            if not speaking:
                try:
                    text = self.q.get(timeout=0.01)
                except queue.Empty:
                    text = None
                else:
                    if text is None:
                        self.q.task_done()
                        break
                    if text == "__flush__":
                        self.q.task_done()
                    else:
                        engine.say(text)
                        speaking = True

            engine.iterate()
            if speaking and not engine.isBusy():
                self.q.task_done()
                speaking = False
            time.sleep(0.01)
        engine.endLoop()

    def say(self, text: str):
        text = clean_for_speech(text)
        if text:
            self.q.put(text)

    def wait(self):
        self.q.join()


def clean_for_speech(text: str) -> str:
    text = re.sub(r"[*_`#>~|]+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------- LLM ----------
def main():
    print("Loading Whisper...")
    whisper = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE)

    print("Loading LLM (first load can take a bit)...")
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=N_CTX,
        n_gpu_layers=N_GPU_LAYERS,
        flash_attn=False,
        verbose=False,
    )

    speaker = Speaker()
    history = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("\nReady. Press Enter, then speak. Say 'exit' or 'quit' to stop.\n")
    sentence_end = re.compile(r"(?<=[.!?])\s+|\n+")

    while True:
        try:
            input("[Enter] to talk...")
        except (KeyboardInterrupt, EOFError):
            break

        print("Listening...")
        audio = record_until_silence()
        user_text = transcribe(whisper, audio)
        if not user_text:
            print("(didn't catch anything)")
            continue
        print(f"You: {user_text}")
        if user_text.lower().strip(" .!?") in {"exit", "quit", "goodbye"}:
            speaker.say("Goodbye!")
            speaker.wait()
            break

        history.append({"role": "user", "content": user_text})

        # Keep context bounded: system prompt + last 8 messages
        if len(history) > 9:
            history = [history[0]] + history[-8:]

        print("AI: ", end="", flush=True)
        reply, buffer = "", ""
        t0 = time.time()
        stream = llm.create_chat_completion(
            messages=history, max_tokens=MAX_TOKENS, temperature=0.7, top_p=0.9, stream=True
        )
        for chunk in stream:
            delta = chunk["choices"][0]["delta"].get("content")
            if not delta:
                continue
            print(delta, end="", flush=True)
            reply += delta
            buffer += delta
            # speak each finished sentence immediately
            parts = sentence_end.split(buffer)
            if len(parts) > 1:
                for s in parts[:-1]:
                    speaker.say(s)
                buffer = parts[-1]
        if buffer.strip():
            speaker.say(buffer)
        print(f"\n({time.time() - t0:.1f}s)")

        history.append({"role": "assistant", "content": reply})
        speaker.wait()  # don't listen while it is talking (avoids mic picking up the speaker)


if __name__ == "__main__":
    main()
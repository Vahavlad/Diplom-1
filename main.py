import json
import logging
import queue
import threading
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import ttk

import pyttsx3

try:
    import sounddevice as sd
    from vosk import KaldiRecognizer, Model
except Exception:  # optional runtime dependency failure is handled in initialization
    sd = None
    Model = None
    KaldiRecognizer = None

BASE_DIR = Path(__file__).resolve().parent
INTENTS_PATH = BASE_DIR / "intents.json"
MODELS_DIR = BASE_DIR / "models"
LOG_PATH = BASE_DIR / "robot_guide.log"


logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


@dataclass
class RuntimeContext:
    language: str
    sample_rate: int = 16000
    retries: int = 0
    max_retries: int = 2


class IntentProcessor:
    def __init__(self, intents_db: dict):
        self.intents_db = intents_db

    def detect(self, language: str, text: str) -> str:
        normalized = text.lower().strip()
        lang_block = self.intents_db[language]
        for intent, payload in lang_block.items():
            if not isinstance(payload, dict) or "keywords" not in payload:
                continue
            if any(keyword in normalized for keyword in payload["keywords"]):
                return intent
        return "fallback"

    def response(self, language: str, intent: str) -> str:
        lang_block = self.intents_db[language]
        if intent == "fallback":
            return lang_block["fallback"]
        return lang_block[intent]["response"]

    def has_wake_word(self, language: str, text: str) -> bool:
        normalized = text.lower().strip()
        wake_words = self.intents_db[language].get("wake_words", [])
        return any(w in normalized for w in wake_words)


class VoiceRobotGuide:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("University Robot Guide")
        self.root.geometry("760x420")

        self.status_var = tk.StringVar(value="Выберите язык / Select language / Тілді таңдаңыз")
        self.text_var = tk.StringVar(value="")

        self.ctx = None
        self.running = False
        self.engine = None
        self.recognizer = None
        self.audio_queue = queue.Queue()

        self.intents_db = self._load_intents()
        self.processor = IntentProcessor(self.intents_db)

        self._build_ui()

    def _build_ui(self):
        frame = ttk.Frame(self.root, padding=16)
        frame.pack(fill="both", expand=True)

        title = ttk.Label(frame, text="Robot Guide", font=("Segoe UI", 24, "bold"))
        title.pack(pady=8)

        status = ttk.Label(frame, textvariable=self.status_var, font=("Segoe UI", 11))
        status.pack(pady=8)

        live_text = ttk.Label(frame, textvariable=self.text_var, font=("Consolas", 11), wraplength=700)
        live_text.pack(pady=12)

        buttons = ttk.Frame(frame)
        buttons.pack(pady=12)

        ttk.Button(buttons, text="Русский", command=lambda: self.start_session("ru")).grid(row=0, column=0, padx=6)
        ttk.Button(buttons, text="English", command=lambda: self.start_session("en")).grid(row=0, column=1, padx=6)
        ttk.Button(buttons, text="Қазақ", command=lambda: self.start_session("kk")).grid(row=0, column=2, padx=6)

        self.stop_btn = ttk.Button(frame, text="Stop session", command=self.stop_session, state="disabled")
        self.stop_btn.pack(pady=10)

    def _load_intents(self):
        with open(INTENTS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    def _set_status(self, text):
        self.status_var.set(text)
        self.root.update_idletasks()

    def _speak(self, text):
        self.text_var.set(f"TTS: {text}")
        if self.engine:
            self.engine.say(text)
            self.engine.runAndWait()

    def _load_model(self, language):
        if Model is None:
            raise RuntimeError("Vosk or sounddevice is not installed.")
        lang_map = {
            "ru": "vosk-model-small-ru-0.22",
            "en": "vosk-model-small-en-us-0.15",
            "kk": "vosk-model-small-kz-0.15",
        }
        model_path = MODELS_DIR / lang_map[language]
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}. Download models from https://alphacephei.com/vosk/models and place in ./models"
            )
        return Model(str(model_path))

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            logging.warning("Audio status: %s", status)
        self.audio_queue.put(bytes(indata))

    def start_session(self, language):
        if self.running:
            return
        self.stop_btn.config(state="normal")
        self.ctx = RuntimeContext(language=language)
        self.running = True
        thread = threading.Thread(target=self._run_session, daemon=True)
        thread.start()

    def stop_session(self):
        self.running = False
        self.stop_btn.config(state="disabled")
        self._set_status("Session stopped. Выберите язык / Select language / Тілді таңдаңыз")

    def _run_session(self):
        try:
            self._set_status("Инициализация модулей...")
            self.engine = pyttsx3.init()
            model = self._load_model(self.ctx.language)
            self.recognizer = KaldiRecognizer(model, self.ctx.sample_rate)

            greeting_map = {
                "ru": "Добро пожаловать! Скажите горячее слово Робот, затем ваш вопрос.",
                "en": "Welcome! Say wake word Robot, then ask your question.",
                "kk": "Қош келдіңіз! Алдымен Робот деп айтып, кейін сұрағыңызды қойыңыз.",
            }
            self._speak(greeting_map[self.ctx.language])
            self._set_status("Сессия активна. Ожидание горячего слова...")

            if sd is None:
                raise RuntimeError("sounddevice is unavailable.")

            with sd.RawInputStream(
                samplerate=self.ctx.sample_rate,
                blocksize=8000,
                dtype="int16",
                channels=1,
                callback=self._audio_callback,
            ):
                while self.running:
                    data = self.audio_queue.get()
                    if self.recognizer.AcceptWaveform(data):
                        result = json.loads(self.recognizer.Result())
                        text = result.get("text", "").strip()
                        if not text:
                            continue
                        self.text_var.set(f"STT: {text}")
                        self._handle_text(text)
        except Exception as exc:
            logging.exception("Session error: %s", exc)
            self._speak("Критическая ошибка. Сессия завершается.")
            self.stop_session()

    def _handle_text(self, text: str):
        if not self.processor.has_wake_word(self.ctx.language, text):
            return

        cleaned = text.lower().replace("робот", "").replace("robot", "").strip()
        if not cleaned:
            self._speak(self.processor.response(self.ctx.language, "help"))
            return

        intent = self.processor.detect(self.ctx.language, cleaned)
        answer = self.processor.response(self.ctx.language, intent)
        self._speak(answer)

        if intent == "fallback":
            self.ctx.retries += 1
            if self.ctx.retries == 1:
                self._speak(answer)
            elif self.ctx.retries >= self.ctx.max_retries:
                self._speak("Сессия завершается. Попробуйте снова позже.")
                logging.error("Too many failed attempts. Ending session.")
                self.stop_session()
        else:
            self.ctx.retries = 0

        if intent == "goodbye":
            self.stop_session()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = VoiceRobotGuide()
    app.run()

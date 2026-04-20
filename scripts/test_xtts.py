"""Quick XTTS-v2 smoke test: synthesize one short Hindi sentence with the
student voice ref, save to /tmp/xtts_smoke.wav, report timing.
"""
import os, time
os.environ["COQUI_TOS_AGREED"] = "1"

from TTS.api import TTS

t0 = time.time()
print("[xtts] loading model ...", flush=True)
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False, gpu=False)
print(f"[xtts] loaded in {time.time() - t0:.1f}s  synthesizing ...", flush=True)

t1 = time.time()
tts.tts_to_file(
    text="आज हम speech recognition और voice cloning के बारे में बात करेंगे।",
    speaker_wav="data/student_voice_ref.wav",
    language="hi",
    file_path="/tmp/xtts_smoke.wav",
)
print(f"[xtts] synthesized in {time.time() - t1:.1f}s  total {time.time() - t0:.1f}s")

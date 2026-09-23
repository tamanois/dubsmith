import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    "./models/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    device_map="cpu", # cpu or "cuda:0" if you have a GPU
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
    local_files_only=True, # Prevents checking the Hugging Face hub entirely
    )

# single inference
wavs, sr = model.generate_custom_voice(
    text="其实我真的有发现，我是一个特别善于观察别人情绪的人。",
    language="Chinese", # Pass `Auto` (or omit) for auto language adaptive; if the target language is known, set it explicitly.
    speaker="Vivian",
    instruct="用特别愤怒的语气说", # Omit if not needed.
)
sf.write("output_custom_voice.wav", wavs[0], sr)
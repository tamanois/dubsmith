"""French voice cloning with the Base checkpoint (Qwen3-TTS-12Hz-1.7B-Base).

Text in -> speech out is French. The donor clip does not have to be French:
Base cloning is cross-lingual, so a clip in any supported language defines the
timbre while the model speaks the French lines below.

The Base model exposes `generate_voice_clone()` and `create_voice_clone_prompt()` only.
It has no named speakers (`talker_config.spk_id` is empty in its config.json), so the
voice is defined entirely by a reference clip you supply.

Two reference-driven modes:
  * ICL mode (x_vector_only_mode=False) -- uses reference speech codes AND the
    reference transcript. Best similarity, but REF_TEXT must match REF_AUDIO exactly.
  * x-vector-only mode (x_vector_only_mode=True) -- speaker embedding only, no
    transcript needed. Lower similarity, but works with any clip.
"""

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

# --- reference voice (donor) -------------------------------------------------
# The donor clip defines the timbre. It is Chinese here, which is fine: cloning
# is cross-lingual, so the model keeps this voice while speaking the French
# lines below.
# REF_TEXT must be the EXACT TRANSCRIPT of REF_AUDIO -- not a filename -- whenever
# x_vector_only_mode=False (ICL mode), and it stays in the donor's own language.
#
# Any decodable format works for REF_AUDIO (.wav/.mp3/.flac/.ogg, a URL, or an
# (ndarray, sr) tuple). To try the MP3 donor instead, just switch to:
#     REF_AUDIO = "ref_test.mp3"
# For a French-native timbre, supply your own clip:
#     REF_AUDIO = "ma_voix_fr.flac"
#     REF_TEXT  = "<exact French words spoken in ma_voix_fr.flac>"
REF_AUDIO = "input/Delphine-Charming.mp3"
REF_TEXT = """
    Bonjour, je suis delphine. Comment tu vas? Bien j'espère. On a une magnifique journée qui s'annonce. Tu ne trouves pas. Ce ciel bleu, ces nuages, Oh my god, C'est.. Juste... Parfait!
    """

# --- French text to synthesize in that voice ---------------------------------
# Must be spelled "French" (checked case-insensitively against the checkpoint's
# codec_language_id keys: chinese/english/german/italian/portuguese/spanish/
# japanese/korean/french/russian, plus "auto").
TARGET_LANGUAGE = "French"
TARGET_TEXTS = [
    """
    Trop froid dehors? Voici le plan familial parfait. Chez Sauve Qui Peut, résolvez les énigmes ensemble avant la fin du chrono. Puis, célébrez autour d’une bonne poutine chez Poutineville. Plus d’idées sur HotPlaces.
    """,
]

model = Qwen3TTSModel.from_pretrained(
    "./models/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="sdpa",  # flash-attn is not installed here
    local_files_only=True,       # Prevents checking the Hugging Face hub entirely
)

# Build the prompt once, then reuse it for as many lines as you want -- this
# avoids re-encoding the reference audio on every call.
prompt = model.create_voice_clone_prompt(
    ref_audio=REF_AUDIO,
    ref_text=REF_TEXT,
    x_vector_only_mode=False,  # ICL mode
)

# A single prompt broadcasts across every line in the batch, so the reference
# audio is encoded only once.
wavs, sr = model.generate_voice_clone(
    text=TARGET_TEXTS,
    language=[TARGET_LANGUAGE] * len(TARGET_TEXTS),
    voice_clone_prompt=prompt,
)

for i, wav in enumerate(wavs, 1):
    path = f"output/output_voice_clone_fr_{i}.wav"
    sf.write(path, wav, sr)
    print(f"wrote {path} ({len(wav) / sr:.2f}s @ {sr} Hz)")

# Models, weights and licenses

This page separates the project license from the licenses of the models it runs.
The code is Apache-2.0; the weights are **not** distributed with the code and
remain under their upstream terms.

## Quick summary

| Component | License | Commercial use |
|---|---|---|
| Project code | Apache-2.0 | Yes |
| Documentation | CC BY 4.0 | Yes, with attribution |
| Qwen3-TTS Base / VoiceDesign / CustomVoice | Apache-2.0 | Yes |
| Whisper / faster-whisper weights | MIT | Yes |
| CosyVoice3 | Apache-2.0 | Yes |
| Chatterbox | MIT | Yes |
| Fish Speech S2-Pro | Fish Audio Research License | **No, not without a separate license** |

The default dubbing path uses Qwen3-TTS and Whisper only. Fish is an optional
backend; leaving it disabled keeps the default path free of the non-commercial
weights.

## Backend matrix

| Backend | Weights | Controls | Rough VRAM |
|---|---|---|---|
| `qwen` (default) | Qwen3-TTS Base | voice clone, accent modes | ~4–6 GB |
| `customvoice` (`say` only) | Qwen3-TTS CustomVoice | preset speaker + instruction | ~4–6 GB |
| `cosyvoice3` | Fun-CosyVoice3 | emotion, pace, volume, `[breath]`, `[laughter]` | ~4–6 GB |
| `chatterbox` | Chatterbox Multilingual | expression intensity | ~3–5 GB |
| `fish` | Fish Speech S2-Pro | inline `[tags]` | ~6 GB resident, ~11 GB peak (int8) |

These are planning figures, not guarantees. Measure with `python dub.py info`
and `report.json` on your hardware.

## What is not included

- Model weights are not committed to the repository; `models/` is gitignored.
- Fish weights are not redistributed; the backend downloads them from the
  upstream publisher.
- Third-party checkouts under `third_party/` are not vendored.

## If you need commercial rights

1. Use the default Qwen path, or CosyVoice3/Chatterbox if they fit your needs.
2. Do not enable the Fish backend unless Fish Audio has granted you a
   commercial license.
3. Audit system binaries if you redistribute them: ffmpeg builds vary between
   LGPL and GPL, and `rubberband`/`sox` are GPL-family.
4. Check `THIRD_PARTY_LICENSES.md` for package-level notes.

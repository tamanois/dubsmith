# Demo media

Small, curated before/after assets used by the README and docs. They are
released under **CC BY 4.0** (see [`../LICENSE.md`](../LICENSE.md))

## Before / after

| File | What it is |
|---|---|
| `demo-before-after.gif` | 6 s side-by-side loop used in the README (720 px wide, 2.4 MB) |
| `demo-poster.png` | 1280×640 social-preview image, also usable as a still |
| `demo-before.mp4` | Full 8 s source excerpt, 1280×720, 24 fps |
| `demo-after.mp4` | Same excerpt dubbed into French; includes a French subtitle track |

| | |
|---|---|
| Source clip | `speaking-corporate` sample supplied by the maintainer |
| Excerpt | full 8 s |
| Language pair | English → French |
| Backend | Qwen default path |
| Command | not recorded; regenerate with the default `python dub.py --video … --target fr` path |
| Date | 2026-11-09 |


The dubbed excerpt contains a **synthetic voice** cloned from the speaker. It is
included to demonstrate the tool, not to represent the speaker.

## The interface

| File | What it is |
|---|---|
| `ui-dub.png` | The Dub page: recent inputs, target and source language, backend, prosody plan, voice and accent mode, with the device, precision and theme controls in the header. Used in the root README |
| `ui-activity.png` | The Activity page: runs with their deliverable, state, duration and per-job actions, plus search and the state filter. Used in `docs/usage.md` |

Both are captures of the app running locally on one machine, with a real job
list. No transcript, reference audio or full prompt is visible in either.

## Regenerating the GIF and poster

```bash
scripts/make_demo.sh assets/speaking-corporate-original.mp4 \
  assets/speaking-corporate-dubbed.mp4
```

The script writes the four files above into `docs/media/`.

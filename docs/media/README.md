# Demo media

Small, curated before/after assets used by the README and docs. They are
released under **CC BY 4.0** (see [`../LICENSE.md`](../LICENSE.md)) **only if the
underlying source clip's rights allow redistribution**; confirm the provenance
below before publishing.

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
| Permission | **Pending confirmation** — see the note below |
| Excerpt | full 8 s |
| Language pair | English → French |
| Backend | Qwen default path |
| Command | not recorded; regenerate with the default `python dub.py --video … --target fr` path |
| Date | 2026-11-09 |

> **Publishing blocker.** Confirm that the source clip may be redistributed and
> that the speaker consented to a cloned synthetic voice. If this is third-party
> stock footage, record its license and attribution here before publishing, or
> replace it with an own recording. Do not merge or release the media until this
> is resolved.

The dubbed excerpt contains a **synthetic voice** cloned from the speaker. It is
included to demonstrate the tool, not to represent the speaker.

## Regenerating the GIF and poster

```bash
scripts/make_demo.sh assets/speaking-corporate-original.mp4 \
  assets/speaking-corporate-dubbed.mp4
```

The script writes the four files above into `docs/media/`.

## GitHub social preview

Upload `demo-poster.png` in **Settings → Social preview** after the provenance
above is confirmed. GitHub does not render `<video>` or `<audio>` from a repo
path, so the GIF is the README hero and the MP4s are links.

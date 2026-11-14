"""ASGI entry point for ``uvicorn server.asgi:app``.

``server.app.create_app`` is a factory so a test can inject ``data_dir`` and an
``executor``; uvicorn wants a module attribute instead. Building the default app
here keeps that split: this module has import-time side effects (it creates the
data and output directories), so nothing else imports it — not the tests, not
the CLI, not ``python -m server``.

Run it from the repository root, or point the environment at absolute paths:

```bash
uvicorn server.asgi:app --host 127.0.0.1 --port 8000 --reload
```
"""

from __future__ import annotations

from .app import create_app

app = create_app()

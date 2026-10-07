"""Entrypoint: `uv run gemma-lens` or `python -m gemma_lens`."""

from __future__ import annotations

import argparse
import threading
import time
import webbrowser

import uvicorn
from rich.console import Console

from gemma_lens import config
from gemma_lens.app import create_app

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(prog="gemma-lens")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true", help="don't open a browser")
    args = parser.parse_args()

    if not config.INDEX_PATH.exists():
        console.print(
            f"[red]No index found at {config.INDEX_PATH}.[/red]\n"
            f"Run: [bold]uv run python -m gemma_lens.ingest[/bold]"
        )
        raise SystemExit(1)

    url = f"http://{args.host}:{args.port}"
    if not args.no_browser:
        threading.Thread(
            target=lambda: (time.sleep(1.5), webbrowser.open(url)), daemon=True
        ).start()

    uvicorn.run(create_app(), host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()

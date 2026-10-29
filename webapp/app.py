"""Compatibility entry point for the unified local AI classification app."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local_tagging_server import create_app, main

app = create_app()

if __name__ == '__main__':
    raise SystemExit(main(['--port', '5000', *sys.argv[1:]]))

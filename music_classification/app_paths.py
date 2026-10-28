"""User-local settings discovery; preserve the earlier local bundle fallback."""
import os
from pathlib import Path


def default_settings_path(root):
    if os.environ.get('MUSIC_CLASSIFICATION_SETTINGS'):
        return Path(os.environ['MUSIC_CLASSIFICATION_SETTINGS']).expanduser().resolve()
    root=Path(root)
    modern=root/'config/model_paths.local.json'
    legacy=root/'outputs/research-release-r1/model_paths.local.json'
    return modern if modern.is_file() or not legacy.is_file() else legacy

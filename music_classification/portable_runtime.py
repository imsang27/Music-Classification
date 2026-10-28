"""Derive request-local runtimes from reviewed bytes without modifying evidence."""

import ast
import hashlib
from importlib.resources import files
import json
from pathlib import Path
import shutil

from .muq_runtime import MANIFEST_PINS, digest, verified


def resource_json(name, sha):
    if name not in ('clap_prompts.json', 'classification_taxonomy.json'):
        raise ValueError('Unknown packaged resource')
    data = files('music_classification').joinpath('resources', name).read_bytes()
    if hashlib.sha256(data).hexdigest() != sha:
        raise ValueError('Packaged resource checksum mismatch')
    return json.loads(data)


def relocate_mert_source(text):
    calls = [node for node in ast.walk(ast.parse(text)) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == 'FeatureExtractorMERT']
    values = [kw.value for call in calls for kw in call.keywords if kw.arg == 'model_name']
    if len(values) != 1 or not isinstance(values[0], ast.Constant) or not isinstance(values[0].value, str):
        raise ValueError('Unexpected reviewed MERT constructor')
    old = ast.get_source_segment(text, values[0])
    if text.count(old) != 1 or 'import os' not in text:
        raise ValueError('Ambiguous reviewed MERT path')
    result = text.replace(old, 'os.environ["MUSIC_CLASSIFICATION_MERT_PATH"]')
    ast.parse(result)
    return result


def relocate_muq_source(text, reviewed_block):
    if text.count(reviewed_block) != 1 or 'MuQ(config=' not in reviewed_block:
        raise ValueError('Unexpected reviewed MuQ constructor')
    replacement = ('import json\n            import os\n            from pathlib import Path\n'
                   '            self.model = MuQ(config=json.loads(Path(os.environ["MUSIC_CLASSIFICATION_MUQ_AUDIO_CONFIG"]).read_text(encoding="utf-8")))')
    result = text.replace(reviewed_block, replacement)
    ast.parse(result)
    return result


def derive_runtime(source, hashes, destination, *, kind, reviewed_block=None):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or destination.is_relative_to(source):
        raise ValueError('Derived runtime must be separate from verified source')
    destination.mkdir()  # Request-local, never overwrite an existing runtime.
    for name, sha in hashes.items():
        original = verified(source, name, sha)
        target = (destination / name).resolve()
        if not target.is_relative_to(destination):
            raise ValueError('Unsafe runtime destination')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, target)
        verified(destination, name, sha)
    if kind == 'music2emo':
        target = destination / 'music2emo.py'
        transformed = relocate_mert_source(target.read_text(encoding='utf-8'))
    elif kind == 'muq':
        target = destination / 'src/muq/muq_mulan/models/audio.py'
        transformed = relocate_muq_source(target.read_text(encoding='utf-8'), reviewed_block)
    else:
        raise ValueError('Unknown runtime kind')
    target.write_text(transformed, encoding='utf-8', newline='\n')
    return {'policy': 'Only verified checkpoint-path construction relocated through explicit worker environment',
            'derived_files_sha256': {target.relative_to(destination).as_posix(): digest(target)}}


def load_muq(base, runtime, device):
    """All sources/weights are verified before executing request-local model code."""
    import os
    import sys
    base, runtime = Path(base).resolve(), Path(runtime).resolve()
    manifests = {name: json.loads(verified(base, name, sha).read_text(encoding='utf-8'))
                 for name, sha in MANIFEST_PINS.items()}
    for item in manifests['assets.json']:
        verified(base, item['file'], item['sha256'])
    manifest = manifests['runtime-manifest.json']
    block = next(p['new'] for p in manifest['patches'] if p['file'] == 'src/muq/muq_mulan/models/audio.py' and 'MuQ(config=' in p['new'])
    relocation = derive_runtime(base / 'runtime', manifest['files_sha256'], runtime, kind='muq', reviewed_block=block)
    os.environ['MUSIC_CLASSIFICATION_MUQ_AUDIO_CONFIG'] = str(base / 'audio-config/config.json')
    import torch
    sys.path.insert(0, str(runtime / 'src'))
    from muq import MuQMuLan
    config = json.loads((base / 'checkpoint/config.json').read_text(encoding='utf-8'))
    config['text_model']['name'] = (base / 'xlm-roberta-base').as_posix()
    model = MuQMuLan(config=config).float().eval()
    state = dict(torch.load(base / 'checkpoint/pytorch_model.bin', map_location='cpu', weights_only=True, mmap=True))
    aliases = {}
    prefix = 'mulan.audio.model.model.conformer.pos_conv_embed.conv.'
    target = model.state_dict()
    for old, new in [('weight_g', 'parametrizations.weight.original0'), ('weight_v', 'parametrizations.weight.original1')]:
        old, new = prefix + old, prefix + new
        if old in state and new in target:
            if new in state:
                raise ValueError('Ambiguous checkpoint keys')
            state[new] = state.pop(old)
            aliases[old] = new
    if set(state) != set(target):
        raise ValueError('Unknown or missing MuQ weights')
    model.load_state_dict(state, strict=True)
    actual = model.state_dict()
    if any(not torch.equal(actual[name], value) for name, value in state.items()):
        raise ValueError('Loaded MuQ tensor differs from checkpoint')
    info = {'name': 'OpenMuQ/MuQ-MuLan-large', 'experimental': True,
            'revision': '2e01c796b71dca71b45251384c04cd7b237c9020',
            'upstream_revision': manifests['source-manifest.json']['revision'],
            'checkpoint_sha256': 'd42ae3f7cb9b66759ee0089ddc70e2f28b130c2d8ba621457358272d32dd0444',
            'artifact_manifests_sha256': MANIFEST_PINS, 'installation': str(base),
            'strict_load': True, 'all_state_tensors_equal': True, 'state_tensors': len(state),
            'legacy_weight_norm_aliases': aliases, 'device': str(device), 'runtime_relocation': relocation}
    del state, target, actual
    return model.to(device), info

"""Private, offline loader for the reviewed local MuQ-MuLan experiment."""

import hashlib
import json
from pathlib import Path
import sys

MANIFEST_PINS = {
    'assets.json': '348bf46d0015c20f5ad920818b58c00a9c46295705fbfeb2501e6ed888c79b9b',
    'runtime-manifest.json': '7520b92ee9bdef758baf2f31a1ba08154fbbdf48b761131b59ad500b548010f2',
    'source-manifest.json': '475574336efc65cefba88675fe04641196c7419a6fa123fdeb5194ce536642a4',
}


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verified(root, name, sha):
    root = Path(root).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root) or digest(path) != sha:
        raise ValueError(f'MuQ integrity check failed: {name}')
    return path


def load_local_model(base, device='cuda'):
    """Called only by the private worker, never in the application interpreter."""
    base = Path(base).resolve()
    # The reviewed runtime's audio constructor embeds this checkpoint-config path.
    if base != Path('C:/Python/Music Classification/work/muq-experiment').resolve():
        raise ValueError('Pinned MuQ runtime requires its reviewed installation path')
    manifests = {name: json.loads(verified(base, name, sha).read_text(encoding='utf-8'))
                 for name, sha in MANIFEST_PINS.items()}
    for item in manifests['assets.json']:
        verified(base, item['file'], item['sha256'])
    for name, sha in manifests['runtime-manifest.json']['files_sha256'].items():
        verified(base / 'runtime', name, sha)
    import torch
    sys.path.insert(0, str(base / 'runtime/src'))
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
                raise ValueError('Ambiguous weight_norm checkpoint keys')
            state[new] = state.pop(old)
            aliases[old] = new
    if set(state) != set(target):
        raise ValueError('Missing or unknown MuQ weights')
    model.load_state_dict(state, strict=True)
    actual = model.state_dict()
    if any(not torch.equal(actual[name], value) for name, value in state.items()):
        raise ValueError('Loaded MuQ tensor differs from checkpoint')
    info = {'name': 'OpenMuQ/MuQ-MuLan-large', 'experimental': True,
            'revision': '2e01c796b71dca71b45251384c04cd7b237c9020',
            'upstream_revision': manifests['source-manifest.json']['revision'],
            'checkpoint_sha256': 'd42ae3f7cb9b66759ee0089ddc70e2f28b130c2d8ba621457358272d32dd0444',
            'artifact_manifests_sha256': MANIFEST_PINS, 'device': str(device),
            'strict_load': True, 'all_state_tensors_equal': True, 'state_tensors': len(state),
            'legacy_weight_norm_aliases': aliases}
    del state, target, actual
    return model.to(device), info

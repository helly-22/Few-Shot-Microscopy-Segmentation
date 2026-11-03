"""Explicit, pre-job acquisition of official model weights."""

import json
from pathlib import Path
import shutil
import urllib.request

from voron_matsam.baseline_data import download_weights, sha256, write_json

SAM2_REVISION = '2b90b9f5ceec907a1c18123530e92e794ad901a4'
SAM2_URL = 'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt'


def fetch(method, output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    registry = output / 'baseline_assets.json'
    entries = json.loads(registry.read_text(encoding='utf-8')) if registry.exists() else {}
    if method in entries:
        entry = entries[method]
        if sha256(output / entry['file']) != entry['sha256']:
            raise ValueError('Previously recorded weights have changed')
        print(f'Already verified: {method}')
        return
    destination = output / {'cellpose_sam':'cpsam', 'cellpose2':'cyto2', 'cellpose3':'cyto3',
                            'sam2':'sam2.1_hiera_small.pt'}.get(method, method)
    if method == 'cellpose_sam':
        download_weights(destination)
        source = 'pinned mouseland/cellpose-sam checkpoint; see configs/upstream.json'
    elif method in ('cellpose2', 'cellpose3'):
        from cellpose import models
        name = 'cyto2' if method == 'cellpose2' else 'cyto3'
        path = Path(models.model_path(name, 0))
        if destination.exists():
            raise FileExistsError(f'Unregistered checkpoint already exists: {destination}')
        shutil.copyfile(path, destination)
        source = f'official Cellpose model_path({name!r}, 0); content SHA256 recorded'
    elif method == 'sam2':
        if destination.exists():
            raise FileExistsError(f'Unregistered checkpoint already exists: {destination}')
        temporary = destination.with_suffix('.download')
        urllib.request.urlretrieve(SAM2_URL, temporary)
        temporary.replace(destination)
        source = SAM2_URL
    else:
        print(f'{method} starts from scratch or needs no weights')
        return
    entries[method] = dict(file=destination.name, sha256=sha256(destination), source=source)
    write_json(registry, entries)

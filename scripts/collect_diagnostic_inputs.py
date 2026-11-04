"""Collect exact held-out inputs and saved masks for a completed baseline run.

Run on the machine that contains the original checkpoint and prepared dataset.
This reads existing artifacts and does not perform inference or training.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    run = json.loads((args.run/'run.json').read_text())
    if run['status'] != 'completed' or args.output.exists():
        raise ValueError('Use a completed run and a new archive output path')
    manifest_path = Path(run['arguments']['manifest'])
    manifest = json.loads(manifest_path.read_text())
    checkpoint_hash = sha256(run['trained_checkpoint'])
    if checkpoint_hash != run['trained_checkpoint_sha256'] or sha256(manifest_path) != run['manifest_sha256']:
        raise ValueError('Checkpoint or manifest checksum differs from recorded run')
    records = {r['image_id']: r for r in manifest['records']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    files = []
    import io
    with tarfile.open(args.output, 'w:gz') as archive:
        archive.add(manifest_path, arcname='manifest.json')
        for sid in manifest['test']:
            record = records[sid]
            members = [(kind, manifest_path.parent/record[kind], 'prepared/'+record[kind])
                       for kind in ('image','mask','raw_mask')]
            members += [('prediction',args.run/'instances'/f'{sid}_masks.tif',f'predictions/{sid}_masks.tif')]
            for kind, path, name in members:
                digest = sha256(path)
                if kind != 'prediction' and digest != record['sha256_'+kind]:
                    raise ValueError(f'Prepared input changed: {sid} {kind}')
                archive.add(path, arcname=name)
                files.append(dict(image_id=sid,kind=kind,path=name,sha256=digest))
            archive.add(args.run/'binary_masks'/f'{sid}_mask.png', arcname=f'binary_predictions/{sid}_mask.png')
        provenance = dict(trained_checkpoint_sha256=checkpoint_hash,manifest_sha256=sha256(manifest_path),files=files)
        data = (json.dumps(provenance,indent=2)+'\n').encode()
        info = tarfile.TarInfo('provenance.json'); info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    print(args.output)


if __name__ == '__main__':
    main()

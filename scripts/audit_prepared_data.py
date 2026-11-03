"""Compare prepared-data GT measurements with reported GT, without a model."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import tifffile

from voron_matsam.baseline_data import load_manifest, write_json
from voron_matsam.jeffries import measure, radial_extents

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--manifest', default='data/processed/paper-bright/manifest.json')
parser.add_argument('--output', default='outputs/prepared_data_audit.json')
args = parser.parse_args()
manifest, records, root = load_manifest(args.manifest)
results = {}
for representation in ('raw', 'processed'):
    rows = []
    for sid in manifest['test']:
        record = records[sid]
        if representation == 'raw':
            raw = np.array(Image.open(root / record['raw_mask']).convert('L'))
            binary = raw >= 128 if manifest['polarity'] == 'bright' else raw < 128
        else:
            binary = tifffile.imread(root / record['mask']) > 0
        rows.append(dict(image_id=sid, **measure(radial_extents(binary))))
    results[representation] = dict(
        means={key:float(np.mean([r[key] for r in rows]))
               for key in ('internal', 'intercepted', 'na', 'g')},
        measurements=rows)
report = dict(test_ids=manifest['test'], results=results,
              paper_gt=dict(internal=60.3, intercepted=31.9, na=752.7, g=6.60),
              conclusion='Reconstructed splits/preprocessing do not exactly match reported GT; not an exact reproduction')
Path(args.output).parent.mkdir(parents=True, exist_ok=True)
write_json(args.output, report)
print(json.dumps({key:value['means'] for key,value in results.items()}, indent=2))

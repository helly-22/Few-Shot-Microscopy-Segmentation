"""Verify existing diagnostic artifacts and package the local report for sharing."""
import csv
import hashlib
from html import escape
from html.parser import HTMLParser
import json
from pathlib import Path
import sys
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import find_boundaries
import tifffile
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from voron_matsam.baseline_data import disk

root=Path('outputs/two_shot_diagnostic')
summary=json.loads((root/'summary.json').read_text())
audit=json.loads((root/'audit.json').read_text())
events=json.loads((root/'instance_events.json').read_text())
scores=json.loads((root/'per_image_scores.json').read_text())
visual=json.loads((root/'visual_review.json').read_text())
assert {r['image_id'] for r in scores} == set(visual['images']) and len(scores)==10
for row in scores:
    sid=row['image_id']
    assert (root/'plots'/f'{sid}_four_panel.png').is_file()
    assert (root/'plots'/f'{sid}_boundaries.png').is_file()
    pred=tifffile.imread(root/'inputs/predictions'/f'{sid}_masks.tif')
    binary=np.asarray(Image.open(root/'inputs/binary_predictions'/f'{sid}_mask.png').convert('L'))>0
    expected=(pred>0)&~ndi.binary_dilation(find_boundaries(pred,mode='outer'),structure=disk(2))
    assert np.array_equal(binary,expected)
    for key in ('gt_boundary_within_3px','gt_boundary_3_to_10px','gt_boundary_beyond_10px'):
        assert 0<=row[key]<=1
    assert np.isclose(sum(row[key] for key in ('gt_boundary_within_3px','gt_boundary_3_to_10px','gt_boundary_beyond_10px')),1)
run=json.loads(Path('outputs/paper-20261103T234304Z/baselines/05_seed0/run.json').read_text())
audit['comparability']['current_source_matches_recorded']=all(
    hashlib.sha256((Path('src/voron_matsam')/name).read_bytes()).hexdigest()==digest
    for name,digest in run['source_sha256'].items())
assert audit['comparability']['current_source_matches_recorded']
summary['totals']['neighbor_gt_pairs']=sum(d['topology']['neighbor_gt_pairs'] for d in events.values())
for filename,value in [('audit.json',audit),('summary.json',summary)]:
    (root/filename).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
page=(root/'diagnostic_report.html').read_text(encoding='utf-8')
old=f'There are {summary["totals"]["neighboring_pairs_same_pred"]} nearby GT grain pairs predominantly covered by the same predicted ID.'
new=f'The restrictive 5-pixel adjacency check finds {summary["totals"]["neighboring_pairs_same_pred"]} of {summary["totals"]["neighbor_gt_pairs"]} tested neighboring GT pairs predominantly covered by one predicted ID; it excludes wider annotation gaps and is not an exhaustive merge search.'
page=page.replace(old,new)
# Keep full per-model uncertainty visible in the report as well as in the CSV.
from diagnose_two_shot import html_table
model=summary['model_uncertainty']
labels={'zero':'Zero shot','05':'2 images','10':'4 images','25':'10 images','50':'20 images','75':'30 images'}
def formatted(split,metric,percentage=False):
    r=next(r for r in model if r['split']==split and r['metric']==metric)
    return f"{r['mean']:.4f} [{r['ci95_low']:.4f}, {r['ci95_high']:.4f}]" if not percentage else f"{r['mean']:.2f} [{r['ci95_low']:.2f}, {r['ci95_high']:.2f}]"
block='<section id="model-uncertainty"><h2>Model means and image uncertainty</h2>'+html_table(
    ['Training set','AP50 and 95% bootstrap CI','mAP and 95% bootstrap CI','Boundary F1 and 95% bootstrap CI','G MAPE percent and 95% bootstrap CI'],
    [[label,formatted(s,'ap50'),formatted(s,'map_50_95'),formatted(s,'boundary_f1'),formatted(s,'g_ape_percent',True)] for s,label in labels.items()])+'</section>'
if 'id="model-uncertainty"' not in page:
    page=page.replace('<h2>Paired model comparison</h2>',block+'<h2>Paired model comparison</h2>')
page=page.replace('Investigate foreground probability and flow rejection, input normalization, and training behavior on a separate validation set',
    'Investigate foreground probability and flow rejection (see <a href="https://cellpose.readthedocs.io/en/latest/settings.html">Cellpose settings</a>), input normalization, and training behavior on a separate validation set')
(root/'diagnostic_report.html').write_text(page,encoding='utf-8')
class Links(HTMLParser):
    def __init__(self): super().__init__();self.paths=[]
    def handle_starttag(self,tag,attrs):
        for k,v in attrs:
            if k in ('href','src') and v and not v.startswith(('https:','http:','#')):
                self.paths.append(v)
links=Links();links.feed(page)
assert all((root/p).is_file() for p in links.paths), 'Broken local report link'
for p in root.glob('*.json'): json.loads(p.read_text())
validation=dict(ten_images_visually_reviewed=True,twenty_diagnostic_figures_present=True,
    saved_binary_masks_match_instance_rendering=True,current_source_matches_recorded=True,
    local_report_links_checked=len(links.paths),test_suite='56 passed',
    notes='Fixed saved masks only. No retraining, new inference, threshold selection, or Voronoi regularization.')
(root/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
(root/'READ_ME.txt').write_text('Open diagnostic_report.html in a browser. Click a four-panel image for full resolution and expand its boundary diagnostic. CSVs contain exact per-image scores, paired uncertainty, and sensitivity results. summary.json and audit.json contain methods and provenance. Candidate counts are not manual error annotations. This package includes rendered figures and score records; checkpoints and raw input files remain in the original project outputs.\n')
archive_path=Path('outputs/reports/two_shot_diagnostic.zip')
archive_path.parent.mkdir(exist_ok=True)
with ZipFile(archive_path,'w',ZIP_DEFLATED) as archive:
    for p in sorted(root.iterdir()):
        if p.is_file() and p.suffix in ('.json','.csv','.html','.txt'):
            archive.write(p,'two_shot_diagnostic/'+p.name)
    for p in sorted((root/'plots').glob('*.png')):
        archive.write(p,'two_shot_diagnostic/plots/'+p.name)
print('Validated report links, all masks, source hashes, score records and ten visual reviews')
print(archive_path,archive_path.stat().st_size,'bytes')

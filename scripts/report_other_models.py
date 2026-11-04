"""Compare completed grain_benchmark_v2 models and build per-image galleries."""
import argparse
import csv
import html
import json
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tifffile
from skimage.segmentation import find_boundaries

from voron_matsam.baseline_data import load_manifest, sha256, write_json
from voron_matsam.diagnostics import overlap_topology

RUNS={'connected_components':'connected_components/test','watershed':'watershed/test',
      'matsam':'matsam/test','unet':'unet/unet','unet_watershed':'unet/unet_watershed',
      'stardist':'stardist/test'}


def gallery(image,gt,pred,method,sid,destination):
    topology=overlap_topology(gt,pred)
    rng=np.random.default_rng(93)
    def colors(labels):
        palette=plt.get_cmap('turbo')(rng.uniform(.03,.97,int(labels.max())+1))[:,:3]
        palette[0]=0
        return palette[labels]
    overlay=image.astype(float)/255
    gb=find_boundaries(gt,mode='inner')
    overlay[gb]=[.8,.85,.9]
    split=np.isin(gt,topology['split_gt_ids'])
    merged=np.isin(pred,topology['merged_pred_ids'])
    overlay[split]=.65*overlay[split]+.35*np.array([1,.57,.05])
    overlay[merged]=.65*overlay[merged]+.35*np.array([.95,.15,.72])
    fig,axes=plt.subplots(2,2,figsize=(12,9))
    for ax,view,title in zip(axes.flat,[image,colors(gt),colors(pred),overlay],
        ['Original',f'GT instances: {topology["gt_count"]}',f'Predicted: {topology["pred_count"]}',
         'Orange: split candidates; magenta: merge candidates']):
        ax.imshow(view)
        ax.set_title(title,fontsize=10)
        ax.axis('off')
    fig.suptitle(f'{method}: {sid}; overlap events are candidates, not verified error annotations')
    fig.tight_layout()
    fig.savefig(destination,dpi=120)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--cellpose-reference',type=Path,required=True)
    parser.add_argument('--partial',action='store_true')
    parser.add_argument('--skip-gallery',action='store_true')
    args=parser.parse_args()
    manifest,records,data=load_manifest(args.manifest)
    expected_hash=sha256(args.manifest)
    out=args.root/('report_partial' if args.partial else 'report')
    out.mkdir(exist_ok=False)
    (out/'plots').mkdir()
    summaries,per_image,pending=[],[],[]
    for method,relative in RUNS.items():
        run=args.root/relative
        if not (run/'summary.json').exists():
            pending.append(method)
            continue
        summary=json.loads((run/'summary.json').read_text())
        metadata=json.loads((run/'run.json').read_text())
        if metadata['status']!='completed' or metadata['manifest_sha256']!=expected_hash:
            raise ValueError(f'Incomplete/different-protocol run: {method}')
        rows=json.loads((run/'per_image.json').read_text())
        if [r['image_id'] for r in rows]!=manifest['test']:
            raise ValueError(f'Test membership/order differs: {method}')
        primary=next(r for r in summary['jeffries'] if r['target']==60 and r['mode']=='superimposed')
        summaries.append(dict(method=method,**{k:summary['metrics'][k] for k in
            ('ap50','map_50_95','boundary_f1','count_error','pq','foreground_dice')},
            g_mape=primary['g_mape'],na_mape=primary['na_mape'],
            ap50_ci95=summary['uncertainty']['ap50'],training=summary['training']))
        for row in rows:
            per_image.append(dict(method=method,**row))
            if not args.skip_gallery:
                sid=row['image_id']
                record=records[sid]
                gallery(np.array(Image.open(data/record['image']).convert('RGB')),
                    tifffile.imread(data/record['mask']),tifffile.imread(run/'instances'/f'{sid}_masks.tif'),
                    method,sid,out/'plots'/f'{method}_{sid}.png')
    if pending and not args.partial:
        raise ValueError(f'Final report still missing models: {pending}')
    reference=json.loads(args.cellpose_reference.read_text())
    if reference['manifest_sha256']!=expected_hash:
        raise ValueError('Cellpose reference uses another manifest')
    cellpose=next(r for r in reference['results'] if r['run']=='05_float32_seed0')
    cellpose_row=dict(method='cellpose_sam_2shot_float32',
        **{k:cellpose['metrics'][k]['mean'] for k in ('ap50','map_50_95','boundary_f1','count_error')},
        g_mape=cellpose['primary_jeffries']['g_mape'],ap50_ci95=cellpose['metrics']['ap50'],
        training='2 images; inference settings fixed before threshold sweeps')
    report=dict(protocol='grain_benchmark_v2',manifest_sha256=expected_hash,test_ids=manifest['test'],
        models=[cellpose_row,*summaries],pending=pending,per_image=per_image,
        gallery_review='automatically generated; manual review pending',
        comparisons='Common test set/evaluator, differing annotation budgets explicitly recorded; '
                    'classical methods selected on five validation images; MatSAM fixed zero-shot settings; '
                    'U-Net/StarDist refit on 30 after development on 25 and selection on five',
        precision='float32 for U-Net; SAM ViT-L default float32; StarDist TensorFlow float32',
        source_sha256=sha256(__file__))
    write_json(out/'results.json',report)
    columns=['method','ap50','map_50_95','boundary_f1','count_error','g_mape']
    with (out/'comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,columns,extrasaction='ignore')
        writer.writeheader()
        writer.writerows(report['models'])
    with (out/'per_image_scores.csv').open('w',newline='') as f:
        fields=list(per_image[0])
        writer=csv.DictWriter(f,fields)
        writer.writeheader()
        writer.writerows(per_image)
    body=['<!doctype html><meta charset="utf-8"><title>Other segmentation models</title>',
        '<style>body{font:16px system-ui;max-width:1200px;margin:30px auto;padding:16px}'
        'table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:9px}'
        'img{max-width:100%;height:auto}</style><h1>Grain segmentation model comparison</h1>',
        '<p>Ten fixed test images and one evaluator. One training seed per learned model; '
        'intervals describe image variability, not seed variance. Budgets/settings differ; '
        'these are documented reconstructions and extensions, not confirmed author experiments.</p>',
        '<p>Pending: '+html.escape(', '.join(pending) or 'none')+'</p>',
        '<table><tr><th>Model</th><th>AP50 [95% CI]</th><th>mAP</th><th>Boundary F1</th>'
        '<th>Signed count error</th><th>G MAPE %</th></tr>']
    for row in report['models']:
        ci=row['ap50_ci95']
        g='undefined' if row['g_mape'] is None else f'{row["g_mape"]:.2f}'
        body.append(f'<tr><td>{row["method"]}</td><td>{row["ap50"]:.4f} '
            f'[{ci["ci95_low"]:.4f}, {ci["ci95_high"]:.4f}]</td>'
            f'<td>{row["map_50_95"]:.4f}</td><td>{row["boundary_f1"]:.4f}</td>'
            f'<td>{row["count_error"]:+.1f}</td><td>{g}</td></tr>')
    body.append('</table><p><a href="results.json">Full metrics and protocol</a> | '
        '<a href="per_image_scores.csv">Per-image CSV</a></p>')
    if not args.skip_gallery:
        for row in per_image:
            name=f'{row["method"]}_{row["image_id"]}'
            body.append(f'<details><summary>{name}: AP50 {row["ap50"]:.4f}</summary>'
                        f'<img src="plots/{name}.png"></details>')
    (out/'report.html').write_text('\n'.join(body),encoding='utf-8')
    archive=args.root/('benchmark_partial.zip' if args.partial else 'benchmark_results.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(out.rglob('*')):
            if path.is_file(): bundle.write(path,'report/'+path.relative_to(out).as_posix())
    print(f'Report: {out/"report.html"}; archive: {archive}')


if __name__=='__main__':
    main()

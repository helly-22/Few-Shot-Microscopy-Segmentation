"""Diagnose saved Cellpose-SAM predictions without training or test-set tuning.

Run from the project root with the prepared diagnostic input archive unpacked:
    python scripts/diagnose_two_shot.py
"""
import argparse
import csv
import hashlib
import html
import importlib.util
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.segmentation import find_boundaries
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from voron_matsam.diagnostics import overlap_topology, boundary_distances, boundary_distance_summary, bootstrap_mean, paired_mean_comparison
from voron_matsam.metrics import evaluate_instances
from voron_matsam.jeffries import evaluate_pair
from voron_matsam.baseline_data import disk

PAPER = 'https://arxiv.org/html/2604.18957v1'


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def label_colors(labels):
    rng = np.random.default_rng(93)
    palette = plt.get_cmap('turbo')(rng.uniform(.03, .97, int(labels.max())+1))[:, :3]
    palette[0] = 0
    return palette[labels]


def paint(rgb, mask, color, alpha=1):
    rgb[mask] = (1-alpha)*rgb[mask] + alpha*np.asarray(color)


def panel(image, gt, pred, topology, row, destination):
    original = image.astype(float)/255
    overlay = original.copy()
    gb = find_boundaries(gt, mode='inner')
    pb = find_boundaries(pred, mode='inner')
    paint(overlay, gb, [.8,.85,.9], .5)
    split = np.isin(gt, topology['split_gt_ids'])
    merge = np.isin(pred, topology['merged_pred_ids'])
    paint(overlay, merge, [.95,.15,.72], .23)
    paint(overlay, split, [1,.57,.05], .32)
    paint(overlay, ndi.binary_dilation(gb & split, iterations=1), [1,.57,.05])
    paint(overlay, ndi.binary_dilation(pb & merge, iterations=1), [.95,.15,.72])
    fig, axes = plt.subplots(2, 2, figsize=(16, 12.6))
    views = [original, label_colors(gt), label_colors(pred), overlay]
    titles = ['Original microscopy', f'Ground truth grain instances  n={row["gt_count"]}',
              f'Predicted instances  n={row["pred_count"]}', 'Overlap candidates on microscopy']
    for ax, view, title in zip(axes.flat, views, titles):
        ax.imshow(view); ax.set_title(title, fontsize=13, pad=9); ax.axis('off')
    axes[1,1].legend(handles=[Patch(color='#ff910d', label='GT grain split candidate'),
                             Patch(color='#f226b8', label='Predicted merge candidate')],
                     loc='lower left', fontsize=9, framealpha=.93)
    fig.suptitle(f'{row["image_id"]}  |  2-shot final checkpoint  |  AP50 {row["ap50"]:.3f}  |  '
                 f'split GT {row["split_gt_count"]}/{row["gt_count"]}  |  '
                 f'merged predictions {row["merged_pred_count"]}/{row["pred_count"]}', fontsize=15, y=.985)
    fig.text(.5, .012, 'Instance colors encode local IDs and do not imply matched objects. '
             'Overlap >=10% of both objects; candidates can coexist and need visual interpretation.',
             ha='center', fontsize=10)
    fig.subplots_adjust(left=.015, right=.985, bottom=.04, top=.94, hspace=.16, wspace=.04)
    fig.savefig(destination, dpi=170, facecolor='white'); plt.close(fig)


def boundary_panel(image, gt, pred, sid, summary, destination):
    gb, pb, dp, _ = boundary_distances(gt, pred)
    overlay = image.astype(float)/255
    # Render GT boundaries by distance to the nearest predicted boundary.
    paint(overlay, ndi.binary_dilation(pb), [.1,.8,1], .7)
    paint(overlay, ndi.binary_dilation(gb & (dp <= 3)), [.25,.9,.35])
    paint(overlay, ndi.binary_dilation(gb & (dp > 3) & (dp <= 10)), [1,.8,.05])
    paint(overlay, ndi.binary_dilation(gb & (dp > 10)), [1,.1,.1])
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.7), gridspec_kw={'width_ratios':[1.6,1]})
    axes[0].imshow(overlay); axes[0].axis('off'); axes[0].set_title(sid+'  Boundary support', fontsize=13)
    axes[0].legend(handles=[Patch(color='#40e659', label='GT boundary within 3 px'),
                            Patch(color='#ffcc0d', label='GT boundary >3 to 10 px'),
                            Patch(color='#ff1a1a', label='GT boundary beyond 10 px'),
                            Patch(color='#1accff', label='Predicted boundary')], loc='lower left', fontsize=9)
    curve = summary['curve']
    for key, label in [('recall','GT boundary recall'),('precision','Predicted boundary precision'),('f1','Boundary F1')]:
        axes[1].plot([r['tolerance_pixels'] for r in curve], [r[key] for r in curve], marker='o', label=label)
    axes[1].set(xlabel='Euclidean boundary tolerance in pixels', ylabel='Score', ylim=(0,1),
                title='Thin boundary distance diagnostic')
    axes[1].grid(alpha=.25); axes[1].legend(fontsize=9, loc='lower right')
    fig.text(.5,.025,'A boundary farther than 10 px is unsupported at this tolerance; '
             'distance alone cannot prove a missing physical boundary. Frame band 10 px excluded.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.055,1,1)); fig.savefig(destination,dpi=155,facecolor='white'); plt.close(fig)


def html_table(headers, rows):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(str(v))+'</th>' for v in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, default=Path('outputs/paper-20261103T234304Z'))
    parser.add_argument('--output', type=Path, default=Path('outputs/two_shot_diagnostic'))
    parser.add_argument('--skip-plots', action='store_true', help='Reuse already rendered plots while rebuilding numerical outputs')
    args = parser.parse_args()
    out, runroot = args.output, args.run_root
    inputs = out/'inputs'; plots = out/'plots'; plots.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((inputs/'manifest.json').read_text())
    records = {r['image_id']: r for r in manifest['records']}
    ids = manifest['test']; seed = 20261104
    resamples = np.random.default_rng(seed).integers(0, len(ids), size=(20000,len(ids)))
    config, per_model, all_gape = {}, {}, {}
    for split in ['zero','05','10','25','50','75']:
        folder = runroot/'baselines'/f'{split}_seed0'
        config[split] = json.loads((folder/'run.json').read_text())
        rows = json.loads((folder/'segmentation_per_image.json').read_text())
        assert [r['image_id'] for r in rows] == ids
        per_model[split] = {r['image_id']:r for r in rows}
        jj = json.loads((folder/'jeffries_per_image.json').read_text())
        all_gape[split] = {}
        for r in jj:
            if r['mode']=='superimposed' and r['target']==60:
                all_gape[split][r['image_id']] = abs(r['pred']['g']-r['gt']['g'])/abs(r['gt']['g'])*100
    provenance = json.loads((inputs/'provenance.json').read_text())
    assert provenance['trained_checkpoint_sha256'] == config['05']['trained_checkpoint_sha256']
    for r in provenance['files']:
        assert hashlib.sha256((inputs/r['path']).read_bytes()).hexdigest() == r['sha256']
    rows, details, curves, sensitivity, measurement_rows = [], {}, [], [], []
    reference_spec = importlib.util.spec_from_file_location('released_jeffries', 'external/ASTM-Grain-Size-Estimator/4_evaluate_jeffries.py')
    reference = importlib.util.module_from_spec(reference_spec); reference_spec.loader.exec_module(reference)
    normalized_gt = out/'normalized_gt'; normalized_gt.mkdir(exist_ok=True)
    for sid in ids:
        rec = records[sid]
        image = np.asarray(Image.open(inputs/'prepared'/rec['image']).convert('RGB'))
        gt = tifffile.imread(inputs/'prepared'/rec['mask'])
        pred = tifffile.imread(inputs/'predictions'/f'{sid}_masks.tif')
        assert image.shape[:2] == gt.shape == pred.shape
        topology = overlap_topology(gt, pred)
        boundary = boundary_distance_summary(gt, pred)
        supplemental = evaluate_instances(gt, pred)
        row = dict(per_model['05'][sid])
        assert row['gt_count']==topology['gt_count'] and row['pred_count']==topology['pred_count']
        row.update(split_gt_count=len(topology['split_gt_ids']), merged_pred_count=len(topology['merged_pred_ids']),
                   gt_in_merge_count=len(topology['gt_in_merge_ids']), split_gt_rate=topology['split_gt_rate'],
                   merged_pred_rate=topology['merged_pred_rate'], gt_in_merge_rate=topology['gt_in_merge_rate'],
                   majority_uncovered_gt_count=len(topology['majority_uncovered_gt_ids']),
                   majority_uncovered_gt_rate=len(topology['majority_uncovered_gt_ids'])/topology['gt_count'],
                   gt_foreground_predicted_background=float(((gt>0)&(pred==0)).sum()/(gt>0).sum()),
                   majority_extra_pred_count=len(topology['majority_extra_pred_ids']),
                   nearly_uncovered_gt_count=len(topology['nearly_uncovered_gt_ids']),
                   fully_uncovered_gt_count=len(topology['fully_uncovered_gt_ids']),
                   neighboring_pairs_in_same_pred=len(topology['neighboring_pairs_in_same_prediction']),
                   pq=supplemental['pq'], matched_dice=supplemental['matched_dice'],
                   gt_boundary_within_3px=boundary['gt_boundary_within_3px'],
                   gt_boundary_3_to_10px=boundary['gt_boundary_3_to_10px'],
                   gt_boundary_beyond_10px=boundary['gt_boundary_beyond_10px'],
                   gt_to_pred_median_px=boundary['gt_to_pred_median_px'],
                   gt_to_pred_p90_px=boundary['gt_to_pred_p90_px'], g_ape_percent=all_gape['05'][sid])
        rows.append(row); details[sid]={'topology':topology,'boundary':boundary}
        curves += [dict(image_id=sid,**r) for r in boundary['curve']]
        for threshold in (.05,.1,.2):
            t = topology if threshold==.1 else overlap_topology(gt,pred,threshold)
            sensitivity.append(dict(image_id=sid,threshold=threshold,split_gt_count=len(t['split_gt_ids']),
                                    merged_pred_count=len(t['merged_pred_ids']),split_gt_rate=t['split_gt_rate'],
                                    merged_pred_rate=t['merged_pred_rate']))
        binary_path=inputs/'binary_predictions'/f'{sid}_mask.png'
        binary=np.asarray(Image.open(binary_path).convert('L'))>0
        expected_binary=(pred>0)&~ndi.binary_dilation(find_boundaries(pred,mode='outer'),structure=disk(2))
        assert np.array_equal(binary,expected_binary), 'Saved binary mask differs from recorded instance rendering'
        truth_path=normalized_gt/f'{sid}.png'
        Image.fromarray((gt>0).astype(np.uint8)*255).save(truth_path)
        # Check the released mathematical implementation on identical binary inputs.
        ours = evaluate_pair(gt>0,binary,targets=[60])[0]
        released_gt=reference.calculate_jeffries(str(truth_path),60)
        released_pred=reference.calculate_jeffries(str(binary_path),60,'superimposed',released_gt['r'])
        for label, value in [('gt',released_gt),('pred',released_pred)]:
            assert all(np.isclose(ours[label][key],value[key],atol=1e-9) for key in ('internal','intercepted','na','g'))
        for calibration in (2.26,2.25):
            for representation in ('processed','raw'):
                raw=np.asarray(Image.open(inputs/'prepared'/rec['raw_mask']).convert('L'))
                gt_binary=(gt>0) if representation=='processed' else raw>=128
                result=evaluate_pair(gt_binary,binary,calibration,targets=[60])[0]
                g,p=result['gt'],result['pred']
                measurement_rows.append(dict(image_id=sid,pixels_per_um=calibration,gt_representation=representation,
                    radius_px=g['radius_px'],gt_g=g['g'],pred_g=p['g'],gt_density=g['na'],pred_density=p['na'],
                    g_ape_percent=abs(p['g']-g['g'])/abs(g['g'])*100,
                    density_ape_percent=abs(p['na']-g['na'])/g['na']*100))
        assert np.isclose(ours['pred']['g'],released_pred['g'])
        assert np.isclose(row['g_ape_percent'],abs(ours['pred']['g']-ours['gt']['g'])/abs(ours['gt']['g'])*100)
        if not args.skip_plots:
            panel(image,gt,pred,topology,row,plots/f'{sid}_four_panel.png')
            boundary_panel(image,gt,pred,sid,boundary,plots/f'{sid}_boundaries.png')
        print(f"{sid}: splits {row['split_gt_count']}, merges {row['merged_pred_count']}, GT beyond10px {row['gt_boundary_beyond_10px']:.3f}",flush=True)

    aggregate={key:bootstrap_mean([r[key] for r in rows],resamples) for key in [
        'ap50','map_50_95','boundary_f1','count_error','pq','matched_dice','split_gt_rate','merged_pred_rate',
        'gt_in_merge_rate','majority_uncovered_gt_rate','gt_foreground_predicted_background',
        'gt_boundary_within_3px','gt_boundary_3_to_10px','gt_boundary_beyond_10px','g_ape_percent']}
    model_ci, paired = [], []
    for split, model in per_model.items():
        for metric in ('ap50','map_50_95','boundary_f1','count_error','g_ape_percent'):
            values = np.array([all_gape[split][sid] if metric=='g_ape_percent' else model[sid][metric] for sid in ids])
            model_ci.append(dict(split=split,metric=metric,**bootstrap_mean(values,resamples)))
            if split not in ('zero','05'):
                left=np.array([all_gape['05'][sid] if metric=='g_ape_percent' else per_model['05'][sid][metric] for sid in ids])
                paired.append(dict(comparison='05 minus '+split,metric=metric,**paired_mean_comparison(left,values,resamples)))
    common_fields=['epochs','batch_size','aug_multiplier','pixels_per_um','jeffries_gt','seed','cpu','float32','limit_test_images']
    comparability=dict(
        common_arguments={k:config['05']['arguments'][k] for k in common_fields},
        common_arguments_equal=all(all(config[s]['arguments'][k]==config['05']['arguments'][k] for k in common_fields) for s in ['10','25','50','75']),
        identical_manifest_hash=len({v['manifest_sha256'] for v in config.values()})==1,
        identical_initial_checkpoint=len({v['checkpoint_sha256'] for v in config.values()})==1,
        identical_recorded_source=len({json.dumps(v['source_sha256'],sort_keys=True) for v in config.values()})==1,
        current_source_matches_recorded=all(hashlib.sha256((Path('src/voron_matsam')/name).read_bytes()).hexdigest()==digest for name,digest in config['05']['source_sha256'].items()),
        identical_package_versions=len({json.dumps(v['environment']['packages'],sort_keys=True) for v in config.values()})==1,
        same_ordered_test_ids=ids,
        nested_training_sets=all(set(manifest['fractions'][a])<=set(manifest['fractions'][b]) for a,b in zip(['05','10','25','50'],['10','25','50','75'])),
        train_test_disjoint=all(not set(v)&set(ids) for v in manifest['fractions'].values()),
        hardware={s:v['environment']['gpu'] for s,v in config.items()},
        nominal_optimizer_batches={s:100*config[s]['augmented_images']//2 for s in ['05','10','25','50','75']},
        selection='Only completed 2-shot run; final epoch checkpoint, not validation-selected best. The test-set winner is descriptive only.')
    audit=dict(calibration_matches_paper_numeric_value=True,pixels_per_um=2.26,
        released_measurement_functions_match_on_identical_binary_inputs=True,
        circle='Image center floor(width/2), floor(height/2); radius 10 px in steps of 2 until 60 internal grains, capped by inscribed radius',
        formulas='NA=(internal+intercepted/2)/(pi*r_px^2/(pixels_per_um^2*1e6)); G=3.321928*log10(NA)-2.954',
        component_connectivity=8,
        differences=['Processed GT is integer labels converted with >0, rather than 8-bit loading and thresholding label intensities.',
          'Exact manifest IDs replace ambiguous filename-prefix matching.',
          'Unattainable targets and empty predictions are retained, rather than skipped.',
          'Original author split, physical scale-bar provenance, and exact foreground preparation remain unconfirmed.'],
        checkpoint_provenance=provenance,comparability=comparability)
    summary=dict(bootstrap=dict(seed=seed,resamples=20000,unit='whole micrograph',interval='95% percentile',
        limitations='Conditional on one training seed; n=10, physical specimen independence unknown; no pixel/grain independence assumption or training-seed uncertainty.'),
        aggregate=aggregate,model_uncertainty=model_ci,paired_differences=paired,
        totals=dict(gt=sum(r['gt_count'] for r in rows),pred=sum(r['pred_count'] for r in rows),
                    split_gt=sum(r['split_gt_count'] for r in rows),merged_pred=sum(r['merged_pred_count'] for r in rows),
                    gt_in_merge=sum(r['gt_in_merge_count'] for r in rows),
                    majority_uncovered_gt=sum(r['majority_uncovered_gt_count'] for r in rows),
                    nearly_uncovered_gt=sum(r['nearly_uncovered_gt_count'] for r in rows),
                    fully_uncovered_gt=sum(r['fully_uncovered_gt_count'] for r in rows),
                    neighbor_gt_pairs=sum(d['topology']['neighbor_gt_pairs'] for d in details.values()),
                    neighboring_pairs_same_pred=sum(r['neighboring_pairs_in_same_pred'] for r in rows)),
        methods=dict(overlap='Each edge requires >=10% of both GT and predicted area; split=GT degree>=2, merge=predicted degree>=2; events can coexist.',
          boundary='Four-neighbor thin label-transition boundary, Euclidean distance, 10 px frame band excluded; >10 px unsupported is a proxy, not proof of absence.',
          neighbor='GT labels expanded at most 5 px across background; 4-connected neighboring label pairs; each GT has >=50% area in same dominant predicted ID.'))
    write_csv(out/'per_image_scores.csv',rows); write_json(out/'per_image_scores.json',rows)
    write_csv(out/'boundary_tolerance_scores.csv',curves); write_csv(out/'overlap_threshold_sensitivity.csv',sensitivity)
    write_csv(out/'measurement_sensitivity.csv',measurement_rows); write_csv(out/'paired_model_differences.csv',paired)
    write_csv(out/'model_uncertainty.csv',model_ci); write_json(out/'summary.json',summary)
    write_json(out/'audit.json',audit); write_json(out/'instance_events.json',details)
    # A static scientific plot shows the effect of pairing and its practical scale.
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for split in per_model:
        values=[per_model[split][sid]['ap50'] for sid in ids]
        axes[0].plot(range(10),values,marker='o',alpha=.7,label=split)
    axes[0].set_xticks(range(10),ids,rotation=45); axes[0].set(ylabel='AP50',title='Same ten images across all models')
    axes[0].legend(title='Split',fontsize=8,ncol=2); axes[0].grid(alpha=.2)
    p=[r for r in paired if r['metric']=='ap50']
    for i,r in enumerate(p):
        axes[1].plot([r['paired_t_ci95_low'],r['paired_t_ci95_high']],[i-.1,i-.1],color='#9a9a9a',linewidth=2,label='Paired t interval' if i==0 else None)
        axes[1].plot([r['ci95_low'],r['ci95_high']],[i,i],color='#275d83',linewidth=2,label='Paired bootstrap' if i==0 else None)
        axes[1].plot(r['mean'],i,'o',color='#275d83')
    axes[1].set_yticks(range(len(p)),[r['comparison'] for r in p]); axes[1].axvline(0,color='black',linestyle='--')
    axes[1].set(xlabel='Paired mean AP50 difference',title='95% paired image intervals');axes[1].grid(alpha=.2);axes[1].legend(fontsize=8)
    fig.tight_layout();fig.savefig(plots/'model_comparison.png',dpi=170);plt.close(fig)
    ci=lambda x:f"{x['mean']:.4f} [{x['ci95_low']:.4f}, {x['ci95_high']:.4f}]"
    visual_review=json.loads((out/'visual_review.json').read_text())['images'] if (out/'visual_review.json').exists() else {}
    totals=summary['totals']; diff=next(r for r in paired if r['comparison']=='05 minus 10' and r['metric']=='ap50')
    sections=['<!doctype html><html lang="en"><meta charset="utf-8"><title>Two shot segmentation diagnostic</title>',
      '<style>body{font:16px/1.55 Arial,sans-serif;max-width:1280px;margin:40px auto;padding:0 24px;color:#19232b}h1,h2,h3{color:#111}img{max-width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border:1px solid #ccd3d9;padding:8px;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#e8edf1}.scroll{overflow:auto}p{max-width:1050px}nav a{margin-right:15px}figure{margin:28px 0}figcaption{font-size:14px;color:#38434a}details{margin:16px 0}a{color:#155d90}</style>',
      '<h1>Two shot Cellpose SAM segmentation diagnostic</h1>',
      '<p>The saved final 2-shot checkpoint was verified by SHA256. Its predictions were generated after checkpoint reload. No model was retrained, no new checkpoint was selected, and no Voronoi loss was added. This is the only completed 2-shot training seed, not a validation-selected best checkpoint.</p>',
      '<nav>'+''.join(f'<a href="#{sid}">{sid}</a>' for sid in ids)+'</nav>',
      '<h2>Answers to the diagnostic questions</h2>',
      f'<p><b>Splits and merges.</b> Across ten images, {totals["split_gt"]} of {totals["gt"]} GT grains are split candidates, while {totals["merged_pred"]} of {totals["pred"]} predictions are merge candidates. {totals["gt_in_merge"]} GT grains participate in candidate merges. The restrictive 5-pixel adjacency check finds {totals["neighboring_pairs_same_pred"]} of {totals["neighbor_gt_pairs"]} tested neighboring GT pairs predominantly covered by one predicted ID; it excludes wider annotation gaps and is not an exhaustive merge search. These counts describe overlapping events, not independent samples. See threshold sensitivity and all overlays below.</p>',
      f'<p><b>Missed and shrunken foreground are separate failure modes.</b> {totals["majority_uncovered_gt"]} of {totals["gt"]} GT grains have more than half their area predicted as background; {totals["nearly_uncovered_gt"]} have less than 10% foreground coverage and {totals["fully_uncovered_gt"]} have zero coverage. The mean fraction of GT foreground pixels predicted as background is {aggregate["gt_foreground_predicted_background"]["mean"]:.1%}. A negative grain count error by itself must not be interpreted as evidence of widespread merging. The visual masks show missing foreground regions as well as splits and contour displacement.</p>',
      f'<p><b>Missing versus displaced boundaries.</b> Mean GT boundary fractions are {aggregate["gt_boundary_within_3px"]["mean"]:.1%} within 3 px, {aggregate["gt_boundary_3_to_10px"]["mean"]:.1%} farther than 3 but within 10 px, and {aggregate["gt_boundary_beyond_10px"]["mean"]:.1%} unsupported within 10 px. The last group is consistent with missing or more severely misplaced contours; the distance test alone does not establish cause. Neighbor merge evidence helps identify missing separators. These thin-boundary diagnostics differ from the dilated paper F1.</p>',
      '<p><b>Measurement audit.</b> Calibration is 2.26 px per micrometre, numerically matching the paper. Center, radius search, 8-connected components, half-counted intercepted grains, physical-area conversion, and G formula agree with the released functions when given identical binary inputs; this was checked on all ten images. The GT input representation, exact original split, and independent physical calibration are not established as identical. The audit and sensitivity CSV make those differences explicit.</p>',
      f'<p><b>Training comparability.</b> Common arguments: {comparability["common_arguments_equal"]}; same manifest, initial checkpoint, source and package versions: {comparability["identical_manifest_hash"] and comparability["identical_initial_checkpoint"] and comparability["identical_recorded_source"] and comparability["identical_package_versions"]}. All five training sets are nested and disjoint from the same ordered ten-image test set. GPU models differ. Equal epochs and augmentation multiplier mean different numbers of optimizer batches; see audit.json. Test labels were not used for optimization or checkpoint selection.</p>',
      f'<p><b>Uncertainty.</b> Mean 2-shot AP50 with image bootstrap 95% interval is {ci(aggregate["ap50"])}. The paired 2-shot minus 4-shot bootstrap result is {ci(diff)}. The paired t interval is [{diff["paired_t_ci95_low"]:.5f}, {diff["paired_t_ci95_high"]:.5f}], and the exact two-sided sign-flip p value is {diff["exact_sign_flip_p_two_sided"]:.4f}. These small-sample methods give different evidence near zero; the slight bootstrap exclusion of zero should not be treated as a robust model-ranking claim. There are only ten micrographs and one training seed; do not treat grains or pixels as independent test units.</p>',
      '<h2>Per image scores and topology</h2>',
      html_table(['Image','AP50','mAP','Paper boundary F1','GT / Pred','Split GT','Merged Pred','GT in merges','Uncovered GT','G APE %'],
         [[r['image_id'],f"{r['ap50']:.4f}",f"{r['map_50_95']:.4f}",f"{r['boundary_f1']:.4f}",f"{r['gt_count']} / {r['pred_count']}",r['split_gt_count'],r['merged_pred_count'],r['gt_in_merge_count'],r['majority_uncovered_gt_count'],f"{r['g_ape_percent']:.2f}"] for r in rows]),
      '<p>Uncovered GT means more than half its pixels are predicted as background. Split and merge candidates require each overlap to cover at least 10% of both objects. Orange highlights split GT instances; magenta highlights merged predicted instances. PQ and matched Dice are also in the CSV; matched Dice excludes unmatched objects and must be read with PQ.</p>',
      '<h2>Paired model comparison</h2><img src="plots/model_comparison.png" alt="Per image AP50 and paired bootstrap intervals">',
      html_table(['Comparison','Metric','Mean difference','Bootstrap 95% low','Bootstrap 95% high','Paired t 95% low','Paired t 95% high','Sign flip p'],[[r['comparison'],r['metric'],f"{r['mean']:.5f}",f"{r['ci95_low']:.5f}",f"{r['ci95_high']:.5f}",f"{r['paired_t_ci95_low']:.5f}",f"{r['paired_t_ci95_high']:.5f}",f"{r['exact_sign_flip_p_two_sided']:.4f}"] for r in paired if r['metric']=='ap50']),
      '<p>Each bootstrap draw resamples the same ten image positions for both models, preserving their pairing. 20000 draws, seed 20261104, percentile intervals. The t interval uses 9 degrees of freedom and assumes a regular distribution of paired differences; the exact sign-flip reference enumerates all 1024 sign patterns and requires sign exchangeability under the null. Intervals describe image sampling conditional on the fitted models, assume exchangeable micrographs, and omit physical-specimen correlation and training-seed variability. Comparisons are exploratory and unadjusted for multiplicity.</p>',
      '<h2>Measurement and configuration records</h2><p>Machine-readable outputs: <a href="per_image_scores.csv">per-image scores</a>, <a href="model_uncertainty.csv">model uncertainty</a>, <a href="paired_model_differences.csv">paired differences</a>, <a href="boundary_tolerance_scores.csv">boundary tolerance curves</a>, <a href="overlap_threshold_sensitivity.csv">overlap sensitivity</a>, <a href="measurement_sensitivity.csv">calibration and mask sensitivity</a>, and <a href="audit.json">full audit and provenance</a>.</p>',
      '<p>Primary grain-size error uses the shared GT-derived circle at target 60. The headline paper result at independent target 90 is a different setting. Converting label IDs to an 8-bit image and applying a foreground intensity threshold is not equivalent to using GT labels &gt;0. See the existing source audit for the reported zero-shot density inconsistency.</p>',
      html_table(['GT representation','Pixels per micrometre','Mean G APE %','Mean density APE %'],[[representation,f'{calibration:.2f}',f"{np.mean([r['g_ape_percent'] for r in measurement_rows if r['gt_representation']==representation and r['pixels_per_um']==calibration]):.3f}",f"{np.mean([r['density_ape_percent'] for r in measurement_rows if r['gt_representation']==representation and r['pixels_per_um']==calibration]):.3f}"] for representation in ['processed','raw'] for calibration in [2.26,2.25]]),
      '<h2>Implications before adding Voronoi regularization</h2><p>The observed undercount is dominated by missing or contracted foreground, with localized fragmentation. The evidence does not support widespread full neighboring-grain fusion as the main error. A geometry term should not be assumed to recover grains absent from the foreground prediction. Investigate foreground probability and flow rejection, input normalization, and training behavior on a separate validation set before choosing a regularizer. Do not tune thresholds or the geometry loss against these ten inspected test images; keep later comparisons paired and include multiple training seeds.</p>',
      '<h2>All ten image inspections</h2>']
    for r in rows:
        sid=r['image_id']; sections += [f'<section id="{sid}"><h3>{sid}</h3>',
          f'<figure><a href="plots/{sid}_four_panel.png"><img loading="lazy" src="plots/{sid}_four_panel.png" alt="Original microscopy, GT, prediction and split merge overlay for {sid}"></a><figcaption>{sid}: AP50 {r["ap50"]:.4f}; {r["split_gt_count"]} split GT candidates; {r["merged_pred_count"]} merged prediction candidates; {r["neighboring_pairs_in_same_pred"]} neighboring GT pairs in the same prediction. Click for full resolution.</figcaption></figure>',
          '<p><b>Visual inspection.</b> '+html.escape(visual_review.get(sid,'See the full-resolution panel and boundary diagnostic.'))+'</p>',
          f'<details><summary>Boundary distance inspection for {sid}</summary><img loading="lazy" src="plots/{sid}_boundaries.png" alt="Boundary displacement and support diagnostic for {sid}"></details></section>']
    sections += [f'<h2>Sources</h2><p><a href="{PAPER}">Mueez and Vyas 2026</a>, sections 3.4 to 3.6 and Tables 1 to 3; pinned released implementation 80f305e9ae10d3f845914f6d773ecd2f8fcea651; saved run metadata and per-image outputs under outputs/paper-20261103T234304Z. Checkpoint SHA256: {provenance["trained_checkpoint_sha256"]}.</p></html>']
    (out/'diagnostic_report.html').write_text('\n'.join(sections),encoding='utf-8')
    print('SUMMARY',json.dumps(summary['totals']),flush=True)
    print('AP CI',aggregate['ap50']);print('2 minus 4',diff)
    print('Written',out/'diagnostic_report.html')


if __name__=='__main__':
    main()

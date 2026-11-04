"""Official MatSAM prompt/mask code with SAM ViT-L and explicit instance export.

The upstream notebook defaults to ViT-H; ViT-L is the grain-paper reconstruction.
The prompt class is unchanged, including its released non-square XY convention.
"""
import ast
import importlib
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
from skimage import measure, morphology

from baseline.benchmark import evaluate_test
from baseline.models.sam2 import compose_masks
from voron_matsam.baseline_data import sha256, write_json

UPSTREAM_COMMIT='cbea7edaada991d88d7dfee656bd7e3dac09863f'


def prompt_class(upstream):
    """Load only the two used classes; unrelated upstream Gala metrics stay out.

    AST extraction preserves the class definitions verbatim and avoids importing
    Gala, which is unused for prompt generation and incompatible with this stack.
    """
    context=dict(np=np,cv2=cv2,morphology=morphology,measure=measure,label=measure.label)
    for relative,name in [('utils/metrics.py','PostPrecess'),('utils/prompt_generator.py','PromptGenerator')]:
        path=upstream/relative
        nodes=[node for node in ast.parse(path.read_text(encoding='utf-8')).body
               if isinstance(node,ast.ClassDef) and node.name==name]
        if len(nodes)!=1:
            raise ValueError(f'Expected one official {name} definition')
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),context)
    return context['PromptGenerator']


def run_matsam(args):
    import torch
    if not args.cpu and not torch.cuda.is_available():
        raise RuntimeError('MatSAM requires an allocated CUDA GPU')
    upstream=Path(args.upstream).resolve(strict=True)
    commit=subprocess.run(['git','-C',str(upstream),'rev-parse','HEAD'],
                           capture_output=True,text=True,check=True).stdout.strip()
    if commit!=UPSTREAM_COMMIT:
        raise ValueError('Official MatSAM checkout differs from the audited commit')
    checkpoint=Path(args.checkpoint).resolve(strict=True)
    registry=json.loads((checkpoint.parent/'matsam_asset.json').read_text())
    if sha256(checkpoint)!=registry['sha256'] or registry['model']!='vit_l':
        raise ValueError('Unverified SAM ViT-L checkpoint')
    sys.path.insert(0,str(upstream))
    sam_module=importlib.import_module('utils.segment_anything_')
    prompter=prompt_class(upstream)
    device='cpu' if args.cpu else 'cuda'
    sam=sam_module.sam_model_registry['vit_l'](checkpoint=str(checkpoint))
    sam.to(device=device)
    sam.eval()
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    (out/'prompts').mkdir()
    parameters=dict(layers=0,scales=3,n_per_side_base=32,method_type=1,
        pred_iou_thresh=.90,stability_score_thresh=.92,box_nms_thresh=.80,
        crop_nms_thresh=.80,points_per_batch=32,min_mask_region_area=0)
    info=dict(upstream_commit=commit,checkpoint_sha256=registry['sha256'],model='SAM ViT-L',
        optimizer_updates=0,training_images=[],annotation_images_for_selection=0,
        parameters=parameters,prompt_class_sha256=sha256(upstream/'utils/prompt_generator.py'),
        prompt_policy='official released Canny ROI centroids plus grid; released XY normalization retained',
        instance_export='score-descending overlap suppression; area 200..50000 pixels',
        deviations='ViT-L in grain paper vs ViT-H in upstream notebook; batch 32 vs 256 for memory; '
            'instance composition needed because official example exports a binary union; author grain settings unavailable',
        optional_import_compatibility='AST-loaded unchanged PostPrecess and PromptGenerator classes; unused Gala metrics omitted')
    write_json(out/'protocol.json',info)
    def predict(image,sid):
        grids=prompter(image,layers=0,scales=3,n_per_side_base=32,method_type=1).generate_prompt_points()
        write_json(out/'prompts'/f'{sid}.json',dict(points_normalized=[g.tolist() for g in grids],source='image_only'))
        generator=sam_module.SamAutomaticMaskGenerator(model=sam,points_per_side=None,
            point_grids=grids,pred_iou_thresh=.90,crop_n_layers=0,crop_n_points_downscale_factor=3,
            box_nms_thresh=.80,crop_nms_thresh=.80,stability_score_thresh=.92,
            points_per_batch=32,min_mask_region_area=0)
        with torch.inference_mode():
            masks=generator.generate(image)
        candidates=[(float(m['predicted_iou']),m['segmentation']) for m in masks
                    if 200<=int(m['area'])<=50000]
        return compose_masks(candidates,image.shape[:2],200,50000/np.prod(image.shape[:2]))
    evaluate_test(predict,args.manifest,out/'test','matsam',info)

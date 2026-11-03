"""SAM 2.1 image inference with reproducible image-derived positive prompts."""

from pathlib import Path
import json
import numpy as np

from baseline.common import prompt_points
from voron_matsam.baseline_data import compact_labels, write_json


def compose_masks(candidates, shape, min_area, max_fraction):
    """Keep high-score regions first; eliminate overlaps without merging IDs."""
    labels = np.zeros(shape, dtype=np.uint32)
    for score, mask in sorted(candidates, key=lambda item:-item[0]):
        area = np.count_nonzero(mask)
        if not min_area <= area <= max_fraction * labels.size:
            continue
        available = mask & (labels == 0)
        if np.count_nonzero(available) >= min_area:
            labels[available] = int(labels.max()) + 1
    return compact_labels(labels)


class SAM2Adapter:
    def __init__(self, config, output, checkpoint, cpu):
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        if not cpu and not torch.cuda.is_available():
            raise RuntimeError('SAM 2 needs a GPU allocation or --cpu')
        if checkpoint is None or not Path(checkpoint).is_file():
            raise FileNotFoundError('Fetch the SAM 2.1 checkpoint first')
        self.config, self.output = config, Path(output)
        self.device = 'cpu' if cpu else 'cuda'
        self.predictor = SAM2ImagePredictor(build_sam2(
            'configs/sam2.1/sam2.1_hiera_s.yaml', str(checkpoint), device=self.device))
        (self.output / 'prompts').mkdir()

    def fit(self, training, validation):
        return dict(trainable_parameters=0, optimizer_updates=0,
                    prompt_source='image-derived watershed centers; no test annotation access',
                    prompt_budget=self.config['max_prompts'], mode='prompted_inference_sam2.1')

    def predict(self, image, image_id):
        import torch
        if self.config.get('prompts_dir'):
            source = Path(self.config['prompts_dir']) / f'{image_id}.json'
            record = json.loads(source.read_text(encoding='utf-8'))
            if record['source'] != 'image_only':
                raise ValueError('This protocol accepts only reusable image-derived prompts')
            points = np.asarray(record['coordinates_xy'], dtype=np.float32).reshape(-1, 2)
            if len(points) > self.config['max_prompts'] or not np.isfinite(points).all():
                raise ValueError('Prompt coordinates exceed the declared budget or are invalid')
            if len(points) and ((points < 0).any() or (points[:, 0] >= image.shape[1]).any()
                                or (points[:, 1] >= image.shape[0]).any()):
                raise ValueError('Prompt coordinates outside the image')
        else:
            points = prompt_points(image, self.config)
        write_json(self.output / 'prompts' / f'{image_id}.json',
                   dict(coordinates_xy=points.tolist(), source='image_only',
                        max_prompts=self.config['max_prompts']))
        candidates = []
        with torch.inference_mode():
            self.predictor.set_image(image)
            for point in points:
                masks, scores, _ = self.predictor.predict(
                    point_coords=point[None], point_labels=np.ones(1, dtype=np.int32),
                    multimask_output=True)
                best = int(np.argmax(scores))
                if scores[best] >= self.config['quality_threshold']:
                    candidates.append((float(scores[best]), masks[best].astype(bool)))
        return compose_masks(candidates, image.shape[:2], self.config['min_area'], self.config['max_mask_fraction'])

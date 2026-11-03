"""Version-specific native APIs. No Voronoi regularizer is used."""

from pathlib import Path
import importlib.metadata

import numpy as np

from voron_matsam.baseline_data import compact_labels, sha256


class CellposeAdapter:
    def __init__(self, method, config, output, checkpoint, cpu):
        from cellpose import models
        import torch
        expected = {'cellpose2':'2.3.2', 'cellpose3':'3.1.1.1', 'cellpose_sam':'4.0.6'}[method]
        if importlib.metadata.version('cellpose') != expected:
            raise RuntimeError(f'{method} requires its separate cellpose=={expected} environment')
        if not cpu and not torch.cuda.is_available():
            raise RuntimeError('Cellpose needs a GPU allocation, or --cpu for debugging')
        if checkpoint is None or not Path(checkpoint).is_file():
            raise FileNotFoundError('Fetch method weights before submitting the job')
        self.method, self.config, self.output = method, config, Path(output)
        self.cpu = cpu
        self.model = self._load(checkpoint)

    def _load(self, checkpoint):
        from cellpose import models
        if self.method == 'cellpose_sam':
            return models.CellposeModel(gpu=not self.cpu, pretrained_model=str(checkpoint), use_bfloat16=True)
        kwargs = dict(gpu=not self.cpu, pretrained_model=str(checkpoint), model_type=None)
        if self.method == 'cellpose2':
            kwargs['net_avg'] = False
        return models.CellposeModel(**kwargs)

    def fit(self, training, validation):
        images, labels = map(list, zip(*training))
        config = self.config
        kwargs = dict(train_data=images, train_labels=labels, save_path=str(self.output),
                      model_name=self.method, n_epochs=config['epochs'], batch_size=config['batch_size'],
                      learning_rate=config['learning_rate'], weight_decay=config['weight_decay'],
                      nimg_per_epoch=config['steps_per_epoch'] * config['batch_size'],
                      min_train_masks=5, normalize=True, save_every=10)
        # Native augmentations are applied on the fly; there is no synthetic pretraining.
        if self.method == 'cellpose2':
            result = self.model.train(channels=[0, 0], SGD=True, rescale=True, **kwargs)
            path = result[0] if isinstance(result, tuple) else result
        else:
            from cellpose import train
            if self.method == 'cellpose_sam':
                result = train.train_seg(self.model.net, channel_axis=-1, rescale=False, **kwargs)
            else:
                result = train.train_seg(self.model.net, channels=[0, 0], rescale=True, SGD=False, **kwargs)
            path = result[0]
            np.savez(self.output / 'losses.npz', train=result[1], validation=result[2])
        if not Path(path).is_file():
            raise FileNotFoundError(f'Training did not produce checkpoint: {path}')
        self.model = self._load(path)
        return dict(checkpoint=str(path), checkpoint_sha256=sha256(path),
                    optimizer_updates_requested=config['epochs'] * config['steps_per_epoch'],
                    checkpoint_selection='final_epoch', voronoi_loss_weight=0.,
                    seed_policy='run seed controls subset/Torch; native Cellpose resets NumPy augmentation RNG by epoch')

    def predict(self, image, image_id):
        kwargs = dict(diameter=None, flow_threshold=self.config['flow_threshold'],
                      cellprob_threshold=self.config['cellprob_threshold'], min_size=self.config['min_area'])
        if self.method == 'cellpose_sam':
            kwargs['channel_axis'] = -1
        else:
            kwargs['channels'] = [0, 0]
        result = self.model.eval(image, **kwargs)
        return compact_labels(result[0])

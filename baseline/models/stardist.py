"""StarDist 2D from scratch, using only the selected few-shot training images."""

from pathlib import Path
import numpy as np

from baseline.common import normalize
from voron_matsam.baseline_data import compact_labels, sha256


class StarDistAdapter:
    def __init__(self, config, output, cpu):
        import tensorflow as tf
        from stardist.models import Config2D, StarDist2D
        if cpu:
            tf.config.set_visible_devices([], 'GPU')
        gpus = [] if cpu else tf.config.list_physical_devices('GPU')
        if not cpu and not gpus:
            raise RuntimeError('TensorFlow cannot see a GPU; use the StarDist environment or --cpu')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.keras.utils.set_random_seed(config['seed'])
        self.config, self.output = config, Path(output)
        network = Config2D(n_rays=config['n_rays'], n_channel_in=3, axes='YXC', grid=(2, 2),
                           train_patch_size=(config['patch_size'],)*2, train_epochs=config['epochs'],
                           train_steps_per_epoch=config['steps_per_epoch'], train_batch_size=config['batch_size'],
                           train_learning_rate=config['learning_rate'], train_reduce_lr=None,
                           train_tensorboard=False, train_checkpoint_last='weights_last.h5', use_gpu=False)
        self.model = StarDist2D(network, name='stardist', basedir=str(self.output))

    def fit(self, training, validation):
        x, y = map(list, zip(*training))
        vx, vy = map(list, zip(*validation))
        def augmenter(image, mask):
            turns = np.random.randint(4)
            image, mask = np.rot90(image, turns), np.rot90(mask, turns)
            if np.random.rand() < .5:
                image, mask = image[::-1], mask[::-1]
            return np.ascontiguousarray(image), np.ascontiguousarray(mask)
        self.model.train([normalize(a) for a in x], y,
                         validation_data=([normalize(a) for a in vx], vy),
                         augmenter=augmenter, seed=self.config['seed'])
        path = self.output / 'stardist' / 'weights_last.h5'
        if not path.is_file():
            raise FileNotFoundError('StarDist did not save the requested final checkpoint')
        self.model.load_weights('weights_last.h5')
        return dict(checkpoint=str(path), checkpoint_sha256=sha256(path), checkpoint_selection='final_epoch',
                    optimizer_updates=self.config['epochs'] * self.config['steps_per_epoch'])

    def predict(self, image, image_id):
        labels, _ = self.model.predict_instances(normalize(image),
                    prob_thresh=self.config['probability_threshold'], nms_thresh=self.config['nms_threshold'],
                    n_tiles=(2, 2, 1))
        return compact_labels(labels)

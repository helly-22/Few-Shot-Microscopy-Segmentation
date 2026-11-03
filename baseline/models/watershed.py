from baseline.common import classical_instances


class WatershedAdapter:
    def __init__(self, config, output):
        self.config = config

    def fit(self, training, validation):
        return {'trainable_parameters': 0, 'optimizer_updates': 0}

    def predict(self, image, image_id):
        return classical_instances(image, self.config)

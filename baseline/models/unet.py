"""Four-level U-Net trained on grain interiors, followed by distance watershed."""

from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from baseline.common import distance_instances, normalize
from voron_matsam.baseline_data import sha256, write_json


class DoubleConv(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self, base=32):
        super().__init__()
        widths = [base * 2**i for i in range(5)]
        self.down = nn.ModuleList(DoubleConv(cin, cout) for cin, cout in zip([3]+widths[:-1], widths))
        self.up = nn.ModuleList(nn.ConvTranspose2d(widths[i], widths[i-1], 2, stride=2) for i in range(4, 0, -1))
        self.merge = nn.ModuleList(DoubleConv(widths[i], widths[i-1]) for i in range(4, 0, -1))
        self.output = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        skip = []
        for i, layer in enumerate(self.down):
            x = layer(x)
            if i < 4:
                skip.append(x)
                x = F.max_pool2d(x, 2)
        for up, merge, residual in zip(self.up, self.merge, reversed(skip)):
            x = up(x)
            if x.shape[-2:] != residual.shape[-2:]:
                x = F.interpolate(x, residual.shape[-2:], mode='bilinear', align_corners=False)
            x = merge(torch.cat((residual, x), dim=1))
        return self.output(x)


def random_batch(data, batch_size, patch, rng):
    images, masks = [], []
    for _ in range(batch_size):
        image, mask = data[int(rng.integers(len(data)))]
        image = normalize(image)
        h, w = mask.shape
        pad = ((0, max(0, patch-h)), (0, max(0, patch-w)))
        image = np.pad(image, pad + ((0, 0),), mode='edge')
        mask = np.pad(mask > 0, pad)
        y = int(rng.integers(image.shape[0]-patch+1))
        x = int(rng.integers(image.shape[1]-patch+1))
        image, mask = image[y:y+patch, x:x+patch], mask[y:y+patch, x:x+patch]
        turns = int(rng.integers(4))
        image, mask = np.rot90(image, turns), np.rot90(mask, turns)
        if rng.random() < .5:
            image, mask = image[::-1], mask[::-1]
        images.append(np.ascontiguousarray(image.transpose(2, 0, 1)))
        masks.append(np.ascontiguousarray(mask[None], dtype=np.float32))
    return torch.from_numpy(np.stack(images)), torch.from_numpy(np.stack(masks))


def segmentation_loss(logits, target):
    probability = logits.sigmoid()
    intersection = (probability * target).sum((1, 2, 3))
    dice = (2 * intersection + 1) / (probability.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1)
    return F.binary_cross_entropy_with_logits(logits, target) + (1-dice).mean()


@torch.inference_mode()
def tiled_probability(model, image, device, patch=256, overlap=64):
    image = normalize(image)
    h, w = image.shape[:2]
    padded = np.pad(image, ((0, max(0, patch-h)), (0, max(0, patch-w)), (0, 0)), mode='edge')
    height, width = padded.shape[:2]
    stride = patch - overlap
    if stride < 1:
        raise ValueError('Overlap must be smaller than the patch size')
    ys = sorted(set(range(0, max(1, height-patch+1), stride)) | {height-patch})
    xs = sorted(set(range(0, max(1, width-patch+1), stride)) | {width-patch})
    total, weight = np.zeros((height, width), np.float32), np.zeros((height, width), np.float32)
    model.eval()
    for y in ys:
        for x in xs:
            tile = padded[y:y+patch, x:x+patch].transpose(2, 0, 1).copy()
            prediction = model(torch.from_numpy(tile[None]).to(device)).sigmoid()[0, 0].cpu().numpy()
            total[y:y+patch, x:x+patch] += prediction
            weight[y:y+patch, x:x+patch] += 1
    return (total / weight)[:h, :w]


class UNetAdapter:
    def __init__(self, config, output, cpu):
        if not cpu and not torch.cuda.is_available():
            raise RuntimeError('U-Net needs a GPU allocation or --cpu')
        self.config, self.output = config, Path(output)
        self.device = torch.device('cpu' if cpu else 'cuda')
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
        self.model = UNet(config['base_channels']).to(self.device)

    def fit(self, training, validation):
        started = time.perf_counter()
        c = self.config
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=c['learning_rate'], weight_decay=c['weight_decay'])
        rng = np.random.default_rng(c['seed'])
        history = []
        for epoch in range(c['epochs']):
            self.model.train()
            losses = []
            for _ in range(c['steps_per_epoch']):
                image, target = random_batch(training, c['batch_size'], c['patch_size'], rng)
                image, target = image.to(self.device), target.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                loss = segmentation_loss(self.model(image), target)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
            history.append(dict(epoch=epoch+1, loss=float(np.mean(losses))))
            print(f'U-Net epoch {epoch+1}: loss={history[-1]["loss"]:.5f}', flush=True)
        path = self.output / 'unet.pt'
        torch.save(self.model.state_dict(), path)
        write_json(self.output / 'training_history.json', history)
        return dict(checkpoint=str(path), checkpoint_sha256=sha256(path), checkpoint_selection='final_epoch',
                    optimizer_updates=c['epochs'] * c['steps_per_epoch'],
                    training_seconds=time.perf_counter()-started, device=str(self.device), precision='float32',
                    peak_cuda_memory_bytes=(torch.cuda.max_memory_allocated(self.device)
                                            if self.device.type == 'cuda' else None),
                    trainable_parameters=sum(p.numel() for p in self.model.parameters()))

    def predict(self, image, image_id):
        c = self.config
        probability = tiled_probability(self.model, image, self.device, c['patch_size'], c['overlap'])
        return distance_instances(probability >= c['foreground_threshold'], c['min_distance'], c['min_area'])

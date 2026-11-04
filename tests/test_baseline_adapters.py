import importlib.metadata
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from baseline.models.sam2 import compose_masks
from baseline.models.unet import UNet, UNetAdapter, random_batch, segmentation_loss, tiled_probability
from baseline.run import load_config


@pytest.fixture
def limited_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_unet_odd_shapes_loss_and_gradient(limited_threads):
    model = UNet(base=2)
    # Keep ReLUs active so this tests gradient propagation rather than whether
    # random initialization happens to leave both final feature channels dead.
    with torch.no_grad():
        for layer in model.modules():
            if isinstance(layer, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
                layer.weight.fill_(.01)
                if layer.bias is not None:
                    layer.bias.fill_(.1)
    x = torch.rand(1,3,33,49, generator=torch.Generator().manual_seed(0))
    prediction = model(x)
    assert prediction.shape == (1,1,33,49)
    loss = segmentation_loss(prediction, torch.ones_like(prediction))
    loss.backward()
    assert torch.isfinite(loss) and model.output.weight.grad.abs().sum() > 0
    assert model.down[0][0].weight.grad.abs().sum() > 0


def test_tiling_covers_small_images_and_uneven_edges(limited_threads):
    class Constant(torch.nn.Module):
        def forward(self, x):
            return torch.zeros(x.shape[0],1,*x.shape[-2:])
    for shape in ((13,17,3),(71,83,3)):
        p = tiled_probability(Constant(), np.zeros(shape,np.uint8), torch.device('cpu'), 32, 8)
        assert p.shape == shape[:2]
        np.testing.assert_allclose(p, .5)


def test_actual_unet_cpu_training_smoke(tmp_path, limited_threads):
    config = load_config('unet_watershed')
    config.update(seed=0, epochs=2, steps_per_epoch=2, batch_size=1,
                  patch_size=32, base_channels=2, overlap=8, min_area=2, min_distance=2)
    image = np.zeros((40,48,3),np.uint8)
    labels = np.zeros((40,48),np.uint32)
    image[5:30,5:30] = 255
    labels[5:30,5:30] = 1
    adapter = UNetAdapter(config, tmp_path, cpu=True)
    result = adapter.fit([(image,labels)], [(image,labels)])
    assert result['optimizer_updates'] == 4
    assert Path(result['checkpoint']).is_file()
    predicted = adapter.predict(image, 'synthetic')
    assert predicted.shape == labels.shape and predicted.dtype == np.uint32


def test_sam_mask_composition_has_no_overlaps_and_filters_giant_masks():
    a, b = np.zeros((20,20),bool), np.zeros((20,20),bool)
    a[2:10,2:10] = True
    b[6:15,6:15] = True
    labels = compose_masks([(.7,b),(.9,a),(1.,np.ones_like(a))], a.shape, 10, .5)
    assert labels.max() == 2
    assert labels[7,7] == 1 and labels[13,13] == 2


@pytest.mark.parametrize('method,version', [('cellpose2','2.3.2'),('cellpose3','3.1.1.1'),('cellpose_sam','4.0.6')])
def test_cellpose_version_specific_training_calls(method, version, monkeypatch, tmp_path):
    from baseline.models.cellpose import CellposeAdapter
    initial, final = tmp_path/'initial', tmp_path/'final'
    initial.write_bytes(b'initial')
    final.write_bytes(b'final')
    calls = []
    class Model:
        def __init__(self, **kwargs):
            self.net = object()
        def train(self, **kwargs):
            calls.append(kwargs)
            return str(final)
        def eval(self, image, **kwargs):
            return np.ones(image.shape[:2],np.uint32), None, None
    cellpose = ModuleType('cellpose')
    def train_seg(net, **kwargs):
        calls.append(kwargs)
        return str(final), np.ones(2), np.zeros(2)
    cellpose.models = SimpleNamespace(CellposeModel=Model)
    cellpose.train = SimpleNamespace(train_seg=train_seg)
    monkeypatch.setitem(sys.modules, 'cellpose', cellpose)
    monkeypatch.setattr(importlib.metadata, 'version', lambda name:version)
    config = load_config(method)
    adapter = CellposeAdapter(method, config, tmp_path, initial, cpu=True)
    data = [(np.zeros((32,32,3),np.uint8),np.ones((32,32),np.uint32))]
    info = adapter.fit(data, data)
    assert info['voronoi_loss_weight'] == 0
    assert 'test_data' not in calls[0] and 'test_labels' not in calls[0]
    assert calls[0]['nimg_per_epoch'] == 60
    assert calls[0]['learning_rate'] == config['learning_rate']
    if method == 'cellpose2':
        assert calls[0]['SGD']
    if method == 'cellpose_sam':
        assert calls[0]['channel_axis'] == -1 and not calls[0]['rescale']


def test_sam2_prompt_record_and_replay_without_labels(monkeypatch, tmp_path):
    import json
    from baseline.models.sam2 import SAM2Adapter
    parent = ModuleType('sam2')
    parent.__path__ = []
    build = ModuleType('sam2.build_sam')
    build.build_sam2 = lambda *a, **kw:object()
    predictor = ModuleType('sam2.sam2_image_predictor')
    points_seen = []
    class Predictor:
        def __init__(self, model):
            pass
        def set_image(self, image):
            self.shape = image.shape[:2]
        def predict(self, point_coords, point_labels, multimask_output):
            points_seen.append(point_coords.copy())
            mask = np.zeros(self.shape, bool)
            mask[5:25,5:25] = True
            return np.stack([mask]*3), np.array([.9,.7,.5]), None
    predictor.SAM2ImagePredictor = Predictor
    for name,module in [('sam2',parent),('sam2.build_sam',build),('sam2.sam2_image_predictor',predictor)]:
        monkeypatch.setitem(sys.modules,name,module)
    checkpoint = tmp_path/'sam.pt'
    checkpoint.write_bytes(b'mocked SAM weights')
    config = load_config('sam2')
    config.update(min_area=10, min_distance=8, max_prompts=1)
    image = np.zeros((64,64,3),np.uint8)
    image[5:25,5:25] = 255
    first = tmp_path/'first'
    first.mkdir()
    adapter = SAM2Adapter(config,first,checkpoint,cpu=True)
    assert adapter.fit([],[])['optimizer_updates'] == 0
    a = adapter.predict(image,'sample')
    record = json.loads((first/'prompts/sample.json').read_text())
    assert record['source'] == 'image_only' and len(record['coordinates_xy']) == 1
    second = tmp_path/'second'
    second.mkdir()
    replay = SAM2Adapter(dict(config,prompts_dir=str(first/'prompts')),second,checkpoint,cpu=True)
    b = replay.predict(image,'sample')
    np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(points_seen[0],points_seen[1])
    record['source'] = 'ground_truth'
    (first/'prompts/sample.json').write_text(json.dumps(record))
    with pytest.raises(ValueError,match='image-derived'):
        replay.predict(image,'sample')


def test_stardist_uses_only_train_validation_and_reloads_last_checkpoint(monkeypatch,tmp_path):
    from baseline.models.stardist import StarDistAdapter
    tensorflow = ModuleType('tensorflow')
    tensorflow.config = SimpleNamespace(set_visible_devices=lambda *a:None,
        list_physical_devices=lambda *a:[], experimental=SimpleNamespace(set_memory_growth=lambda *a:None))
    tensorflow.keras = SimpleNamespace(utils=SimpleNamespace(set_random_seed=lambda seed:None))
    parent = ModuleType('stardist')
    parent.__path__ = []
    models = ModuleType('stardist.models')
    calls = {}
    def Config2D(**kwargs):
        calls['config'] = kwargs
        return kwargs
    class Model:
        def __init__(self, config, name, basedir):
            self.path = Path(basedir)/name
            self.path.mkdir()
        def train(self,x,y,validation_data,augmenter,seed):
            calls['training'] = (len(x),len(y))
            calls['validation'] = (len(validation_data[0]),len(validation_data[1]))
            (self.path/'weights_last.h5').write_bytes(b'mocked StarDist weights')
        def load_weights(self,path):
            calls['loaded'] = path
        def predict_instances(self,image,**kwargs):
            return np.ones(image.shape[:2],np.uint32),None
    models.Config2D,models.StarDist2D = Config2D,Model
    for name,module in [('tensorflow',tensorflow),('stardist',parent),('stardist.models',models)]:
        monkeypatch.setitem(sys.modules,name,module)
    config = dict(load_config('stardist'),seed=0)
    adapter = StarDistAdapter(config,tmp_path,cpu=True)
    image,label = np.zeros((32,32,3),np.uint8),np.ones((32,32),np.uint32)
    result = adapter.fit([(image,label)]*2,[(image,label)])
    assert calls['training'] == (2,2) and calls['validation'] == (1,1)
    assert calls['loaded'] == 'weights_last.h5'
    assert result['checkpoint_selection'] == 'final_epoch'
    assert calls['config']['n_rays'] == 32
    assert adapter.predict(image,'sample').shape == label.shape

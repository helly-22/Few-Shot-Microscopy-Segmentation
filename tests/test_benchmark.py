import numpy as np
import torch
import pytest

from baseline.benchmark import classical, connected, select_configuration
from baseline.benchmark_neural import unet_reconstruct
from baseline.models.unet import UNet, random_batch


def test_validation_selection_only_calls_supplied_validation_ids():
    mask=np.zeros((30,40),np.uint16)
    mask[3:25,3:30]=1
    seen=[]
    def prediction(sid,config):
        assert sid in ('val1','val2')
        seen.append(sid)
        return mask if config['good'] else np.zeros_like(mask)
    chosen,rows=select_configuration([dict(good=False),dict(good=True)],
        ['val1','val2'],prediction,lambda sid:mask)
    assert chosen['good'] and len(rows)==2 and len(seen)==4


def test_connected_and_watershed_reuse_same_probability_and_split_touching_regions():
    yy,xx=np.ogrid[:64,:64]
    probability=(((yy-32)**2+(xx-22)**2<14**2)|((yy-32)**2+(xx-43)**2<14**2)).astype(float)
    saved=probability.copy()
    cc=unet_reconstruct(probability,dict(threshold=.5,erosion=0),'unet')
    ws=unet_reconstruct(probability,dict(threshold=.5,distance=12),'unet_watershed')
    assert cc.max()==1 and ws.max()==2
    np.testing.assert_array_equal(probability,saved)
    assert connected(np.zeros((20,20),bool)).max()==0


def test_four_stage_unet_odd_shape_and_cached_normalization_equivalence():
    previous=torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        model=UNet(base=2,levels=4)
        assert len(model.down)==4 and len(model.up)==3
        assert model(torch.zeros((1,3,33,49))).shape==(1,1,33,49)
        image=np.zeros((40,48,3),np.uint8)
        image[4:35,4:40]=255
        mask=(image[...,0]>0).astype(np.uint16)
        from baseline.common import normalize
        a=random_batch([(image,mask)],1,32,np.random.default_rng(7))
        b=random_batch([(normalize(image),mask)],1,32,np.random.default_rng(7),normalized=True)
        for x,y in zip(a,b): torch.testing.assert_close(x,y)
    finally:
        torch.set_num_threads(previous)


def test_classical_constant_input_is_empty_for_both_polarities():
    for method in ('connected_components','watershed'):
        for bright in (True,False):
            result=classical(np.zeros((32,40)),dict(sigma=1.,offset=0.,bright=bright,min_area=2,distance=4),method)
            assert not result.any()


def test_common_evaluator_rejects_probabilities_before_label_compaction(tmp_path,monkeypatch):
    import baseline.benchmark as benchmark
    mask=np.zeros((30,40),np.uint16)
    mask[3:25,3:30]=1
    monkeypatch.setattr(benchmark,'load_manifest',lambda path:(
        dict(test=['heldout'],validation=['validation']),dict(heldout={}),tmp_path))
    monkeypatch.setattr(benchmark,'read_pair',lambda record,root:(np.zeros((30,40,3),np.uint8),mask))
    with pytest.raises(ValueError,match='nonnegative integers'):
        benchmark.evaluate_test(lambda image,sid:np.full(mask.shape,.5),
                                tmp_path/'manifest.json',tmp_path/'result','synthetic')

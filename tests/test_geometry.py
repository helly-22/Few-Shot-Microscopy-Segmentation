import pytest
import torch
from voron_matsam.geometry import soft_voronoi, voronoi_consistency_loss


def test_partition_and_nearest_seed():
    c = torch.tensor([[[0., .5], [1., .5]]])
    p = soft_voronoi(c, 5, 9, temperature=.01)
    assert p.shape == (1, 2, 5, 9)
    torch.testing.assert_close(p.sum(1), torch.ones(1, 5, 9))
    assert (p.argmax(1)[:, :, :4] == 0).all()
    assert (p.argmax(1)[:, :, 5:] == 1).all()


def test_positive_power_weight_expands_cell():
    c = torch.tensor([[[0., .5], [1., .5]]])
    before = soft_voronoi(c, 3, 11)
    after = soft_voronoi(c, 3, 11, weights=torch.tensor([[.5, 0.]]))
    assert after[:, 0].sum() > before[:, 0].sum()


def test_gradients_reach_centers_weights_and_segmentation():
    c = torch.tensor([[[.2, .3], [.7, .8]]], requires_grad=True)
    w = torch.tensor([[.1, -.1]], requires_grad=True)
    logits = torch.zeros(1, 2, 7, 9, requires_grad=True)
    loss = voronoi_consistency_loss(logits, c, weights=w)
    loss.backward()
    for t in (c, w, logits):
        assert torch.isfinite(t.grad).all()
        assert t.grad.abs().sum() > 0


def test_numerical_gradient():
    c = torch.tensor([[[.2, .3], [.7, .8]]], dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x: soft_voronoi(x, 3, 4), (c,))


def test_single_seed_and_extreme_temperature():
    torch.testing.assert_close(soft_voronoi(torch.zeros(1, 1, 2), 2, 3), torch.ones(1, 1, 2, 3))
    p = soft_voronoi(torch.tensor([[[0., 0.], [1., 1.]]]), 3, 3, 1e-8)
    assert torch.isfinite(p).all()


@pytest.mark.parametrize('temperature', [0, -1, float('nan'), float('inf')])
def test_bad_temperature(temperature):
    with pytest.raises(ValueError):
        soft_voronoi(torch.zeros(1, 2, 2), 3, 3, temperature)

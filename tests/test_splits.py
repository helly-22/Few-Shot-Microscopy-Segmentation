import pytest
from voron_matsam.splits import make_experiments


RECORDS = [dict(image_id=f'{s}_{i}', specimen_id=s)
           for s in ('train_a', 'train_b', 'val', 'test') for i in range(4)]


def test_paired_reproducible_and_leak_free():
    runs = make_experiments(RECORDS, ['val'], ['test'])
    assert runs == make_experiments(list(reversed(RECORDS)), ['val'], ['test'])
    assert len(runs) == 20
    for a, b in zip(runs[::2], runs[1::2]):
        assert a['train'] == b['train']
        assert len(a['train']) == a['shots']
        assert not set(a['train']) & set(a['test'] + a['validation'])
        assert all(x.startswith('train_') for x in a['train'])
    for offset in range(0, 20, 4):
        assert set(runs[offset]['train']) <= set(runs[offset + 2]['train'])


def test_reject_overlapping_specimens():
    with pytest.raises(ValueError):
        make_experiments(RECORDS, ['val'], ['val'])


def test_reject_too_few_training_images():
    with pytest.raises(ValueError):
        make_experiments(RECORDS, ['val'], ['test'], shots=(9,))

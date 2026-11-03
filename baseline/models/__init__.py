def create(method, config, output, checkpoint=None, cpu=False):
    if method in ('cellpose_sam', 'cellpose2', 'cellpose3'):
        from .cellpose import CellposeAdapter
        return CellposeAdapter(method, config, output, checkpoint, cpu)
    if method == 'unet_watershed':
        from .unet import UNetAdapter
        return UNetAdapter(config, output, cpu)
    if method == 'stardist':
        from .stardist import StarDistAdapter
        return StarDistAdapter(config, output, cpu)
    if method == 'sam2':
        from .sam2 import SAM2Adapter
        return SAM2Adapter(config, output, checkpoint, cpu)
    if method == 'watershed':
        from .watershed import WatershedAdapter
        return WatershedAdapter(config, output)
    raise ValueError(f'Unknown method: {method}')

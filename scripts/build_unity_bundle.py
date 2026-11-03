"""Package source for transfer; exclude all datasets, weights and local outputs."""

from pathlib import Path
import tarfile

root = Path(__file__).resolve().parents[1]
destination = root / 'dist' / 'voron-matsam-unity.tar.gz'
destination.parent.mkdir(exist_ok=True)
with tarfile.open(destination, 'w:gz') as archive:
    for name in ('src', 'tests', 'hpc', 'requirements', 'scripts', 'docs', 'configs', 'baseline'):
        for path in sorted((root / name).rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts:
                archive.add(path, arcname=f'voron-matsam/{path.relative_to(root).as_posix()}')
    for name in ('README.md', 'pyproject.toml', '.gitignore', '.gitattributes', 'LICENSE'):
        path = root / name
        if path.is_file():
            archive.add(path, arcname=f'voron-matsam/{name}')
print(f'{destination} ({destination.stat().st_size / 1e6:.1f} MB)')

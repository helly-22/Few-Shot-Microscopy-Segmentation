"""Fetch official MatSAM code and SAM ViT-L weights outside version control."""
import argparse
import json
from pathlib import Path
import socket
import subprocess
import urllib.request

from baseline.benchmark_matsam import UPSTREAM_COMMIT
from voron_matsam.baseline_data import sha256, write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args()
    root=args.project_root.resolve()
    repository=root/'external/matsam-benchmark'
    repository.parent.mkdir(exist_ok=True)
    if not repository.exists():
        subprocess.run(['git','clone','--filter=blob:none','--sparse',
            'https://github.com/USTB-AI3DVIP/matsam.git',str(repository)],check=True)
        subprocess.run(['git','-C',str(repository),'sparse-checkout','set','utils'],check=True)
        subprocess.run(['git','-C',str(repository),'checkout','--detach',UPSTREAM_COMMIT],check=True)
    revision=subprocess.run(['git','-C',str(repository),'rev-parse','HEAD'],
                             check=True,capture_output=True,text=True).stdout.strip()
    if revision!=UPSTREAM_COMMIT:
        raise ValueError('Existing MatSAM checkout differs; inspect it without overwriting')
    destination=root/'data/models/matsam_sam_vit_l.pth'
    destination.parent.mkdir(parents=True,exist_ok=True)
    registry=destination.parent/'matsam_asset.json'
    url='https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth'
    if not destination.exists():
        socket.setdefaulttimeout(90)
        temporary=destination.with_suffix('.download')
        urllib.request.urlretrieve(url,temporary)
        if temporary.stat().st_size<1000000000:
            raise ValueError('Unexpected SAM ViT-L checkpoint size')
        temporary.replace(destination)
    entry=dict(file=destination.name,url=url,sha256=sha256(destination),
        bytes=destination.stat().st_size,model='vit_l',upstream_commit=UPSTREAM_COMMIT)
    if registry.exists() and json.loads(registry.read_text())!=entry:
        raise ValueError('Recorded checkpoint differs; inspect without replacing the registry')
    write_json(registry,entry)
    print(f'Verified MatSAM {revision}; SAM ViT-L SHA256 {entry["sha256"]}')


if __name__=='__main__':
    main()

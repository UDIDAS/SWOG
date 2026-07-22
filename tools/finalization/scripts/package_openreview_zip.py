from __future__ import annotations
import argparse,fnmatch,zipfile
from pathlib import Path
EXCLUDES=['.git/*','**/.git/*','data/raw/*','data/cache/*','embeddings/*','checkpoints/*','docs/*','*.nii','*.nii.gz','*.dcm','*.pt','*.pth','*.ckpt','*.npz','__pycache__/*','**/__pycache__/*','.pytest_cache/*','**/.pytest_cache/*']
def main():
    p=argparse.ArgumentParser(); p.add_argument('--repo-root',required=True); p.add_argument('--out',required=True); p.add_argument('--exclude',action='append',default=[]); a=p.parse_args()
    repo=Path(a.repo_root).resolve(); patterns=EXCLUDES+a.exclude; included=[]
    with zipfile.ZipFile(a.out,'w',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(repo.rglob('*')):
            if not path.is_file(): continue
            rel=path.relative_to(repo).as_posix()
            if any(fnmatch.fnmatch(rel,x) for x in patterns): continue
            z.write(path,rel); included.append(rel)
    print(f'Wrote {a.out} with {len(included)} files')
if __name__=='__main__': main()

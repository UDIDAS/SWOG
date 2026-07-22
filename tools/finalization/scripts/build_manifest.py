from __future__ import annotations
import argparse, hashlib
from pathlib import Path
import pandas as pd

def sha256(path: Path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(); p.add_argument('--repo-root',required=True); p.add_argument('--mapping',required=True); p.add_argument('--out',required=True); a=p.parse_args()
    repo=Path(a.repo_root).resolve(); mapping=pd.read_csv(a.mapping); rows=[]
    for _,r in mapping.iterrows():
        rel=str(r.output_file); path=repo/rel
        if not path.exists(): rows.append({'output_file':rel,'sha256':'MISSING','size_bytes':'','rows':'','paper_table':r.paper_table}); continue
        nrows=''
        if path.suffix.lower()=='.csv':
            try: nrows=len(pd.read_csv(path))
            except Exception: nrows='ERROR'
        rows.append({'output_file':rel,'sha256':sha256(path),'size_bytes':path.stat().st_size,'rows':nrows,'paper_table':r.paper_table})
    pd.DataFrame(rows).to_csv(a.out,index=False); print(f'Wrote {a.out}')
if __name__=='__main__': main()

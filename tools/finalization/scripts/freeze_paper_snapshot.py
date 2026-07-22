from __future__ import annotations
import argparse, hashlib, json, subprocess
from pathlib import Path
import yaml

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def git_commit(repo: Path):
    try:
        return subprocess.check_output(['git','rev-parse','HEAD'], cwd=repo, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None

def main():
    p=argparse.ArgumentParser(); p.add_argument('--repo-root',required=True); p.add_argument('--config',required=True); p.add_argument('--out',required=True); a=p.parse_args()
    repo=Path(a.repo_root).resolve(); cfg=yaml.safe_load(Path(a.config).read_text())
    tracked=[]
    for rel in cfg.get('paper_snapshot',{}).get('tracked_files',[]):
        path=repo/rel
        if not path.exists(): raise FileNotFoundError(rel)
        tracked.append({'path':rel,'sha256':sha256(path),'size_bytes':path.stat().st_size})
    ev=cfg['evaluation']
    snap={'snapshot_version':1,'git_commit':git_commit(repo),'primary_policy':ev['primary_policy'],'policy_selection_rule':ev['policy_selection_rule'],'query_convention':ev['query_convention'],'incomparable_policy':ev['incomparable_policy'],'bootstrap':ev['bootstrap'],'hypothesis_families':ev['hypothesis_families'],'tracked_files':tracked}
    Path(a.out).write_text(json.dumps(snap,indent=2)+'\n')
    print(f'Wrote {a.out}')
if __name__=='__main__': main()

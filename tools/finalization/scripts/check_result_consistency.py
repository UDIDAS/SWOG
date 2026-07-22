from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd

def value(repo: Path, spec: dict) -> float:
    df=pd.read_csv(repo/spec['file'])
    for col, expected in spec.get('filters',{}).items():
        df=df[df[col].astype(str)==str(expected)]
    if len(df)!=1: raise ValueError(f"{len(df)} rows for {spec}")
    return float(df.iloc[0][spec['value_column']])

def main():
    p=argparse.ArgumentParser(); p.add_argument('--repo-root',required=True); p.add_argument('--rules',required=True); a=p.parse_args()
    repo=Path(a.repo_root).resolve(); rules=json.loads(Path(a.rules).read_text()); failures=0
    for rule in rules['checks']:
        vals=[]; print('\n'+rule['name'])
        for spec in rule['sources']:
            try:
                v=value(repo,spec); vals.append(v); print(f"  {spec['label']}: {v:.6f}")
            except Exception as e:
                failures+=1; print(f"  ERROR {spec['label']}: {e}")
        if len(vals)>=2:
            spread=max(vals)-min(vals); tol=float(rule.get('tolerance',1e-9))
            if spread>tol: failures+=1; print(f'  FAIL spread={spread:.6g} > {tol:.6g}')
            else: print('  OK')
    raise SystemExit(1 if failures else 0)
if __name__=='__main__': main()

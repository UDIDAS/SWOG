from __future__ import annotations
import argparse,re
from pathlib import Path
PATTERNS={
'email':re.compile(r'\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b',re.I),
'github_owner':re.compile(r'https?://(?:www\.)?github\.com/[^/\s]+',re.I),
'unix_path':re.compile(r'(?:/Users/|/home/|/scratch/)[^\s\"\']+'),
'windows_path':re.compile(r'\b[A-Z]:\\[^\s\"\']+',re.I),
'internal_name':re.compile(r'\b(?:llmft|sam3_delivery|UDIDAS|UMKC)\b',re.I),
'google_drive':re.compile(r'https?://(?:drive|docs)\.google\.com/[^\s\"\']+',re.I)}
SUFFIX={'.py','.md','.txt','.yaml','.yml','.json','.jsonl','.csv','.tex','.toml','.ini','.cfg','.ipynb'}
def main():
    p=argparse.ArgumentParser(); p.add_argument('root'); p.add_argument('--allow',action='append',default=[]); a=p.parse_args(); findings=[]
    for path in Path(a.root).rglob('*'):
        if not path.is_file() or path.suffix.lower() not in SUFFIX or '.git' in path.parts: continue
        text=path.read_text(errors='ignore')
        for kind,pat in PATTERNS.items():
            for m in pat.finditer(text):
                v=m.group(0)
                if not any(x in v for x in a.allow): findings.append((path,kind,v))
    if findings:
        print('Potential anonymity leaks:')
        for path,kind,v in findings: print(f'- {path}: {kind}: {v}')
        raise SystemExit(1)
    print('No obvious anonymity leaks found.')
if __name__=='__main__': main()

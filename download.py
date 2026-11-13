#!/usr/bin/env python3
"""Download one verified 24-layer stage; no Hub SDK or credentials required."""
import argparse,hashlib,json,os,pathlib,urllib.request
REPOSITORY='kortexa-ai/Inkling-2-Small-EXL3-2.59bpw'
REVISION='main'  # Replaced with the verified immutable publication commit before release.
MANIFEST_SHA='97d75d66d3883514450772f8a3efb348467d35157925c908ea97315ae6e3feca'


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while block:=f.read(4*1024**2):h.update(block)
        os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)
    return h.hexdigest()


def safe_path(root,name):
    relative=pathlib.PurePosixPath(name)
    if relative.is_absolute() or '..' in relative.parts or not relative.parts:
        raise ValueError('Invalid artifact path')
    target=root.joinpath(*relative.parts)
    if target.is_symlink() or not target.resolve().is_relative_to(root.resolve()):
        raise ValueError('Artifact traverses a symlink outside the model directory')
    return target


def download(name,target,sha=None,size=None):
    if target.exists() and (size is None or target.stat().st_size==size) and sha and digest(target)==sha:return
    target.parent.mkdir(parents=True,exist_ok=True)
    temporary=target.with_name(target.name+'.part')
    if temporary.is_symlink():raise ValueError('Refusing a symlink download target')
    url=f'https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{name}'
    h=hashlib.sha256();count=0
    with urllib.request.urlopen(url,timeout=120) as response,temporary.open('wb') as out:
        while block:=response.read(4*1024**2):
            out.write(block);h.update(block);count+=len(block)
        out.flush();os.fsync(out.fileno())
        os.posix_fadvise(out.fileno(),0,0,os.POSIX_FADV_DONTNEED)
    if sha and h.hexdigest()!=sha:raise ValueError(f'Checksum mismatch: {name}')
    if size is not None and count!=size:raise ValueError(f'Size mismatch: {name}')
    temporary.replace(target)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rank',type=int,choices=(0,1),required=True)
    p.add_argument('--directory',type=pathlib.Path,required=True)
    args=p.parse_args();root=args.directory.expanduser().resolve();root.mkdir(parents=True,exist_ok=True)
    download('checkpoint.json',root/'checkpoint.json',MANIFEST_SHA)
    download(f'ranks/rank-{args.rank}.json',root/f'rank-{args.rank}.json')
    manifest=json.loads((root/'checkpoint.json').read_text())
    rank=json.loads((root/f'rank-{args.rank}.json').read_text())
    if rank['checkpoint_sha256']!=MANIFEST_SHA:raise ValueError('Rank/checkpoint mismatch')
    files=rank['files']
    if any(manifest['files'].get(name)!=receipt for name,receipt in files.items()):
        raise ValueError('Rank file differs from the checkpoint inventory')
    remaining=sum(r['bytes'] for n,r in files.items() if not safe_path(root,n).exists())
    free=os.statvfs(root).f_bavail*os.statvfs(root).f_frsize
    if free<remaining+2*1024**3:raise RuntimeError('Insufficient disk for this rank plus 2 GiB headroom')
    for index,(name,r) in enumerate(files.items(),1):
        download(name,safe_path(root,name),r['sha256'],r['bytes'])
        if index%25==0:print(f'Rank {args.rank}: verified {index}/{len(files)} files',flush=True)
    print(json.dumps({'rank':args.rank,'files':len(files),'bytes':sum(r['bytes'] for r in files.values()),'revision':REVISION}),flush=True)

if __name__=='__main__':main()

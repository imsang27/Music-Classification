"""Install pinned public MuQ assets; never distribute weights in this repo."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from music_classification.muq_runtime import MANIFEST_PINS


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def child(root,name):
    target=(Path(root)/name.replace('\\','/')).resolve()
    if not target.is_relative_to(Path(root).resolve()):raise ValueError('Unsafe manifest path')
    return target


def metadata():
    base=ROOT/'setup_assets/muq';result={}
    for name,sha in MANIFEST_PINS.items():
        path=base/name
        if digest(path)!=sha:raise ValueError('Public integrity metadata changed: '+name)
        result[name]=json.loads(path.read_text(encoding='utf-8'))
    return result


def materialize(target,expected,url,*,cache=None,offline=False,size=None):
    target=Path(target)
    if target.exists():
        if digest(target)!=expected:raise ValueError('Existing file differs; do not overwrite: '+str(target))
        return
    target.parent.mkdir(parents=True,exist_ok=True)
    if cache is not None and Path(cache).is_file() and digest(cache)==expected:
        try:os.link(cache,target)
        except OSError:shutil.copyfile(cache,target)
        return
    if offline:raise ValueError('Verified cached file not found: '+str(target))
    partial=target.with_name(target.name+'.download')
    offset=partial.stat().st_size if partial.exists() else 0
    request=urllib.request.Request(url,headers={'Range':f'bytes={offset}-'} if offset else {})
    print('Downloading '+target.name,flush=True)
    with urllib.request.urlopen(request,timeout=90) as response:
        append=bool(offset and response.status==206)
        with partial.open('ab' if append else 'wb') as output:
            received=offset if append else 0
            while chunk:=response.read(1024*1024):
                received+=len(chunk)
                if size is not None and received>size:raise ValueError('Asset exceeds expected size')
                output.write(chunk)
    if (size is not None and partial.stat().st_size!=size) or digest(partial)!=expected:
        raise ValueError('Downloaded checksum/size mismatch: '+str(target))
    partial.replace(target)


def verify_installation(destination,records):
    destination=Path(destination)
    for name,sha in MANIFEST_PINS.items():
        if digest(destination/name)!=sha:raise ValueError('Wrong installation manifest: '+name)
    for item in records['assets.json']:
        if digest(child(destination,item['file']))!=item['sha256']:raise ValueError('Wrong model asset')
    for name,sha in records['runtime-manifest.json']['files_sha256'].items():
        if digest(child(destination/'runtime',name))!=sha:raise ValueError('Wrong runtime source: '+name)


def install(destination,*,asset_cache=None,offline=False):
    records=metadata();destination=Path(destination).resolve();destination.mkdir(parents=True,exist_ok=True)
    cache=Path(asset_cache).resolve() if asset_cache else None
    for item in records['assets.json']:
        materialize(child(destination,item['file']),item['sha256'],item['url'],
            cache=child(cache,item['file']) if cache else None,offline=offline,size=item['bytes'])
    source=records['source-manifest.json'];runtime=records['runtime-manifest.json']
    for item in source['files']:
        name=item['path']
        url=f"https://raw.githubusercontent.com/tencent-ailab/MuQ/{source['revision']}/{name}"
        materialize(child(destination/'upstream',name),item['sha256'],url,
            cache=child(cache/'upstream',name) if cache else None,offline=offline)
    for name,expected in runtime['files_sha256'].items():
        target=child(destination/'runtime',name)
        if target.exists():
            if digest(target)!=expected:raise ValueError('Existing patched source differs: '+name)
            continue
        text=child(destination/'upstream',name).read_text(encoding='utf-8')
        for patch in runtime['patches']:
            if patch['file']==name:
                if not text.count(patch['old']):raise ValueError('Unexpected upstream source: '+name)
                text=text.replace(patch['old'],patch['new'])
        encodings=[text.encode('utf-8'),text.replace('\n','\r\n').encode('utf-8')]
        matched=next((data for data in encodings if hashlib.sha256(data).hexdigest()==expected),None)
        if matched is None:
            raise ValueError('Patched runtime hash mismatch: '+name)
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(matched)
    for name in MANIFEST_PINS:
        body=(ROOT/'setup_assets/muq'/name).read_bytes()
        path=destination/name
        if path.exists() and path.read_bytes()!=body:raise ValueError('Different installation manifest: '+name)
        path.write_bytes(body)
    verify_installation(destination,records)
    return destination


def write_settings(path,installation,python):
    path=Path(path);root=path.parent.resolve()
    data={'version':1,'muq':{'installation':str(Path(installation).resolve()),
        'python_executable':str(Path(python).resolve()),'work_directory':str(ROOT/'work/inference-requests'),
        'cache_directory':str(ROOT/'work/inference-cache'),'timeout':600}}
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8'))!=data:raise ValueError('Local settings already exist; choose --settings with a new path')
        return
    root.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description='Prepare the pinned MuQ installation and local-only settings.')
    parser.add_argument('--installation',type=Path,default=ROOT/'work/muq-experiment')
    parser.add_argument('--settings',type=Path,default=ROOT/'config/model_paths.local.json')
    parser.add_argument('--python',type=Path,default=Path(sys.executable))
    parser.add_argument('--asset-cache',type=Path,help='Optional verified installation to reuse via hard links or copies')
    parser.add_argument('--offline',action='store_true');parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args()
    if args.verify_only:verify_installation(args.installation,metadata())
    else:
        install(args.installation,asset_cache=args.asset_cache,offline=args.offline)
        write_settings(args.settings,args.installation,args.python)
    print('Verified MuQ installation: '+str(args.installation.resolve()))


if __name__=='__main__':main()

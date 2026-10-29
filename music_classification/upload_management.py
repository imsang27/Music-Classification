"""Inventory and reversible cleanup of the explicitly configured upload folder."""
import hashlib
import json
from pathlib import Path
import shutil
from uuid import uuid4

EXTENSIONS={'.wav','.mp3','.flac','.ogg','.m4a','.aac'}


class Uploads:
    def __init__(self, root, directory):
        self.root=Path(root).resolve()
        self.directory=Path(directory).resolve()
        self.trash=self.root/'work/upload-trash'

    def inventory(self):
        if self.directory.resolve()!=self.directory:
            raise ValueError('업로드 폴더 경로가 바뀌었습니다.')
        rows=[]
        if self.directory.exists():
            for path in sorted(self.directory.iterdir()):
                if path.is_symlink() or not path.is_file() or path.suffix.lower() not in EXTENSIONS:continue
                stat=path.stat()
                rows.append({'name':path.name,'bytes':stat.st_size,'modified_ns':stat.st_mtime_ns})
        version=hashlib.sha256(json.dumps(rows,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
        return {'files':rows,'count':len(rows),'bytes':sum(r['bytes'] for r in rows),'version':version,
                'retention':'새 분석용 오디오는 처리 후 삭제합니다. 아래 목록은 기존 보관 파일입니다.'}

    def recycle(self, version):
        state=self.inventory()
        if state['version']!=version:raise ValueError('파일 목록이 바뀌었습니다. 새로고침한 뒤 다시 시도해주세요.')
        if not state['files']:raise ValueError('정리할 파일이 없습니다.')
        if len(state['files'])>1000:raise ValueError('한 번에 정리할 수 있는 파일 수를 초과했습니다.')
        identifier=uuid4().hex; target=self.trash/identifier; target.mkdir(parents=True,exist_ok=False)
        moved=[]
        try:
            for row in state['files']:
                path=self.directory/row['name']
                if path.resolve().parent!=self.directory or path.is_symlink():raise ValueError('파일 경로를 확인할 수 없습니다.')
                shutil.move(str(path),str(target/path.name));moved.append(path.name)
            (target/'record.json').write_text(json.dumps({'directory':str(self.directory),'names':moved},ensure_ascii=False),encoding='utf-8')
        except Exception:
            for name in reversed(moved):shutil.move(str(target/name),str(self.directory/name))
            raise
        return {'id':identifier,'count':len(moved),'message':'보관 파일을 복구 가능한 보관함으로 이동했습니다.'}

    def restore(self, identifier):
        if not isinstance(identifier,str) or len(identifier)!=32 or any(c not in '0123456789abcdef' for c in identifier):
            raise ValueError('복구 기록을 확인해주세요.')
        target=self.trash/identifier
        if target.is_symlink() or target.resolve().parent!=self.trash.resolve():raise ValueError('복구 폴더 경로를 확인할 수 없습니다.')
        record=json.loads((target/'record.json').read_text(encoding='utf-8'))
        if record['directory']!=str(self.directory):raise ValueError('현재 업로드 폴더와 다른 기록입니다.')
        self.inventory()
        names=record['names']
        if not names:raise ValueError('이미 복구한 기록입니다.')
        for name in names:
            if Path(name).name!=name or '\\' in name or (self.directory/name).exists() or not (target/name).is_file() or (target/name).is_symlink():
                raise ValueError('같은 이름의 파일이 있거나 복구할 파일을 확인할 수 없습니다.')
        self.directory.mkdir(parents=True,exist_ok=True);moved=[]
        try:
            for name in names:shutil.move(str(target/name),str(self.directory/name));moved.append(name)
        except Exception:
            for name in reversed(moved):shutil.move(str(self.directory/name),str(target/name))
            raise
        record['names']=[]
        (target/'record.json').write_text(json.dumps(record,ensure_ascii=False),encoding='utf-8')
        return {'count':len(moved),'message':'업로드 파일을 원래 폴더로 복구했습니다.'}

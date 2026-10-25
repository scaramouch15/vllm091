"""Package audit/clone logs and kit, excluding the large cloned environment."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();root=a.work.resolve();out=a.out.resolve()
    if out.is_relative_to(root):raise ValueError('Put the return archive outside the work directory')
    files=sorted(x for x in root.rglob('*') if x.is_file() and '__pycache__' not in x.parts)
    if any(x.is_symlink() for x in root.rglob('*')):raise ValueError('Do not package symlinks or environment directories')
    manifest={x.relative_to(root).as_posix():hashlib.sha256(x.read_bytes()).hexdigest() for x in files}
    with tarfile.open(out,'x:gz') as t:
        for x in files:t.add(x,arcname='r1_service_compat_return/'+x.relative_to(root).as_posix(),recursive=False)
        data=json.dumps(manifest,indent=2).encode('utf-8');info=tarfile.TarInfo('r1_service_compat_return/returned_files_sha256.json');info.size=len(data)
        t.addfile(info,io.BytesIO(data))
    digest=hashlib.sha256(out.read_bytes()).hexdigest()
    out.with_name(out.name+'.sha256').write_text(digest+'  '+out.name+'\n',encoding='ascii',newline='\n')
    print(str(out)+'\n'+str(out)+'.sha256')


if __name__=='__main__':main()

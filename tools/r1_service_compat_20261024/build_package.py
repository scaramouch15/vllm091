"""Build a source-only kit from exact committed Git blobs, never working files."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

ROOT=Path(__file__).resolve().parent
PACKAGE_ROOT='r1_service_compat_20261024'
FILES=(
    '.gitignore','README.md','review_report_20261025.md','TESTING.md',
    'inspect_service.py','environment_probe.py','cpu_checks.py',
    'clone_validate.py','pack_checks.py','test_checks.py','build_package.py',
    'predictor_requirements.txt','source_baseline.json',
)


def git(repo,*args):
    return subprocess.check_output(['git','-C',str(repo),*args])


def payloads(repo,revision,prefix):
    commit=git(repo,'rev-parse','--verify','--end-of-options',revision+'^{commit}').decode().strip()
    result={name:git(repo,'show',commit+':'+prefix+'/'+name) for name in FILES}
    baseline=json.loads(result['source_baseline.json'])
    provenance=dict(repository='https://github.com/scaramouch15/vllm091',source_commit=commit,
        source_directory=prefix,runtime_baseline_commit=baseline['state']['head'],
        content='Committed source/docs only; no server results, environments, models or data')
    result['SOURCE_COMMIT.json']=(json.dumps(provenance,indent=2)+'\n').encode('utf-8')
    manifest={name:hashlib.sha256(data).hexdigest() for name,data in sorted(result.items())}
    result['files_sha256.json']=(json.dumps(manifest,indent=2)+'\n').encode('utf-8')
    return commit,result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--revision',default='HEAD',help='Commit/ref containing this tool directory (default HEAD)')
    p.add_argument('--out-dir',type=Path,required=True,help='New directory outside the repository; parent must exist')
    a=p.parse_args()
    repo=Path(git(ROOT,'rev-parse','--show-toplevel').decode().strip()).resolve()
    prefix=ROOT.relative_to(repo).as_posix()
    commit,contents=payloads(repo,a.revision,prefix)
    out=a.out_dir.resolve()
    if out.is_relative_to(repo):raise ValueError('Keep generated packages outside the repository')
    out.mkdir(exist_ok=False)
    archive=out/'r1_service_compat_20261025_tools.tar.gz'
    with tarfile.open(archive,'x:gz') as t:
        for name,data in sorted(contents.items()):
            member=tarfile.TarInfo(PACKAGE_ROOT+'/'+name)
            member.size=len(data);member.mode=0o644;member.mtime=0
            t.addfile(member,io.BytesIO(data))
    digest=hashlib.sha256(archive.read_bytes()).hexdigest()
    with archive.with_name(archive.name+'.sha256').open('x',encoding='ascii',newline='\n') as f:
        f.write(digest+'  '+archive.name+'\n')
    print('Source commit:',commit)
    print('Runtime baseline:',json.loads(contents['SOURCE_COMMIT.json'])['runtime_baseline_commit'])
    print(archive)
    print(str(archive)+'.sha256')


if __name__=='__main__':main()

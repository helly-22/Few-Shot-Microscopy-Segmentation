"""Monitor only recorded jobs and retrieve the finished, checksummed report.

No new jobs are submitted. Run from the local project after creating
outputs/runtime/unity-checklist.json. Uses the user's existing ssh unity alias.
"""
import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
METADATA = ROOT / 'outputs/runtime/unity-checklist.json'
LOG = ROOT / 'outputs/runtime/checklist-watch.log'


def update(**changes):
    metadata = json.loads(METADATA.read_text())
    metadata.update(changes, last_verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    temporary = METADATA.with_suffix('.tmp')
    temporary.write_text(json.dumps(metadata, indent=2) + '\n')
    temporary.replace(METADATA)


def log(message):
    line = datetime.datetime.now(datetime.timezone.utc).isoformat() + ' ' + message
    with LOG.open('a', encoding='utf-8') as output:
        output.write(line + '\n')
    print(line, flush=True)


def ssh(command):
    return subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                           'unity', command], check=True, capture_output=True,
                          text=True, timeout=45).stdout.strip()


def main():
    metadata = json.loads(METADATA.read_text())
    jobs = metadata['jobs']
    run_root = metadata['run_root']
    if (metadata['ssh_alias'] != 'unity'
            or not all(re.fullmatch(r'\d+', str(value)) for value in jobs.values())
            or not re.fullmatch(r'/work/[A-Za-z0-9_./-]+', run_root)
            or '..' in Path(run_root).parts):
        raise ValueError('Unexpected SSH target, job IDs or remote run path')
    query = 'sacct -Xn -j ' + ','.join(jobs.values()) + ' --format=JobIDRaw,State,ExitCode -P'
    previous, errors = None, 0
    deadline = time.monotonic() + 24 * 3600
    update(watcher_status='monitoring')
    while time.monotonic() < deadline:
        try:
            states = {}
            for line in ssh(query).splitlines():
                fields = line.split('|')
                if len(fields) >= 3 and fields[0].strip() in jobs.values():
                    states[fields[0].strip()] = dict(state=fields[1].strip(), exit_code=fields[2].strip())
            errors = 0
            update(slurm_states=states)
            if states != previous:
                log(json.dumps(states, sort_keys=True))
                previous = states
            if any(value['state'].split()[0].rstrip('+') in
                   ('FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'DEPENDENCY')
                   for value in states.values()):
                update(watcher_status='stopped_job_failure')
                log('A recorded job failed or was cancelled; inspect Slurm logs. No resubmission performed.')
                return
            report = states.get(jobs['report'], {})
            if report.get('state') == 'COMPLETED' and report.get('exit_code') == '0:0':
                break
        except (subprocess.SubprocessError, OSError) as error:
            errors += 1
            log(f'Transient SSH error {errors}/5: {type(error).__name__}')
            if errors >= 5:
                update(watcher_status='stopped_connection_errors')
                return
        time.sleep(60)
    else:
        update(watcher_status='stopped_24_hour_timeout')
        return
    remote_archive = run_root + '/checklist_results.zip'
    expected = ssh('sha256sum ' + remote_archive).split()[0]
    if not re.fullmatch(r'[a-f0-9]{64}', expected):
        raise ValueError('Invalid remote checksum')
    destination = ROOT / 'outputs/checklist_results' / run_root.rsplit('/', 1)[1]
    destination.mkdir(parents=True, exist_ok=True)
    temporary = destination / 'checklist_results.download'
    subprocess.run(['scp', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                    'unity:' + remote_archive, str(temporary)], check=True, timeout=600)
    digest = hashlib.sha256()
    with temporary.open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(block)
    actual = digest.hexdigest()
    if actual != expected:
        raise ValueError('Downloaded report checksum differs')
    archive = destination / 'checklist_results.zip'
    if archive.exists() and hashlib.sha256(archive.read_bytes()).hexdigest() != actual:
        raise FileExistsError('A different report archive already exists')
    temporary.replace(archive)
    with zipfile.ZipFile(archive) as file:
        for entry in file.infolist():
            if not (destination / entry.filename).resolve().is_relative_to(destination.resolve()):
                raise ValueError('Unsafe report archive path')
        file.extractall(destination)
    update(watcher_status='completed_report_downloaded', local_report=str(destination / 'report/report.html'),
           local_report_archive=str(archive), report_archive_sha256=actual)
    log(f'Report downloaded and verified: {destination / "report/report.html"}')


if __name__ == '__main__':
    main()

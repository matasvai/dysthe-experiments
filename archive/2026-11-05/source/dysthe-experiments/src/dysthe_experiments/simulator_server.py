"""Loopback-only workbench for bounded spatial-Dysthe simulations.

Precomputed playback reads saved complex trajectories. POST /api/run launches
the same CPU solver in one worker, with fixed bounded settings by default.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import re
import threading
import time
from urllib.parse import parse_qs, urlsplit
import uuid

from . import simulator


class Workbench:
    def __init__(self, data):
        self.data = Path(data).resolve()
        self.web = Path(__file__).with_name('simulator_web')
        self.jobs = {}
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='dysthe-local')
        self.preview_slots = threading.BoundedSemaphore(2)

    def precomputed(self, case_id):
        if case_id not in simulator.CASE_IDS:
            raise ValueError('Unknown envelope case')
        return self.data/case_id

    def cases(self):
        cases = simulator.case_info()
        for case in cases:
            folder = self.precomputed(case['id'])
            case['precomputed_available'] = (folder/'viewer.json').is_file()
            if case['precomputed_available']:
                case['qualification'] = json.loads((folder/'viewer.json').read_text()).get('qualification', {})
        return {'cases': cases, 'runtime': {'model_id': 'water-wave-dysthe-spatial-v1',
            'execution': 'Local CPU; one bounded run at a time',
            'data_directory': str(self.data)}}

    def load_viewer(self, folder, source, download_url):
        file = folder/'viewer.json'
        if not file.is_file():
            raise FileNotFoundError('Precomputed trajectory is not available; run the local example first')
        result = json.loads(file.read_text())
        result['source'] = source
        result['download_url'] = download_url
        return result

    def preview(self, payload):
        if not isinstance(payload,dict) or set(payload)-{'case_id','parameters','settings'}:
            raise ValueError('Expected custom case_id, parameters and optional bounded settings')
        if payload.get('case_id')!='custom':
            raise ValueError('Formula preview is available for the custom envelope case')
        if not self.preview_slots.acquire(blocking=False):
            raise RuntimeError('Two formula previews are already being evaluated; try again shortly')
        try:
            return simulator.preview_custom(payload.get('parameters'),payload.get('settings'))
        finally:
            self.preview_slots.release()

    def submit(self, payload):
        if not isinstance(payload, dict) or set(payload)-{'case_id','parameters','settings','preview_fingerprint'}:
            raise ValueError('Expected case_id, parameters, optional bounded settings and custom preview_fingerprint')
        case = payload.get('case_id')
        if not isinstance(case,str) or case not in simulator.CASE_IDS:
            raise ValueError('Unknown envelope case')
        parameters, settings = simulator.validate_request(case,payload.get('parameters'),payload.get('settings'))
        if case=='custom':
            # A hash prevents accidentally running a stale preview. It is not
            # trusted as validation: resample and recheck independently here.
            preview=self.preview(dict(case_id=case,parameters=parameters,settings=settings))
            if payload.get('preview_fingerprint')!=preview['fingerprint']:
                raise ValueError('Preview this exact formula, parameter values and numerical setup before Run')
            if not preview['can_run']:
                raise ValueError('Custom input blocked: '+'; '.join(w['message'] for w in preview['warnings'] if w['blocks_run']))
        with self.lock:
            if any(j['status'] in ('queued','running') for j in self.jobs.values()):
                raise RuntimeError('A local simulation is already running; wait for it to finish')
            job_id = uuid.uuid4().hex
            self.jobs[job_id] = {'job_id':job_id,'case_id':case,'status':'queued',
                'progress':None,'created_at':time.time(),'source':'live'}
        self.executor.submit(self._run,job_id,case,parameters,settings)
        return self.job(job_id)

    def _run(self, job_id, case, parameters, settings):
        folder = self.data/'live'/job_id
        def progress(value):
            with self.lock:
                self.jobs[job_id]['progress'] = value
        with self.lock:
            self.jobs[job_id]['status'] = 'running'
        try:
            simulator.simulate(case,folder,parameters=parameters,settings=settings,progress=progress)
            with self.lock:
                self.jobs[job_id].update(status='complete',finished_at=time.time(),
                    result_url=f'/api/results/{job_id}',download_url=f'/api/jobs/{job_id}/download')
        except Exception as exc:
            with self.lock:
                self.jobs[job_id].update(status='error',error=str(exc),finished_at=time.time())

    def job(self, job_id):
        with self.lock:
            if job_id not in self.jobs:
                # Keep completed download/result links working across this
                # service restart. Only server-generated UUID directory names
                # may resolve inside the existing live-results directory.
                if not isinstance(job_id,str) or re.fullmatch(r'[0-9a-f]{32}',job_id) is None:
                    raise KeyError('Unknown local run')
                folder=self.data/'live'/job_id
                if not all((folder/name).is_file() for name in ('report.json','viewer.json','fields.npz')):
                    raise KeyError('Unknown local run')
                report=json.loads((folder/'report.json').read_text())
                self.jobs[job_id]=dict(job_id=job_id,case_id=report['case_id'],
                    status='complete',source='live',restored_from_disk=True,
                    result_url=f'/api/results/{job_id}',download_url=f'/api/jobs/{job_id}/download')
            return dict(self.jobs[job_id])


def make_handler(workbench):
    class Handler(BaseHTTPRequestHandler):
        server_version = 'DystheWorkbench'

        def send_json(self, data, status=200):
            body = json.dumps(data,allow_nan=False).encode()
            self.send_response(status); self.send_header('Content-Type','application/json; charset=utf-8')
            self.send_header('Content-Length',str(len(body))); self.send_header('Cache-Control','no-store')
            self.end_headers(); self.wfile.write(body)

        def send_file(self,path,download=False):
            if not path.is_file():
                raise FileNotFoundError('File is not available')
            self.send_response(200)
            self.send_header('Content-Type',mimetypes.guess_type(str(path))[0] or 'application/octet-stream')
            self.send_header('Content-Length',str(path.stat().st_size))
            if download:
                self.send_header('Content-Disposition','attachment; filename="dysthe-complex-fields.npz"')
            self.end_headers()
            with path.open('rb') as stream:
                while block := stream.read(1024*1024):
                    self.wfile.write(block)

        def do_GET(self):
            parsed=urlsplit(self.path); path=parsed.path; query=parse_qs(parsed.query)
            try:
                if path=='/api/cases':
                    return self.send_json(workbench.cases())
                if path in ('/api/data','/api/download'):
                    case=query.get('case',[''])[0]; folder=workbench.precomputed(case)
                    if path=='/api/download':
                        return self.send_file(folder/'fields.npz',download=True)
                    return self.send_json(workbench.load_viewer(folder,'precomputed',f'/api/download?case={case}'))
                if path.startswith('/api/jobs/'):
                    suffix=path[len('/api/jobs/'):].split('/')
                    job=workbench.job(suffix[0])
                    if len(suffix)==1:
                        return self.send_json(job)
                    if len(suffix)==2 and suffix[1]=='download' and job['status']=='complete':
                        return self.send_file(workbench.data/'live'/suffix[0]/'fields.npz',download=True)
                    raise FileNotFoundError('Run data is not available yet')
                if path.startswith('/api/results/'):
                    job_id=path[len('/api/results/'):]; job=workbench.job(job_id)
                    if job['status']!='complete':
                        raise FileNotFoundError('Run is not complete')
                    return self.send_json(workbench.load_viewer(workbench.data/'live'/job_id,
                        'live',f'/api/jobs/{job_id}/download'))
                assets={'/':'index.html','/index.html':'index.html','/app.js':'app.js','/style.css':'style.css'}
                if path in assets:
                    return self.send_file(workbench.web/assets[path])
                raise FileNotFoundError('Route not found')
            except (ValueError,KeyError) as exc:
                self.send_json({'error':str(exc)},400)
            except FileNotFoundError as exc:
                self.send_json({'error':str(exc)},404)

        def do_POST(self):
            if self.path not in ('/api/run','/api/preview'):
                return self.send_json({'error':'Route not found'},404)
            origin=self.headers.get('Origin')
            host=self.headers.get('Host')
            if origin and origin not in (f'http://{host}',f'https://{host}'):
                return self.send_json({'error':'Local workbench requests must use the same origin'},403)
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=16384:
                    raise ValueError('A small JSON run request is required')
                def reject_constant(value):
                    raise ValueError(f'JSON must contain finite numbers; {value} is not supported')
                payload=json.loads(self.rfile.read(size),parse_constant=reject_constant)
                if self.path=='/api/preview':
                    return self.send_json(workbench.preview(payload))
                return self.send_json(workbench.submit(payload),202)
            except RuntimeError as exc:
                self.send_json({'error':str(exc)},409)
            except (ValueError,TypeError,RecursionError,OverflowError) as exc:
                self.send_json({'error':str(exc)},400)

    return Handler


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True,help='Folder with profile/perturbed/gaussian saved examples')
    parser.add_argument('--port',type=int,default=8765)
    args=parser.parse_args()
    if not 1024<=args.port<=65535:
        parser.error('Choose a local port between 1024 and 65535')
    workbench=Workbench(args.data)
    server=ThreadingHTTPServer(('127.0.0.1',args.port),make_handler(workbench))
    print(f'Dysthe workbench: http://127.0.0.1:{args.port}\nPrecomputed directory: {workbench.data}\nLive runs use the local CPU; Ctrl-C stops the server.',flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close();workbench.executor.shutdown(wait=True)


if __name__=='__main__':
    main()

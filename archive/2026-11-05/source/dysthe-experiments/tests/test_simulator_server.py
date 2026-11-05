"""HTTP contract tests exercise a real short Gaussian solve and complex download."""
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
from dysthe_experiments.simulator_server import Workbench, make_handler
from dysthe_experiments.simulator import CASE_IDS


class SimulatorServerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.workbench=Workbench(Path(self.temp.name))
        self.server=ThreadingHTTPServer(('127.0.0.1',0),make_handler(self.workbench))
        self.base=f'http://127.0.0.1:{self.server.server_port}'
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown();self.server.server_close()
        self.workbench.executor.shutdown(wait=True)
        self.temp.cleanup()

    def request(self,path,payload=None,headers=None):
        req=Request(self.base+path,data=None if payload is None else json.dumps(payload).encode(),
                    headers={'Content-Type':'application/json',**(headers or {})})
        with urlopen(req,timeout=30) as response:
            return response.status,response.read()

    def test_real_local_run_then_download_preserves_complex_field(self):
        status,body=self.request('/api/cases')
        self.assertEqual(status,200)
        self.assertEqual(len(json.loads(body)['cases']),len(CASE_IDS))
        status,body=self.request('/api/run',dict(case_id='gaussian',settings=dict(
            n=128,length=40.,dxi=.0025,xi_end=.02,samples=3)))
        self.assertEqual(status,202);job=json.loads(body)
        deadline=time.monotonic()+30
        while time.monotonic()<deadline:
            _,body=self.request('/api/jobs/'+job['job_id']);job=json.loads(body)
            if job['status'] in ('complete','error'):
                break
            threading.Event().wait(.01)
        self.assertEqual(job['status'],'complete',job)
        _,body=self.request(job['result_url']);viewer=json.loads(body)
        self.assertEqual(viewer['source'],'live')
        self.assertIn('unqualified',viewer['qualification']['status'])
        self.assertEqual(len(viewer['real']),3)
        _,body=self.request(job['download_url'])
        with np.load(io.BytesIO(body),allow_pickle=False) as archive:
            self.assertEqual(archive['u'].dtype,np.dtype('complex128'))
            self.assertEqual(archive['u'].shape,(3,128))
            self.assertGreater(np.max(abs(archive['u'].imag)),.1)

    def test_invalid_work_and_cross_origin_requests_are_rejected(self):
        for payload in (dict(case_id='optical'),dict(case_id='gaussian',parameters={'epsilon':.15}),
                        dict(case_id='gaussian',settings={'xi_end':100000})):
            with self.assertRaises(HTTPError) as context:
                self.request('/api/run',payload)
            self.assertEqual(context.exception.code,400)
        with self.assertRaises(HTTPError) as context:
            self.request('/api/run',dict(case_id='gaussian'),{'Origin':'https://unrelated.invalid'})
        self.assertEqual(context.exception.code,403)


if __name__=='__main__':
    unittest.main()

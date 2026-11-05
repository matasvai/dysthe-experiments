#!/usr/bin/env python3
"""Approved local-only Firefox WebDriver BiDi UI test.

Requires Firefox and Python websockets/numpy already installed. Start the local
simulator server, then run this script. It creates a fresh /tmp Firefox profile;
never reads a personal profile. No external site is opened. One bounded custom
Gaussian run is launched through the actual Run locally button.
Protocol references: https://firefox-source-docs.mozilla.org/remote/ and
https://w3c.github.io/webdriver-bidi/ (session, browsingContext, script, input).
"""
import argparse, asyncio, base64, json, os, signal, subprocess, tempfile, time
from pathlib import Path
import numpy as np
import websockets

class BiDi:
    def __init__(self, ws): self.ws=ws; self.id=0; self.context=None; self.events=[]
    async def command(self, method, params):
        self.id+=1; ident=self.id
        await self.ws.send(json.dumps(dict(id=ident,method=method,params=params)))
        while True:
            message=json.loads(await self.ws.recv())
            if message.get('id')!=ident: self.events.append(message); continue
            if message.get('type')=='error': raise RuntimeError(f'{method}: {message}')
            return message['result']
    async def evaluate(self, expression):
        result=await self.command('script.evaluate',dict(expression=f'(async()=>JSON.stringify(await ({expression})))()',target=dict(context=self.context),awaitPromise=True,userActivation=True))
        if result.get('type')=='exception': raise RuntimeError(str(result))
        value=result['result'].get('value')
        return json.loads(value) if value is not None else None
    async def body(self, body): return await self.evaluate(f'(async()=>{{{body}}})()')
    async def click(self, selector):
        point=await self.body(f"const e=document.querySelector({json.dumps(selector)});if(!e)throw Error('Missing control');e.scrollIntoView({{block:'center'}});const r=e.getBoundingClientRect();return {{x:Math.round(r.x+r.width/2),y:Math.round(r.y+r.height/2)}};")
        await self.command('input.performActions',dict(context=self.context,actions=[dict(type='pointer',id='qa-mouse',parameters=dict(pointerType='mouse'),actions=[dict(type='pointerMove',x=point['x'],y=point['y'],origin='viewport',duration=0),dict(type='pointerDown',button=0),dict(type='pointerUp',button=0)])]))
    async def change(self, selector, value, event='change'):
        return await self.body(f"const e=document.querySelector({json.dumps(selector)});e.value={json.dumps(str(value))};e.dispatchEvent(new Event({json.dumps(event)},{{bubbles:true}}));return e.value;")
    async def wait(self, expression, timeout=25):
        stop=time.monotonic()+timeout
        while time.monotonic()<stop:
            if await self.evaluate(expression): return
            await asyncio.sleep(.25)
        raise TimeoutError(expression)
    async def shot(self, path):
        result=await self.command('browsingContext.captureScreenshot',dict(context=self.context,origin='document'))
        path.write_bytes(base64.b64decode(result['data']))
    async def viewport(self,w,h):
        await self.command('browsingContext.setViewport',dict(context=self.context,viewport=dict(width=w,height=h),devicePixelRatio=1))
        await asyncio.sleep(.3)

async def run(args):
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True);downloads=out/'downloads';downloads.mkdir(exist_ok=True)
    profile=Path(tempfile.mkdtemp(prefix='dysthe-firefox-ui-'))
    prefs={'focusmanager.testmode':True,'browser.download.folderList':2,'browser.download.dir':str(downloads),'browser.download.useDownloadDir':True,'browser.download.alwaysOpenPanel':False,'browser.helperApps.neverAsk.saveToDisk':'application/octet-stream,application/zip','browser.shell.checkDefaultBrowser':False,'browser.startup.homepage_override.mstone':'ignore','datareporting.policy.dataSubmissionEnabled':False,'toolkit.telemetry.reportingpolicy.firstRun':False,'toolkit.telemetry.enabled':False,'toolkit.telemetry.unified':False,'toolkit.telemetry.server':'http://127.0.0.1:9','app.normandy.enabled':False,'app.shield.optoutstudies.enabled':False,'datareporting.healthreport.uploadEnabled':False,'extensions.update.enabled':False,'services.settings.server':'http://127.0.0.1:9','network.captive-portal-service.enabled':False,'network.connectivity-service.enabled':False,'browser.newtabpage.enabled':False,'browser.newtabpage.activity-stream.feeds.telemetry':False}
    (profile/'user.js').write_text(''.join(f'user_pref({json.dumps(k)}, {json.dumps(v)});\n' for k,v in prefs.items()))
    report={'url':args.url,'profile':str(profile),'method':'Firefox WebDriver BiDi; actual rendered page and input events','screenshots':[],'checks':[]}
    log=(out/'firefox.log').open('w')
    process=subprocess.Popen([args.firefox,'--headless','--no-remote','--profile',str(profile),f'--remote-debugging-port={args.port}','about:blank'],stdout=log,stderr=log)
    async def record(name,details):
        report['checks'].append(dict(name=name,passed=True,details=details));print(name+': PASS',flush=True)
    async def shot(browser,name):
        await browser.body('window.scrollTo(0,0);return true;');await asyncio.sleep(.15)
        path=out/name;await browser.shot(path);report['screenshots'].append(str(path));print('Screenshot '+name,flush=True)
    try:
        ws=None
        for _ in range(80):
            try: ws=await websockets.connect(f'ws://127.0.0.1:{args.port}/session',max_size=64*1024*1024);break
            except OSError: await asyncio.sleep(.25)
        if ws is None: raise RuntimeError('Temporary Firefox automation endpoint did not start')
        async with ws:
            browser=BiDi(ws)
            session=await browser.command('session.new',dict(capabilities={}));report['browser']=session['capabilities']
            tree=await browser.command('browsingContext.getTree',{});browser.context=tree['contexts'][0]['context']
            await browser.command('session.subscribe',dict(events=['log.entryAdded']))
            await browser.viewport(1440,1000)
            await browser.command('browsingContext.navigate',dict(context=browser.context,url=args.url,wait='complete'))
            await browser.wait("!document.getElementById('viewer-content').hidden && document.getElementById('source-badge').textContent.includes('PRECOMPUTED')")
            contrast=await browser.body("const lin=v=>{v/=255;return v<=.04045?v/12.92:((v+.055)/1.055)**2.4};const lum=c=>{const a=c.match(/[0-9.]+/g).slice(0,3).map(Number).map(lin);return .2126*a[0]+.7152*a[1]+.0722*a[2]};return ['.intro-copy','.case-button small','.panel-title h2','.button.primary'].map(sel=>{const e=document.querySelector(sel),s=getComputedStyle(e);let parent=e,bg=s.backgroundColor;while(bg==='rgba(0, 0, 0, 0)'&&parent.parentElement){parent=parent.parentElement;bg=getComputedStyle(parent).backgroundColor}const a=lum(s.color),b=lum(bg);return {selector:sel,foreground:s.color,background:bg,ratio:(Math.max(a,b)+.05)/(Math.min(a,b)+.05)}});")
            assert all(item['ratio']>=4.5 for item in contrast),contrast
            await record('Rendered text contrast at least 4.5:1',contrast)
            await record('Initial reference loads',await browser.body("return {title:document.getElementById('result-title').textContent,source:document.getElementById('source-badge').textContent,canvas:[document.getElementById('envelope-chart').width,document.getElementById('envelope-chart').height]};"))
            await shot(browser,'desktop-profile-initial.png')
            for case_id in ['profile','perturbed','gaussian']:
                await browser.click(f'[data-case-id="{case_id}"]')
                await browser.wait(f"state.data?.case_id === '{case_id}' && document.getElementById('source-badge').textContent.includes('PRECOMPUTED')")
                await browser.change('#snapshot',80,'input')
                await browser.wait("document.getElementById('xi-value').textContent.includes('4')")
                for mode in ['amplitude','complex','phase']: await browser.change('#field-mode',mode)
                assert not await browser.evaluate("document.getElementById('phase-note').hidden")
                await browser.change('#field-mode','amplitude')
                for domain in ['domain','packet']: await browser.change('#view-range',domain)
                await browser.change('#heatmap-scale','log')
                details=await browser.body("return {title:document.getElementById('result-title').textContent,source:document.getElementById('source-badge').textContent,peak:document.getElementById('stat-peak').textContent,shape:document.getElementById('stat-shape').textContent,qualification:document.getElementById('qualification-badge').textContent,rows:document.querySelectorAll('.qualification-table tbody tr').length};")
                assert details['qualification']=='CHECKS PASSED';assert details['rows']==3
                await record('Case and display controls: '+case_id,details)
                await shot(browser,f'desktop-{case_id}-final.png')
                if case_id=='gaussian':
                    await browser.change('#field-mode','phase');await shot(browser,'desktop-gaussian-phase-final.png');await browser.change('#field-mode','amplitude')
            await browser.change('#snapshot',0,'input');await browser.click('#play-pause');await asyncio.sleep(.8)
            playing=await browser.body("return {frame:Number(document.getElementById('snapshot').value),label:document.getElementById('play-pause').getAttribute('aria-label')};")
            assert playing['frame']>0 and 'Pause' in playing['label'];await browser.click('#play-pause')
            before=await browser.evaluate("document.getElementById('snapshot').value");await asyncio.sleep(.4);after=await browser.evaluate("document.getElementById('snapshot').value");assert before==after
            await record('Play, pause and scrub',playing)
            await browser.click('#heatmap-chart')
            clicked=await browser.evaluate("Number(document.getElementById('snapshot').value)");assert 25<clicked<60
            await record('Heatmap selects snapshot',clicked)
            # Activate the headless window before testing CSS focus pseudo-classes.
            await browser.command('browsingContext.activate',dict(context=browser.context))
            # Test a real keyboard Tab sequence and visible focus.
            await browser.body("document.activeElement.blur();document.querySelector('[data-case-id=profile]').focus();return true;")
            await browser.command('input.performActions',dict(context=browser.context,actions=[dict(type='key',id='qa-keyboard',actions=[dict(type='keyDown',value='\ue004'),dict(type='keyUp',value='\ue004')])]))
            focus=await browser.body("const s=getComputedStyle(document.activeElement);return {documentFocused:document.hasFocus(),focused:document.activeElement.matches(':focus'),element:document.activeElement.dataset.caseId||document.activeElement.id,visible:document.activeElement.matches(':focus-visible'),outline:s.outlineStyle,width:s.outlineWidth};")
            assert focus['element']=='perturbed',focus
            await record('Keyboard Tab navigation',focus)
            if focus['documentFocused']:
                assert focus['outline']!='none' and focus['width']!='0px',focus
                await record('Keyboard focus visibility',focus)
            else:
                report.setdefault('limitations',[]).append('Headless Firefox reports document.hasFocus()=false even after browsingContext.activate and the recommended focusmanager.testmode preference. Tab navigation works, but focus-outline rendering cannot be verified in this environment. The UI provides explicit 2px :focus outlines.')
            await browser.viewport(390,844)
            for case_id in ['profile','perturbed','gaussian']:
                await browser.click(f'[data-case-id="{case_id}"]');await browser.wait(f"state.data?.case_id === '{case_id}'")
                await browser.change('#snapshot',80,'input')
                overflow=await browser.body("return {viewport:window.innerWidth,scrollWidth:document.documentElement.scrollWidth,overflow:[...document.querySelectorAll('main *')].filter(e=>{const r=e.getBoundingClientRect();return r.width>0&&(r.right>innerWidth+1||r.left < -1)}).map(e=>({tag:e.tagName,id:e.id,class:e.className})).slice(0,10)};")
                assert overflow['scrollWidth']<=overflow['viewport'],overflow
                await record('Narrow layout: '+case_id,overflow)
            await shot(browser,'mobile-gaussian-final.png')
            await browser.click('details summary');await shot(browser,'mobile-definitions-open.png')
            if not args.skip_live:
                await browser.viewport(1440,1000)
                await browser.change('#param-amplitude',.8,'input');await browser.change('#param-width',2.2,'input');await browser.change('#param-wavenumber',.25,'input');await browser.change('#param-chirp',.01,'input')
                assert await browser.evaluate("[...document.querySelectorAll('#parameter-fields input')].every(e=>e.checkValidity())")
                await browser.click('#run-live')
                await browser.wait("document.getElementById('job-status-text').textContent.includes('Solver') || document.getElementById('job-status-text').textContent.includes('propag') || document.getElementById('source-badge').textContent.includes('LIVE')",timeout=25)
                progress=[];stop=time.monotonic()+100
                while time.monotonic()<stop:
                    status=await browser.body("const p=document.getElementById('job-progress');return {text:document.getElementById('job-status-text').textContent,source:document.getElementById('source-badge').textContent,value:p.hidden?null:p.value,max:p.hidden?null:p.max};")
                    if not progress or status!=progress[-1]:progress.append(status)
                    if status['source'].startswith('LIVE') and 'complete' in status['text'].lower():break
                    if 'failed' in status['text'].lower():raise RuntimeError(status['text'])
                    await asyncio.sleep(.5)
                else:raise TimeoutError('Live CPU run did not complete in 100 seconds')
                live=await browser.body("return {parameters:state.data.parameters,qualification:state.data.qualification.status,source:document.getElementById('source-badge').textContent,url:document.getElementById('download-data').href};")
                assert live['parameters']==dict(amplitude=.8,width=2.2,wavenumber=.25,chirp=.01),live
                assert 'unqualified' in live['qualification'].lower(),live
                live['progress']=progress;await record('Custom Gaussian local CPU run',live)
                await browser.change('#snapshot',80,'input');await browser.change('#field-mode','complex');await shot(browser,'desktop-live-custom-gaussian.png')
                oldfiles=set(downloads.glob('*.npz'));await browser.click('#download-data')
                stop=time.monotonic()+15;download=None
                while time.monotonic()<stop:
                    newfiles=set(downloads.glob('*.npz'))-oldfiles
                    if newfiles:
                        candidate=max(newfiles,key=lambda p:p.stat().st_mtime)
                        try:
                            with np.load(candidate) as arrays:
                                keys=list(arrays.files);complexes={k:dict(shape=list(arrays[k].shape),dtype=str(arrays[k].dtype)) for k in keys if np.iscomplexobj(arrays[k])}
                            if complexes:download=dict(file=str(candidate),keys=keys,complex_arrays=complexes);break
                        except (ValueError,OSError):pass
                    await asyncio.sleep(.25)
                assert download,'Complex NPZ was not downloaded through the browser';await record('Browser complex NPZ download',download)
            runtime_errors=[e for e in browser.events if e.get('method')=='log.entryAdded' and e.get('params',{}).get('level')=='error'];report['browser_errors']=runtime_errors
            assert not runtime_errors,runtime_errors
            await record('No browser script errors',[])
            report['passed']=True
            await browser.command('session.end',{})
    except Exception as exc:
        report['passed']=False;report['error']=str(exc);raise
    finally:
        (out/'browser-e2e.json').write_text(json.dumps(report,indent=2)+'\n')
        if process.poll() is None:
            process.terminate()
            try:process.wait(timeout=8)
            except subprocess.TimeoutExpired:process.kill();process.wait(timeout=5)
        log.close()

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--skip-live',action='store_true',help='Check rendered UI without another numerical run');parser.add_argument('--url',default='http://127.0.0.1:8765/');parser.add_argument('--firefox',default='/usr/sbin/firefox');parser.add_argument('--port',type=int,default=9223);parser.add_argument('--output',type=Path,default=Path(__file__).parent);asyncio.run(run(parser.parse_args()))

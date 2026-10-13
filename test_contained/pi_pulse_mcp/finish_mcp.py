"""Continue the completed four-case MCP experiment through transfer/persistence checks."""
import asyncio,json,os,pathlib,sys,time,traceback
import runpy
D=runpy.run_path('/Users/billabobz/OPTIMIZER/test_contained/pi_pulse_mcp/run_mcp.py',run_name='campaign_helpers')
globals().update({k:v for k,v in D.items() if not k.startswith('__')})
# Functions in the shared driver reference their owning namespace.
CASES=json.loads((OUT/'campaign-state.json').read_text());CHECKS=json.loads((OUT/'checks.json').read_text())
TRAJECTORIES={}
for name,c in CASES.items():
 u=json.loads((OUT/(c['binding']+'-controls.json')).read_text()) if name.startswith('g_') else [c['p0']]*N
 TRAJECTORIES[name]={'u':u,'pop':replay(u)['pop']}
for f in [log,check,render,plot]:f.__globals__.update(CASES=CASES,CHECKS=CHECKS,TRAJECTORIES=TRAJECTORIES)
async def continuation():
 global STATUS
 env={**os.environ,'PYTHONPATH':str(WORK),'TMPDIR':'/private/tmp'}
 params=StdioServerParameters(command='/Users/billabobz/.local/bin/optimizer-mcp',cwd=str(WORK),env=env)
 async with stdio_client(params) as (read,write):
  async with ClientSession(read,write) as client:
   await client.initialize();wire=Wire(client)
   attached=await wire.tool('optimizer_attach',{'path':str(EXPERIMENT)});assert attached['status']=='ok',attached
   try:
    configured=await wire.call('storage.configure',{'keep_recent':256})
    state=await wire.call('session.state');heads={name:state['state_after']['contexts'][name]['head'] for name in CASES}
    await wire.call('optimizers.adam',{'maxiter':2},context='s_hard',error='wrong_family')
    await wire.call('optimizers.projected_gradient',{'maxiter':2},context='g_soft',error='capability_missing')
    await wire.call('session.use',{'value':ref('s_hard_seed_0')},context='g_soft',error='incompatible_layout')

    for name in ['g_soft','g_hard','s_soft','s_hard']:
     await wire.call('checkpoints.save',{'value':ref(heads[name]),'name':'qualified_'+name,'kind':'controls' if name.startswith('g_') else 'variables','pin':True})
    detached=await wire.tool('optimizer_detach',{});check('MCP detach',detached['status']=='ok')
    attached=await wire.tool('optimizer_attach',{'path':str(EXPERIMENT)});check('MCP reconnect to live owner',attached['status']=='ok')
    for name in ['g_soft','g_hard']:
     u=np.asarray(await wire.inspect(heads[name]));check(name+' retained controls',np.array_equal(u,np.asarray(TRAJECTORIES[name]['u'])))
    await wire.call('jobs.shutdown')
    attached=await wire.tool('optimizer_attach',{'path':str(EXPERIMENT),'start_coordinator':True});check('saved experiment reopened',attached['status']=='ok')
    for name in ['g_soft','g_hard']:
     u=np.asarray(await wire.inspect(heads[name]));check(name+' reopened physical values',np.array_equal(u,np.asarray(TRAJECTORIES[name]['u'])))
    await wire.call('jobs.shutdown')

   finally:
    try:await wire.tool('optimizer_detach',{})
    except Exception:pass
 save('trajectories.json',TRAJECTORIES)
 history=EXPERIMENT/'history.jsonl';origins={}
 for line in history.read_text().splitlines():
  entry=json.loads(line)
  if entry.get('kind')!='submitted':continue
  origin=entry.get('data',{}).get('origin');origins[str(origin)]=origins.get(str(origin),0)+1
 save('history-origins.json',origins)
 STATUS=f'Completed: {sum(c["passed"] for c in CHECKS)}/{len(CHECKS)} checks passed'
 render.__globals__['STATUS']=STATUS
 save('results.json',{'status':STATUS,'cases':CASES,'checks':CHECKS,'seconds':time.monotonic()-START,'experiment':str(EXPERIMENT),'history_origins':origins,'transport':'official MCP client stdio; native assistant reload not verified'})
 plot();log(STATUS)
 if not all(c['passed'] for c in CHECKS):raise SystemExit(1)
if __name__=='__main__':
 try:asyncio.run(continuation())
 except BaseException as exc:
  STATUS='Continuation stopped: '+str(exc);render.__globals__['STATUS']=STATUS;log(STATUS);(OUT/'continuation-failure.txt').write_text(traceback.format_exc());raise

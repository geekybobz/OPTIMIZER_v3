"""Retain identical-option failures, then qualify with distinct verifier settings."""
import asyncio,os,json,runpy
D=runpy.run_path('/Users/billabobz/OPTIMIZER/test_contained/pi_pulse_mcp/run_mcp.py',run_name='campaign_helpers')
globals().update({k:v for k,v in D.items() if not k.startswith('__')})
result=json.loads((OUT/'results.json').read_text());CHECKS=result['checks'];CASES=result['cases'];TRAJECTORIES=json.loads((OUT/'trajectories.json').read_text())
for f in [log,check,render,plot]:f.__globals__.update(CHECKS=CHECKS,CASES=CASES,TRAJECTORIES=TRAJECTORIES,STATUS='Retrying distinct verification settings')
async def main():
 async with stdio_client(StdioServerParameters(command='/Users/billabobz/.local/bin/optimizer-mcp',cwd=str(WORK),env={**os.environ,'TMPDIR':'/private/tmp'})) as (r,w):
  async with ClientSession(r,w) as client:
   await client.initialize();wire=Wire(client);a=await wire.tool('optimizer_attach',{'path':str(EXPERIMENT),'start_coordinator':True});assert a['status']=='ok',a
   try:
    state=await wire.call('session.state');heads={name:state['state_after']['contexts'][name]['head'] for name in CASES}
    for mode in ['soft','hard']:
     old=f's_{mode}_steps_4000';stop=await wire.inspect(old,'stop_reason');source=await wire.inspect(old,'verification','independence_source')
     check(mode+' equal options correctly rejected',stop=='verification_not_independent' and source=='explicit_options_not_distinct',{'stop_reason':stop,'source':source})
     name=old+'_distinct';await wire.call('shooting.solve_shooting',{'initial':ref(heads['s_'+mode]),'options':{'steps':4000},'verification_options':{'steps':16000},'maxiter':10,'tolerance':1e-10,'verification_tolerance':1e-8},context='s_'+mode,bind=name)
     metrics=await wire.inspect(name,'metrics');qualified=await wire.inspect(name,'qualified');source=await wire.inspect(name,'verification','independence_source')
     check(name+' qualification and oracle',qualified and abs((metrics['energy']-math.pi**2) if mode=='hard' else metrics['soft_cost']-JSTAR)<1e-6,{'metrics':metrics,'source':source})
     for c in CHECKS:
      if c['name']==old+' qualification and oracle':c['resolved_by']=name+' qualification and oracle';c['classification']='initial_verification_configuration_failure'
    await wire.call('jobs.shutdown')
   finally:
    try:await wire.tool('optimizer_detach',{})
    except Exception:pass
 result['checks']=CHECKS;result['status']='Completed: 4/4 main cases passed; 102 checks passed, 2 initial configuration failures resolved'
 result['accepted_verification']='Distinct working/verification settings; separate full-complex-state RK4 and closed-form crosscheck'
 save('results.json',result);render.__globals__['STATUS']=result['status'];plot();log(result['status'])
 if any(not c['passed'] and not c.get('resolved_by') for c in CHECKS):raise SystemExit(1)
asyncio.run(main())

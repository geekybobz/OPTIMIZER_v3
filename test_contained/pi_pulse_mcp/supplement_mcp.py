"""Remaining planned mesh, recovered-control warm start and singular-start checks."""
import asyncio,os,json,pathlib,runpy
D=runpy.run_path('/Users/billabobz/OPTIMIZER/test_contained/pi_pulse_mcp/run_mcp.py',run_name='campaign_helpers')
globals().update({k:v for k,v in D.items() if not k.startswith('__')})
result=json.loads((OUT/'results.json').read_text());CHECKS=result['checks'];CASES=result['cases']
TRAJECTORIES=json.loads((OUT/'trajectories.json').read_text())
for f in [log,check,render,plot]:f.__globals__.update(CHECKS=CHECKS,CASES=CASES,TRAJECTORIES=TRAJECTORIES,STATUS='Final planned checks')
async def main():
 env={**os.environ,'TMPDIR':'/private/tmp'}
 async with stdio_client(StdioServerParameters(command='/Users/billabobz/.local/bin/optimizer-mcp',cwd=str(WORK),env=env)) as (r,w):
  async with ClientSession(r,w) as client:
   await client.initialize();wire=Wire(client);a=await wire.tool('optimizer_attach',{'path':str(EXPERIMENT),'start_coordinator':True});assert a['status']=='ok',a
   try:
    state=await wire.call('session.state');heads={name:state['state_after']['contexts'][name]['head'] for name in CASES}
    for mode in ['soft','hard']:
     for steps in [500,2000,4000]:
      name=f's_{mode}_steps_{steps}'
      await wire.call('shooting.solve_shooting',{'initial':ref(heads['s_'+mode]),'options':{'steps':steps},'verification_options':{'steps':4000},'maxiter':10,'tolerance':1e-10},context='s_'+mode,bind=name)
      metrics=await wire.inspect(name,'metrics');qualified=await wire.inspect(name,'qualified')
      check(name+' qualification and oracle',qualified and abs((metrics['energy']-math.pi**2) if mode=='hard' else metrics['soft_cost']-JSTAR)<1e-6,metrics)
     name='recovered_'+mode+'_warmstart'
     await wire.call('optimizers.projected_gradient' if mode=='hard' else 'optimizers.line_search',{'controls':ref(mode+'_recovered','controls'),'maxiter':2,'step_size':250 if mode=='hard' else 500},context='g_'+mode,bind=name)
     metrics=await wire.inspect(name,'metrics');check(name+' remains minimum',abs((metrics['energy']-math.pi**2) if mode=='hard' else metrics['J']-JSTAR)<1e-6,metrics)
    await wire.load('hard_zero','g','hard');await wire.call('guesses.constant_guess',{'value':0.},context='hard_zero',bind='zero_control');await wire.use('hard_zero',ref('zero_control'))
    rr=await wire.call('optimizers.projected_gradient',{'maxiter':3,'step_size':250},context='hard_zero',bind='zero_hard_trial')
    m=await wire.inspect('zero_hard_trial','metrics');stop=await wire.inspect('zero_hard_trial','stop_reason')
    check('singular hard zero start not accepted as feasible',abs(m['c0_re'])>.9 and rr['state_before']['contexts']['hard_zero']['head']==rr['state_after']['contexts']['hard_zero']['head'],{'metrics':m,'stop_reason':stop})
    await wire.call('jobs.shutdown')
   finally:
    try:await wire.tool('optimizer_detach',{})
    except Exception:pass
 result['checks']=CHECKS;result['status']=f'Completed: {sum(c["passed"] for c in CHECKS)}/{len(CHECKS)} checks passed';save('results.json',result)
 render.__globals__['STATUS']=result['status'];plot();log(result['status'])
 if not all(c['passed'] for c in CHECKS):raise SystemExit(1)
asyncio.run(main())

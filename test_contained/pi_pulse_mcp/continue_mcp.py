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
    STATUS='Control exchange and mesh checks'
    await wire.call('utils.uniform_control_grid',{'start':0,'stop':1,'count':1000,'placement':'cell_centers'},bind='grid1000')
    for mode in ['soft','hard']:
     s,g='s_'+mode,'g_'+mode;best=CASES[s]['binding'];await wire.call('utils.recover_shooting_controls',{'variables':ref(heads[s]),'grid':ref('grid1000')},context=s,bind=mode+'_recovered')
     await wire.call('base.evaluate',{'controls':ref(mode+'_recovered','controls')},context=g,bind=mode+'_replayed');metrics=await wire.inspect(mode+'_replayed','metrics');check(mode+' shooting -> gradient physical replay',abs(metrics['energy']-CASES[s]['E'])<1e-9 and abs(metrics['fidelity']-CASES[s]['F'])<1e-9,metrics)
     # Explicit mean-angle projection supplies a seed; never claims general inverse.
     mean=CASES[g]['theta']/T;seed=await wire.shoot_seed(s,mode+'_from_gradient',mean)
     await wire.call('shooting.solve_shooting',{'initial':seed},context=s,bind=mode+'_exchange_root');metrics=await wire.inspect(mode+'_exchange_root','metrics');check(mode+' gradient mean -> shooting',abs(metrics['soft_cost']-JSTAR)<1e-6 if mode=='soft' else abs(metrics['energy']-math.pi**2)<1e-6,metrics)
     for count in [500,2000,4000]:
      context=f'{g}_{count}';await wire.load(context,'g',mode,N=count);await wire.call('base.control_spec',context=context,bind=context+'_spec');await wire.call('utils.uniform_control_grid',{'start':0,'stop':1,'count':count,'placement':'cell_centers'},bind=context+'_grid')
      await wire.call('utils.resample_controls',{'controls':ref(mode+'_recovered','controls'),'source_grid':ref('grid1000'),'target_grid':ref(context+'_grid'),'target':ref(context+'_spec'),'extrapolation':'hold'},bind=context+'_mapped')
      await wire.call('base.evaluate',{'controls':ref(context+'_mapped','controls')},context=context,bind=context+'_eval');metrics=await wire.inspect(context+'_eval','metrics');check(context+' resample',abs(metrics['energy']-CASES[s]['E'])<1e-8 and abs(metrics['fidelity']-CASES[s]['F'])<1e-9,metrics)
    STATUS='Duration scaling'
    for mode in ['hard','soft']:
     source='g_'+mode;source_control=ref(heads[source])
     for duration in [.5,2.]:
      tag=str(duration).replace('.','_');context=f'{source}_T{tag}';await wire.load(context,'g',mode,T=duration)
      await wire.call('utils.scale_controls',{'controls':source_control,'factors':1/duration},bind=context+'_scaled')
      values=np.asarray(await wire.inspect(context+'_scaled','controls'));new=await wire.controls(context,context+'_physical',values)
      await wire.call('base.evaluate',{'controls':new},context=context,bind=context+'_eval');metrics=await wire.inspect(context+'_eval','metrics');check(context+' F and energy scaling',abs(metrics['fidelity']-CASES[source]['F'])<1e-9 and abs(metrics['energy']-CASES[source]['E']/duration)<1e-8,metrics)
      if mode=='soft':
       matched=context+'_matched';await wire.load(matched,'g',mode,T=duration,alpha=ALPHA*duration);await wire.call('base.evaluate',{'controls':new},context=matched,bind=matched+'_eval');metrics=await wire.inspect(matched+'_eval','metrics');check(matched+' tradeoff',abs(metrics['soft_cost']-CASES[source]['J'])<1e-9,metrics)
       # At fixed alpha, optimize the warm start; scaled control is not presumed optimal.
       await wire.use(context,new);await wire.call('optimizers.line_search',{'variant':'armijo','maxiter':100,'step_size':500,'normalize':False,'gradient_tolerance':1e-12},context=context,bind=context+'_optimized');metrics=await wire.inspect(context+'_optimized','metrics');_,target=soft_oracle(ALPHA,duration);check(context+' fixed-alpha reoptimization',abs(metrics['J']-target)<1e-6,metrics['J'],target)
    # Structured failure paths, checkpoint, detach/reconnect and disk physical data.
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

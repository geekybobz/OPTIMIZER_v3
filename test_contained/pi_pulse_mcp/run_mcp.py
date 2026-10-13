"""Four-case campaign through the installed MCP stdio server; no local solver calls."""
from __future__ import annotations
import asyncio,datetime,hashlib,html,json,math,os,pathlib,sys,time,traceback
import numpy as np
import optimizer
from mcp import ClientSession,StdioServerParameters
from mcp.client.stdio import stdio_client
from optimizer.utils.resources import available_cpus,cpu_budget

OUT=pathlib.Path('/Users/billabobz/OPTIMIZER/test_contained/pi_pulse_mcp/runs/20261013')
WORK=pathlib.Path('/tmp/optimizer-pi-pulse-mcp-20261013')
EXPERIMENT=WORK/'.optimizer'/os.getenv('PI_MCP_EXPERIMENT','pi_mcp_qualified_20261013')
ALPHA=.01; N=1000; T=1.; DT=T/N
EVENTS=[];CHECKS=[];CASES={};TRAJECTORIES={};STATUS='Connecting'
START=time.monotonic()

def save(name,value):
 p=OUT/name;tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def render():
 rows=''.join('<tr>'+''.join(f'<td>{html.escape(str(x))}</td>' for x in [name,data.get('stage',''),data.get('F',''),data.get('E',''),data.get('J',''),data.get('density_norm','')])+'</tr>' for name,data in CASES.items())
 events='\n'.join(html.escape(e) for e in EVENTS[-35:])
 plot='<img src="plots.png" style="width:100%" alt="Measured controls, populations, convergence and residuals">' if (OUT/'plots.png').exists() else ''
 content=f'''<!doctype html><html><head><meta charset="utf-8">{'' if STATUS.startswith('Completed') else '<meta http-equiv="refresh" content="3">'}<title>π-pulse MCP progress</title><style>body{{font:16px system-ui;background:#10161f;color:#e5edf6;margin:30px auto;max-width:1180px;padding:0 22px}}h1{{font-size:27px}}small{{color:#acbccd}}table{{border-collapse:collapse;width:100%;margin:24px 0}}td,th{{padding:12px;text-align:left;border-bottom:1px solid #334151}}pre{{background:#182331;padding:20px;white-space:pre-wrap;font:13px monospace}}img{{background:white}}a{{color:#9acbff}}</style></head><body><h1>π-pulse • installed MCP experiment</h1><p>{html.escape(STATUS)} · elapsed {time.monotonic()-START:.1f}s</p><small>Release {optimizer.__version__} · 1000 intervals · dt=.001 · sequential execution · native threads 1 · CPU ceiling {cpu_budget()}<br>Transport: official MCP client → global optimizer-mcp → installed coordinator. This page shows actual requests/results.</small><table><tr><th>Case</th><th>Stage</th><th>Fidelity</th><th>Energy</th><th>Soft cost</th><th>Stationarity density</th></tr>{rows}</table>{plot}<h2>Live requests and checks</h2><pre>{events}</pre><p><a href="results.json">Structured results</a> · <a href="progress.log">Complete progress</a></p></body></html>'''
 tmp=OUT/'index.html.tmp';tmp.write_text(content);tmp.replace(OUT/'index.html')
def log(message):
 stamp=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds');line=f'{stamp} {message}'
 EVENTS.append(line);print(line,flush=True)
 with (OUT/'progress.log').open('a') as f:f.write(line+'\n')
 render()
def check(name,ok,observed=None,expected=None):
 CHECKS.append(dict(name=name,passed=bool(ok),observed=observed,expected=expected));log(f'{"PASS" if ok else "FAIL"} {name}: {observed}')
 save('checks.json',CHECKS)
def ref(name,*path):return {'$ref':{'object_id':name,'path':list(path)}}
def soft_oracle(alpha=ALPHA,T=1):
 lo,hi=math.pi/2,math.pi
 for _ in range(100):
  mid=(lo+hi)/2
  if math.sin(mid)>4*alpha*mid/T:lo=mid
  else:hi=mid
 theta=(lo+hi)/2;return theta,1-math.sin(theta/2)**2+alpha*theta**2/T
THETA,JSTAR=soft_oracle()
def replay(u,T=1,substeps=4):
 u=np.asarray(u);h=T/(len(u)*substeps);psi=np.array([1+0j,0j]);pop=[0.]
 for value in u:
  def rhs(x):return -.5j*value*x[::-1]
  for _ in range(substeps):
   a=rhs(psi);b=rhs(psi+h*a/2);c=rhs(psi+h*b/2);d=rhs(psi+h*c);psi+=h*(a+2*b+2*c+d)/6
  pop.append(float(abs(psi[1])**2))
 return dict(F=float(abs(psi[1])**2),c0=float(psi[0].real),norm_error=abs(float(np.vdot(psi,psi).real)-1),pop=pop)
def facts(u,hard=False,T=1,alpha=ALPHA):
 u=np.asarray(u);dt=T/len(u);theta=dt*u.sum();F=float(np.sin(theta/2)**2);E=float(dt*(u*u).sum());density=2*(u-u.mean()) if hard else 2*alpha*u-.5*np.sin(theta)
 return dict(F=F,E=E,J=1-F+alpha*E,theta=float(theta),density_norm=float(np.sqrt(dt*(density*density).sum())),flatness=float(u.std()))
def plot():
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,axes=plt.subplots(2,2,figsize=(12,7))
 for name,data in TRAJECTORIES.items():
  axes[0,0].plot((np.arange(len(data['u']))+.5)/len(data['u']),data['u'],label=name)
  axes[0,1].plot(np.linspace(0,1,len(data['pop'])),data['pop'],label=name)
 for name,data in CASES.items():
  if data.get('samples'):
   x=np.arange(len(data['samples']));ys=[r['J'] if 'soft' in name else r['E'] for r in data['samples']]
   axes[1,0].plot(x,ys,'o-',label=name)
   axes[1,1].semilogy(x,[max(r.get('density_norm',r.get('residual',0)),1e-16) for r in data['samples']],'o-',label=name)
 for ax,title,xlabel,ylabel in zip(axes.flat,['Recovered control','Independent state replay','Inspected chunk objectives','Stationarity or shooting residual'],['t','t','Chunk / shooting seed','Chunk / shooting seed'],['u(t)','Population |1⟩','J (soft) or E (hard)','Residual norm']):
  ax.set(title=title,xlabel=xlabel,ylabel=ylabel);ax.grid(alpha=.2);ax.legend(fontsize=8)
 fig.tight_layout();fig.savefig(OUT/'plots.png',dpi=140);plt.close(fig)
 render()

class Wire:
 def __init__(self,client):self.client=client;self.count=0
 async def tool(self,name,args):
  self.count+=1;start=time.monotonic();log(f'MCP {self.count}: {name} '+(str(args.get('operation','')) or str(args.get('reference',''))[:90]))
  result=await self.client.call_tool(name,args);data=result.structuredContent
  if data is None:raise RuntimeError(f'No structured reply: {result}')
  with (OUT/'requests.jsonl').open('a') as f:f.write(json.dumps(dict(run=EXPERIMENT.name,number=self.count,tool=name,arguments=args,response=data,seconds=time.monotonic()-start),allow_nan=False)+'\n')
  return data
 async def call(self,operation,arguments=None,context=None,bind=None,activate=False,error=None):
  args={'operation':operation,'arguments':arguments or {}}
  if not operation.startswith(('session.','project.','checkpoints.','storage.','jobs.')):args['activate']=activate
  if context:args['context']=context
  if bind:args['bind']=bind
  before=(await self.call('session.state'))['state_after']['contexts'] if error else None
  data=await self.tool('optimizer_execute',args)
  if error:
   check(f'error {operation} -> {error}',data['status']=='error' and data['error']['code']==error,data.get('error'))
   after=(await self.call('session.state'))['state_after']['contexts']
   check('failed request preserves active state',before==after)
  elif data['status']!='ok':raise RuntimeError(f'{operation}: {data.get("error")}')
  return data
 async def inspect(self,name,*path,limit=1024):
  items=[];offset=0
  while True:
   data=await self.tool('optimizer_inspect',{'reference':{'object_id':name,'path':list(path)},'detail':'values','offset':offset,'limit':limit})
   if data['status']!='ok':raise RuntimeError(data)
   summary=data['metadata']['summary']
   if 'value' in summary:return summary['value']
   part=summary.get('items',summary)
   if isinstance(part,dict):return part
   items.extend(part)
   if summary.get('next_offset') is None:return items
   offset=summary['next_offset']
 async def load(self,name,family,mode,N=1000,T=1,alpha=ALPHA,steps=1000):
  primary={'N':N,'T':T} if family=='g' else {'T':T,'steps':steps}
  return await self.call('session.load',{'name':name,'factory':str(WORK/'factories'/('pi_pulse_olgs.py' if family=='g' else 'pi_pulse_olss.py')),'factory_name':mode,'parameters':{'config':{'primary':primary,'secondary':{'alpha':alpha}}},'checkpoint':{'stateless':True,'dependencies':[]}},activate=False)
 async def controls(self,context,name,u):
  await self.call('base.control_spec',context=context,bind=name+'_spec')
  await self.call('base.controls',{'spec':ref(name+'_spec'),'values':[np.asarray(u).tolist()]},bind=name)
  return ref(name)
 async def use(self,context,value):return await self.call('session.use',{'value':value},context=context)
 async def shoot_seed(self,context,name,p):
  await self.call('system_olss.validate_shooting_spec',context=context,bind=name+'_spec')
  await self.call('system_olss.ShootingVariables',{'spec':ref(name+'_spec'),'values':[float(p)]},bind=name)
  return ref(name)

async def campaign():
 global STATUS
 OUT.mkdir(parents=True,exist_ok=True);STATUS='Attaching experiment';log('Beginning four-case campaign')
 receipt=json.loads(pathlib.Path('/Users/billabobz/.local/share/optimizer/install.json').read_text())
 check('installed release 4.0.6',optimizer.__version__=='4.0.6',optimizer.__file__)
 check('installed source isolated','site-packages/optimizer' in optimizer.__file__)
 save('provenance.json',{'install':receipt,'package':optimizer.__file__,'available_cpus':available_cpus(),'cpu_budget':cpu_budget(),'factory_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (WORK/'factories').glob('pi_pulse_*.py')},'transport':'official MCP client stdio','assistant_native_tools_exposed':False,'T':T,'N':N,'dt':DT,'soft_oracle':{'theta':THETA,'J':JSTAR},'thread_limits':{k:os.getenv(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')}})
 env={**os.environ,'PYTHONPATH':str(WORK),'TMPDIR':'/private/tmp'}
 params=StdioServerParameters(command='/Users/billabobz/.local/bin/optimizer-mcp',cwd=str(WORK),env=env)
 async with stdio_client(params) as (read,write):
  async with ClientSession(read,write) as client:
   init=await client.initialize();tools=await client.list_tools();check('six MCP tools and server version',len(tools.tools)==6 and init.serverInfo.version=='4.0.6')
   wire=Wire(client);attached=await wire.tool('optimizer_attach',{'path':str(EXPERIMENT),'start_coordinator':True});assert attached['status']=='ok',attached
   try:
    await wire.call('storage.configure',{'keep_recent':256})
    for op in ['optimizers.adam','optimizers.line_search','optimizers.projected_gradient','shooting.solve_shooting','base.controls','session.save','utils.recover_shooting_controls']:
     info=await wire.tool('optimizer_info',{'operation':op});assert info['status']=='ok',info
    # Define project systems after the action derivation in MCP_TEST_PLAN.md.
    for family,mode,name in [('g','soft','g_soft'),('g','hard','g_hard'),('s','soft','s_soft'),('s','hard','s_hard')]:await wire.load(name,family,mode)
    for mode in ['soft','hard']:
     context='g_'+mode;u=.5+.15*np.sin(2*np.pi*(np.arange(N)+.5)/N) if mode=='soft' else 2.5+.5*np.sin(2*np.pi*(np.arange(N)+.5)/N)
     value=await wire.controls(context,context+'_seed',u);await wire.use(context,value)
     for eps in [1e-4,1e-5]:
      r=await wire.call('utils.verify_gradient',{'eps':eps,'directions':5,'rtol':1e-6,'atol':1e-8},context=context,bind=f'{context}_gradient_{eps}')
      report=await wire.inspect(f'{context}_gradient_{eps}');check(f'{context} derivative eps={eps}',report['passed'],report)
      if mode=='hard':
       await wire.call('utils.verify_jacobian',{'eps':eps,'directions':5,'rtol':1e-6,'atol':1e-8},context=context,bind=f'{context}_jac_{eps}');report=await wire.inspect(f'{context}_jac_{eps}');check(f'hard constraint derivative eps={eps}',report['passed'],report)
     await wire.call('base.gradient',{},context=context,bind=context+'_gradient')
     actual=np.asarray(await wire.inspect(context+'_gradient'))
     expected=DT*(2*u if mode=='hard' else 2*ALPHA*u-.5*np.sin(DT*u.sum()))
     check(f'{context} exact action derivative',np.max(abs(actual-expected))<=1e-12,float(np.max(abs(actual-expected))))
    # Optimize in short chunks and inspect before explicit acceptance.
    for context in ['g_soft','g_hard']:
     hard=context.endswith('hard');STATUS='Optimizing '+context
     CASES[context]={'stage':'initial','samples':[]}
     for chunk in range(24):
      if not hard and chunk<2:method='adam';options={'maxiter':10,'step_size':.08,'gradient_tolerance':1e-12}
      elif not hard:method='line_search';options={'variant':'armijo','maxiter':100,'step_size':500.,'normalize':False,'gradient_tolerance':1e-12}
      else:method='projected_gradient';options={'maxiter':5,'step_size':250.,'feasibility_tolerance':1e-10,'stationarity_tolerance':1e-12,'damping':1e-12,'restoration_maxiter':8}
      name=f'{context}_chunk_{chunk}';r=await wire.call('optimizers.'+method,options,context=context,bind=name,activate=False)
      check(name+' candidate leaves head unchanged',r['state_before']['contexts'][context]['head']==r['state_after']['contexts'][context]['head'])
      metrics=await wire.inspect(name,'metrics');u=np.asarray(await wire.inspect(name,'controls'));record=await wire.inspect(name,'run_record','rows')
      f=facts(u,hard);samples=CASES[context]['samples'];samples.append(f)
      CASES[context]={**f,'samples':samples,'stage':f'{method} chunk {chunk+1}','binding':name,'native_metrics':metrics}
      save(name+'.json',{'metrics':metrics,'facts':f,'record':record});save(name+'-controls.json',u.tolist())
      physical=replay(u);TRAJECTORIES[context]={'u':u.tolist(),'pop':physical['pop']};plot();log(f'{context}: F={f["F"]:.12f} E={f["E"]:.12f} J={f["J"]:.12f} density={f["density_norm"]:.3g}')
      # Explicitly select inspected candidate; finite data, and hard feasibility.
      if hard and abs(metrics['c0_re'])>1e-8:raise RuntimeError('Hard candidate is infeasible')
      await wire.use(context,ref(name,'controls'))
      if f['density_norm']<1e-6 and abs((f['E']-math.pi**2) if hard else f['J']-JSTAR)<1e-6:break
     target=math.pi**2 if hard else JSTAR;observed=f['E'] if hard else f['J']
     check(context+' minimum cost',abs(observed-target)<1e-6,observed,target)
     check(context+' continuum stationarity',f['density_norm']<1e-6,f['density_norm'])
     check(context+' independent fidelity',abs(physical['F']-f['F'])<1e-9,physical['F'],f['F'])
     check(context+' independent norm',physical['norm_error']<1e-9,physical['norm_error'])
     if hard:check('hard independent endpoint',abs(physical['c0'])<1e-8,physical['c0'])
     CASES[context]['stage']='qualified' if all(x['passed'] for x in CHECKS if x['name'].startswith(context)) else 'check failures'
    for context in ['s_soft','s_hard']:
     hard=context.endswith('hard');STATUS='Shooting '+context;roots=[]
     for j,p in enumerate([2.5,-2.5,3*math.pi] if hard else [2.5,-2.5,0.]):
      seed=await wire.shoot_seed(context,f'{context}_seed_{j}',p)
      for eps in [1e-4,1e-5]:
       name=f'{context}_jac_{j}_{eps}';await wire.call('shooting.verify_shooting_jacobian',{'variables':seed,'eps':eps,'rtol':1e-6,'atol':1e-8},context=context,bind=name);report=await wire.inspect(name);check(name,report['passed'],report)
      name=f'{context}_root_{j}';r=await wire.call('shooting.solve_shooting',{'initial':seed,'maxiter':30,'method':'newton','tolerance':1e-11,'verification_tolerance':1e-8},context=context,bind=name)
      metrics=await wire.inspect(name,'metrics');pv=float((await wire.inspect(name,'variables','values'))[0]);qualified=await wire.inspect(name,'qualified')
      independent=await wire.inspect(name,'verification','independence_source');record=await wire.inspect(name,'run_record','rows')
      check(name+' root qualification',qualified,metrics)
      f=facts(np.full(N,pv),hard);roots.append({'binding':name,'p0':pv,**f,'qualified':qualified,'metrics':metrics,'verification_source':independent})
      save(name+'.json',{'root':roots[-1],'record':record})
      log(f'{context} seed {p:.6g}: p0={pv:.12f} F={f["F"]:.12f} E={f["E"]:.12f} J={f["J"]:.12f}')
     best=min(roots,key=lambda x:x['E'] if hard else x['J']);await wire.use(context,ref(best['binding'],'variables'))
     CASES[context]={**best,'stage':'qualified','samples':[dict(r,residual=r['metrics']['residual_norm']) for r in roots]}
     u=np.full(N,best['p0']);physical=replay(u);TRAJECTORIES[context]={'u':u.tolist(),'pop':physical['pop']};plot()
     check(context+' minimum branch',abs((best['E']-math.pi**2) if hard else best['J']-JSTAR)<1e-6,best['p0'])
     check(context+' independent fidelity',abs(physical['F']-best['F'])<1e-9,physical['F'])
     if not hard:
      zero=roots[-1];curvature=2*ALPHA-.5*T;check('zero soft root rejected as minimum',abs(zero['p0'])<1e-10 and curvature<0 and zero['J']>best['J'],{'zero_cost':zero['J'],'uniform_curvature':curvature})
     else:check('higher hard branch rejected by energy',roots[-1]['E']>8*best['E'],roots[-1]['E'])
    STATUS='Control exchange and mesh checks'
    await wire.call('utils.uniform_control_grid',{'start':0,'stop':1,'count':1000,'placement':'cell_centers'},bind='grid1000')
    for mode in ['soft','hard']:
     s,g='s_'+mode,'g_'+mode;best=CASES[s]['binding'];await wire.call('utils.recover_shooting_controls',{'variables':ref(best,'variables'),'grid':ref('grid1000')},context=s,bind=mode+'_recovered')
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
     source='g_'+mode;source_control=ref(CASES[source]['binding'],'controls')
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
    state=await wire.call('session.state');heads={name:state['state_after']['contexts'][name]['head'] for name in CASES}
    for name in CASES:
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
    if 'attached' in locals():
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
 save('results.json',{'status':STATUS,'cases':CASES,'checks':CHECKS,'seconds':time.monotonic()-START,'experiment':str(EXPERIMENT),'history_origins':origins})
 plot();log(STATUS)
 if not all(c['passed'] for c in CHECKS):raise SystemExit(1)

if __name__=='__main__':
 try:asyncio.run(campaign())
 except BaseException as exc:
  STATUS='Stopped: '+str(exc);log(STATUS);(OUT/'failure.txt').write_text(traceback.format_exc());raise

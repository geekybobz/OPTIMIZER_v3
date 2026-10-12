"""Optional official-SDK MCP server over bounded local standard I/O."""
from __future__ import annotations
import argparse
from contextlib import asynccontextmanager
import sys
from optimizer.checkpoints.codec import _json
from optimizer.mcp.adapter import _Adapter
from optimizer.operations.contracts import _ServiceError

def _schema(properties,required=()):
    return {'type':'object','properties':properties,'required':list(required),'additionalProperties':False}

_STRING={'type':'string'}
_REF=_schema({'object_id':_STRING,'path':{'type':'array','items':{'anyOf':[{'type':'string','minLength':1,'pattern':'^[^_]'}, {'type':'integer','minimum':0}]},'maxItems':24}},('object_id',))
_REQUEST=_schema({'operation':_STRING,'arguments':{'type':'object'},'context':_STRING,'bind':_STRING,
                 'activate':{'type':'boolean'},'request_id':_STRING,'expected_revision':{'type':'integer','minimum':0},'experiment_id':_STRING},('operation',))
_TOOLS=(
    ('optimizer_attach','Attach an existing experiment. Explicit start_coordinator starts/recover its owner; never silently restarts numerical work.',
     _schema({'path':_STRING,'start_coordinator':{'type':'boolean','default':False},'restore_mode':{'type':'string','enum':['exact','controls_only','variables_only'],'default':'exact'}},('path',))),
    ('optimizer_detach','Detach this client while leaving the experiment and jobs running.',_schema({})),
    ('optimizer_list','List registered operation identities, including unsupported entries and reasons. Paginated; use optimizer_info for signatures.',
     _schema({'group':_STRING,'classification':_STRING,'offset':{'type':'integer','minimum':0},'limit':{'type':'integer','minimum':1,'maximum':50}})),
    ('optimizer_info','Describe one catalog or lifecycle operation, parameters, injected inputs, activation and transport limitations.',_schema({'operation':_STRING},('operation',))),
    ('optimizer_execute','Execute one native operation transaction. arguments accept finite JSON, {"$ref":{"object_id":"name-or-id","path":[]}} and {"$tuple":[]}. Solvers inherit service activation defaults; use activate=false for candidates. Use jobs.submit and short jobs.wait for long work. No shell, CLI text, imports or live callbacks. expected_revision rejects stale decisions.',_REQUEST),
    ('optimizer_inspect','Inspect retained data by reference. Summary is default; values pages flatten numeric arrays and page mappings/sequences. Never autoactivates.',
     _schema({'reference':_REF,'detail':{'type':'string','enum':['summary','values']},'offset':{'type':'integer','minimum':0},'limit':{'type':'integer','minimum':1,'maximum':1024}},('reference',))),
)

@asynccontextmanager
async def _stdio():
    """SDK message streams with explicit line budgets and UTF-8 framing."""
    import anyio
    from mcp.types import JSONRPCMessage
    from mcp.shared.message import SessionMessage
    incoming,receive=anyio.create_memory_object_stream(0)
    send,outgoing=anyio.create_memory_object_stream(0)
    async def read():
        async with incoming:
            while True:
                raw=await anyio.to_thread.run_sync(lambda:sys.stdin.buffer.readline(131073),abandon_on_cancel=True)
                if not raw: break
                if len(raw)>131072 or not raw.endswith(b'\n'):
                    await incoming.send(ValueError('MCP message exceeds 128 KiB or lacks newline framing.'));break
                try:
                    message=JSONRPCMessage.model_validate_json(raw.decode('utf-8',errors='strict'))
                except (ValueError,UnicodeError) as exc:
                    await incoming.send(exc);continue
                await incoming.send(SessionMessage(message))
    async def write():
        async with outgoing:
            async for message in outgoing:
                raw=message.message.model_dump_json(by_alias=True,exclude_none=True).encode()+b'\n'
                if len(raw)>262144: raise ValueError('MCP response exceeds 256 KiB.')
                await anyio.to_thread.run_sync(lambda:(sys.stdout.buffer.write(raw),sys.stdout.buffer.flush()))
    async with anyio.create_task_group() as group:
        group.start_soon(read);group.start_soon(write)
        try: yield receive,send
        finally: group.cancel_scope.cancel()

async def _serve(adapter):
    import anyio
    from mcp.server.lowlevel import Server
    from mcp import types
    server=Server('optimizer',version='4.0.5',instructions='Attach a named experiment, discover operations, execute one native request at a time, inspect references. CLI/MCP share state and history. Detach preserves jobs; jobs.shutdown explicitly stops them.')
    lock=anyio.Lock()
    @server.list_tools()
    async def listing():
        return [types.Tool(name=name,description=description,inputSchema=schema,
                           annotations=types.ToolAnnotations(readOnlyHint=False,destructiveHint=True,openWorldHint=True))
                for name,description,schema in _TOOLS]
    @server.call_tool()
    async def call(name,arguments):
        async with lock:
            try:
                if name=='optimizer_attach': result=await anyio.to_thread.run_sync(lambda:adapter.attach(**arguments))
                elif name=='optimizer_detach': result=await anyio.to_thread.run_sync(adapter.detach)
                elif name in {'optimizer_list','optimizer_info','optimizer_execute','optimizer_inspect'}:
                    result=await anyio.to_thread.run_sync(lambda:adapter.call(name.removeprefix('optimizer_'),arguments))
                else: raise _ServiceError('unknown_operation','Unknown MCP tool.')
            except (OSError,TypeError,ValueError,_ServiceError) as exc:
                result={'status':'error','error':{'code':getattr(exc,'code','invalid_arguments'),'message':str(exc)[:2048]}}
            return types.CallToolResult(content=[types.TextContent(type='text',text=_json(result).decode())],
                                        structuredContent=result,isError=result.get('status')!='ok')
    try:
        async with _stdio() as (read,write):
            await server.run(read,write,server.create_initialization_options())
    finally:
        adapter.close()

def main(argv=None):
    parser=argparse.ArgumentParser(description='Local MCP standard-I/O server for named optimizer experiments.')
    parser.add_argument('--experiment',help='Attach this existing experiment on startup.')
    parser.add_argument('--start-coordinator',action='store_true',help='Explicitly start/recover the selected experiment owner.')
    parser.add_argument('--restore-mode',choices=('exact','controls_only','variables_only'),default='exact')
    options=parser.parse_args(argv)
    if options.start_coordinator and not options.experiment: parser.error('--start-coordinator requires --experiment')
    try: import anyio; import mcp
    except ImportError:
        print('Install optimizer-v4[mcp] to enable the MCP server.',file=sys.stderr);return 2
    adapter=_Adapter()
    try:
        if options.experiment:
            result=adapter.attach(options.experiment,start_coordinator=options.start_coordinator,restore_mode=options.restore_mode)
            if result['status']!='ok': raise _ServiceError(result['error']['code'],result['error']['message'])
        anyio.run(_serve,adapter)
    except (OSError,ValueError,_ServiceError) as exc:
        print(str(exc),file=sys.stderr);return 2
    finally: adapter.close()
    return 0

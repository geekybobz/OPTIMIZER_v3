---
role: topic
parent: ../README.md
summary: "Learn the installed optimizer through a small, reproducible terminal example."
read_when: "Start here for your first installation and interactive run."
status: active
reviewed: 2026-10-13
---
# Getting started

Install the current approved release using [Installation](INSTALLATION.md), then
open a terminal in your problem folder and run `optimizer`. Enter structured
Python-shaped calls ending in `;`. You launch the interface once per session.
The supported namespace is lowercase `opt`; arbitrary Python is not accepted.

A project supplies its equations as an OLGS gradient system or OLSS shooting
system. The optimizer supplies numerical methods and inspection tools. Begin
with the small quadratic example before adapting the workflow to your own theory.

## Details

### First gradient experiment

From the cloned repository, copy the self-contained quadratic test system into
an independent temporary project:

```bash
OPTIMIZER_DEMO_DIR="$(mktemp -d)"
cp tests/fixtures/quadratic_system.py "$OPTIMIZER_DEMO_DIR/system.py"
cd "$OPTIMIZER_DEMO_DIR"
optimizer
```

For your own equations, `optimizer-system --family olgs system.py` creates an
unfinished scaffold. Complete its TODOs before running numerical methods; the
scaffold does not invent your physical model.

Inside the interface:

```python
opt.project.init(name="first-run", root=".");
opt.session.load(name="main", factory="./system.py",
    checkpoint={"stateless":True, "dependencies":[]});
u = opt.guesses.constant_guess(value=0.1);
opt.session.use(u);
r = opt.adam(maxiter=5, step_size=0.01);
r.metrics;
base = opt.session.save(name="baseline");
opt.session.exit();
```

`stateless` is a promise about your factory and all its local dependencies; it
is appropriate for the quadratic example. Declare additional files when your
own system uses them. Loading a factory executes trusted project Python.
Results expose fields for inspection; a metric's physical meaning comes from
the system you loaded. A successful solver call is not a global optimality proof.

Restart `optimizer`, then reopen:

```python
opt.project.open(path=".optimizer/first-run");
opt.session.state();
opt.checkpoints.list();
opt.session.restore("baseline");
opt.session.exit();
```

The hidden `.optimizer/first-run/` folder holds this independent experiment.
Commands and outcomes are retained in its `history.jsonl`; checkpoints hold
numerical state. Keep factory inputs and matching dependencies for reproduction.
See [Persistence](../optimizer/persistence/PERSISTENCE.md) for exact versus
physical restoration and disk quotas.

### Choose an interface

| Interface | You provide | Execution |
|---|---|---|
| Python | A Python program importing `optimizer` | Direct numerical calls; no automatic command history |
| CLI | Semicolon-terminated calls in `optimizer`, or an `.opt` script | Validated operation service; optional durable experiment |
| MCP | Typed tool requests from an MCP client | Same service and experiment coordinator; shared history |

MCP means Model Context Protocol. The client (for example an assistant app)
starts `optimizer-mcp` as a local server process. The server advertises tools;
the client calls a tool with JSON arguments. `optimizer_execute` requests one
optimizer operation, such as `adam`, rather than sending shell text. The service
validates it and the coordinator owns shared state. Numerical systems and
methods remain the bottom layer.

```text
human or assistant -> CLI / MCP client
                          -> terminal adapter / MCP server
                              -> operation service + experiment coordinator
                                  -> OLGS / OLSS numerical tools and project system
```

The MCP dependency is optional for Python and CLI users. The complete installation
in this guide includes it. Connecting an MCP client is a separate client setting;
installation alone does not register a server in every assistant app. See the
[MCP interface](../optimizer/mcp/MCP.md) for configuration, its six tools,
reference envelopes, limits and request examples.

Continue with [Workflows](WORKFLOWS.md), or consult the
[Structured terminal](../optimizer/cli/TERMINAL.md) for the exact language.

---

[⌂ Home](../README.md) · [Next →](WORKFLOWS.md)

---
role: deep
parent: ../../README.md
summary: Structured terminal syntax, launch modes, retained names, display and foreground lifetime.
read_when: Read when running optimizer interactively or writing a command script.
---

# Structured optimizer terminal

Open `optimizer` once, then enter Python-shaped commands ending in `;`.
The terminal sends registered operations to the existing
[operation service](../operations/SERVICE.md). OLGS and OLSS retain their own
contracts and numerical implementations.

## Launch and lifetime

```bash
optimizer
optimizer --command ':help adam'
optimizer --script workflow.opt --json
python -B -m optimizer.cli
```

At the prompt, `:help` lists the commands, `:help <name>` shows one command and
`:exit` leaves; Ctrl-D also leaves. A colon command is a whole line and needs
no `;`. It runs as its `opt.*` equivalent (`opt.help();`, `opt.help("adam");`,
`opt.session.exit();`), which history records. Bare `exit`, `quit` and `help`
get a hint pointing to the colon commands and are not submitted.

`optimizer` is the console entry point after installation from this source.
`python -m optimizer.cli` works from the V4 checkout using the configured
`optimizer_v4` Python. An older installed release does not acquire the new entry
point until it is updated. One-shot and script modes start a fresh session:
load a system and perform work in the same submission/file.

The terminal defaults to foreground RAM. Select an independent durable experiment
with `opt.project.init(name="experiment");` or `opt.project.open(path=...);`.
[Persistent experiments](../persistence/PERSISTENCE.md) owns save/reopen/fork,
typed checkpoints, restore/undo/redo, exact-continuation requirements and cleanup.
Exit/EOF preserves an open experiment's committed data and releases its lease.
[Command history](../history/HISTORY.md) records frontend statements and outcomes.
Ordinary Python receives no automatic history writer. [Managed jobs](../jobs/JOBS.md)
support bounded parallel runs, persistent owners and live pause/resume. The
[MCP interface](../mcp/MCP.md) shares that owner and history through typed requests.

## Calls and retained objects

```python
opt.session.load(name="main", factory="./system.py", parameters={"N": 1001});
u0 = opt.guesses.random_fourier_guess(seed=7, amplitude=0.1);
opt.session.use(u0);
r = opt.adam(maxiter=20, step_size=0.01);
r.metrics;
saved = r.state;
trial = opt.lbfgs(maxiter=5, activate=False);
trial.metrics;
opt.session.use(trial);
opt.session.exit();
```

`factory` is an explicit trusted Python source path; `factory_name="system"`
selects its exported factory. `parameters` supplies factory arguments, including
their recorded effective defaults. Loading executes project Python and is not a
sandbox. Select independent contexts with `opt.session.select("main");`.
Change secondary options through `opt.session.update(updates={...});`.

The selected context supplies the native `system` parameter. Other positional
arguments retain their native order; keyword spelling is the native API's.
For example `opt.adam(u0, maxiter=5);` explicitly starts from `u0`.
`context="main"`, `activate=False` and `bind="name"` are service options, not
numerical parameters. Use either assignment or `bind`, never both.

In RAM-only sessions, assignments protect retained object versions within the byte/object
quotas. `old = r;` protects the full old result when `r` is rebound.
`saved = r.state;` retains an owned state field with its original
context/configuration/method provenance. It does not keep every other field of
the original result resident. Release a name with `opt.session.release("old");`.
In durable experiments, pins protect disk payloads; names can become explicit
`payload_pruned` tombstones after retention or cleanup. Binding storage is included in retained-data accounting; a quota error leaves
the previous head and names intact. These quotas do not measure process RAM.

Same-method implicit calls continue compatible native state. A retained state
alias can also be supplied explicitly: `opt.adam(state=saved, maxiter=5);`.
`opt.session.use(r.state);` activates compatible state for implicit continuation.
Method/configuration changes reject incompatible full state. Passing
`r.controls` or OLSS `r.variables` starts from physical data and resets method
memory. Transforms, diagnostics and candidates leave heads unchanged unless
explicitly activated. Complex reports require selecting a numerical member.

## Language and validation

Supported statements are registered calls, simple-name assignments and data
inspection. Arguments support finite numbers, literal complex numbers such as
`2-3j`, strings, booleans, `None`, lists, tuples and dictionaries. Nested registered
constructors work, for example
`step_size=opt.schedules.ConstantSchedule(value=0.01)`.

Inspect public dataclass fields, mapping keys and nonnegative list/tuple/array
indices. Controls expose declared channels: `u0["ux"][0];` when `ux` exists.
No channel names/count are assumed by the parser. Arrays display shape and dtype.
Private names/attributes, slices, arbitrary methods, generic arithmetic, imports,
loops, comprehensions, definitions, lambdas and unpacking are rejected. `opt` is
reserved. Session load/select/update/release operations run without assignment;
inspect `opt.session.state();` afterward. `session.use` may be assigned.

Semicolons inside strings/comments do not end statements. Calls may span lines
inside brackets. A submission is limited to 65,536 UTF-8 bytes, 8,192 AST nodes,
32 expression nesting levels and 128 registered calls. Command scripts use this
same restricted grammar, not general Python. The whole submitted batch is
syntax-validated before execution. Runtime failure stops subsequent statements;
earlier completed statements keep their committed transitions. Nested calls have
explicit responses but share the parent statement transaction and history pair.

Live callbacks must already be registered by trusted Python through
`service.register_reference("step", callback)` when embedding the coordinator;
the command language cannot define or import arbitrary callback objects.

## Help, discovery and output

```python
opt.help();
opt.info("adam");
opt.list(group="optimizers", classification="tool");
opt.inspect(r);
opt.session.discover(root=".", depth=3);
```

Tool discovery/classification/signatures come from the existing catalog/service.
`info` reports unavailable reasons, and unavailable entries cannot execute.
Tab completion suggests supported operations and retained names when standard
`readline` is available. `info`/`help` provide signatures; completion does not
insert argument values. A usable prompt remains when readline is unavailable.
Filesystem discovery scans at most 512 Python files, at most depth 12, and
parses files at most 256 KiB. It skips hidden/cache/build directories and
symlinks. Candidates are top-level `system` functions or `System` classes;
custom factory names require explicit paths. Candidates are never imported or
automatically selected. Multiple candidates require an explicit path.

On a terminal, results print as panels drawn with `rich`. Gradient
results show iterations, stop reason and metrics with `J` first. Shooting
results keep `qualified`, `verified` and `solver_converged` as separate facts.
Multistart shows its counts and the five best `ranked[i]` roots. Session and
project operations print one confirmation line. Errors print the code, the
message and a hint. A panel's footer names any context whose head changed and
the call's duration.

Piped or redirected output prints one compact text line per operation. `--json`
emits one service response envelope per operation, with compact metadata and no
full numerical array/result expansion. If `rich` is missing the terminal prints
the compact text, and `NO_COLOR` turns colour off. A numerical iteration budget
ending at `maxiter` is a valid result, not a global optimality certificate.

While a call runs longer than half a second, a live line on standard error shows
progress: a bar with the iteration count, the latest objective value and the
remaining time when the budget is known; a spinner otherwise. Multistart counts
seeds and continuation counts steps. A timer keeps the elapsed time moving during
a long single evaluation, and the line clears before the result prints. It
appears only when standard error is a terminal and `--json` is off. It is
display-only: nothing is recorded or written.

Exit codes are `0` success, `2` input/dispatch failure, and `130` interruption.
Interactive Ctrl-C clears incomplete input. During native execution, a valid
native interrupted result may be retained; otherwise the last committed head
remains. This does not provide cooperative pause/resume or background execution.

## Source and verification

[main.py](main.py) owns launch/prompt; [input.py](input.py) frames submissions;
[language.py](language.py) validates the AST and binds service requests;
[display.py](display.py) summarizes data and draws panels; [progress.py](progress.py)
draws the live line; [discovery.py](discovery.py) scans paths.
The recorder in [history/](../history/HISTORY.md) owns permanent statement history.
The coordinator's optional explicit observer remains an informational embedding
hook and cannot replace the recorder's commit protocol.

The [language tests](../../tests/test_cli_language.py) cover native parity,
aliases, continuation, candidates, validation/quotas and isolated OLGS/OLSS
contexts. [Process tests](../../tests/test_cli.py) exercise module/script/pipe
modes, compact JSON, exit codes, EOF and real TTY Ctrl-C. The
[terminal UX tests](../../tests/test_cli_ux.py) cover colon commands, panels,
display fallbacks, the progress reporter and the live line. They use neutral
fixtures; theory-first article benchmarks and MCP remain separately reviewed.

## Connections

- Parent: [V4 entry](../../README.md)
- Related: [Foreground service](../operations/SERVICE.md)
- Related: [API reference](../../API_REFERENCE.md)

---

[⌂ Home](../../README.md)

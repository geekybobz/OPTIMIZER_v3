"""Console entry point and foreground input loop."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from optimizer.cli.display import _Display
from optimizer.cli.input import _MAX_BYTES, _frame
from optimizer.cli.language import _Terminal
from optimizer.cli.progress import _live_line_factory
from optimizer.operations.contracts import _ServiceError


def _read_script(path: str) -> str:
    with Path(path).expanduser().open("rb") as handle:
        payload = handle.read(_MAX_BYTES + 1)
    if len(payload) > _MAX_BYTES:
        raise _ServiceError("input_limit", f"Script exceeds {_MAX_BYTES} UTF-8 bytes.")
    return payload.decode("utf-8")


def _completion(terminal: _Terminal):
    try:
        import readline
    except ImportError:
        return lambda: None
    previous, delimiters = readline.get_completer(), readline.get_completer_delims()
    matches = []
    def complete(text, state):
        if state == 0:
            matches[:] = [name for name in terminal.completions() if name.startswith(text)]
        return matches[state] if state < len(matches) else None
    readline.set_completer_delims(" \t\n(=,;")
    readline.set_completer(complete)
    readline.parse_and_bind("bind ^I rl_complete" if "libedit" in (readline.__doc__ or "") else "tab: complete")
    def restore():
        readline.set_completer(previous)
        readline.set_completer_delims(delimiters)
    return restore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Structured optimizer foreground terminal; end commands with ';'.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--command", help="Run one structured semicolon-terminated submission.")
    mode.add_argument("--script", help="Run a UTF-8 .opt command file (not general Python).")
    parser.add_argument("--json", action="store_true", help="Emit one compact response JSON object per operation.")
    connection=parser.add_mutually_exclusive_group()
    connection.add_argument('--connect',metavar='EXPERIMENT',help='Attach to an existing persistent coordinator.')
    connection.add_argument('--start-coordinator',metavar='EXPERIMENT',help='Explicitly start/recover and attach to a persistent experiment coordinator.')
    parser.add_argument('--restore-mode',choices=('exact','controls_only','variables_only'),default='exact',help='Explicit restore mode when starting a coordinator.')
    options = parser.parse_args(argv)
    display = _Display(json_output=options.json)
    try:
        if options.connect or options.start_coordinator:
            from optimizer.jobs.coordinator import _RemoteTerminal, start
            path=start(options.start_coordinator,options.restore_mode) if options.start_coordinator else options.connect
            terminal=_RemoteTerminal(path,emit=display.emit,progress=None if options.json else _live_line_factory())
        else:
            terminal = _Terminal(emit=display.emit, progress=None if options.json else _live_line_factory())
    except (OSError,_ServiceError) as exc:
        from optimizer.operations.contracts import OperationResponse
        display.emit(OperationResponse('coordinator-connect',None,'error',error={'code':getattr(exc,'code','coordinator_unavailable'),'message':str(exc)}))
        return 2
    display.timings = terminal.timings
    try:
        return _run(terminal,options)
    finally:
        terminal.finish('frontend_close')


def _run(terminal,options):
    if options.command is not None:
        handled = terminal.frontend(options.command)
        return terminal.submit(options.command) if handled is None else handled
    if options.script is not None:
        try:
            source = _read_script(options.script)
        except (OSError, UnicodeError, _ServiceError) as exc:
            from optimizer.operations.contracts import OperationResponse
            response = OperationResponse("script-input", None, "error", error={
                "code": exc.code if isinstance(exc, _ServiceError) else "script_input", "message": str(exc)})
            terminal.emit(response)
            return 2
        return terminal.submit(source)
    interactive = sys.stdin.isatty()
    restore = _completion(terminal) if interactive else lambda: None
    if interactive and not options.json:
        print("OPTIMIZER — structured session; project.init/open enables persistence. "
              "End commands with ';'.  :help for commands, :exit to leave.")
    buffer, status = "", 0
    try:
        while not terminal.closed:
            try:
                line = input(("...> " if buffer else "opt> ") if interactive and not options.json else "")
            except EOFError:
                return terminal.submit(buffer) if buffer.strip() else status
            except KeyboardInterrupt:
                if buffer.strip():
                    terminal.cancel_input(buffer)
                buffer = ""
                if not interactive:
                    return 130
                print("\nInput cleared.", file=sys.stderr)
                continue
            if not buffer:
                handled = terminal.frontend(line)
                if handled is not None:
                    status = handled
                    if status and not interactive:
                        return status
                    continue
            buffer += line + "\n"
            try:
                complete = _frame(buffer, incomplete_ok=True) is not None
            except _ServiceError:
                complete = True  # submit emits the structured error.
            if complete:
                status = terminal.submit(buffer)
                buffer = ""
                if status and not interactive:
                    return status
    finally:
        restore()
    return status

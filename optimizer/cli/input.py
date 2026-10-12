"""Bounded lexical framing of semicolon-terminated submissions."""

from __future__ import annotations

import io
import tokenize

from optimizer.operations.contracts import _ServiceError

_MAX_BYTES = 65536


def _frame(source: str, *, incomplete_ok: bool = False) -> list[str] | None:
    if len(source.encode("utf-8")) > _MAX_BYTES:
        raise _ServiceError("input_limit", f"Submission exceeds {_MAX_BYTES} UTF-8 bytes.")
    lines = source.splitlines(keepends=True)
    offsets, cursor = [], 0
    for line in lines:
        offsets.append(cursor)
        cursor += len(line)
    offsets.append(cursor)
    def offset(position):
        row, col = position
        return offsets[min(row - 1, len(offsets) - 1)] + col
    depth, start, parts, pending = 0, 0, [], False
    try:
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type in {tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE,
                              tokenize.ENDMARKER, tokenize.ENCODING}:
                continue
            if token.type == tokenize.OP:
                if token.string in "([{":
                    depth += 1
                elif token.string in ")]}":
                    depth -= 1
                elif token.string == ";" and depth == 0:
                    end = offset(token.end)
                    if pending:
                        parts.append(source[start:end])
                    start, pending = end, False
                    continue
            if token.type == tokenize.ERRORTOKEN and token.string.isspace():
                continue
            pending = True
    except (tokenize.TokenError, IndentationError) as exc:
        if incomplete_ok and isinstance(exc, tokenize.TokenError) and str(exc).startswith("('EOF"):
            return None
        raise _ServiceError("syntax_error", str(exc)) from exc
    if pending:
        if incomplete_ok:
            return None
        raise _ServiceError("missing_semicolon", "Every command must end with ';'.")
    return parts
